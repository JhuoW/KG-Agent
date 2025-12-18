"""
Training Script for QC-Agent Critic Model with Multi-GPU Support.

Trains the critic to predict whether a reasoning path can lead to the correct answer.
Uses BCE loss on (question, path) pairs with optional margin ranking loss.

Usage:
    # Single GPU
    python Critic/train_critic.py --train_data data/critic_training/RoG-webqsp/train

    # Multi-GPU with accelerate
    accelerate launch --config_file Critic/configs/accelerate_config.yaml \
        Critic/train_critic.py --train_data data/critic_training/RoG-webqsp/train
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import random
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import load_from_disk
from tqdm import tqdm
from typing import Dict, List, Optional
import wandb

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("Warning: accelerate not available, falling back to single GPU mode")

from Critic.models.critic import CriticModel


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader."""
    questions = [item["question"] for item in batch]
    paths = [item["path"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    return {
        "questions": questions,
        "paths": paths,
        "labels": labels,
    }


def create_ranking_pairs(batch: List[Dict]) -> Optional[Dict]:
    """
    Create positive-negative pairs from a batch for ranking loss.

    Returns None if no valid pairs can be created.
    """
    positive_samples = [s for s in batch if s["label"] == 1]
    negative_samples = [s for s in batch if s["label"] == 0]

    if len(positive_samples) == 0 or len(negative_samples) == 0:
        return None

    # Create pairs by matching questions
    pairs = []
    question_to_positives = {}
    question_to_negatives = {}

    for s in positive_samples:
        q = s["question"]
        if q not in question_to_positives:
            question_to_positives[q] = []
        question_to_positives[q].append(s)

    for s in negative_samples:
        q = s["question"]
        if q not in question_to_negatives:
            question_to_negatives[q] = []
        question_to_negatives[q].append(s)

    # Create pairs for each question
    for q in question_to_positives:
        if q not in question_to_negatives:
            continue
        pos_list = question_to_positives[q]
        neg_list = question_to_negatives[q]

        for pos in pos_list:
            neg = random.choice(neg_list)
            pairs.append((pos, neg))

    if len(pairs) == 0:
        return None

    # Unpack pairs
    pos_questions = [p[0]["question"] for p in pairs]
    pos_paths = [p[0]["path"] for p in pairs]
    neg_questions = [p[1]["question"] for p in pairs]
    neg_paths = [p[1]["path"] for p in pairs]

    return {
        "pos_questions": pos_questions,
        "pos_paths": pos_paths,
        "neg_questions": neg_questions,
        "neg_paths": neg_paths,
    }


class CriticTrainer:
    """Trainer for the Critic model with multi-GPU support via accelerate."""

    def __init__(
        self,
        model: CriticModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        accelerator: Optional["Accelerator"] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_ranking_loss: bool = True,
        ranking_loss_weight: float = 0.5,
        ranking_margin: float = 0.5,
        output_dir: str = "Critic/trained_models",
        log_interval: int = 10,
        eval_interval: int = 100,
        save_interval: int = 500,
        use_wandb: bool = False,
        wandb_project: str = "qc-agent-critic",
    ):
        self.accelerator = accelerator
        self.use_accelerate = accelerator is not None

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.use_ranking_loss = use_ranking_loss
        self.ranking_loss_weight = ranking_loss_weight
        self.ranking_margin = ranking_margin

        self.output_dir = output_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        # Setup optimizer (only for trainable parameters)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Calculate total steps
        total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)

        # Setup scheduler with warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Prepare with accelerator if available
        if self.use_accelerate:
            self.model, self.optimizer, self.train_dataloader, self.scheduler = accelerator.prepare(
                model, self.optimizer, train_dataloader, self.scheduler
            )
            if val_dataloader is not None:
                self.val_dataloader = accelerator.prepare(val_dataloader)
            else:
                self.val_dataloader = None
        else:
            self.model = model
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader

        # Create output directory (only on main process)
        if self.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

        # Initialize wandb if enabled (only on main process)
        if use_wandb and self.is_main_process:
            wandb.init(project=wandb_project, config={
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": train_dataloader.batch_size,
                "use_ranking_loss": use_ranking_loss,
                "ranking_loss_weight": ranking_loss_weight,
                "num_gpus": accelerator.num_processes if self.use_accelerate else 1,
            })

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        if self.use_accelerate:
            return self.accelerator.is_main_process
        return True

    @property
    def device(self):
        """Get the device."""
        if self.use_accelerate:
            return self.accelerator.device
        return next(self.model.parameters()).device

    def print_main(self, *args, **kwargs):
        """Print only on main process."""
        if self.is_main_process:
            print(*args, **kwargs)

    def get_unwrapped_model(self):
        """Get the unwrapped model for saving or forward pass."""
        # For DDP wrapped models, access the underlying module directly
        # This avoids triggering DeepSpeed imports via accelerate.unwrap_model()
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Get the unwrapped model for forward pass
        unwrapped_model = self.get_unwrapped_model()

        # Forward pass for BCE loss
        result = unwrapped_model(
            questions=batch["questions"],
            paths=batch["paths"],
            labels=batch["labels"].to(self.device),
        )

        loss = result["loss"]
        losses = {"bce_loss": loss.item()}

        # Optional ranking loss
        if self.use_ranking_loss:
            # Create ranking pairs from the batch data
            batch_list = [
                {"question": q, "path": p, "label": int(l)}
                for q, p, l in zip(batch["questions"], batch["paths"], batch["labels"])
            ]
            ranking_pairs = create_ranking_pairs(batch_list)

            if ranking_pairs is not None:
                ranking_loss = unwrapped_model.compute_ranking_loss(
                    positive_questions=ranking_pairs["pos_questions"],
                    positive_paths=ranking_pairs["pos_paths"],
                    negative_questions=ranking_pairs["neg_questions"],
                    negative_paths=ranking_pairs["neg_paths"],
                    margin=self.ranking_margin,
                )
                loss = loss + self.ranking_loss_weight * ranking_loss
                losses["ranking_loss"] = ranking_loss.item()

        losses["total_loss"] = loss.item()

        return loss, losses

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        unwrapped_model = self.get_unwrapped_model()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_pos_correct = 0
        total_pos_samples = 0
        total_neg_correct = 0
        total_neg_samples = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating", disable=not self.is_main_process):
            result = unwrapped_model(
                questions=batch["questions"],
                paths=batch["paths"],
                labels=batch["labels"].to(self.device),
            )

            total_loss += result["loss"].item() * len(batch["labels"])

            # Compute accuracy
            predictions = (result["values"] > 0.5).float()
            labels = batch["labels"].to(result["values"].device)
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)

            # Per-class accuracy
            pos_mask = labels == 1
            neg_mask = labels == 0
            total_pos_correct += (predictions[pos_mask] == labels[pos_mask]).sum().item()
            total_pos_samples += pos_mask.sum().item()
            total_neg_correct += (predictions[neg_mask] == labels[neg_mask]).sum().item()
            total_neg_samples += neg_mask.sum().item()

        # Gather metrics across all processes
        if self.use_accelerate:
            total_loss = self.accelerator.gather(torch.tensor([total_loss], device=self.device)).sum().item()
            total_correct = self.accelerator.gather(torch.tensor([total_correct], device=self.device)).sum().item()
            total_samples = self.accelerator.gather(torch.tensor([total_samples], device=self.device)).sum().item()
            total_pos_correct = self.accelerator.gather(torch.tensor([total_pos_correct], device=self.device)).sum().item()
            total_pos_samples = self.accelerator.gather(torch.tensor([total_pos_samples], device=self.device)).sum().item()
            total_neg_correct = self.accelerator.gather(torch.tensor([total_neg_correct], device=self.device)).sum().item()
            total_neg_samples = self.accelerator.gather(torch.tensor([total_neg_samples], device=self.device)).sum().item()

        metrics = {
            "val_loss": total_loss / total_samples if total_samples > 0 else 0,
            "val_accuracy": total_correct / total_samples if total_samples > 0 else 0,
            "val_pos_accuracy": total_pos_correct / total_pos_samples if total_pos_samples > 0 else 0,
            "val_neg_accuracy": total_neg_correct / total_neg_samples if total_neg_samples > 0 else 0,
        }

        return metrics

    def train(self):
        """Full training loop."""
        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(self.num_epochs):
            self.print_main(f"\n{'='*50}")
            self.print_main(f"Epoch {epoch + 1}/{self.num_epochs}")
            self.print_main(f"{'='*50}")

            self.model.train()
            epoch_losses = []

            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Training Epoch {epoch+1}",
                disable=not self.is_main_process
            )

            for step, batch in enumerate(progress_bar):
                loss, losses = self.train_step(batch)

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

                # Backward pass
                if self.use_accelerate:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                epoch_losses.append(losses["total_loss"])

                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.use_accelerate:
                        self.accelerator.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.max_grad_norm,
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.max_grad_norm,
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    global_step += 1

                    # Logging
                    if global_step % self.log_interval == 0 and self.is_main_process:
                        avg_loss = sum(epoch_losses[-self.log_interval:]) / min(len(epoch_losses), self.log_interval)
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                        })

                        if self.use_wandb:
                            wandb.log({
                                "train/loss": losses["total_loss"],
                                "train/bce_loss": losses.get("bce_loss", 0),
                                "train/ranking_loss": losses.get("ranking_loss", 0),
                                "train/learning_rate": self.scheduler.get_last_lr()[0],
                                "global_step": global_step,
                            })

                    # Evaluation
                    if self.val_dataloader is not None and global_step % self.eval_interval == 0:
                        val_metrics = self.evaluate()
                        self.print_main(f"\nStep {global_step} - Validation: {val_metrics}")

                        if self.use_wandb and self.is_main_process:
                            wandb.log({f"val/{k}": v for k, v in val_metrics.items()})

                        # Save best model
                        if val_metrics["val_loss"] < best_val_loss:
                            best_val_loss = val_metrics["val_loss"]
                            self.save_checkpoint("best")
                            self.print_main(f"Saved best model with val_loss={best_val_loss:.4f}")

                    # Periodic save
                    if global_step % self.save_interval == 0:
                        self.save_checkpoint(f"step_{global_step}")

            # End of epoch
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.print_main(f"\nEpoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")

            # Evaluate at end of epoch (only if validation data provided)
            if self.val_dataloader is not None:
                val_metrics = self.evaluate()
                self.print_main(f"Epoch {epoch+1} - Validation: {val_metrics}")

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best")

                # Save epoch checkpoint only when doing validation
                self.save_checkpoint(f"epoch_{epoch+1}")

        # Final save (always save final model)
        self.save_checkpoint("final")
        if self.val_dataloader is not None:
            self.print_main(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")
        else:
            self.print_main(f"\nTraining complete! Final model saved.")

        if self.use_wandb and self.is_main_process:
            wandb.finish()

    def save_checkpoint(self, name: str):
        """Save model checkpoint (only on main process)."""
        # Wait for all processes to reach this point BEFORE the is_main_process check
        # to avoid deadlocks
        if self.use_accelerate:
            self.accelerator.wait_for_everyone()

        if not self.is_main_process:
            return

        save_path = os.path.join(self.output_dir, name)
        unwrapped_model = self.get_unwrapped_model()
        unwrapped_model.save_pretrained(save_path)

        # Save training state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, os.path.join(save_path, "training_state.pt"))


def main():
    parser = argparse.ArgumentParser(description="Train QC-Agent Critic")

    # Model arguments
    parser.add_argument("--model_path", type=str, default="rmanluo/GCR-Meta-Llama-3.1-8B-Instruct",
                       help="Path to pretrained LLM")
    parser.add_argument("--hidden_dim", type=int, default=512,
                       help="Hidden dimension for projection")
    parser.add_argument("--use_question_pooling", action="store_true",
                       help="Use explicit question pooling")
    parser.add_argument("--dtype", type=str, default="bf16",
                       choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                       choices=["eager", "sdpa", "flash_attention_2"])

    # Data arguments
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data directory")
    parser.add_argument("--val_data", type=str, default=None,
                       help="Path to validation data directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Loss arguments
    parser.add_argument("--use_ranking_loss", action="store_true",
                       help="Use margin ranking loss in addition to BCE")
    parser.add_argument("--ranking_loss_weight", type=float, default=0.5)
    parser.add_argument("--ranking_margin", type=float, default=0.5)

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="Critic/trained_models")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)

    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="qc-agent-critic")

    # Other
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Initialize accelerator if available
    accelerator = None
    if ACCELERATE_AVAILABLE:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="bf16" if args.dtype == "bf16" else ("fp16" if args.dtype == "fp16" else "no"),
        )
        set_seed(args.seed)
        is_main = accelerator.is_main_process
        device = accelerator.device
        num_processes = accelerator.num_processes
    else:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        is_main = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_processes = 1

    # Save args (only on main process)
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        print("=" * 60)
        print("QC-Agent Critic Training")
        print("=" * 60)
        print(f"Model: {args.model_path}")
        print(f"Training data: {args.train_data}")
        print(f"Output directory: {args.output_dir}")
        print(f"Number of GPUs: {num_processes}")
        print(f"Device: {device}")
        print("=" * 60)

    # Load data
    if is_main:
        print("\nLoading training data...")
    train_dataset = load_from_disk(args.train_data)
    if is_main:
        print(f"Loaded {len(train_dataset)} training samples")

    val_dataset = None
    if args.val_data:
        if is_main:
            print(f"\nLoading validation data from {args.val_data}...")
        val_dataset = load_from_disk(args.val_data)
        if is_main:
            print(f"Loaded {len(val_dataset)} validation samples")

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # Initialize model
    if is_main:
        print("\nInitializing Critic model...")
    model = CriticModel(
        model_path=args.model_path,
        hidden_dim=args.hidden_dim,
        use_question_pooling=args.use_question_pooling,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        freeze_llm=True,  # Always freeze LLM
    )

    # Move model to device (only if not using accelerate)
    if not ACCELERATE_AVAILABLE:
        model.to(device)

    # Count parameters
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Initialize trainer
    trainer = CriticTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        accelerator=accelerator,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        use_ranking_loss=args.use_ranking_loss,
        ranking_loss_weight=args.ranking_loss_weight,
        ranking_margin=args.ranking_margin,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )

    # Train
    if is_main:
        print("\nStarting training...")
    trainer.train()


if __name__ == "__main__":
    main()
