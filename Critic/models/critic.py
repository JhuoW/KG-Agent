"""
Critic Model for QC-Agent.

The Critic evaluates the quality of a reasoning path given a question.
It uses a frozen LLM to encode (question, path) pairs and projects the
last hidden state to compute a value V(P_t, q) in (0, 1).

Architecture:
    - Frozen LLM backbone (e.g., Llama-3.1-8B)
    - Trainable projection layer W_z: d_llm -> d_z
    - Trainable value head w_v: d_z -> 1
    - Optional action scoring head Q for neighbor pruning
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List, Dict, Tuple, Optional
import dotenv

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


class CriticModel(nn.Module):
    """
    Critic model that evaluates reasoning paths.

    Given a question q and a partial path P_t, computes:
        z_t = LayerNorm(W_z * h_{L_t})
        V(P_t, q) = sigmoid(w_v^T * z_t)

    where h_{L_t} is the last hidden state from the frozen LLM.
    """

    # Special tokens for marking question and path boundaries
    Q_START_TOKEN = "[Q_START]"
    Q_END_TOKEN = "[Q_END]"
    PATH_START_TOKEN = "[PATH_START]"
    PATH_END_TOKEN = "[PATH_END]"

    DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    def __init__(
        self,
        model_path: str,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_question_pooling: bool = False,
        dtype: str = "bf16",
        attn_implementation: str = "sdpa",
        freeze_llm: bool = True,
    ):
        """
        Initialize the Critic model.

        Args:
            model_path: Path to the pretrained LLM
            hidden_dim: Dimension of the projected embedding z_t (d_z)
            dropout: Dropout rate for regularization
            use_question_pooling: If True, concatenate pooled question embedding
            dtype: Data type for model weights
            attn_implementation: Attention implementation (eager, sdpa, flash_attention_2)
            freeze_llm: Whether to freeze the LLM backbone
        """
        super().__init__()

        self.model_path = model_path
        self.hidden_dim = hidden_dim
        self.use_question_pooling = use_question_pooling
        self.freeze_llm = freeze_llm
        self.dtype = self.DTYPE_MAP.get(dtype, torch.bfloat16)

        # Load tokenizer and add special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, token=HF_TOKEN, trust_remote_code=True
        )
        self._add_special_tokens()

        # Load the LLM backbone
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation=attn_implementation,
        )

        # Resize embeddings if special tokens were added
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # Freeze LLM if specified
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        # Get LLM hidden dimension
        config = AutoConfig.from_pretrained(model_path, token=HF_TOKEN, trust_remote_code=True)
        self.llm_hidden_dim = config.hidden_size

        # Projection dimension based on whether we use question pooling
        proj_input_dim = self.llm_hidden_dim * 2 if use_question_pooling else self.llm_hidden_dim

        # Trainable projection layer: W_z
        self.projection = nn.Sequential(
            nn.Linear(proj_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Value head: w_v
        self.value_head = nn.Linear(hidden_dim, 1)

        # Get special token IDs
        self.q_start_id = self.tokenizer.convert_tokens_to_ids(self.Q_START_TOKEN)
        self.q_end_id = self.tokenizer.convert_tokens_to_ids(self.Q_END_TOKEN)
        self.path_start_id = self.tokenizer.convert_tokens_to_ids(self.PATH_START_TOKEN)
        self.path_end_id = self.tokenizer.convert_tokens_to_ids(self.PATH_END_TOKEN)

    def _add_special_tokens(self):
        """Add special tokens to the tokenizer."""
        special_tokens = {
            "additional_special_tokens": [
                self.Q_START_TOKEN,
                self.Q_END_TOKEN,
                self.PATH_START_TOKEN,
                self.PATH_END_TOKEN,
            ]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            print(f"Added {num_added} special tokens to tokenizer")

    def format_input(self, question: str, path: str) -> str:
        """
        Format question and path into the input sequence.

        Format:
            [Q_START]
            {question_text}
            [Q_END]

            [PATH_START]
            {linearized path P_t}
            [PATH_END]

        Args:
            question: The natural language question
            path: The linearized reasoning path (e.g., "entity1 -> relation1 -> entity2")

        Returns:
            Formatted input string
        """
        return f"{self.Q_START_TOKEN}\n{question}\n{self.Q_END_TOKEN}\n\n{self.PATH_START_TOKEN}\n{path}\n{self.PATH_END_TOKEN}"

    def _get_question_span_indices(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the start and end indices of the question span for each sequence.

        Args:
            input_ids: [batch_size, seq_len]

        Returns:
            Tuple of (start_indices, end_indices) for question span
        """
        batch_size = input_ids.shape[0]
        start_indices = []
        end_indices = []

        for i in range(batch_size):
            # Find Q_START and Q_END positions
            q_start_pos = (input_ids[i] == self.q_start_id).nonzero(as_tuple=True)[0]
            q_end_pos = (input_ids[i] == self.q_end_id).nonzero(as_tuple=True)[0]

            if len(q_start_pos) > 0 and len(q_end_pos) > 0:
                start_indices.append(q_start_pos[0].item() + 1)  # After Q_START
                end_indices.append(q_end_pos[0].item())  # Before Q_END
            else:
                # Fallback: use first few tokens
                start_indices.append(0)
                end_indices.append(min(10, input_ids.shape[1]))

        return torch.tensor(start_indices), torch.tensor(end_indices)

    def _pool_question_embedding(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool hidden states over the question span.

        q^enc = (1/|I^Q|) * sum_{i in I^Q} h_i

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            input_ids: [batch_size, seq_len]

        Returns:
            Pooled question embedding [batch_size, hidden_dim]
        """
        batch_size = hidden_states.shape[0]
        start_indices, end_indices = self._get_question_span_indices(input_ids)

        pooled = []
        for i in range(batch_size):
            start = start_indices[i]
            end = end_indices[i]
            span_hidden = hidden_states[i, start:end, :]
            pooled.append(span_hidden.mean(dim=0))

        return torch.stack(pooled, dim=0)

    def encode(
        self,
        questions: List[str],
        paths: List[str],
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode question-path pairs and compute path embeddings.

        Args:
            questions: List of question strings
            paths: List of path strings (linearized)
            return_hidden_states: If True, return the raw hidden states

        Returns:
            Dictionary containing:
                - z_t: Path embeddings [batch_size, hidden_dim]
                - hidden_states (optional): Raw LLM hidden states
        """
        # Format inputs
        formatted_inputs = [
            self.format_input(q, p) for q, p in zip(questions, paths)
        ]

        # Tokenize
        encodings = self.tokenizer(
            formatted_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=2048,
        )

        input_ids = encodings.input_ids.to(self.llm.device)
        attention_mask = encodings.attention_mask.to(self.llm.device)

        # Forward pass through frozen LLM
        with torch.no_grad() if self.freeze_llm else torch.enable_grad():
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get last layer hidden states
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, d_llm]

        # Get the last token's hidden state (h_{L_t})
        # Find the last non-padding position for each sequence
        seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        h_last = hidden_states[batch_indices, seq_lengths]  # [batch_size, d_llm]

        # Optionally concatenate pooled question embedding
        if self.use_question_pooling:
            q_enc = self._pool_question_embedding(hidden_states, input_ids)
            h_concat = torch.cat([h_last, q_enc], dim=-1)  # [batch_size, 2*d_llm]
        else:
            h_concat = h_last

        # Project to get z_t
        z_t = self.projection(h_concat.to(self.projection[0].weight.dtype))  # [batch_size, hidden_dim]

        result = {"z_t": z_t}
        if return_hidden_states:
            result["hidden_states"] = hidden_states
            result["input_ids"] = input_ids

        return result

    def compute_value(
        self,
        questions: List[str],
        paths: List[str],
    ) -> torch.Tensor:
        """
        Compute the value V(P_t, q) for question-path pairs.

        V(P_t, q) = sigmoid(w_v^T * z_t)

        Args:
            questions: List of question strings
            paths: List of path strings

        Returns:
            Values in (0, 1) [batch_size]
        """
        encoding_result = self.encode(questions, paths)
        z_t = encoding_result["z_t"]

        # Compute value through value head
        logits = self.value_head(z_t).squeeze(-1)  # [batch_size]
        values = torch.sigmoid(logits)

        return values

    def forward(
        self,
        questions: List[str],
        paths: List[str],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.

        Args:
            questions: List of question strings
            paths: List of path strings
            labels: Optional binary labels (1 for positive, 0 for negative)

        Returns:
            Dictionary containing:
                - values: Predicted values [batch_size]
                - z_t: Path embeddings [batch_size, hidden_dim]
                - loss (optional): BCE loss if labels provided
        """
        encoding_result = self.encode(questions, paths)
        z_t = encoding_result["z_t"]

        # Compute value
        logits = self.value_head(z_t).squeeze(-1)
        values = torch.sigmoid(logits)

        result = {
            "values": values,
            "z_t": z_t,
            "logits": logits,
        }

        # Compute loss if labels provided
        if labels is not None:
            labels = labels.to(logits.device).float()
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            result["loss"] = loss

        return result

    def compute_ranking_loss(
        self,
        positive_questions: List[str],
        positive_paths: List[str],
        negative_questions: List[str],
        negative_paths: List[str],
        margin: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute margin ranking loss between positive and negative paths.

        L_rank = sum max(0, margin - V(P+, q) + V(P-, q))

        Args:
            positive_questions: Questions for positive paths
            positive_paths: Positive (ground truth) paths
            negative_questions: Questions for negative paths (same questions)
            negative_paths: Negative (corrupted/random) paths
            margin: Margin for ranking loss

        Returns:
            Ranking loss scalar
        """
        pos_values = self.compute_value(positive_questions, positive_paths)
        neg_values = self.compute_value(negative_questions, negative_paths)

        # Margin ranking loss: max(0, margin - pos + neg)
        ranking_loss = F.relu(margin - pos_values + neg_values).mean()

        return ranking_loss

    def save_pretrained(self, save_path: str):
        """Save the trainable components."""
        os.makedirs(save_path, exist_ok=True)

        # Save projection and value head
        torch.save({
            "projection": self.projection.state_dict(),
            "value_head": self.value_head.state_dict(),
            "config": {
                "model_path": self.model_path,
                "hidden_dim": self.hidden_dim,
                "use_question_pooling": self.use_question_pooling,
                "llm_hidden_dim": self.llm_hidden_dim,
            }
        }, os.path.join(save_path, "critic_weights.pt"))

        # Save tokenizer (with special tokens)
        self.tokenizer.save_pretrained(save_path)

        print(f"Saved critic model to {save_path}")

    @classmethod
    def from_pretrained(
        cls,
        save_path: str,
        dtype: str = "bf16",
        attn_implementation: str = "sdpa",
    ) -> "CriticModel":
        """Load a pretrained critic model."""
        # Load config
        checkpoint = torch.load(
            os.path.join(save_path, "critic_weights.pt"),
            map_location="cpu"
        )
        config = checkpoint["config"]

        # Initialize model
        model = cls(
            model_path=config["model_path"],
            hidden_dim=config["hidden_dim"],
            use_question_pooling=config["use_question_pooling"],
            dtype=dtype,
            attn_implementation=attn_implementation,
        )

        # Load tokenizer with special tokens
        model.tokenizer = AutoTokenizer.from_pretrained(save_path, trust_remote_code=True)

        # Load trained weights
        model.projection.load_state_dict(checkpoint["projection"])
        model.value_head.load_state_dict(checkpoint["value_head"])

        print(f"Loaded critic model from {save_path}")
        return model


class ActionScoringHead(nn.Module):
    """
    Optional action scoring head for pre-filtering candidate neighbors.

    Q(P_t, a, q) = w_Q^T * c_a
    where c_a = psi([z_t || e' || r])

    This is used when |N(e_t)| is large to pre-select top-K neighbors.
    """

    def __init__(
        self,
        path_hidden_dim: int,
        entity_dim: int,
        relation_dim: int,
        hidden_dim: int = 256,
    ):
        """
        Args:
            path_hidden_dim: Dimension of z_t (from CriticModel)
            entity_dim: Dimension of entity embeddings
            relation_dim: Dimension of relation embeddings
            hidden_dim: Hidden dimension for the MLP
        """
        super().__init__()

        input_dim = path_hidden_dim + entity_dim + relation_dim

        self.psi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.w_Q = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z_t: torch.Tensor,
        entity_embeddings: torch.Tensor,
        relation_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score candidate actions.

        Args:
            z_t: Path embeddings [batch_size, path_hidden_dim]
            entity_embeddings: Candidate entity embeddings [batch_size, num_candidates, entity_dim]
            relation_embeddings: Candidate relation embeddings [batch_size, num_candidates, relation_dim]

        Returns:
            Action scores [batch_size, num_candidates]
        """
        _, num_candidates, _ = entity_embeddings.shape

        # Expand z_t to match candidates
        z_t_expanded = z_t.unsqueeze(1).expand(-1, num_candidates, -1)

        # Concatenate [z_t || e' || r]
        concat = torch.cat([z_t_expanded, entity_embeddings, relation_embeddings], dim=-1)

        # Apply psi and w_Q
        c_a = self.psi(concat)
        scores = self.w_Q(c_a).squeeze(-1)

        return scores
