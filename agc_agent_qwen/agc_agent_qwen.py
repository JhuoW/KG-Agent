"""
Main Qwen3-8B AGC-Agent Class.

Adaptive Graph-Constrained Agentic Reasoning (AGC-Agent) for Knowledge Graph
Question Answering, specifically adapted for Qwen3-8B.

Key adaptations for Qwen3:
- Uses enable_thinking=False for structured KG reasoning
- Proper handling of Qwen3 tokenizer and vocab size
- Generation parameters tuned for non-thinking mode
"""

from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass
import torch

from .kg_index_qwen import QwenKGIndex
from .agentic_controller_qwen import QwenAgenticController

# Import common components from base agc_agent
from agc_agent.beam_state import BeamState, BeamManager, PathAccumulator, BeamStatus
from agc_agent.agentic_controller import TerminationAction, TerminationResult
from agc_agent.agc_agent import AGCAgentResult


@dataclass
class QwenAGCAgentConfig:
    """Configuration for Qwen3-8B AGC-Agent."""
    # Beam search parameters
    beam_width: int = 10
    max_depth: int = 2  # Aligned with GCR index_path_length
    max_backtracks: int = 3
    backtrack_penalty: float = 0.8

    # Selection parameters
    relation_top_k: int = 3
    entity_top_k: int = 3

    # Termination parameters
    answer_threshold: float = 0.5
    min_completed_beams: int = 1

    # Generation parameters (Qwen3-specific)
    use_constrained_generation: bool = True
    max_new_tokens: int = 1024
    generation_mode: str = "beam"  # greedy, beam, sampling

    # Output parameters
    output_top_k: int = 10  # Number of paths to return (K in GCR)

    # Performance optimization
    skip_termination_at_depth_zero: bool = True  # Skip termination check at depth 0


class QwenAGCAgent:
    """
    Qwen3-8B Adaptive Graph-Constrained Agentic Reasoning Agent.

    Main interface for performing KGQA with step-wise constrained reasoning
    using Qwen3-8B as the backbone LLM.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[QwenAGCAgentConfig] = None
    ):
        """
        Initialize the Qwen3-8B AGC-Agent.

        Args:
            model: The Qwen3-8B model
            tokenizer: The Qwen3 tokenizer
            config: Agent configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or QwenAGCAgentConfig()

        # These will be initialized per-question
        self.kg_index: Optional[QwenKGIndex] = None
        self.controller: Optional[QwenAgenticController] = None
        self.beam_manager: Optional[BeamManager] = None

    def _build_kg_index(self, graph_triples: List[Tuple[str, str, str]]) -> QwenKGIndex:
        """Build Qwen3-specific KG index from graph triples."""
        kg_index = QwenKGIndex(tokenizer=self.tokenizer)
        kg_index.build_from_triples(graph_triples)
        return kg_index

    def _initialize_for_question(
        self,
        graph_triples: List[Tuple[str, str, str]],
        topic_entities: List[str]
    ) -> None:
        """Initialize agent components for a new question."""
        # Build Qwen3-specific KG index
        self.kg_index = self._build_kg_index(graph_triples)

        # Create Qwen3-specific controller
        self.controller = QwenAgenticController(
            model=self.model,
            tokenizer=self.tokenizer,
            kg_index=self.kg_index,
            use_constrained_generation=self.config.use_constrained_generation,
            relation_top_k=self.config.relation_top_k,
            entity_top_k=self.config.entity_top_k,
            answer_threshold=self.config.answer_threshold,
            generation_mode=self.config.generation_mode
        )

        # Create beam manager (uses base implementation)
        self.beam_manager = BeamManager(
            beam_width=self.config.beam_width,
            max_depth=self.config.max_depth,
            max_backtracks=self.config.max_backtracks,
            backtrack_penalty=self.config.backtrack_penalty,
            answer_threshold=self.config.answer_threshold,
            min_completed_beams=self.config.min_completed_beams
        )

        # Initialize beams from topic entities
        self.beam_manager.initialize(topic_entities)

    def _run_beam_search(
        self,
        question: str,
        topic_entities: List[str]
    ) -> List[BeamState]:
        """
        Run the adaptive beam search algorithm.

        Returns:
            List of completed beams
        """
        for depth in range(self.config.max_depth):
            if self.beam_manager.should_terminate():
                break

            active_beams = self.beam_manager.get_active_beams()
            if not active_beams:
                break

            self.beam_manager.clear_active_beams()

            candidate_beams = []

            for beam in active_beams:
                if not self.beam_manager.should_continue_beam(beam):
                    candidate_beams.append(beam.complete())
                    continue

                # Perform one reasoning step
                term_result, new_beams = self.controller.step(
                    question, topic_entities, beam,
                    skip_termination_at_depth_zero=self.config.skip_termination_at_depth_zero
                )

                if term_result.action == TerminationAction.ANSWER:
                    for b in new_beams:
                        self.beam_manager.add_candidate(b)

                elif term_result.action == TerminationAction.CONTINUE:
                    candidate_beams.extend(new_beams)

                elif term_result.action == TerminationAction.BACKTRACK:
                    if new_beams:
                        candidate_beams.extend(new_beams)

            self.beam_manager.set_active_beams(candidate_beams)
            self.beam_manager.prune_to_top_k()

        return self.beam_manager.get_all_results()

    def reason(
        self,
        question: str,
        graph_triples: List[Tuple[str, str, str]],
        topic_entities: List[str]
    ) -> AGCAgentResult:
        """
        Perform reasoning to answer a question.

        Args:
            question: The natural language question
            graph_triples: List of (head, relation, tail) triples
            topic_entities: Topic entities extracted from the question

        Returns:
            AGCAgentResult with predictions and reasoning trace
        """
        # Initialize for this question
        self._initialize_for_question(graph_triples, topic_entities)

        # Run beam search
        result_beams = self._run_beam_search(question, topic_entities)

        # Accumulate paths
        accumulator = PathAccumulator()
        accumulator.add_paths(result_beams)

        # Format for evaluation using LLM-based answer extraction
        predictions = accumulator.format_for_evaluation_with_llm(
            question=question,
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=self.config.output_top_k,
            topic_entities=topic_entities
        )
        raw_paths = accumulator.get_paths()[:self.config.output_top_k]
        answers = accumulator.get_answers_with_llm(
            question=question,
            model=self.model,
            tokenizer=self.tokenizer,
            topic_entities=topic_entities
        )

        # Get statistics
        stats = self.beam_manager.get_statistics()

        return AGCAgentResult(
            question=question,
            predictions=predictions,
            answers=answers,
            reasoning_trace={
                "total_paths_explored": stats["active_beams"] + stats["completed_beams"] + stats["pruned_beams"],
                "completed_paths": stats["completed_beams"],
                "max_depth_reached": stats["max_depth_reached"],
                "backtrack_count": stats["total_backtracks"]
            },
            raw_paths=raw_paths
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "Qwen/Qwen3-8B",
        config: Optional[QwenAGCAgentConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa"
    ) -> 'QwenAGCAgent':
        """
        Load QwenAGCAgent from a pretrained model.

        Args:
            model_path: Path to the Qwen3-8B model
            config: Agent configuration
            device: Device to load model on
            dtype: Model dtype
            attn_implementation: Attention implementation

        Returns:
            Initialized QwenAGCAgent
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True
        ).to(device)

        return cls(model=model, tokenizer=tokenizer, config=config)


class QwenSimplifiedAGCAgent:
    """
    Simplified Qwen3-8B AGC-Agent that uses a single LLM call per step.

    This version is more efficient but may be less accurate than the full
    QwenAgenticController approach.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[QwenAGCAgentConfig] = None
    ):
        """Initialize the simplified agent."""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or QwenAGCAgentConfig()

        # System prompt for combined selection
        self.system_prompt = """You are a Knowledge Graph Reasoning Agent. Navigate the graph step-by-step to answer questions.

At each step, you will see:
1. The question
2. Current entity and path history
3. Available (relation -> entity) pairs

Select the best hop to take, or STOP if the current entity answers the question.

Output format:
- To stop: "STOP"
- To select: Output the index number (1, 2, 3, etc.)"""

    def _build_kg_index(self, graph_triples: List[Tuple[str, str, str]]) -> QwenKGIndex:
        """Build Qwen3-specific KG index from graph triples."""
        kg_index = QwenKGIndex(tokenizer=self.tokenizer)
        kg_index.build_from_triples(graph_triples)
        return kg_index

    def _get_available_hops(
        self,
        kg_index: QwenKGIndex,
        current_entity: str,
        visited: set
    ) -> List[Tuple[str, str]]:
        """Get available (relation, entity) pairs from current entity."""
        hops = []
        relations = kg_index.get_valid_relations(current_entity)

        for rel in relations:
            if (current_entity, rel) in visited:
                continue
            entities = kg_index.get_valid_entities(current_entity, rel)
            for ent in entities:
                hops.append((rel, ent))

        return hops

    def _format_hops(self, hops: List[Tuple[str, str]]) -> str:
        """Format available hops for the prompt."""
        if not hops:
            return "(No available hops - must STOP)"

        lines = []
        for i, (rel, ent) in enumerate(hops, 1):
            lines.append(f"({i}) via [{rel}] -> {ent}")
        return "\n".join(lines)

    def _format_path(self, path: List[Tuple[str, str, str]]) -> str:
        """Format path for prompt."""
        if not path:
            return "(Starting position)"

        parts = [path[0][0]]
        for h, r, t in path:
            parts.extend([r, t])
        return " -> ".join(parts)

    @torch.inference_mode()
    def _select_hop(
        self,
        question: str,
        current_entity: str,
        path: List[Tuple[str, str, str]],
        hops: List[Tuple[str, str]]
    ) -> Tuple[bool, int]:
        """
        Select the next hop or decide to stop.

        Returns:
            (should_stop, hop_index) - hop_index is 0-indexed
        """
        user_prompt = f"""Question: {question}

Current Entity: {current_entity}

Path So Far: {self._format_path(path)}

Available Hops:
{self._format_hops(hops)}

Select the best hop (enter number) or STOP if current entity answers the question:"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Apply Qwen3 chat template with enable_thinking=False
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.model.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip().upper()

        if "STOP" in response:
            return True, -1

        # Extract number
        import re
        match = re.search(r'\d+', response)
        if match:
            idx = int(match.group()) - 1  # Convert to 0-indexed
            if 0 <= idx < len(hops):
                return False, idx

        # Default: stop
        return True, -1

    def reason(
        self,
        question: str,
        graph_triples: List[Tuple[str, str, str]],
        topic_entities: List[str]
    ) -> AGCAgentResult:
        """
        Perform simplified reasoning.

        Args:
            question: The question
            graph_triples: KG triples
            topic_entities: Starting entities

        Returns:
            AGCAgentResult
        """
        kg_index = self._build_kg_index(graph_triples)

        all_paths = []
        total_explored = 0

        for start_entity in topic_entities:
            stack = [(start_entity, [], set())]

            while stack and len(all_paths) < self.config.output_top_k:
                current, path, visited = stack.pop()
                total_explored += 1

                if len(path) >= self.config.max_depth:
                    all_paths.append((path.copy(), current))
                    continue

                hops = self._get_available_hops(kg_index, current, visited)

                if not hops:
                    all_paths.append((path.copy(), current))
                    continue

                should_stop, hop_idx = self._select_hop(question, current, path, hops)

                if should_stop:
                    all_paths.append((path.copy(), current))
                else:
                    rel, ent = hops[hop_idx]
                    new_path = path + [(current, rel, ent)]
                    new_visited = visited.copy()
                    new_visited.add((current, rel))
                    stack.append((ent, new_path, new_visited))

        # Format results
        predictions = []
        raw_paths = []

        for path, answer in all_paths:
            if path:
                parts = [path[0][0]]
                for h, r, t in path:
                    parts.extend([r, t])
                path_str = " -> ".join(parts)
            else:
                path_str = answer

            formatted = f"# Reasoning Path:\n{path_str}\n# Answer:\n{answer}"
            predictions.append(formatted)
            raw_paths.append((path_str, 1.0))

        answers = [(answer, 1.0) for _, answer in all_paths]

        return AGCAgentResult(
            question=question,
            predictions=predictions,
            answers=answers,
            reasoning_trace={
                "total_paths_explored": total_explored,
                "completed_paths": len(all_paths),
                "max_depth_reached": max((len(p) for p, _ in all_paths), default=0),
                "backtrack_count": 0
            },
            raw_paths=raw_paths
        )
