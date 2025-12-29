"""
Agentic Reasoning Controller for Qwen3-8B AGC-Agent.

This module provides Qwen3-specific implementations of the reasoning components,
with proper handling of Qwen3's chat template and generation parameters.

Key adaptations for Qwen3:
- Uses enable_thinking=False for structured KG reasoning (no <think> blocks)
- Generation parameters: temp=0.7, top_p=0.8, top_k=20 for non-thinking mode
- Handles Qwen3's special tokens and template format
"""

from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass
from enum import Enum
import torch
import re

from agc_agent.beam_state import BeamState
from agc_agent.constraint_engine import StepwiseConstrainedDecoding
from .kg_index_qwen import QwenKGIndex

# Import base types
from agc_agent.agentic_controller import (
    TerminationAction,
    SelectionResult,
    TerminationResult,
)


# =============================================================================
# Prompt Templates (same as base, but with Qwen3 template application)
# =============================================================================

RELATION_SELECTOR_SYSTEM_PROMPT = """You are a Knowledge Graph Reasoning Agent specialized in navigating structured knowledge graphs to answer questions. Your task is to select the most promising relation to follow at each step of the reasoning process.

A knowledge graph consists of entities connected by relations, forming triples: (head_entity, relation, tail_entity). For example:
- (Barack Obama, spouse_of, Michelle Obama) means Barack Obama's spouse is Michelle Obama
- (USA, president, Joe Biden) means the president of USA is Joe Biden

When selecting a relation, you should consider:
1. SEMANTIC RELEVANCE: How well does the relation's meaning align with what the question is asking?
2. PATH PROGRESS: Does this relation move closer to the type of entity the question seeks?
3. REASONING CHAIN: How does this relation connect to the previous reasoning steps?

You must ONLY select from the available relations listed. Any relation not in the list does not exist for the current entity in the knowledge graph.

Output your selected relation between <REL> and </REL> tags, exactly as it appears in the available list."""


RELATION_SELECTOR_USER_TEMPLATE = """# Question:
{question}

# Topic Entities:
{topic_entities}

# Current Reasoning State:
- Path So Far: {path_so_far}
- Current Entity: {current_entity}
- Reasoning Depth: {depth} step(s) taken

# Available Relations from "{current_entity}":
{available_relations}

# Task:
Analyze the question and current reasoning state. Select the single best relation to follow that will make progress toward answering the question.
Consider:
1. Which relation semantically connects to the question's intent?
2. Which relation logically continues the reasoning path?
3. Which relation is likely to reach answer-type entities?

# Selected Relation:
<REL>"""


ENTITY_SELECTOR_SYSTEM_PROMPT = """You are a Knowledge Graph Reasoning Agent. Given a reasoning path and a selected relation, your task is to choose the target entity that is most likely to lead toward answering the question.

You must ONLY select from the available target entities listed. Do NOT invent or suggest entities not in the list.

Output your selected entity between <ENT> and </ENT> tags, exactly as it appears in the available list."""


ENTITY_SELECTOR_USER_TEMPLATE = """# Question:
{question}

# Topic Entities:
{topic_entities}

# Reasoning Path So Far:
{path_so_far}

# Current Step:
From entity "{current_entity}", following relation [{selected_relation}]

# Available Target Entities:
{available_entities}

# Task:
Select the entity that best continues the reasoning toward the answer.
Consider:
1. Which entity's type matches what the question asks for?
2. Which entity is most semantically relevant to the question?
3. If multiple entities seem valid, which is most specific?

# Selected Entity:
<ENT>"""


TERMINATION_PREDICTOR_SYSTEM_PROMPT = """You are a Knowledge Graph Reasoning Agent evaluating whether to continue exploration or stop.

You must decide one of three actions:
1. ANSWER: The current entity is the TYPE of thing the question asks for. Stop here.
2. CONTINUE: The current entity is an intermediate step, not the answer type. Keep exploring.
3. BACKTRACK: The current path is unlikely to lead to the answer. Go back.

KEY PRINCIPLE: If the question asks for a specific TYPE of entity (person, place, language, date, etc.), check if the current entity matches that type.

Output exactly one word: ANSWER, CONTINUE, or BACKTRACK."""


TERMINATION_PREDICTOR_USER_TEMPLATE = """# Question:
{question}

# Reasoning Path Taken:
{path_so_far}

# Current Entity: {current_entity}

# Evaluate the current state:

First, identify what TYPE of entity the question asks for.
Then, check if "{current_entity}" is that type of entity.

1. ANSWER: Choose this if "{current_entity}" is the TYPE of entity the question asks for.

2. CONTINUE: Choose this if "{current_entity}" is an intermediate step, not the answer type.

3. BACKTRACK: Choose this if the reasoning has gone in a wrong direction.

# Decision (ANSWER/CONTINUE/BACKTRACK):"""


class QwenTrieConstraint:
    """
    Wrapper class that adapts Qwen trie objects to the StepConstraint interface.

    This provides the get_allowed_tokens method expected by StepwiseConstrainedDecoding.
    """

    def __init__(self, tokenizer: Any, trie: Any):
        """
        Initialize the constraint wrapper.

        Args:
            tokenizer: Qwen3 tokenizer
            trie: The trie structure (QwenRelationTokenTrie or QwenEntityTokenTrie)
        """
        self.tokenizer = tokenizer
        self.trie = trie
        self.all_tokens = list(range(len(tokenizer)))

    def get_allowed_tokens(self, prefix_sequence: List[int]) -> List[int]:
        """
        Get allowed next tokens given the generated prefix.

        Args:
            prefix_sequence: List of token IDs generated so far for this item

        Returns:
            List of allowed next token IDs
        """
        if self.trie is None:
            return self.all_tokens

        allowed = self.trie.get(prefix_sequence)
        if len(allowed) == 0:
            # Fallback to all tokens if trie is exhausted
            return self.all_tokens
        return allowed


class QwenConstraintEngine:
    """
    Qwen3-specific Constraint Engine.

    Manages the creation and application of constraints during
    the AGC-Agent's reasoning process using Qwen3 tokenizer.
    """

    def __init__(self, kg_index: QwenKGIndex):
        """
        Initialize the constraint engine.

        Args:
            kg_index: The Qwen3-specific KG index
        """
        self.kg_index = kg_index
        self.tokenizer = kg_index.tokenizer

    def create_relation_constraint(self, current_entity: str):
        """Create a relation constraint for the current entity."""
        valid_relations = self.kg_index.get_valid_relations(current_entity)
        if not valid_relations:
            return None

        trie = self.kg_index.get_relation_constraint_trie(current_entity)
        if trie is None:
            return None
        return QwenTrieConstraint(self.tokenizer, trie)

    def create_entity_constraint(self, current_entity: str, selected_relation: str):
        """Create an entity constraint for (current_entity, selected_relation)."""
        valid_entities = self.kg_index.get_valid_entities(current_entity, selected_relation)
        if not valid_entities:
            return None

        trie = self.kg_index.get_entity_constraint_trie(current_entity, selected_relation)
        if trie is None:
            return None
        return QwenTrieConstraint(self.tokenizer, trie)

    def get_valid_relations(self, entity: str) -> List[str]:
        """Get valid relations from an entity."""
        return self.kg_index.get_valid_relations(entity)

    def get_valid_entities(self, entity: str, relation: str) -> List[str]:
        """Get valid entities for (entity, relation)."""
        return self.kg_index.get_valid_entities(entity, relation)


class QwenRelationSelector:
    """
    Selects which relation(s) to explore next, using Qwen3-specific generation.

    Key Qwen3 adaptations:
    - Uses enable_thinking=False in chat template
    - Temperature=0.7, top_p=0.8 for non-thinking mode
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        constraint_engine: QwenConstraintEngine,
        use_constrained_generation: bool = True,
        top_k: int = 3,
        generation_mode: str = "beam"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.constraint_engine = constraint_engine
        self.use_constrained_generation = use_constrained_generation
        self.top_k = top_k
        self.generation_mode = generation_mode

        # Special tokens for constrained region
        self.rel_start_token = "<REL>"
        self.rel_end_token = "</REL>"

    def _format_path(self, beam: BeamState) -> str:
        """Format the path history for the prompt."""
        if not beam.path:
            return "(Starting position - no previous steps)"
        return f"<PATH> {beam.path_to_string()} </PATH>"

    def _format_relations(self, relations: List[str]) -> str:
        """Format available relations for the prompt."""
        if not relations:
            return "(No available relations)"
        return "\n".join(f"- {r}" for r in relations)

    def _build_prompt(
        self,
        question: str,
        topic_entities: List[str],
        beam: BeamState,
        valid_relations: List[str]
    ) -> str:
        """Build the complete prompt for relation selection with Qwen3 template."""
        user_prompt = RELATION_SELECTOR_USER_TEMPLATE.format(
            question=question,
            topic_entities=", ".join(topic_entities),
            path_so_far=self._format_path(beam),
            current_entity=beam.current_entity,
            depth=beam.depth,
            available_relations=self._format_relations(valid_relations)
        )

        messages = [
            {"role": "system", "content": RELATION_SELECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # Apply Qwen3 chat template with enable_thinking=False
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Disable thinking mode for structured KG reasoning
            )
        except TypeError:
            # Fallback if enable_thinking is not supported
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

    def _clean_qwen_output(self, output: str) -> str:
        """Remove Qwen3 thinking blocks and clean the output."""
        # Remove <think>...</think> blocks if present
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL | re.IGNORECASE)
        # Remove any remaining thinking-related tokens
        output = re.sub(r'<\|im_start\|>think.*?<\|im_end\|>', '', output, flags=re.DOTALL)
        return output.strip()

    def _parse_relation(self, output: str, valid_relations: List[str]) -> Optional[str]:
        """Parse the selected relation from LLM output."""
        # Clean Qwen3-specific tokens first
        output = self._clean_qwen_output(output)

        # Try to extract from <REL>...</REL> tags
        match = re.search(r'<REL>\s*([^<]+?)\s*</REL>', output, re.IGNORECASE)
        if match:
            relation = match.group(1).strip()
            if relation in valid_relations:
                return relation
            # Try case-insensitive match
            for rel in valid_relations:
                if rel.lower() == relation.lower():
                    return rel

        # Fallback: check if any valid relation appears in output
        # Sort by length (longest first) to match most specific relation
        sorted_relations = sorted(valid_relations, key=len, reverse=True)
        output_lower = output.lower()
        for rel in sorted_relations:
            if rel.lower() in output_lower:
                return rel

        return None

    @torch.inference_mode()
    def select(
        self,
        question: str,
        topic_entities: List[str],
        beam: BeamState
    ) -> List[SelectionResult]:
        """Select relation(s) to explore from the current entity."""
        valid_relations = self.constraint_engine.get_valid_relations(beam.current_entity)

        if not valid_relations:
            return []

        # Filter out already visited relations at this entity
        unvisited_relations = [
            r for r in valid_relations
            if not beam.has_visited(beam.current_entity, r)
        ]

        if not unvisited_relations:
            return []

        # Build prompt
        prompt = self._build_prompt(question, topic_entities, beam, unvisited_relations)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        # Setup constrained decoding if enabled
        prefix_allowed_fn = None
        if self.use_constrained_generation:
            constraint = self.constraint_engine.create_relation_constraint(beam.current_entity)
            if constraint:
                start_id = self.tokenizer.convert_tokens_to_ids(self.rel_start_token)
                end_id = self.tokenizer.convert_tokens_to_ids(self.rel_end_token)

                if start_id != self.tokenizer.unk_token_id:
                    handler = StepwiseConstrainedDecoding(
                        tokenizer=self.tokenizer,
                        constraint=constraint,
                        start_token_id=start_id,
                        end_token_id=end_id
                    )
                    prefix_allowed_fn = handler.allowed_tokens_fn

        # Generate with Qwen3-specific parameters
        num_return = min(self.top_k, len(unvisited_relations))

        # Build generation kwargs based on mode with Qwen3 parameters
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 64,  # Reduced from 1024 - relation names are short
            "prefix_allowed_tokens_fn": prefix_allowed_fn,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        if self.generation_mode == "greedy":
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_return_sequences"] = 1
        elif self.generation_mode == "beam":
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_beams"] = num_return
            gen_kwargs["num_return_sequences"] = num_return
        elif self.generation_mode == "sampling":
            # Qwen3 non-thinking mode parameters
            gen_kwargs["do_sample"] = True
            gen_kwargs["num_return_sequences"] = num_return
            gen_kwargs["temperature"] = 0.7
            gen_kwargs["top_p"] = 0.8
            gen_kwargs["top_k"] = 20

        try:
            outputs = self.model.generate(**gen_kwargs)
        except Exception as e:
            print(f"QwenRelationSelector generation error: {e}")
            return []

        # Parse results
        results = []
        seen_relations = set()

        for i, seq in enumerate(outputs.sequences):
            output_text = self.tokenizer.decode(
                seq[input_ids.shape[1]:],
                skip_special_tokens=True
            )

            relation = self._parse_relation(output_text, unvisited_relations)
            if relation and relation not in seen_relations:
                seen_relations.add(relation)
                prob = 1.0 / (i + 1)
                results.append(SelectionResult(
                    item=relation,
                    probability=prob,
                    raw_output=output_text
                ))

        # Fill up to top_k with remaining relations
        remaining = [r for r in unvisited_relations if r not in seen_relations]
        for rel in remaining[:self.top_k - len(results)]:
            results.append(SelectionResult(
                item=rel,
                probability=0.5 / (len(results) + 1),
                raw_output=""
            ))

        return results


class QwenEntitySelector:
    """
    Selects which target entity to traverse to, using Qwen3-specific generation.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        constraint_engine: QwenConstraintEngine,
        use_constrained_generation: bool = True,
        top_k: int = 3,
        max_entities_in_prompt: int = 50,
        generation_mode: str = "beam"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.constraint_engine = constraint_engine
        self.use_constrained_generation = use_constrained_generation
        self.top_k = top_k
        self.max_entities_in_prompt = max_entities_in_prompt
        self.generation_mode = generation_mode

        self.ent_start_token = "<ENT>"
        self.ent_end_token = "</ENT>"

    def _format_path(self, beam: BeamState) -> str:
        if not beam.path:
            return "(Starting position)"
        return f"<PATH> {beam.path_to_string()} </PATH>"

    def _format_entities(self, entities: List[str]) -> str:
        if not entities:
            return "(No available entities)"

        display_entities = entities[:self.max_entities_in_prompt]
        result = "\n".join(f"- {e}" for e in display_entities)

        if len(entities) > self.max_entities_in_prompt:
            result += f"\n... and {len(entities) - self.max_entities_in_prompt} more"

        return result

    def _build_prompt(
        self,
        question: str,
        topic_entities: List[str],
        beam: BeamState,
        selected_relation: str,
        valid_entities: List[str]
    ) -> str:
        user_prompt = ENTITY_SELECTOR_USER_TEMPLATE.format(
            question=question,
            topic_entities=", ".join(topic_entities),
            path_so_far=self._format_path(beam),
            current_entity=beam.current_entity,
            selected_relation=selected_relation,
            available_entities=self._format_entities(valid_entities)
        )

        messages = [
            {"role": "system", "content": ENTITY_SELECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

    def _clean_qwen_output(self, output: str) -> str:
        """Remove Qwen3 thinking blocks and clean the output."""
        # Remove <think>...</think> blocks if present
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL | re.IGNORECASE)
        # Remove any remaining thinking-related tokens
        output = re.sub(r'<\|im_start\|>think.*?<\|im_end\|>', '', output, flags=re.DOTALL)
        return output.strip()

    def _parse_entity(self, output: str, valid_entities: List[str]) -> Optional[str]:
        # Clean Qwen3-specific tokens first
        output = self._clean_qwen_output(output)

        match = re.search(r'<ENT>\s*([^<]+?)\s*</ENT>', output, re.IGNORECASE)
        if match:
            entity = match.group(1).strip()
            if entity in valid_entities:
                return entity
            # Try case-insensitive match
            for ent in valid_entities:
                if ent.lower() == entity.lower():
                    return ent

        # Fallback: check if any valid entity appears in output
        # Sort by length (longest first) to match most specific entity
        sorted_entities = sorted(valid_entities, key=len, reverse=True)
        output_lower = output.lower()
        for ent in sorted_entities:
            if ent.lower() in output_lower:
                return ent

        return None

    @torch.inference_mode()
    def select(
        self,
        question: str,
        topic_entities: List[str],
        beam: BeamState,
        selected_relation: str
    ) -> List[SelectionResult]:
        valid_entities = self.constraint_engine.get_valid_entities(
            beam.current_entity, selected_relation
        )

        if not valid_entities:
            return []

        valid_entities = [e for e in valid_entities if not beam.has_visited_entity(e)]

        if not valid_entities:
            return []

        if len(valid_entities) == 1:
            return [SelectionResult(item=valid_entities[0], probability=1.0, raw_output="")]

        prompt = self._build_prompt(
            question, topic_entities, beam, selected_relation, valid_entities
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        prefix_allowed_fn = None
        if self.use_constrained_generation:
            constraint = self.constraint_engine.create_entity_constraint(
                beam.current_entity, selected_relation
            )
            if constraint:
                start_id = self.tokenizer.convert_tokens_to_ids(self.ent_start_token)
                end_id = self.tokenizer.convert_tokens_to_ids(self.ent_end_token)

                if start_id != self.tokenizer.unk_token_id:
                    handler = StepwiseConstrainedDecoding(
                        tokenizer=self.tokenizer,
                        constraint=constraint,
                        start_token_id=start_id,
                        end_token_id=end_id
                    )
                    prefix_allowed_fn = handler.allowed_tokens_fn

        num_return = min(self.top_k, len(valid_entities))

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 128,
            "prefix_allowed_tokens_fn": prefix_allowed_fn,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        if self.generation_mode == "greedy":
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_return_sequences"] = 1
        elif self.generation_mode == "beam":
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_beams"] = num_return
            gen_kwargs["num_return_sequences"] = num_return
        elif self.generation_mode == "sampling":
            gen_kwargs["do_sample"] = True
            gen_kwargs["num_return_sequences"] = num_return
            gen_kwargs["temperature"] = 0.7
            gen_kwargs["top_p"] = 0.8
            gen_kwargs["top_k"] = 20

        try:
            outputs = self.model.generate(**gen_kwargs)
        except Exception as e:
            print(f"QwenEntitySelector generation error: {e}")
            return []

        results = []
        seen_entities = set()

        for i, seq in enumerate(outputs.sequences):
            output_text = self.tokenizer.decode(
                seq[input_ids.shape[1]:],
                skip_special_tokens=True
            )

            entity = self._parse_entity(output_text, valid_entities)
            if entity and entity not in seen_entities:
                seen_entities.add(entity)
                prob = 1.0 / (i + 1)
                results.append(SelectionResult(
                    item=entity,
                    probability=prob,
                    raw_output=output_text
                ))

        remaining = [e for e in valid_entities if e not in seen_entities]
        for ent in remaining[:self.top_k - len(results)]:
            results.append(SelectionResult(
                item=ent,
                probability=0.5 / (len(results) + 1),
                raw_output=""
            ))

        return results


class QwenTerminationPredictor:
    """
    Decides whether to continue reasoning, return an answer, or backtrack.
    Uses Qwen3-specific generation.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        answer_threshold: float = 0.5
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.answer_threshold = answer_threshold
        self._action_token_ids = None

    def _get_action_token_ids(self) -> Dict[str, int]:
        if self._action_token_ids is None:
            self._action_token_ids = {}
            for action in ["ANSWER", "CONTINUE", "BACKTRACK"]:
                tokens = self.tokenizer(action, add_special_tokens=False).input_ids
                if tokens:
                    self._action_token_ids[action] = tokens[0]
        return self._action_token_ids

    def _format_path(self, beam: BeamState) -> str:
        if not beam.path:
            return f"(Starting at {beam.current_entity})"
        return f"<PATH> {beam.path_to_string()} </PATH>"

    def _build_prompt(self, question: str, beam: BeamState) -> str:
        user_prompt = TERMINATION_PREDICTOR_USER_TEMPLATE.format(
            question=question,
            path_so_far=self._format_path(beam),
            current_entity=beam.current_entity
        )

        messages = [
            {"role": "system", "content": TERMINATION_PREDICTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

    def _clean_qwen_output(self, output: str) -> str:
        """Remove Qwen3 thinking blocks and clean the output."""
        # Remove <think>...</think> blocks if present
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL | re.IGNORECASE)
        # Remove any remaining thinking-related tokens
        output = re.sub(r'<\|im_start\|>think.*?<\|im_end\|>', '', output, flags=re.DOTALL)
        return output.strip()

    def _parse_action(self, output: str) -> TerminationAction:
        # Clean Qwen3-specific tokens first
        output = self._clean_qwen_output(output)
        output_upper = output.upper().strip()

        if "ANSWER" in output_upper:
            return TerminationAction.ANSWER
        elif "BACKTRACK" in output_upper:
            return TerminationAction.BACKTRACK
        else:
            return TerminationAction.CONTINUE

    @torch.inference_mode()
    def predict(self, question: str, beam: BeamState) -> TerminationResult:
        prompt = self._build_prompt(question, beam)

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        try:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=32,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        except Exception as e:
            print(f"QwenTerminationPredictor generation error: {e}")
            return TerminationResult(
                action=TerminationAction.CONTINUE,
                confidence=0.5,
                raw_output=""
            )

        output_text = self.tokenizer.decode(
            outputs.sequences[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        action = self._parse_action(output_text)

        confidence = 0.8
        if outputs.scores:
            first_logits = outputs.scores[0][0]
            probs = torch.softmax(first_logits, dim=-1)
            action_tokens = self._get_action_token_ids()
            if action.value in action_tokens:
                token_id = action_tokens[action.value]
                if token_id < len(probs):
                    confidence = probs[token_id].item()

        return TerminationResult(
            action=action,
            confidence=confidence,
            raw_output=output_text
        )

    def should_answer(self, result: TerminationResult) -> bool:
        return (
            result.action == TerminationAction.ANSWER and
            result.confidence >= self.answer_threshold
        )


class QwenAgenticController:
    """
    Main controller that orchestrates the reasoning components for Qwen3.

    Combines QwenRelationSelector, QwenEntitySelector, and QwenTerminationPredictor
    to perform step-by-step reasoning.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        kg_index: QwenKGIndex,
        use_constrained_generation: bool = True,
        relation_top_k: int = 3,
        entity_top_k: int = 3,
        answer_threshold: float = 0.5,
        generation_mode: str = "beam"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.kg_index = kg_index

        # Create Qwen-specific constraint engine
        self.constraint_engine = QwenConstraintEngine(kg_index)

        # Create Qwen-specific selectors
        self.relation_selector = QwenRelationSelector(
            model=model,
            tokenizer=tokenizer,
            constraint_engine=self.constraint_engine,
            use_constrained_generation=use_constrained_generation,
            top_k=relation_top_k,
            generation_mode=generation_mode
        )

        self.entity_selector = QwenEntitySelector(
            model=model,
            tokenizer=tokenizer,
            constraint_engine=self.constraint_engine,
            use_constrained_generation=use_constrained_generation,
            top_k=entity_top_k,
            generation_mode=generation_mode
        )

        self.termination_predictor = QwenTerminationPredictor(
            model=model,
            tokenizer=tokenizer,
            answer_threshold=answer_threshold
        )

    def step(
        self,
        question: str,
        topic_entities: List[str],
        beam: BeamState,
        skip_termination_at_depth_zero: bool = True
    ) -> Tuple[TerminationResult, List[BeamState]]:
        """
        Perform one reasoning step.

        Args:
            question: The natural language question
            topic_entities: Topic entities from the question
            beam: Current beam state
            skip_termination_at_depth_zero: Skip termination check at depth 0 (always continue)

        Returns:
            (termination_result, new_beams)
        """
        # At depth 0, skip termination check - always continue from starting entities
        if skip_termination_at_depth_zero and beam.depth == 0:
            term_result = TerminationResult(
                action=TerminationAction.CONTINUE,
                confidence=1.0,
                raw_output="[Skipped at depth 0]"
            )
        else:
            # Check termination via LLM
            term_result = self.termination_predictor.predict(question, beam)

        if term_result.action == TerminationAction.ANSWER:
            if self.termination_predictor.should_answer(term_result):
                return term_result, [beam.complete()]
            else:
                term_result = TerminationResult(
                    action=TerminationAction.CONTINUE,
                    confidence=term_result.confidence,
                    raw_output=term_result.raw_output
                )

        if term_result.action == TerminationAction.BACKTRACK:
            backtracked = beam.backtrack()
            if backtracked:
                return term_result, [backtracked]
            else:
                return term_result, []

        # CONTINUE: Select relations and entities
        relations = self.relation_selector.select(question, topic_entities, beam)
        if not relations:
            return TerminationResult(
                action=TerminationAction.ANSWER,
                confidence=0.5,
                raw_output="No valid relations"
            ), [beam.complete()]

        new_beams = []
        for rel_result in relations:
            entities = self.entity_selector.select(
                question, topic_entities, beam, rel_result.item
            )

            for ent_result in entities:
                extended_beam = beam.extend(
                    relation=rel_result.item,
                    next_entity=ent_result.item,
                    relation_prob=rel_result.probability,
                    entity_prob=ent_result.probability
                )
                new_beams.append(extended_beam)

        return term_result, new_beams
