from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass
from enum import Enum
import torch
import re

from .beam_state import BeamState
from .constraint_engine import ConstraintEngine, StepwiseConstrainedDecoding
from .kg_index import KGIndex


class TerminationAction(Enum):
    ANSWER = "ANSWER"
    CONTINUE = "CONTINUE"
    BACKTRACK = "BACKTRACK"


@dataclass
class SelectionResult:
    item: str  # The selected relation or entity
    probability: float  # P(item | context)
    raw_output: str = ""  # Raw LLM output for debugging


@dataclass
class TerminationResult:
    action: TerminationAction
    confidence: float  # P(action | context)
    raw_output: str = ""


# =============================================================================
# Prompt Templates
# =============================================================================

RELATION_SELECTOR_SYSTEM_PROMPT = """You are a Knowledge Graph Reasoning Agent specialized in navigating structured knowledge graphs to answer questions. Your task is to select the most promising relation to follow at each step of the reasoning process.

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

# Decision (ANSWER/CONTINUE/BACKTRACK):"""


class RelationSelector:
    """
    Selects which relation(s) to explore next given the current reasoning state.

    Input:
        - Question q
        - Current entity e_t
        - Path history p_{0:t-1}
        - Available relations R_valid from Constraint Engine

    Output:
        - Set of candidate relations {r_t^(1), r_t^(2), ...} with probabilities
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        constraint_engine: ConstraintEngine,
        use_constrained_generation: bool = True,
        top_k: int = 3,
        generation_mode: str = "beam"
    ):
        """
        Initialize the RelationSelector.

        Args:
            model: The KG-specialized LLM
            tokenizer: The tokenizer
            constraint_engine: For creating relation constraints
            use_constrained_generation: Whether to use trie-constrained generation
            top_k: Number of relations to return
            generation_mode: 'greedy', 'beam', or 'sampling' (aligned with GCR)
        """
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
        """Build the complete prompt for relation selection."""
        user_prompt = RELATION_SELECTOR_USER_TEMPLATE.format(
            question=question,
            topic_entities=", ".join(topic_entities),
            path_so_far=self._format_path(beam),
            current_entity=beam.current_entity,
            depth=beam.depth,
            available_relations=self._format_relations(valid_relations)
        )

        # Apply chat template
        messages = [
            {"role": "system", "content": RELATION_SELECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _parse_relation(self, output: str, valid_relations: List[str]) -> Optional[str]:
        """Parse the selected relation from LLM output."""
        # Try to extract from <REL>...</REL> tags
        match = re.search(r'<REL>\s*([^<]+?)\s*</REL>', output, re.IGNORECASE)
        if match:
            relation = match.group(1).strip()
            if relation in valid_relations:
                return relation

        # Fallback: check if any valid relation appears in output
        output_lower = output.lower()
        for rel in valid_relations:
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
        """
        Select relation(s) to explore from the current entity.

        Args:
            question: The natural language question
            topic_entities: Topic entities from the question
            beam: Current beam state

        Returns:
            List of SelectionResult with selected relations and probabilities
        """
        # Get valid relations from constraint engine
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
                # Get start/end token IDs
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

        # Generate with the specified generation mode (aligned with GCR config)
        num_return = min(self.top_k, len(unvisited_relations))

        # Build generation kwargs based on mode
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 64,  # Reduced from 1024 - relation names are short
            "prefix_allowed_tokens_fn": prefix_allowed_fn,
            "pad_token_id": self.tokenizer.eos_token_id,
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

        try:
            outputs = self.model.generate(**gen_kwargs)
        except Exception as e:
            print(f"RelationSelector generation error: {e}")
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
                # Estimate probability (simplified - using 1.0 for now)
                prob = 1.0 / (i + 1)  # Higher rank = higher prob estimate
                results.append(SelectionResult(
                    item=relation,
                    probability=prob,
                    raw_output=output_text
                ))

        # Fill up to top_k with remaining relations to ensure diversity
        remaining = [r for r in unvisited_relations if r not in seen_relations]
        for rel in remaining[:self.top_k - len(results)]:
            results.append(SelectionResult(
                item=rel,
                probability=0.5 / (len(results) + 1),
                raw_output=""
            ))

        return results


class EntitySelector:
    """
    Selects which target entity to traverse to given the selected relation.

    Input:
        - Question q
        - Current state (entity e_t and path history)
        - Selected relation r_t
        - Valid target entities E_valid from Constraint Engine

    Output:
        - Set of candidate next entities {e_{t+1}^(1), e_{t+1}^(2), ...} with probabilities
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        constraint_engine: ConstraintEngine,
        use_constrained_generation: bool = True,
        top_k: int = 3,
        max_entities_in_prompt: int = 50,
        generation_mode: str = "beam"
    ):
        """
        Initialize the EntitySelector.

        Args:
            model: The KG-specialized LLM
            tokenizer: The tokenizer
            constraint_engine: For creating entity constraints
            use_constrained_generation: Whether to use trie-constrained generation
            top_k: Number of entities to return
            max_entities_in_prompt: Maximum entities to show in prompt
            generation_mode: 'greedy', 'beam', or 'sampling' (aligned with GCR)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.constraint_engine = constraint_engine
        self.use_constrained_generation = use_constrained_generation
        self.top_k = top_k
        self.max_entities_in_prompt = max_entities_in_prompt
        self.generation_mode = generation_mode

        # Special tokens
        self.ent_start_token = "<ENT>"
        self.ent_end_token = "</ENT>"

    def _format_path(self, beam: BeamState) -> str:
        """Format the path history for the prompt."""
        if not beam.path:
            return "(Starting position)"
        return f"<PATH> {beam.path_to_string()} </PATH>"

    def _format_entities(self, entities: List[str]) -> str:
        """Format available entities for the prompt."""
        if not entities:
            return "(No available entities)"

        # Limit entities shown in prompt
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
        """Build the complete prompt for entity selection."""
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

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _parse_entity(self, output: str, valid_entities: List[str]) -> Optional[str]:
        """Parse the selected entity from LLM output."""
        # Try to extract from <ENT>...</ENT> tags
        match = re.search(r'<ENT>\s*([^<]+?)\s*</ENT>', output, re.IGNORECASE)
        if match:
            entity = match.group(1).strip()
            if entity in valid_entities:
                return entity

        # Fallback: check if any valid entity appears in output
        output_lower = output.lower()
        for ent in valid_entities:
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
        """
        Select entity/entities to traverse to via the selected relation.

        Args:
            question: The natural language question
            topic_entities: Topic entities from the question
            beam: Current beam state
            selected_relation: The relation selected by RelationSelector

        Returns:
            List of SelectionResult with selected entities and probabilities
        """
        # Get valid entities from constraint engine
        valid_entities = self.constraint_engine.get_valid_entities(
            beam.current_entity,
            selected_relation
        )

        if not valid_entities:
            return []

        # Filter out already-visited entities to prevent cycles
        valid_entities = [e for e in valid_entities if not beam.has_visited_entity(e)]

        if not valid_entities:
            return []

        # If only one entity, return it directly
        if len(valid_entities) == 1:
            return [SelectionResult(
                item=valid_entities[0],
                probability=1.0,
                raw_output=""
            )]

        # Build prompt
        prompt = self._build_prompt(
            question, topic_entities, beam, selected_relation, valid_entities
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        # Setup constrained decoding if enabled
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

        # Generate with the specified generation mode (aligned with GCR config)
        num_return = min(self.top_k, len(valid_entities))

        # Build generation kwargs based on mode
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 128,
            "prefix_allowed_tokens_fn": prefix_allowed_fn,
            "pad_token_id": self.tokenizer.eos_token_id,
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

        try:
            outputs = self.model.generate(**gen_kwargs)
        except Exception as e:
            print(f"EntitySelector generation error: {e}")
            return []

        # Parse results
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

        # Fill up to top_k with remaining entities to ensure diversity
        remaining = [e for e in valid_entities if e not in seen_entities]
        for ent in remaining[:self.top_k - len(results)]:
            results.append(SelectionResult(
                item=ent,
                probability=0.5 / (len(results) + 1),
                raw_output=""
            ))

        return results


class TerminationPredictor:
    """
    Decides whether to continue reasoning, return an answer, or backtrack.

    Input:
        - Question q
        - Complete current state: entity e_t, path p_{0:t}
        - Current entity's meta (type, description if available)

    Output:
        - One of three actions: ANSWER, CONTINUE, or BACKTRACK
        - Confidence score P(action | context)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        answer_threshold: float = 0.5
    ):
        """
        Initialize the TerminationPredictor.

        Args:
            model: The KG-specialized LLM
            tokenizer: The tokenizer
            answer_threshold: Minimum confidence to accept ANSWER
        """
        self.model = model
        self.tokenizer = tokenizer
        self.answer_threshold = answer_threshold

        # Token IDs for the three actions
        self._action_token_ids = None

    def _get_action_token_ids(self) -> Dict[str, int]:
        """Get token IDs for action words."""
        if self._action_token_ids is None:
            self._action_token_ids = {}
            for action in ["ANSWER", "CONTINUE", "BACKTRACK"]:
                tokens = self.tokenizer(action, add_special_tokens=False).input_ids
                if tokens:
                    self._action_token_ids[action] = tokens[0]
        return self._action_token_ids

    def _format_path(self, beam: BeamState) -> str:
        """Format the path for the prompt."""
        if not beam.path:
            return f"(Starting at {beam.current_entity})"
        return f"<PATH> {beam.path_to_string()} </PATH>"

    def _build_prompt(self, question: str, beam: BeamState) -> str:
        """Build the prompt for termination prediction."""
        user_prompt = TERMINATION_PREDICTOR_USER_TEMPLATE.format(
            question=question,
            path_so_far=self._format_path(beam),
            current_entity=beam.current_entity
        )

        messages = [
            {"role": "system", "content": TERMINATION_PREDICTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _parse_action(self, output: str) -> TerminationAction:
        """Parse the action from LLM output."""
        output_upper = output.upper().strip()

        if "ANSWER" in output_upper:
            return TerminationAction.ANSWER
        elif "BACKTRACK" in output_upper:
            return TerminationAction.BACKTRACK
        else:
            return TerminationAction.CONTINUE

    @torch.inference_mode()
    def predict(self, question: str, beam: BeamState) -> TerminationResult:
        """
        Predict the termination action for the current state.

        Args:
            question: The natural language question
            beam: Current beam state

        Returns:
            TerminationResult with action and confidence
        """
        # Build prompt
        prompt = self._build_prompt(question, beam)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        # Generate
        try:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=32,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        except Exception as e:
            print(f"TerminationPredictor generation error: {e}")
            return TerminationResult(
                action=TerminationAction.CONTINUE,
                confidence=0.5,
                raw_output=""
            )

        # Decode output
        output_text = self.tokenizer.decode(
            outputs.sequences[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Parse action
        action = self._parse_action(output_text)

        # Estimate confidence from first token logits
        confidence = 0.8  # Default confidence
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
        """Check if we should accept the ANSWER action."""
        return (
            result.action == TerminationAction.ANSWER and
            result.confidence >= self.answer_threshold
        )


class AgenticController:
    """
    Main controller that orchestrates the reasoning components.

    Combines RelationSelector, EntitySelector, and TerminationPredictor
    to perform step-by-step reasoning.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        kg_index: KGIndex,
        use_constrained_generation: bool = True,
        relation_top_k: int = 3,
        entity_top_k: int = 3,
        answer_threshold: float = 0.5,
        generation_mode: str = "beam"
    ):
        """
        Initialize the AgenticController.

        Args:
            model: The KG-specialized LLM
            tokenizer: The tokenizer
            kg_index: The KG index for structural information
            use_constrained_generation: Whether to use trie constraints
            relation_top_k: Number of relations to return per step
            entity_top_k: Number of entities to return per step
            answer_threshold: Minimum confidence for ANSWER action
            generation_mode: 'greedy', 'beam', or 'sampling' (aligned with GCR)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.kg_index = kg_index

        # Create constraint engine
        self.constraint_engine = ConstraintEngine(kg_index)

        # Create selectors with generation mode
        self.relation_selector = RelationSelector(
            model=model,
            tokenizer=tokenizer,
            constraint_engine=self.constraint_engine,
            use_constrained_generation=use_constrained_generation,
            top_k=relation_top_k,
            generation_mode=generation_mode
        )

        self.entity_selector = EntitySelector(
            model=model,
            tokenizer=tokenizer,
            constraint_engine=self.constraint_engine,
            use_constrained_generation=use_constrained_generation,
            top_k=entity_top_k,
            generation_mode=generation_mode
        )

        self.termination_predictor = TerminationPredictor(
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
            - If ANSWER: new_beams contains the completed beam
            - If CONTINUE: new_beams contains extended beams
            - If BACKTRACK: new_beams contains the backtracked beam (or empty)
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
                # Low confidence answer -> continue
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
            # No valid relations -> complete as answer
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
