"""
Step-wise Constraint Engine for AGC-Agent.

This is the technical core that enables faithful generation without pre-computing
all paths like GCR. It operates on a simple principle:

GCR Constraint: The entire generated path must exist in KG-Trie.
AGC-Agent: Each generated step must be valid given the current position.

This decomposition transforms an exponential constraint into a sequence of linear constraints.
"""

from typing import List, Optional, Any, Tuple
import torch
from .kg_index import KGIndex, RelationTokenTrie, EntityTokenTrie


class StepConstraint:
    """
    Base class for step-wise constraints during LLM decoding.

    During constrained decoding, we modify the LLM's output distribution:
    P'(t_i | t_{<i}, prompt) = P(t_i | t_{<i}, prompt) / Z  if t_{1:i} ∈ ValidPrefixes(C)
                            = 0                             otherwise

    where Z is the normalization constant and C is the active constraint.
    """

    def __init__(self, tokenizer: Any, trie: Any):
        """
        Initialize the constraint.

        Args:
            tokenizer: HuggingFace tokenizer
            trie: The trie structure (RelationTokenTrie or EntityTokenTrie)
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


class RelationConstraint(StepConstraint):
    """
    Constraint for relation generation.

    Input: Current entity e_t
    Process:
        1. Query RelationIndex to obtain valid relations: R_valid = RelationIndex[e_t]
        2. Extract the subtrie from Relation Token Trie containing only relations in R_valid
        3. Return this subtrie as the decoding constraint

    Formal Definition: C_r(e_t) = SubTrie(RelationTrie, R_valid)
    """

    def __init__(
        self,
        tokenizer: Any,
        relation_trie: RelationTokenTrie,
        valid_relations: Optional[List[str]] = None
    ):
        """
        Initialize the relation constraint.

        Args:
            tokenizer: HuggingFace tokenizer
            relation_trie: The full relation token trie
            valid_relations: If provided, constraint is limited to these relations
        """
        if valid_relations is not None and relation_trie is not None:
            # Extract subtrie for valid relations only
            trie = relation_trie.get_subtrie(valid_relations)
        else:
            trie = relation_trie

        super().__init__(tokenizer, trie)

    @classmethod
    def from_kg_index(
        cls,
        kg_index: KGIndex,
        current_entity: str
    ) -> 'RelationConstraint':
        """
        Create a RelationConstraint from KGIndex for a specific entity.

        Args:
            kg_index: The KG index
            current_entity: The entity to generate relations for

        Returns:
            A RelationConstraint configured for this entity
        """
        valid_relations = kg_index.get_valid_relations(current_entity)
        constraint_trie = kg_index.get_relation_constraint_trie(current_entity)

        return cls(
            tokenizer=kg_index.tokenizer,
            relation_trie=constraint_trie if constraint_trie else kg_index.relation_trie,
            valid_relations=valid_relations if constraint_trie is None else None
        )


class EntityConstraint(StepConstraint):
    """
    Constraint for entity generation.

    Input: Current entity e_t, selected relation r_t
    Process:
        1. Query NeighborIndex: E_valid = NeighborIndex[(e_t, r_t)]
        2. Dynamically construct a mini-trie from tokenized forms of entities in E_valid
        3. Return this mini-trie as the decoding constraint

    Formal Definition: C_e(e_t, r_t) = BuildTrie({Tokenize(e') : e' ∈ E_valid})

    Key Efficiency Insight: Unlike relations (static, KG-wide), entity constraints are:
    - Local: Only entities reachable via one specific (entity, relation) pair
    - Small: Typically 1-100 entities, rarely more
    - Fast to build: Linear in the number of valid entities
    """

    def __init__(
        self,
        tokenizer: Any,
        entity_trie: EntityTokenTrie
    ):
        """
        Initialize the entity constraint.

        Args:
            tokenizer: HuggingFace tokenizer
            entity_trie: The dynamically-built entity trie
        """
        super().__init__(tokenizer, entity_trie)

    @classmethod
    def from_kg_index(
        cls,
        kg_index: KGIndex,
        current_entity: str,
        selected_relation: str
    ) -> 'EntityConstraint':
        """
        Create an EntityConstraint from KGIndex for a specific (entity, relation) pair.

        Args:
            kg_index: The KG index
            current_entity: The current entity
            selected_relation: The selected relation

        Returns:
            An EntityConstraint configured for this (entity, relation) pair
        """
        entity_trie = kg_index.get_entity_constraint_trie(current_entity, selected_relation)

        return cls(
            tokenizer=kg_index.tokenizer,
            entity_trie=entity_trie
        )


class StepwiseConstrainedDecoding:
    """
    Manages step-wise constrained decoding for the AGC-Agent.

    This class handles the dynamic switching between relation and entity constraints
    during generation, and provides the prefix_allowed_tokens_fn for HuggingFace's
    model.generate().
    """

    def __init__(
        self,
        tokenizer: Any,
        constraint: Optional[StepConstraint] = None,
        start_token_id: Optional[int] = None,
        end_token_id: Optional[int] = None
    ):
        """
        Initialize the constrained decoding handler.

        Args:
            tokenizer: HuggingFace tokenizer
            constraint: The active constraint (RelationConstraint or EntityConstraint)
            start_token_id: Token ID that starts constrained region (e.g., <REL> or <ENT>)
            end_token_id: Token ID that ends constrained region (e.g., </REL> or </ENT>)
        """
        self.tokenizer = tokenizer
        self.constraint = constraint
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.all_tokens = list(range(len(tokenizer)))

        # Track input length for each batch item
        self._input_lengths: dict = {}

    def set_constraint(self, constraint: StepConstraint) -> None:
        """Update the active constraint."""
        self.constraint = constraint

    def _check_constrained_region(self, sent: torch.Tensor) -> Tuple[bool, int]:
        """
        Check if we're inside a constrained region.

        Args:
            sent: The current token sequence

        Returns:
            (is_constrained, start_position)
        """
        if self.start_token_id is None or self.end_token_id is None:
            return True, 0

        # Find all start tokens
        start_positions = torch.where(sent == self.start_token_id)[0]
        if len(start_positions) == 0:
            return False, len(sent)

        # Get the last start token position
        last_start = start_positions[-1].item()

        # Count end tokens after the last start token
        tokens_after_start = sent[last_start:]
        end_count = len(torch.where(tokens_after_start == self.end_token_id)[0])

        if end_count == 0:
            # Inside a constrained region
            return True, last_start + 1
        else:
            # Outside constrained region (after end token)
            return False, len(sent)

    def allowed_tokens_fn(self, batch_id: int, sent: torch.Tensor) -> List[int]:
        """
        Callback function for model.generate() to restrict allowed tokens.

        This is passed to prefix_allowed_tokens_fn in model.generate().

        Args:
            batch_id: The batch index
            sent: Current token sequence

        Returns:
            List of allowed next token IDs
        """
        if self.constraint is None:
            return self.all_tokens

        # Check if we're in a constrained region
        is_constrained, start_pos = self._check_constrained_region(sent)

        if not is_constrained:
            return self.all_tokens

        # Get the tokens generated within the constrained region
        constrained_prefix = sent[start_pos:].tolist()

        # Get allowed tokens from the constraint
        allowed = self.constraint.get_allowed_tokens(constrained_prefix)

        if len(allowed) == 0:
            return self.all_tokens

        return allowed


class ConstraintEngine:
    """
    Main interface for the Step-wise Constraint Engine.

    This engine manages the creation and application of constraints during
    the AGC-Agent's reasoning process.
    """

    def __init__(self, kg_index: KGIndex):
        """
        Initialize the constraint engine.

        Args:
            kg_index: The KG index containing all structural information
        """
        self.kg_index = kg_index
        self.tokenizer = kg_index.tokenizer

    def create_relation_constraint(self, current_entity: str) -> Optional[RelationConstraint]:
        """
        Create a relation constraint for the current entity.

        C_r(e_t) = SubTrie(RelationTrie, RelationIndex[e_t])

        Args:
            current_entity: The entity to generate relations for

        Returns:
            RelationConstraint if valid relations exist, None otherwise
        """
        valid_relations = self.kg_index.get_valid_relations(current_entity)
        if not valid_relations:
            return None

        return RelationConstraint.from_kg_index(self.kg_index, current_entity)

    def create_entity_constraint(
        self,
        current_entity: str,
        selected_relation: str
    ) -> Optional[EntityConstraint]:
        """
        Create an entity constraint for (current_entity, selected_relation).

        C_e(e_t, r_t) = BuildTrie({Tokenize(e') : e' ∈ NeighborIndex[(e_t, r_t)]})

        Args:
            current_entity: The current entity
            selected_relation: The selected relation

        Returns:
            EntityConstraint if valid entities exist, None otherwise
        """
        valid_entities = self.kg_index.get_valid_entities(current_entity, selected_relation)
        if not valid_entities:
            return None

        return EntityConstraint.from_kg_index(
            self.kg_index,
            current_entity,
            selected_relation
        )

    def create_decoding_handler(
        self,
        constraint: StepConstraint,
        start_token_id: Optional[int] = None,
        end_token_id: Optional[int] = None
    ) -> StepwiseConstrainedDecoding:
        """
        Create a constrained decoding handler for use with model.generate().

        Args:
            constraint: The constraint to apply
            start_token_id: Optional start token for constrained region
            end_token_id: Optional end token for constrained region

        Returns:
            StepwiseConstrainedDecoding handler
        """
        return StepwiseConstrainedDecoding(
            tokenizer=self.tokenizer,
            constraint=constraint,
            start_token_id=start_token_id,
            end_token_id=end_token_id
        )

    def get_valid_relations(self, entity: str) -> List[str]:
        """Get valid relations from an entity."""
        return self.kg_index.get_valid_relations(entity)

    def get_valid_entities(self, entity: str, relation: str) -> List[str]:
        """Get valid entities for (entity, relation)."""
        return self.kg_index.get_valid_entities(entity, relation)
