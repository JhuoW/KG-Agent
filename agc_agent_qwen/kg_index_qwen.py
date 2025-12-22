"""
KG Index Structures for Qwen3-8B AGC-Agent.

This module provides Qwen3-specific implementations of the KG index structures,
with proper handling of Qwen3's tokenizer vocabulary size and special tokens.

Key adaptations for Qwen3:
- Qwen3 has a vocab size of ~151,936 tokens
- Uses different BPE encoding than Llama
- Handles special tokens differently
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import marisa_trie
import networkx as nx

# Import base classes that don't need modification
from agc_agent.kg_index import (
    RelationMetadata,
    RelationIndex,
    NeighborIndex,
)


class QwenRelationTokenTrie:
    """
    A trie data structure encoding the tokenized forms of all relations in the KG,
    specifically adapted for Qwen3's tokenizer.

    Key differences from Llama version:
    - Qwen3 vocab size is ~151,936 vs Llama's ~128,256
    - Different character mapping range to accommodate larger vocab
    """

    def __init__(
        self,
        tokenizer: Any,
        relations: Optional[List[str]] = None,
        max_token_id: int = 160000  # Qwen3 vocab ~151,936
    ):
        """
        Initialize the QwenRelationTokenTrie.

        Args:
            tokenizer: Qwen3 tokenizer
            relations: Optional list of relations to add immediately
            max_token_id: Maximum token ID for the character mapping (Qwen3 specific)
        """
        self.tokenizer = tokenizer
        self.max_token_id = max_token_id

        # Character mapping for marisa_trie
        # Adjusted for Qwen3's larger vocabulary
        self.int2char = [chr(i) for i in range(min(max_token_id, 55000))] + (
            [chr(i) for i in range(65000, max_token_id + 10000)]
            if max_token_id >= 55000
            else []
        )
        self.char2int = {self.int2char[i]: i for i in range(min(len(self.int2char), max_token_id))}

        # Mapping from tokenized sequences to relation names
        self._token_to_relation: Dict[Tuple[int, ...], str] = {}

        # The underlying marisa_trie
        self._trie: Optional[marisa_trie.Trie] = None

        # Cache for first-level tokens
        self._first_tokens: List[int] = []

        # Build if relations provided
        if relations:
            self.build(relations)

    def build(self, relations: List[str]) -> None:
        """
        Build the trie from a list of relations.

        Args:
            relations: List of relation strings
        """
        sequences = []
        first_tokens_set = set()

        for relation in relations:
            # Tokenize the relation using Qwen3 tokenizer
            tokens = self.tokenizer(
                relation,
                padding=False,
                add_special_tokens=False
            ).input_ids

            if tokens:
                # Validate tokens are within our mapping range
                valid_tokens = [t for t in tokens if t < self.max_token_id]
                if valid_tokens:
                    sequences.append(valid_tokens)
                    self._token_to_relation[tuple(valid_tokens)] = relation
                    first_tokens_set.add(valid_tokens[0])

        self._first_tokens = list(first_tokens_set)

        if sequences:
            # Build the marisa_trie
            try:
                self._trie = marisa_trie.Trie(
                    "".join([self.int2char[i] for i in seq]) for seq in sequences
                )
            except (KeyError, IndexError) as e:
                print(f"Warning: Failed to build trie for some relations: {e}")
                self._trie = None

    def get(self, prefix_sequence: List[int]) -> List[int]:
        """
        Get valid next tokens given a prefix sequence.

        Args:
            prefix_sequence: List of token IDs generated so far

        Returns:
            List of valid next token IDs
        """
        if self._trie is None:
            return []

        if len(prefix_sequence) == 0:
            return self._first_tokens

        # Filter out tokens beyond our mapping range
        valid_prefix = [t for t in prefix_sequence if t < self.max_token_id]
        if len(valid_prefix) != len(prefix_sequence):
            return []

        try:
            key = "".join([self.int2char[i] for i in valid_prefix])
            return list({
                self.char2int[e[len(key)]]
                for e in self._trie.keys(key)
                if len(e) > len(key) and e[len(key)] in self.char2int
            })
        except (KeyError, IndexError):
            return []

    def get_subtrie(self, valid_relations: List[str]) -> 'QwenRelationTokenTrie':
        """
        Extract a subtrie containing only the specified relations.

        Args:
            valid_relations: List of relation names to include

        Returns:
            A new QwenRelationTokenTrie containing only the specified relations
        """
        subtrie = QwenRelationTokenTrie(
            tokenizer=self.tokenizer,
            max_token_id=self.max_token_id
        )
        subtrie.build(valid_relations)
        return subtrie

    def is_complete(self, token_sequence: List[int]) -> bool:
        """Check if a token sequence forms a complete relation."""
        return tuple(token_sequence) in self._token_to_relation

    def get_relation(self, token_sequence: List[int]) -> Optional[str]:
        """Get the relation name for a complete token sequence."""
        return self._token_to_relation.get(tuple(token_sequence))

    def __len__(self) -> int:
        return len(self._trie) if self._trie else 0


class QwenEntityTokenTrie:
    """
    A dynamically-built mini-trie for entity constraint generation,
    adapted for Qwen3's tokenizer.
    """

    def __init__(
        self,
        tokenizer: Any,
        entities: List[str],
        max_token_id: int = 160000  # Qwen3 vocab ~151,936
    ):
        """
        Initialize and build the QwenEntityTokenTrie.

        Args:
            tokenizer: Qwen3 tokenizer
            entities: List of valid entity names
            max_token_id: Maximum token ID for character mapping
        """
        self.tokenizer = tokenizer
        self.max_token_id = max_token_id

        # Character mapping (same approach as QwenRelationTokenTrie)
        self.int2char = [chr(i) for i in range(min(max_token_id, 55000))] + (
            [chr(i) for i in range(65000, max_token_id + 10000)]
            if max_token_id >= 55000
            else []
        )
        self.char2int = {self.int2char[i]: i for i in range(min(len(self.int2char), max_token_id))}

        # Mapping from tokenized sequences to entity names
        self._token_to_entity: Dict[Tuple[int, ...], str] = {}

        # Build the trie
        self._trie: Optional[marisa_trie.Trie] = None
        self._first_tokens: List[int] = []

        self._build(entities)

    def _build(self, entities: List[str]) -> None:
        """Build the trie from entity list."""
        sequences = []
        first_tokens_set = set()

        for entity in entities:
            tokens = self.tokenizer(
                entity,
                padding=False,
                add_special_tokens=False
            ).input_ids

            if tokens:
                # Validate tokens are within our mapping range
                valid_tokens = [t for t in tokens if t < self.max_token_id]
                if valid_tokens:
                    sequences.append(valid_tokens)
                    self._token_to_entity[tuple(valid_tokens)] = entity
                    first_tokens_set.add(valid_tokens[0])

        self._first_tokens = list(first_tokens_set)

        if sequences:
            try:
                self._trie = marisa_trie.Trie(
                    "".join([self.int2char[i] for i in seq]) for seq in sequences
                )
            except (KeyError, IndexError) as e:
                print(f"Warning: Failed to build entity trie: {e}")
                self._trie = None

    def get(self, prefix_sequence: List[int]) -> List[int]:
        """Get valid next tokens given a prefix sequence."""
        if self._trie is None:
            return []

        if len(prefix_sequence) == 0:
            return self._first_tokens

        # Filter out tokens beyond our mapping range
        valid_prefix = [t for t in prefix_sequence if t < self.max_token_id]
        if len(valid_prefix) != len(prefix_sequence):
            return []

        try:
            key = "".join([self.int2char[i] for i in valid_prefix])
            return list({
                self.char2int[e[len(key)]]
                for e in self._trie.keys(key)
                if len(e) > len(key) and e[len(key)] in self.char2int
            })
        except (KeyError, IndexError):
            return []

    def is_complete(self, token_sequence: List[int]) -> bool:
        """Check if a token sequence forms a complete entity."""
        return tuple(token_sequence) in self._token_to_entity

    def get_entity(self, token_sequence: List[int]) -> Optional[str]:
        """Get the entity name for a complete token sequence."""
        return self._token_to_entity.get(tuple(token_sequence))

    def __len__(self) -> int:
        return len(self._trie) if self._trie else 0


class QwenKGIndex:
    """
    Combined KG Index for Qwen3, managing all three index structures.

    This is the main interface for Qwen3-based AGC-Agent to query the KG structure.
    Uses Qwen3-specific trie implementations.
    """

    def __init__(self, tokenizer: Any = None):
        """
        Initialize the QwenKGIndex.

        Args:
            tokenizer: Qwen3 tokenizer
        """
        # Use shared base classes for relation and neighbor indices
        self.relation_index = RelationIndex()
        self.neighbor_index = NeighborIndex()

        # Use Qwen3-specific relation trie
        self.relation_trie: Optional[QwenRelationTokenTrie] = None
        self.tokenizer = tokenizer
        self._all_relations: Set[str] = set()

        # Qwen3 vocab size
        self._max_token_id = 160000 if tokenizer is None else len(tokenizer) + 1

    def build_from_triples(self, triples: List[Tuple[str, str, str]]) -> None:
        """
        Build all indices from a list of triples.

        Args:
            triples: List of (head, relation, tail) tuples
        """
        # Build relation and neighbor indices
        self.relation_index.build_from_triples(triples)
        self.neighbor_index.build_from_triples(triples)

        # Collect all unique relations
        for _, relation, _ in triples:
            self._all_relations.add(relation)

        # Build Qwen3-specific relation trie if tokenizer is available
        if self.tokenizer is not None:
            self._max_token_id = len(self.tokenizer) + 1
            self.relation_trie = QwenRelationTokenTrie(
                tokenizer=self.tokenizer,
                relations=list(self._all_relations),
                max_token_id=self._max_token_id
            )

    def build_from_graph(self, graph: nx.DiGraph) -> None:
        """
        Build all indices from a NetworkX graph.

        Args:
            graph: NetworkX DiGraph with 'relation' edge attribute
        """
        triples = []
        for u, v, data in graph.edges(data=True):
            relation = data.get('relation', '')
            triples.append((u, relation, v))
        self.build_from_triples(triples)

    def get_valid_relations(self, entity: str) -> List[str]:
        """Get all valid relations from an entity."""
        return self.relation_index.get_relation_names(entity)

    def get_valid_entities(self, entity: str, relation: str) -> List[str]:
        """Get all valid tail entities for (entity, relation)."""
        return self.neighbor_index.get_neighbors_list(entity, relation)

    def get_relation_constraint_trie(self, entity: str) -> Optional[QwenRelationTokenTrie]:
        """Get a subtrie for constraining relation generation at the current entity."""
        if self.relation_trie is None:
            return None

        valid_relations = self.get_valid_relations(entity)
        if not valid_relations:
            return None

        return self.relation_trie.get_subtrie(valid_relations)

    def get_entity_constraint_trie(self, entity: str, relation: str) -> Optional[QwenEntityTokenTrie]:
        """Get a mini-trie for constraining entity generation."""
        if self.tokenizer is None:
            return None

        valid_entities = self.get_valid_entities(entity, relation)
        if not valid_entities:
            return None

        return QwenEntityTokenTrie(
            tokenizer=self.tokenizer,
            entities=valid_entities,
            max_token_id=self._max_token_id
        )

    @property
    def all_relations(self) -> Set[str]:
        """Get all unique relations in the KG."""
        return self._all_relations
