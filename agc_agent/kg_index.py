"""
KG Index Structures for AGC-Agent.

This module implements the three lightweight indices that replace GCR's monolithic KG-Trie:
1. RelationIndex: Maps entities to their outgoing relations with metadata
2. NeighborIndex: Maps (entity, relation) pairs to reachable tail entities
3. RelationTokenTrie: Trie over tokenized relations for constrained decoding

Key insight: Build a trie over relations (O(|R|) ~ 7,000) instead of all paths (exponential).
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import marisa_trie
import networkx as nx


@dataclass
class RelationMetadata:
    """Metadata for a relation from a specific entity."""
    relation: str
    count: int  # Number of tail entities reachable via this relation
    frequency: float = 0.0  # Global frequency of this relation in the KG


class RelationIndex:
    """
    Maps each entity to its set of outgoing relations with metadata.

    RelationIndex: E -> {(r, count, freq)}

    For each entity e, stores all relations r such that ∃e': (e, r, e') ∈ G.
    """

    def __init__(self):
        # entity -> list of RelationMetadata
        self._index: Dict[str, List[RelationMetadata]] = defaultdict(list)
        # Global relation frequency counts
        self._relation_counts: Dict[str, int] = defaultdict(int)
        self._total_triples: int = 0

    def build_from_triples(self, triples: List[Tuple[str, str, str]]) -> None:
        """
        Build the index from a list of (head, relation, tail) triples.

        Args:
            triples: List of (head_entity, relation, tail_entity) tuples
        """
        # First pass: count relations per (entity, relation) pair
        entity_relation_tails: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        for head, relation, tail in triples:
            entity_relation_tails[(head, relation)].add(tail)
            self._relation_counts[relation] += 1
            self._total_triples += 1

        # Second pass: build the index
        entity_relations: Dict[str, Dict[str, int]] = defaultdict(dict)
        for (entity, relation), tails in entity_relation_tails.items():
            entity_relations[entity][relation] = len(tails)

        # Create RelationMetadata objects
        for entity, relations in entity_relations.items():
            for relation, count in relations.items():
                freq = self._relation_counts[relation] / self._total_triples if self._total_triples > 0 else 0.0
                self._index[entity].append(RelationMetadata(
                    relation=relation,
                    count=count,
                    frequency=freq
                ))

    def build_from_graph(self, graph: nx.DiGraph) -> None:
        """
        Build the index from a NetworkX directed graph.

        Args:
            graph: NetworkX DiGraph with 'relation' edge attribute
        """
        triples = []
        for u, v, data in graph.edges(data=True):
            relation = data.get('relation', '')
            triples.append((u, relation, v))
        self.build_from_triples(triples)

    def get_relations(self, entity: str) -> List[RelationMetadata]:
        """
        Get all outgoing relations from an entity.

        Args:
            entity: The entity to query

        Returns:
            List of RelationMetadata for all outgoing relations
        """
        return self._index.get(entity, [])

    def get_relation_names(self, entity: str) -> List[str]:
        """
        Get just the relation names for an entity.

        Args:
            entity: The entity to query

        Returns:
            List of relation names
        """
        return [rm.relation for rm in self._index.get(entity, [])]

    def has_entity(self, entity: str) -> bool:
        """Check if entity exists in the index."""
        return entity in self._index

    def __contains__(self, entity: str) -> bool:
        return self.has_entity(entity)

    def __len__(self) -> int:
        return len(self._index)


class NeighborIndex:
    """
    Maps (entity, relation) pairs to the set of reachable tail entities.

    NeighborIndex: (E × R) -> P(E)

    For each valid (entity, relation) pair, stores all reachable tail entities.
    """

    def __init__(self):
        # (entity, relation) -> set of tail entities
        self._index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    def build_from_triples(self, triples: List[Tuple[str, str, str]]) -> None:
        """
        Build the index from a list of (head, relation, tail) triples.

        Args:
            triples: List of (head_entity, relation, tail_entity) tuples
        """
        for head, relation, tail in triples:
            self._index[(head, relation)].add(tail)

    def build_from_graph(self, graph: nx.DiGraph) -> None:
        """
        Build the index from a NetworkX directed graph.

        Args:
            graph: NetworkX DiGraph with 'relation' edge attribute
        """
        for u, v, data in graph.edges(data=True):
            relation = data.get('relation', '')
            self._index[(u, relation)].add(v)

    def get_neighbors(self, entity: str, relation: str) -> Set[str]:
        """
        Get all tail entities reachable from (entity, relation).

        Args:
            entity: The head entity
            relation: The relation to follow

        Returns:
            Set of tail entities
        """
        return self._index.get((entity, relation), set())

    def get_neighbors_list(self, entity: str, relation: str) -> List[str]:
        """
        Get neighbors as a sorted list for deterministic behavior.

        Args:
            entity: The head entity
            relation: The relation to follow

        Returns:
            Sorted list of tail entities
        """
        return sorted(self.get_neighbors(entity, relation))

    def has_edge(self, entity: str, relation: str) -> bool:
        """Check if (entity, relation) pair exists."""
        return (entity, relation) in self._index

    def __len__(self) -> int:
        return len(self._index)


class RelationTokenTrie:
    """
    A trie data structure encoding the tokenized forms of all relations in the KG.

    Key insight: While GCR builds a trie over all paths (exponential), we build
    a trie over all relations (linear in |R|). Since |R| << |paths| (e.g., ~7,000
    relations vs. millions of 2-hop paths), this is dramatically more efficient.

    This trie is used to constrain LLM generation to only output valid relations.
    """

    def __init__(
        self,
        tokenizer: Any,
        relations: Optional[List[str]] = None,
        max_token_id: int = 256001
    ):
        """
        Initialize the RelationTokenTrie.

        Args:
            tokenizer: HuggingFace tokenizer for tokenizing relations
            relations: Optional list of relations to add immediately
            max_token_id: Maximum token ID for the character mapping
        """
        self.tokenizer = tokenizer
        self.max_token_id = max_token_id

        # Character mapping for marisa_trie (same as GCR's approach)
        self.int2char = [chr(i) for i in range(min(max_token_id, 55000))] + (
            [chr(i) for i in range(65000, max_token_id + 10000)]
            if max_token_id >= 55000
            else []
        )
        self.char2int = {self.int2char[i]: i for i in range(max_token_id)}

        # Mapping from tokenized sequences to relation names
        self._token_to_relation: Dict[Tuple[int, ...], str] = {}

        # The underlying marisa_trie
        self._trie: Optional[marisa_trie.Trie] = None

        # Cache for first-level tokens (optimization)
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
            # Tokenize the relation
            tokens = self.tokenizer(
                relation,
                padding=False,
                add_special_tokens=False
            ).input_ids

            if tokens:
                sequences.append(tokens)
                self._token_to_relation[tuple(tokens)] = relation
                first_tokens_set.add(tokens[0])

        self._first_tokens = list(first_tokens_set)

        # Build the marisa_trie
        self._trie = marisa_trie.Trie(
            "".join([self.int2char[i] for i in seq]) for seq in sequences
        )

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

        key = "".join([self.int2char[i] for i in prefix_sequence])
        return list({
            self.char2int[e[len(key)]]
            for e in self._trie.keys(key)
            if len(e) > len(key)
        })

    def get_subtrie(self, valid_relations: List[str]) -> 'RelationTokenTrie':
        """
        Extract a subtrie containing only the specified relations.

        This is the key operation for step-wise constraints:
        Given the valid relations at a step, extract a subtrie for constrained decoding.

        Args:
            valid_relations: List of relation names to include

        Returns:
            A new RelationTokenTrie containing only the specified relations
        """
        subtrie = RelationTokenTrie(
            tokenizer=self.tokenizer,
            max_token_id=self.max_token_id
        )
        subtrie.build(valid_relations)
        return subtrie

    def is_complete(self, token_sequence: List[int]) -> bool:
        """
        Check if a token sequence forms a complete relation.

        Args:
            token_sequence: List of token IDs

        Returns:
            True if the sequence corresponds to a complete relation
        """
        return tuple(token_sequence) in self._token_to_relation

    def get_relation(self, token_sequence: List[int]) -> Optional[str]:
        """
        Get the relation name for a complete token sequence.

        Args:
            token_sequence: List of token IDs

        Returns:
            The relation name, or None if not a complete relation
        """
        return self._token_to_relation.get(tuple(token_sequence))

    def __len__(self) -> int:
        return len(self._trie) if self._trie else 0

    def __iter__(self):
        if self._trie:
            for sequence in self._trie.iterkeys():
                yield [self.char2int[e] for e in sequence]


class EntityTokenTrie:
    """
    A dynamically-built mini-trie for entity constraint generation.

    Unlike the RelationTokenTrie (static, KG-wide), entity constraints are:
    - Local: Only entities reachable via one specific (entity, relation) pair
    - Small: Typically 1-100 entities, rarely more
    - Fast to build: Linear in the number of valid entities

    This is built on-the-fly for each entity selection step.
    """

    def __init__(
        self,
        tokenizer: Any,
        entities: List[str],
        max_token_id: int = 256001
    ):
        """
        Initialize and build the EntityTokenTrie.

        Args:
            tokenizer: HuggingFace tokenizer
            entities: List of valid entity names
            max_token_id: Maximum token ID for character mapping
        """
        self.tokenizer = tokenizer
        self.max_token_id = max_token_id

        # Character mapping (same as RelationTokenTrie)
        self.int2char = [chr(i) for i in range(min(max_token_id, 55000))] + (
            [chr(i) for i in range(65000, max_token_id + 10000)]
            if max_token_id >= 55000
            else []
        )
        self.char2int = {self.int2char[i]: i for i in range(max_token_id)}

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
                sequences.append(tokens)
                self._token_to_entity[tuple(tokens)] = entity
                first_tokens_set.add(tokens[0])

        self._first_tokens = list(first_tokens_set)

        if sequences:
            self._trie = marisa_trie.Trie(
                "".join([self.int2char[i] for i in seq]) for seq in sequences
            )

    def get(self, prefix_sequence: List[int]) -> List[int]:
        """Get valid next tokens given a prefix sequence."""
        if self._trie is None:
            return []

        if len(prefix_sequence) == 0:
            return self._first_tokens

        key = "".join([self.int2char[i] for i in prefix_sequence])
        return list({
            self.char2int[e[len(key)]]
            for e in self._trie.keys(key)
            if len(e) > len(key)
        })

    def is_complete(self, token_sequence: List[int]) -> bool:
        """Check if a token sequence forms a complete entity."""
        return tuple(token_sequence) in self._token_to_entity

    def get_entity(self, token_sequence: List[int]) -> Optional[str]:
        """Get the entity name for a complete token sequence."""
        return self._token_to_entity.get(tuple(token_sequence))

    def __len__(self) -> int:
        return len(self._trie) if self._trie else 0


class KGIndex:
    """
    Combined KG Index that manages all three index structures.

    This is the main interface for the AGC-Agent to query the KG structure.
    """

    def __init__(self, tokenizer: Any = None):
        """
        Initialize the KGIndex.

        Args:
            tokenizer: HuggingFace tokenizer (required for RelationTokenTrie)
        """
        self.relation_index = RelationIndex()
        self.neighbor_index = NeighborIndex()
        self.relation_trie: Optional[RelationTokenTrie] = None
        self.tokenizer = tokenizer
        self._all_relations: Set[str] = set()

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

        # Build relation trie if tokenizer is available
        if self.tokenizer is not None:
            self.relation_trie = RelationTokenTrie(
                tokenizer=self.tokenizer,
                relations=list(self._all_relations),
                max_token_id=len(self.tokenizer) + 1
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
        """
        Get all valid relations from an entity.

        Args:
            entity: The current entity

        Returns:
            List of valid relation names
        """
        return self.relation_index.get_relation_names(entity)

    def get_valid_entities(self, entity: str, relation: str) -> List[str]:
        """
        Get all valid tail entities for (entity, relation).

        Args:
            entity: The current entity
            relation: The selected relation

        Returns:
            List of valid tail entity names
        """
        return self.neighbor_index.get_neighbors_list(entity, relation)

    def get_relation_constraint_trie(self, entity: str) -> Optional[RelationTokenTrie]:
        """
        Get a subtrie for constraining relation generation at the current entity.

        Args:
            entity: The current entity

        Returns:
            A RelationTokenTrie containing only valid relations from this entity
        """
        if self.relation_trie is None:
            return None

        valid_relations = self.get_valid_relations(entity)
        if not valid_relations:
            return None

        return self.relation_trie.get_subtrie(valid_relations)

    def get_entity_constraint_trie(self, entity: str, relation: str) -> Optional[EntityTokenTrie]:
        """
        Get a mini-trie for constraining entity generation.

        Args:
            entity: The current entity
            relation: The selected relation

        Returns:
            An EntityTokenTrie containing only valid tail entities
        """
        if self.tokenizer is None:
            return None

        valid_entities = self.get_valid_entities(entity, relation)
        if not valid_entities:
            return None

        return EntityTokenTrie(
            tokenizer=self.tokenizer,
            entities=valid_entities,
            max_token_id=len(self.tokenizer) + 1
        )

    @property
    def all_relations(self) -> Set[str]:
        """Get all unique relations in the KG."""
        return self._all_relations
