"""
BeamState and Path Accumulator for AGC-Agent.

The Path Accumulator serves as the memory system of the agentic reasoning process.
While other components (Relation Selector, Entity Selector, Termination Predictor)
make decisions at each moment, the Path Accumulator maintains the complete history
and state that enables those decisions to be contextually informed.

- Relation Selector: "Where should I go next?"
- Entity Selector: "Which specific destination?"
- Termination Predictor: "Should I stop here?"
- Path Accumulator: "Where have I been, and what's my current situation?"
"""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy
import heapq


class BeamStatus(Enum):
    """Status of a beam in the search."""
    ACTIVE = "active"       # Still exploring
    COMPLETED = "completed"  # Found an answer
    PRUNED = "pruned"       # Eliminated due to low score


@dataclass
class BeamState:
    """
    Represents the state of a single reasoning trajectory (beam).

    BeamState = {
        current_entity: e_t,
        path: [(e_0, r_1, e_1), ..., (e_{t-1}, r_t, e_t)],
        cumulative_score: Π P(r_i, e_i | context),
        depth: t,
        backtrack_count: number of backtracks taken,
        status: ACTIVE | COMPLETED | PRUNED
    }
    """
    current_entity: str
    path: List[Tuple[str, str, str]] = field(default_factory=list)
    cumulative_score: float = 1.0
    depth: int = 0
    backtrack_count: int = 0
    status: BeamStatus = BeamStatus.ACTIVE

    # Track visited (entity, relation) pairs to avoid cycles
    visited: Set[Tuple[str, str]] = field(default_factory=set)

    # Track explored relations at each entity to enable proper backtracking
    explored_relations: List[Set[str]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize visited set from path."""
        if self.path:
            for head, rel, tail in self.path:
                self.visited.add((head, rel))

    def extend(
        self,
        relation: str,
        next_entity: str,
        relation_prob: float = 1.0,
        entity_prob: float = 1.0
    ) -> 'BeamState':
        """
        Add a new reasoning step to the path (CONTINUE action).

        Called after Relation Selector and Entity Selector produce
        a new (relation, entity) pair.

        Args:
            relation: The selected relation
            next_entity: The target entity
            relation_prob: P(relation | context)
            entity_prob: P(entity | context, relation)

        Returns:
            A new BeamState with the extended path
        """
        new_path = self.path + [(self.current_entity, relation, next_entity)]
        new_visited = self.visited.copy()
        new_visited.add((self.current_entity, relation))

        # Track explored relations for backtracking
        new_explored = [s.copy() for s in self.explored_relations]
        if len(new_explored) <= self.depth:
            new_explored.append(set())
        new_explored[self.depth].add(relation)

        new_score = self.cumulative_score * relation_prob * entity_prob

        return BeamState(
            current_entity=next_entity,
            path=new_path,
            cumulative_score=new_score,
            depth=self.depth + 1,
            backtrack_count=self.backtrack_count,
            status=BeamStatus.ACTIVE,
            visited=new_visited,
            explored_relations=new_explored
        )

    def backtrack(self, penalty: float = 0.8) -> Optional['BeamState']:
        """
        Remove the last step and mark the discarded branch (BACKTRACK action).

        Called when Termination Predictor outputs BACKTRACK.

        Args:
            penalty: Backtrack penalty factor γ (default 0.8)

        Returns:
            A new BeamState with the backtracked path, or None if can't backtrack
        """
        if self.depth == 0 or len(self.path) == 0:
            return None

        # Get the last step
        last_head, last_rel, _ = self.path[-1]

        # Create new path without last step
        new_path = self.path[:-1]

        # Update visited (keep the explored relation to avoid revisiting)
        new_visited = self.visited.copy()
        # Don't remove from visited - we want to avoid this path again

        # Track explored relations
        new_explored = [s.copy() for s in self.explored_relations]

        # Apply backtrack penalty
        new_score = self.cumulative_score * penalty

        # Determine new current entity
        if new_path:
            new_current = new_path[-1][2]  # tail of last triple
        else:
            new_current = last_head  # Go back to starting entity

        return BeamState(
            current_entity=new_current,
            path=new_path,
            cumulative_score=new_score,
            depth=self.depth - 1,
            backtrack_count=self.backtrack_count + 1,
            status=BeamStatus.ACTIVE,
            visited=new_visited,
            explored_relations=new_explored
        )

    def complete(self) -> 'BeamState':
        """
        Mark the beam as having found an answer (ANSWER action).

        Called when Termination Predictor outputs ANSWER with sufficient confidence.

        Returns:
            A new BeamState with COMPLETED status
        """
        return BeamState(
            current_entity=self.current_entity,
            path=self.path.copy(),
            cumulative_score=self.cumulative_score,
            depth=self.depth,
            backtrack_count=self.backtrack_count,
            status=BeamStatus.COMPLETED,
            visited=self.visited.copy(),
            explored_relations=[s.copy() for s in self.explored_relations]
        )

    def prune(self) -> 'BeamState':
        """Mark the beam as pruned."""
        return BeamState(
            current_entity=self.current_entity,
            path=self.path.copy(),
            cumulative_score=self.cumulative_score,
            depth=self.depth,
            backtrack_count=self.backtrack_count,
            status=BeamStatus.PRUNED,
            visited=self.visited.copy(),
            explored_relations=[s.copy() for s in self.explored_relations]
        )

    def get_answer_entity(self) -> str:
        """Get the answer entity (current entity for completed beams)."""
        return self.current_entity

    def get_starting_entity(self) -> str:
        """Get the starting entity of this beam."""
        if self.path:
            return self.path[0][0]
        return self.current_entity

    def has_visited(self, entity: str, relation: str) -> bool:
        """Check if (entity, relation) has been visited."""
        return (entity, relation) in self.visited

    def path_to_string(self) -> str:
        """
        Convert path to string format matching GCR output.

        Format: "e_0 -> r_1 -> e_1 -> r_2 -> e_2 -> ..."
        """
        if not self.path:
            return self.current_entity

        parts = [self.path[0][0]]  # Start with first entity
        for head, rel, tail in self.path:
            parts.extend([rel, tail])

        return " -> ".join(parts)

    def __lt__(self, other: 'BeamState') -> bool:
        """For heap operations - higher score = better."""
        return self.cumulative_score > other.cumulative_score

    def __repr__(self) -> str:
        path_str = self.path_to_string()
        return f"BeamState(entity={self.current_entity}, score={self.cumulative_score:.4f}, depth={self.depth}, status={self.status.value}, path={path_str})"


class BeamManager:
    """
    Coordinates multiple parallel exploration paths to find diverse reasoning routes.

    Implements adaptive beam search with:
    - Dynamic beam width management
    - Backtrack support with penalty
    - Early termination when enough answers found
    - Score-based pruning
    """

    def __init__(
        self,
        beam_width: int = 10,
        max_depth: int = 10,
        max_backtracks: int = 3,
        backtrack_penalty: float = 0.8,
        answer_threshold: float = 0.5,
        min_completed_beams: int = 1
    ):
        """
        Initialize the BeamManager.

        Args:
            beam_width: Maximum number of active beams (K)
            max_depth: Maximum reasoning depth (D_max)
            max_backtracks: Maximum backtracks per beam
            backtrack_penalty: Penalty factor for backtracking (γ)
            answer_threshold: Minimum confidence to accept as answer
            min_completed_beams: Minimum completed beams before early termination
        """
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.max_backtracks = max_backtracks
        self.backtrack_penalty = backtrack_penalty
        self.answer_threshold = answer_threshold
        self.min_completed_beams = min_completed_beams

        # Beam storage
        self.active_beams: List[BeamState] = []
        self.completed_beams: List[BeamState] = []
        self.pruned_beams: List[BeamState] = []

    def initialize(self, topic_entities: List[str]) -> None:
        """
        Initialize beams from topic entities.

        Args:
            topic_entities: List of starting entities from the question
        """
        self.active_beams = []
        self.completed_beams = []
        self.pruned_beams = []

        for entity in topic_entities:
            beam = BeamState(
                current_entity=entity,
                path=[],
                cumulative_score=1.0,
                depth=0,
                backtrack_count=0,
                status=BeamStatus.ACTIVE
            )
            self.active_beams.append(beam)

    def add_candidate(self, beam: BeamState) -> None:
        """Add a candidate beam for consideration."""
        if beam.status == BeamStatus.COMPLETED:
            self.completed_beams.append(beam)
        elif beam.status == BeamStatus.ACTIVE:
            self.active_beams.append(beam)
        elif beam.status == BeamStatus.PRUNED:
            self.pruned_beams.append(beam)

    def prune_to_top_k(self) -> None:
        """Keep only the top-K beams by score."""
        if len(self.active_beams) > self.beam_width:
            # Sort by score (descending)
            self.active_beams.sort(key=lambda b: b.cumulative_score, reverse=True)
            # Keep top K
            pruned = self.active_beams[self.beam_width:]
            self.active_beams = self.active_beams[:self.beam_width]
            # Track pruned beams
            for beam in pruned:
                self.pruned_beams.append(beam.prune())

    def should_terminate(self) -> bool:
        """Check if search should terminate."""
        # No more active beams
        if not self.active_beams:
            return True

        # Enough completed beams found
        if len(self.completed_beams) >= self.beam_width:
            return True

        return False

    def should_continue_beam(self, beam: BeamState) -> bool:
        """Check if a beam should continue exploration."""
        if beam.status != BeamStatus.ACTIVE:
            return False
        if beam.depth >= self.max_depth:
            return False
        return True

    def can_backtrack(self, beam: BeamState) -> bool:
        """Check if a beam can backtrack."""
        return (
            beam.depth > 0 and
            beam.backtrack_count < self.max_backtracks
        )

    def handle_answer(self, beam: BeamState, confidence: float) -> BeamState:
        """
        Handle ANSWER action from termination predictor.

        Args:
            beam: The current beam
            confidence: Confidence of the ANSWER action

        Returns:
            Updated beam (COMPLETED if confidence sufficient, else stays ACTIVE)
        """
        if confidence >= self.answer_threshold:
            return beam.complete()
        else:
            # Confidence too low, continue exploring
            return beam

    def handle_continue(
        self,
        beam: BeamState,
        relation: str,
        entity: str,
        relation_prob: float = 1.0,
        entity_prob: float = 1.0
    ) -> BeamState:
        """
        Handle CONTINUE action.

        Args:
            beam: The current beam
            relation: Selected relation
            entity: Selected entity
            relation_prob: Probability of relation selection
            entity_prob: Probability of entity selection

        Returns:
            Extended beam
        """
        return beam.extend(relation, entity, relation_prob, entity_prob)

    def handle_backtrack(self, beam: BeamState) -> Optional[BeamState]:
        """
        Handle BACKTRACK action.

        Args:
            beam: The current beam

        Returns:
            Backtracked beam, or None if can't backtrack
        """
        if not self.can_backtrack(beam):
            return None
        return beam.backtrack(self.backtrack_penalty)

    def get_active_beams(self) -> List[BeamState]:
        """Get all active beams."""
        return self.active_beams

    def get_completed_beams(self) -> List[BeamState]:
        """Get all completed beams, sorted by score."""
        return sorted(self.completed_beams, key=lambda b: b.cumulative_score, reverse=True)

    def get_all_results(self) -> List[BeamState]:
        """Get all beams (completed + active), sorted by score."""
        all_beams = self.completed_beams + self.active_beams
        return sorted(all_beams, key=lambda b: b.cumulative_score, reverse=True)

    def clear_active_beams(self) -> None:
        """Clear active beams for next iteration."""
        self.active_beams = []

    def set_active_beams(self, beams: List[BeamState]) -> None:
        """Set active beams."""
        self.active_beams = beams

    def get_statistics(self) -> dict:
        """Get search statistics."""
        max_depth = 0
        total_backtracks = 0

        for beam in self.completed_beams + self.active_beams + self.pruned_beams:
            max_depth = max(max_depth, beam.depth)
            total_backtracks += beam.backtrack_count

        return {
            "active_beams": len(self.active_beams),
            "completed_beams": len(self.completed_beams),
            "pruned_beams": len(self.pruned_beams),
            "max_depth_reached": max_depth,
            "total_backtracks": total_backtracks
        }


class PathAccumulator:
    """
    Helper class to accumulate and format paths for output.

    Maintains the history of reasoning steps and provides formatting
    utilities for evaluation output.
    """

    def __init__(self):
        self.paths: List[Tuple[str, float]] = []  # (path_string, score)

    def add_path(self, beam: BeamState) -> None:
        """Add a completed beam's path."""
        path_str = beam.path_to_string()
        self.paths.append((path_str, beam.cumulative_score))

    def add_paths(self, beams: List[BeamState]) -> None:
        """Add multiple beams' paths."""
        for beam in beams:
            self.add_path(beam)

    def get_paths(self) -> List[Tuple[str, float]]:
        """Get all paths sorted by score."""
        return sorted(self.paths, key=lambda x: x[1], reverse=True)

    def format_for_evaluation(self, top_k: int = -1) -> List[str]:
        """
        Format paths for evaluation (matching GCR output format).

        Format: "# Reasoning Path:\n{path}\n# Answer:\n{answer}"

        Args:
            top_k: Number of paths to return (-1 for all)

        Returns:
            List of formatted path strings
        """
        paths = self.get_paths()
        if top_k > 0:
            paths = paths[:top_k]

        results = []
        for path_str, score in paths:
            # Extract answer (last entity in path)
            parts = path_str.split(" -> ")
            answer = parts[-1] if parts else ""

            formatted = f"# Reasoning Path:\n{path_str}\n# Answer:\n{answer}"
            results.append(formatted)

        return results

    def get_answers(self) -> List[Tuple[str, float]]:
        """
        Get unique answers with their best scores.

        Returns:
            List of (answer, score) tuples
        """
        answer_scores: dict = {}
        for path_str, score in self.paths:
            parts = path_str.split(" -> ")
            answer = parts[-1] if parts else ""
            if answer not in answer_scores or score > answer_scores[answer]:
                answer_scores[answer] = score

        return sorted(answer_scores.items(), key=lambda x: x[1], reverse=True)

    def clear(self) -> None:
        """Clear all accumulated paths."""
        self.paths = []
