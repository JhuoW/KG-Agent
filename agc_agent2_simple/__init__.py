"""
AGC-Agent: Adaptive Graph-Constrained Agentic Reasoning

A novel framework for Knowledge Graph Question Answering (KGQA) that replaces
the static KG-Trie approach from Graph-Constrained Reasoning (GCR) with a
dynamic, step-wise constrained reasoning agent.

Key Innovation: Instead of building a trie over all paths (exponential),
AGC-Agent builds:
- A static trie over relations (O(|R|) ~ 7,000)
- Dynamic subtrie extraction per step
- On-the-fly entity mini-tries (typically 1-100 entities)

Components:
- KGIndex: Lightweight index structures (RelationIndex, NeighborIndex, RelationTokenTrie)
- ConstraintEngine: Step-wise constraint generation
- BeamState/BeamManager: Path accumulation and beam search
- AgenticController: Relation/Entity selection and termination prediction
- AGCAgent: Main reasoning interface
"""

from .kg_index import (
    KGIndex,
    RelationIndex,
    NeighborIndex,
    RelationTokenTrie,
    EntityTokenTrie,
    RelationMetadata
)

from .constraint_engine import (
    ConstraintEngine,
    StepConstraint,
    RelationConstraint,
    EntityConstraint,
    StepwiseConstrainedDecoding
)

from .beam_state import (
    BeamState,
    BeamStatus,
    BeamManager,
    PathAccumulator
)

from .agentic_controller import (
    AgenticController,
    RelationSelector,
    EntitySelector,
    TerminationPredictor,
    TerminationAction,
    TerminationResult,
    SelectionResult
)

from .agc_agent import (
    AGCAgent,
    AGCAgentConfig,
    AGCAgentResult,
    SimplifiedAGCAgent
)

__all__ = [
    # KG Index
    "KGIndex",
    "RelationIndex",
    "NeighborIndex",
    "RelationTokenTrie",
    "EntityTokenTrie",
    "RelationMetadata",

    # Constraint Engine
    "ConstraintEngine",
    "StepConstraint",
    "RelationConstraint",
    "EntityConstraint",
    "StepwiseConstrainedDecoding",

    # Beam State
    "BeamState",
    "BeamStatus",
    "BeamManager",
    "PathAccumulator",

    # Agentic Controller
    "AgenticController",
    "RelationSelector",
    "EntitySelector",
    "TerminationPredictor",
    "TerminationAction",
    "TerminationResult",
    "SelectionResult",

    # Main Agent
    "AGCAgent",
    "AGCAgentConfig",
    "AGCAgentResult",
    "SimplifiedAGCAgent",
]
