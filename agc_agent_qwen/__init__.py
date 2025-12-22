"""
AGC-Agent for Qwen3-8B: Adaptive Graph-Constrained Agentic Reasoning

This module provides Qwen3-8B specific implementations for the AGC-Agent framework.
Key adaptations:
- Qwen3-specific chat template with enable_thinking=False for structured KG reasoning
- Adjusted generation parameters (temp=0.7, top_p=0.8 for non-thinking mode)
- Proper handling of Qwen3's tokenizer and vocab size

Components:
- QwenKGIndex: KG index with Qwen3 tokenizer handling
- QwenAgenticController: Controller with Qwen3 chat templates
- QwenAGCAgent: Main agent class for Qwen3
"""

from .kg_index_qwen import (
    QwenKGIndex,
    QwenRelationTokenTrie,
    QwenEntityTokenTrie,
)

from .agentic_controller_qwen import (
    QwenAgenticController,
    QwenRelationSelector,
    QwenEntitySelector,
    QwenTerminationPredictor,
)

from .agc_agent_qwen import (
    QwenAGCAgent,
    QwenAGCAgentConfig,
    QwenSimplifiedAGCAgent,
)

# Re-export common components from base agc_agent
from agc_agent import (
    BeamState,
    BeamStatus,
    BeamManager,
    PathAccumulator,
    ConstraintEngine,
    StepConstraint,
    RelationConstraint,
    EntityConstraint,
    StepwiseConstrainedDecoding,
    AGCAgentResult,
    TerminationAction,
    TerminationResult,
    SelectionResult,
)

__all__ = [
    # Qwen3-specific
    "QwenKGIndex",
    "QwenRelationTokenTrie",
    "QwenEntityTokenTrie",
    "QwenAgenticController",
    "QwenRelationSelector",
    "QwenEntitySelector",
    "QwenTerminationPredictor",
    "QwenAGCAgent",
    "QwenAGCAgentConfig",
    "QwenSimplifiedAGCAgent",

    # Re-exported from base
    "BeamState",
    "BeamStatus",
    "BeamManager",
    "PathAccumulator",
    "ConstraintEngine",
    "StepConstraint",
    "RelationConstraint",
    "EntityConstraint",
    "StepwiseConstrainedDecoding",
    "AGCAgentResult",
    "TerminationAction",
    "TerminationResult",
    "SelectionResult",
]
