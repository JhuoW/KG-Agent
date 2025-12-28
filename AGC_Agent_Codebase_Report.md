# AGC-Agent Codebase Analysis Report

## 1. Project Overview

**AGC-Agent (Adaptive Graph-Constrained Agentic Reasoning)** is a novel framework for Knowledge Graph Question Answering (KGQA) that replaces the static KG-Trie approach from Graph-Constrained Reasoning (GCR) with a dynamic, step-wise constrained reasoning agent.

### Core Innovation

The key innovation is the transformation from **path-level constraints** to **step-level constraints**:

| Approach | Constraint Type | Complexity |
|----------|----------------|------------|
| GCR (Original) | Pre-computed trie over all paths | Exponential O(paths) |
| AGC-Agent | Dynamic per-step constraints | Linear O(\|R\|) + small entity tries |

### Target Datasets
- **WebQSP** (`RoG-webqsp`)
- **CWQ** (`RoG-cwq`)

### Base Model
- `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` (KG-specialized LLM)

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AGC-Agent System                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐                       │
│  │  agc_reasoning2  │───▶│     AGCAgent     │                       │
│  │    (Entry Point) │    │  (Main Interface)│                       │
│  └──────────────────┘    └────────┬─────────┘                       │
│                                   │                                  │
│         ┌─────────────────────────┼─────────────────────────┐       │
│         ▼                         ▼                         ▼       │
│  ┌─────────────┐         ┌─────────────────┐       ┌─────────────┐  │
│  │   KGIndex   │         │AgenticController│       │ BeamManager │  │
│  │ (3 Indices) │         │ (LLM Selectors) │       │ (Search)    │  │
│  └─────┬───────┘         └────────┬────────┘       └─────────────┘  │
│        │                          │                                  │
│  ┌─────┴─────────────────────────┴───────────────────┐              │
│  │            ConstraintEngine                        │              │
│  │  (Relation + Entity Constraint Generation)        │              │
│  └───────────────────────────────────────────────────┘              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Components and Responsibilities

### 3.1 Entry Point: `agc_reasoning2.py`

**Location:** [agc_reasoning2.py](agc_reasoning2.py)

**Responsibilities:**
- Command-line interface and argument parsing
- Multi-GPU orchestration via subprocess spawning
- Dataset loading from HuggingFace (`rmanluo/RoG-webqsp`, `rmanluo/RoG-cwq`)
- Model initialization wrapper (`AGCReasoningModel`)
- Result merging and evaluation

**Key Features:**
- Supports single-GPU and multi-GPU execution
- Resume capability for interrupted runs
- Configurable beam search parameters

### 3.2 Core Agent: `agc_agent2/agc_agent.py`

**Location:** [agc_agent2/agc_agent.py](agc_agent2/agc_agent.py)

**Classes:**

| Class | Purpose |
|-------|---------|
| `AGCAgentConfig` | Dataclass holding all configuration (beam_width, max_depth, top_k values, thresholds) |
| `AGCAgentResult` | Dataclass for returning predictions, answers, and reasoning traces |
| `AGCAgent` | Main agent orchestrating beam search with step-wise constraints |
| `SimplifiedAGCAgent` | Lightweight variant using single LLM call per step |

**Key Method: `reason()`**
```python
def reason(self, question, graph_triples, topic_entities) -> AGCAgentResult:
    # 1. Build KG index from triples
    # 2. Initialize controller and beam manager
    # 3. Run adaptive beam search
    # 4. Format results for evaluation
```

### 3.3 KG Index Structures: `agc_agent2/kg_index.py`

**Location:** [agc_agent2/kg_index.py](agc_agent2/kg_index.py)

Implements the three lightweight indices replacing GCR's monolithic trie:

| Index | Mapping | Purpose |
|-------|---------|---------|
| `RelationIndex` | Entity → {(relation, count, freq)} | What relations exist from an entity |
| `NeighborIndex` | (Entity, Relation) → {Entities} | What entities are reachable |
| `RelationTokenTrie` | Tokenized relations → Trie | Constrained decoding for relations |
| `EntityTokenTrie` | Tokenized entities → Trie | Dynamically built per-step |

**Key Insight:** Relations are tokenized once globally (~7,000), while entity tries are built on-demand (typically 1-100 entities).

### 3.4 Constraint Engine: `agc_agent2/constraint_engine.py`

**Location:** [agc_agent2/constraint_engine.py](agc_agent2/constraint_engine.py)

**Classes:**

| Class | Purpose |
|-------|---------|
| `StepConstraint` | Base class for token-level constraints |
| `RelationConstraint` | Constrains LLM to valid relations from current entity |
| `EntityConstraint` | Constrains LLM to valid entities for (entity, relation) pair |
| `StepwiseConstrainedDecoding` | Callback for `model.generate()` |
| `ConstraintEngine` | Factory for creating constraints |

**Key Formula:**
```
P'(t_i | t_{<i}) = P(t_i) / Z  if t_{1:i} ∈ ValidPrefixes(Trie)
                 = 0           otherwise
```

### 3.5 Agentic Controller: `agc_agent2/agentic_controller.py`

**Location:** [agc_agent2/agentic_controller.py](agc_agent2/agentic_controller.py)

**Components:**

| Component | Responsibility |
|-----------|----------------|
| `RelationSelector` | Selects top-k relations using LLM + constraints |
| `EntitySelector` | Selects top-k entities using LLM + constraints |
| `TerminationPredictor` | Decides ANSWER / CONTINUE / BACKTRACK |
| `AgenticController` | Orchestrates one reasoning step |

**Prompt Templates:** Well-structured system and user prompts for each selector (lines 36-107).

### 3.6 Beam State Management: `agc_agent2/beam_state.py`

**Location:** [agc_agent2/beam_state.py](agc_agent2/beam_state.py)

| Class | Purpose |
|-------|---------|
| `BeamStatus` | Enum: ACTIVE, COMPLETED, PRUNED |
| `BeamState` | Immutable state of a reasoning trajectory |
| `BeamManager` | Manages active/completed/pruned beams, pruning logic |
| `PathAccumulator` | Formats paths for evaluation output |

**BeamState Operations:**
- `extend()` - Add new step (returns new BeamState)
- `backtrack()` - Remove last step with penalty
- `complete()` - Mark as answer found

### 3.7 Evaluation Utilities: `utils/gcr_utils.py`

**Location:** [utils/gcr_utils.py](utils/gcr_utils.py)

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `eval_path_result_w_ans()` | Main evaluation (Accuracy, Hit, F1, Path F1, Path Answer F1) |
| `eval_f1()` | Computes precision, recall, F1 |
| `extract_topk_prediction()` | Deduplicates and ranks predictions |
| `replace_mid_answers_with_path_entity()` | Handles Freebase MID filtering |
| `MarisaTrie` | Efficient trie using marisa_trie library |

---

## 4. Data Flow

```
Input Question
      │
      ▼
┌──────────────────┐
│ Build KG Index   │  ← graph_triples from dataset
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Initialize Beams │  ← one beam per topic entity
└────────┬─────────┘
         │
    ┌────┴────┐
    │  Loop   │ (max_depth iterations)
    └────┬────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ For each active beam:                     │
│  1. TerminationPredictor → action        │
│  2. If CONTINUE:                          │
│     a. RelationSelector → top-k relations │
│     b. EntitySelector → top-k entities    │
│     c. Create extended beams              │
│  3. If ANSWER: mark completed             │
│  4. If BACKTRACK: backtrack with penalty  │
└──────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│ Prune to top-K   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Format Output    │  → predictions.jsonl
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Evaluate         │  → Accuracy, Hit, F1, etc.
└──────────────────┘
```

---

## 5. Code Quality Observations

### 5.1 Strengths

1. **Modular Design:** Clean separation of concerns (index, constraint, controller, beam management)

2. **Comprehensive Documentation:** Each module has detailed docstrings explaining the theory from CLAUDE.md

3. **GCR Compatibility:** Maintains same evaluation metrics and output format for fair comparison

4. **Flexible Configuration:** `AGCAgentConfig` dataclass allows easy experimentation

5. **Multi-GPU Support:** Proper subprocess-based parallelization with result merging

6. **Immutable BeamState:** `extend()`, `backtrack()`, `complete()` return new objects, avoiding mutation bugs

7. **Type Hints:** Extensive use of type annotations improves readability

### 5.2 Areas for Improvement

1. **Error Handling:** Some try/except blocks have broad exceptions:
   ```python
   # agc_agent2/agentic_controller.py:298-301
   except Exception as e:
       print(f"RelationSelector generation error: {e}")
       return []
   ```
   Could benefit from more specific exception types.

2. **Debug Print Statement:** Leftover debug code in class definition:
   ```python
   # agc_reasoning2.py:43
   print("SPECIAL_TOKENS:", SPECIAL_TOKENS)  # Should be removed
   ```

3. **Hardcoded Values:** Some magic numbers could be configurable:
   - `max_entities_in_prompt: int = 50` (line 358 in agentic_controller.py)
   - `max_token_id: int = 256001` (multiple locations)

4. **Code Duplication:** `RelationTokenTrie` and `EntityTokenTrie` share nearly identical logic for character mapping and trie operations. Could be refactored into a base class.

5. **Probability Estimation:** The probability assignment is simplified:
   ```python
   # agentic_controller.py:318
   prob = 1.0 / (i + 1)  # Higher rank = higher prob estimate
   ```
   Could be improved with actual logit-based probabilities.

6. **Missing Tests:** No unit tests found in the repository for the agc_agent2 module.

---

## 6. CLAUDE.md vs. Implementation Comparison

### 6.1 Well-Aligned Components

| CLAUDE.md Section | Implementation | Status |
|-------------------|----------------|--------|
| 3.1.1 RelationIndex | `kg_index.py:RelationIndex` | Fully implemented |
| 3.1.2 NeighborIndex | `kg_index.py:NeighborIndex` | Fully implemented |
| 3.1.3 Relation Token Trie | `kg_index.py:RelationTokenTrie` | Fully implemented |
| 3.2 Constraint Engine | `constraint_engine.py` | Fully implemented |
| 3.4.1 Relation Selector | `agentic_controller.py:RelationSelector` | Implemented with prompts |
| 3.4.2 Entity Selector | `agentic_controller.py:EntitySelector` | Implemented with prompts |
| 3.4.3 Termination Predictor | `agentic_controller.py:TerminationPredictor` | Implemented |
| 3.4.4 Path Accumulator | `beam_state.py:PathAccumulator` | Implemented |
| 3.4.5 Beam Manager | `beam_state.py:BeamManager` | Implemented |

### 6.2 Differences/Simplifications

| CLAUDE.md Specification | Implementation | Notes |
|------------------------|----------------|-------|
| Entity Ranking Pre-filter (Section 3.4.2) | `max_entities_in_prompt=50` | Simplified to truncation |
| Confidence Scoring (Section 3.4.3) | Approximated from first token logits | Not full action probability distribution |
| Multi-Path Aggregation (Section 3.5) | `PathAccumulator.format_for_evaluation_with_llm()` | Uses single-entity extraction per path |
| Answer Synthesis (Section 3.6) | Basic formatting in `AGCAgentResult` | No structured JSON with confidence scores |

### 6.3 Prompt Templates

The prompts in `agentic_controller.py` closely follow CLAUDE.md specifications but are simplified:

**CLAUDE.md (Relation Selector):**
```
Consider:
1. SEMANTIC RELEVANCE: How well does the relation's meaning align with what the question is asking?
2. PATH PROGRESS: Does this relation move closer to the type of entity the question seeks?
3. REASONING CHAIN: How does this relation connect to the previous reasoning steps?
```

**Implementation:**
```python
RELATION_SELECTOR_SYSTEM_PROMPT = """You are a Knowledge Graph Reasoning Agent...
You must ONLY select from the available relations listed...
Output your selected relation between <REL> and </REL> tags..."""
```

The implementation is more concise but captures the essential guidance.

---

## 7. Performance Results (from `agc_reasoning2.sh`)

Based on recorded experiment results:

| Metric | Value |
|--------|-------|
| Accuracy | 71.16% |
| Hit | 86.0% |
| F1 | 43.44% |
| Path F1 | 39.84% |
| Path Answer F1 | 46.01% |

These results use: `BEAM_WIDTH=10`, `RELATION_TOP_K=3`, `ENTITY_TOP_K=3`, `INDEX_LEN=2`

---

## 8. File Structure Summary

```
LLM-KG-Reasoning/
├── agc_reasoning2.py          # Main entry point
├── agc_reasoning2.sh          # Run script with config
├── agc_agent2/                # Core AGC-Agent module
│   ├── __init__.py            # Public API exports
│   ├── agc_agent.py           # AGCAgent, SimplifiedAGCAgent
│   ├── kg_index.py            # KGIndex, RelationIndex, NeighborIndex, Tries
│   ├── constraint_engine.py   # StepConstraint, RelationConstraint, EntityConstraint
│   ├── beam_state.py          # BeamState, BeamManager, PathAccumulator
│   └── agentic_controller.py  # RelationSelector, EntitySelector, TerminationPredictor
├── utils/
│   ├── gcr_utils.py           # Evaluation functions, MarisaTrie
│   └── utils.py               # Graph utilities
├── AA_Trie_Reasoning/         # Reference GCR implementation
│   └── reasoning_trie_multigpu.py
└── CLAUDE.md                  # Specification document
```

---

## 9. Recommendations

1. **Add Unit Tests:** Create tests for `kg_index.py`, `constraint_engine.py`, and `beam_state.py`

2. **Remove Debug Prints:** Clean up leftover `print()` statements

3. **Refactor Trie Classes:** Extract common logic into `BaseTokenTrie`

4. **Improve Probability Estimation:** Use actual generation scores instead of rank-based approximation

5. **Add Logging:** Replace `print()` with proper logging framework

6. **Document Configuration:** Add a CONFIG.md explaining all hyperparameters

7. **Benchmark Constrained vs. Unconstrained:** Add ablation study flag for comparing with/without trie constraints

---

## 10. Conclusion

The AGC-Agent implementation faithfully realizes the CLAUDE.md specification with a clean, modular architecture. The core innovation of step-wise constraints is well-implemented through the `ConstraintEngine` and dynamic trie construction. The codebase is production-ready for research experiments, though would benefit from additional testing and some code cleanup before broader release.

The implementation achieves competitive performance (86% Hit, ~71% Accuracy) on the WebQSP benchmark, validating the theoretical approach of replacing exponential path-level constraints with linear step-level constraints.
