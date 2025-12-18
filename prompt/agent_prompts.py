"""
Actor Prompts for QC-Agent.

Contains system and user prompts for the LLM policy that decides
where to navigate in the knowledge graph.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# Actor (Navigation Policy) Prompts
# =============================================================================

ACTOR_SYSTEM_PROMPT = """You are a knowledge graph reasoning agent performing CONSTRAINED graph traversal. Your task is to navigate a knowledge graph step-by-step to find entities that answer a given question.

CRITICAL CONSTRAINTS:
- You can ONLY select from the provided candidate hops. NO exceptions.
- You CANNOT invent, imagine, or suggest entities/relations not in the candidate list.
- If no candidate seems relevant, you MUST still choose the most promising one OR output "STOP".

At each step, you will see:
1. The natural language question.
2. The reasoning path history (reasoning steps taken so far).
3. A list of VALID next hops in the graph (relations and neighbor entities).

Your goal is to select the hop most likely to lead to the answer, OR to stop if:
- The current entity directly answers the question, OR
- You have reached a dead end with no promising candidates.

DECISION PROCESS:
1. Parse the question to identify what type of entity is being asked for (person, place, time, etc.).
2. Examine each candidate relation - does it semantically lead toward the answer type?
3. If the current entity matches the answer type being asked, output "STOP".
4. Otherwise, select the candidate with the relation most relevant to the question.
5. If multiple candidates seem equally good, prefer shorter relation names (more direct).

OUTPUT FORMAT:
- To stop: "STOP"
- To select action k: "ACTION [k]" where k is the candidate number (1-indexed)

ANTI-HALLUCINATION RULES:
- NEVER output an entity name that is not in the candidate list.
- NEVER suggest a relation that is not in the candidate list.
- ONLY use numbers that correspond to valid candidates.
- If unsure, prefer "STOP" over making up information."""


ACTOR_USER_PROMPT_TEMPLATE = """Question:
{question}

Current Entity:
{current_entity}

Reasoning Path So Far:
{path_history}

Available Candidate Hops (ONLY choose from these):
{candidates}

REMINDER: You can ONLY select from the candidates above. Output exactly one line:
- "STOP" if current entity answers the question
- "ACTION [k]" where k is a valid candidate number from the list above"""


ACTOR_USER_PROMPT_TEMPLATE_TOP_K = """Question:
{question}

Current Entity:
{current_entity}

Reasoning Path So Far:
{path_history}

Available Candidate Hops (ONLY choose from these):
{candidates}

Select the top {k} most promising candidates from the list above.
RULES:
- ONLY use numbers that appear in the candidate list
- Numbers must be separated by commas
- Include "STOP" if you believe the current entity answers the question

Output format: "3, 1, 5" or "STOP" or "STOP, 2, 4" (numbers in order of preference)"""


# =============================================================================
# Answer Aggregation Prompts
# =============================================================================

AGGREGATOR_SYSTEM_PROMPT = """You are an expert knowledge graph reasoner performing GROUNDED answer extraction.

CRITICAL: You can ONLY output answers that appear as the FINAL ENTITY in one of the provided reasoning paths.
You CANNOT invent, imagine, or suggest any entity that is not explicitly present in the paths.

Your task is to extract answers from the provided Candidate Reasoning Paths that correctly answer the question.

VERIFICATION PROCESS:
1. Read the Question and identify what type of entity is being asked for.
2. For EACH candidate path, examine:
   a) Does the FINAL relation in the path match what the question asks?
   b) Does the FINAL entity (end of path) answer the question?
   c) Is the reasoning chain logically valid?
3. ONLY include answers where the path logically supports the answer.

VALIDITY CRITERIA:
- The final relation must be semantically relevant to the question.
- Example: For "who is X's brother", the path should end with a sibling/brother relation.
- Example: For "where was X born", the path should end with a birthplace relation.

OUTPUT RULES:
- ONLY output entities that appear at the END of a candidate path.
- Do NOT output the starting entity unless it genuinely answers the question.
- Do NOT hallucinate or infer entities not present in the paths.
- If NO path validly answers the question, output: "# Answer: NONE, #Reasoning Path: No valid path found"
- For multi-answer questions (e.g., "list all..."), include ALL valid distinct end entities.

Output format (one line per answer):
# Answer: [Exact Entity Name from Path End], #Reasoning Path: [Full Path String]"""


AGGREGATOR_USER_PROMPT_TEMPLATE = """Question:
{question}

Candidate Reasoning Paths (ranked by confidence):
{candidate_paths}

INSTRUCTIONS:
1. Examine each path above. The FINAL entity in each path is a potential answer.
2. Select paths where the final relation semantically matches the question.
3. Output ONLY entities that appear at the END of valid paths.

Output format (no other text):
# Answer: [entity from path end], #Reasoning Path: [full path string]"""


# =============================================================================
# Helper Classes
# =============================================================================

@dataclass
class CandidateAction:
    """Represents a candidate action (next hop) in the KG."""
    relation: str
    entity: str
    index: int

    def format(self) -> str:
        return f"({self.index}) via {self.relation} to {self.entity}"


@dataclass
class ActorInput:
    """Input structure for the Actor."""
    question: str
    current_entity: str
    path_history: List[Tuple[str, str, str]]  # [(entity, relation, entity), ...]
    candidates: List[CandidateAction]


class ActorPromptBuilder:
    """Builds prompts for the Actor (LLM navigation policy)."""

    def __init__(self, top_k: int = 1):
        """
        Args:
            top_k: Number of actions to request from the LLM.
                   If 1, uses single-action prompt.
                   If > 1, uses multi-action prompt for diversity.
        """
        self.top_k = top_k
        self.system_prompt = ACTOR_SYSTEM_PROMPT

    def format_path_history(self, path_history: List[Tuple[str, str, str]]) -> str:
        """Format the reasoning path history."""
        if not path_history:
            return "(Starting position - no previous steps)"

        lines = []
        for i, (h, r, t) in enumerate(path_history, 1):
            lines.append(f"{i}. {h} -> ({r}) -> {t}")
        return "\n".join(lines)

    def format_candidates(self, candidates: List[CandidateAction]) -> str:
        """Format candidate next hops."""
        if not candidates:
            return "(No available candidates - must STOP)"

        return "\n".join(c.format() for c in candidates)

    def build_user_prompt(self, actor_input: ActorInput) -> str:
        """Build the user prompt for the Actor."""
        path_history_str = self.format_path_history(actor_input.path_history)
        candidates_str = self.format_candidates(actor_input.candidates)

        if self.top_k == 1:
            template = ACTOR_USER_PROMPT_TEMPLATE
            return template.format(
                question=actor_input.question,
                current_entity=actor_input.current_entity,
                path_history=path_history_str,
                candidates=candidates_str
            )
        else:
            template = ACTOR_USER_PROMPT_TEMPLATE_TOP_K
            return template.format(
                question=actor_input.question,
                current_entity=actor_input.current_entity,
                path_history=path_history_str,
                candidates=candidates_str,
                k=self.top_k
            )

    def build_messages(self, actor_input: ActorInput) -> List[dict]:
        """Build the chat messages for the Actor."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.build_user_prompt(actor_input)}
        ]


class AggregatorPromptBuilder:
    """Builds prompts for the Answer Aggregator (inductive reasoner)."""

    def __init__(self):
        self.system_prompt = AGGREGATOR_SYSTEM_PROMPT

    def format_candidate_paths(
        self,
        paths: List[Tuple[str, float]]  # [(path_string, score), ...]
    ) -> str:
        """Format candidate paths with their scores.

        Note: path_str already contains the full path including the end entity
        in format: "e0 -> r1 -> e1 -> r2 -> e2"
        """
        lines = []
        for i, (path_str, score) in enumerate(paths, 1):
            # Extract the end entity from the path for display
            parts = path_str.split(" -> ")
            end_entity = parts[-1] if parts else "Unknown"
            # Format: [Path i] (Score: x.xxx): path -> END_ENTITY
            lines.append(f"[Path {i}] (Score: {score:.3f}): {path_str}")
        return "\n".join(lines)

    def build_user_prompt(
        self,
        question: str,
        candidate_paths: List[Tuple[str, float]]
    ) -> str:
        """Build the user prompt for answer aggregation."""
        paths_str = self.format_candidate_paths(candidate_paths)
        return AGGREGATOR_USER_PROMPT_TEMPLATE.format(
            question=question,
            candidate_paths=paths_str
        )

    def build_messages(
        self,
        question: str,
        candidate_paths: List[Tuple[str, float]]
    ) -> List[dict]:
        """Build the chat messages for the Aggregator."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.build_user_prompt(question, candidate_paths)}
        ]


# =============================================================================
# Response Parsing
# =============================================================================

class ActorResponseParser:
    """Parses LLM responses from the Actor."""

    @staticmethod
    def parse_single_action(response: str) -> Tuple[bool, Optional[int]]:
        """
        Parse a single-action response.

        Returns:
            (is_stop, action_index)
            If is_stop is True, action_index is None.
        """
        response = response.strip().upper()

        if "STOP" in response:
            return True, None

        # Look for "ACTION k" or just the number
        import re
        match = re.search(r'ACTION\s*\[?(\d+)\]?', response)
        if match:
            return False, int(match.group(1))

        # Try to find any number
        match = re.search(r'\d+', response)
        if match:
            return False, int(match.group())

        # Default to STOP if can't parse
        return True, None

    @staticmethod
    def parse_top_k_actions(response: str, k: int) -> Tuple[bool, List[int]]:
        """
        Parse a top-k action response.

        Returns:
            (includes_stop, action_indices)
        """
        response = response.strip().upper()
        includes_stop = "STOP" in response

        import re
        numbers = re.findall(r'\d+', response)
        action_indices = [int(n) for n in numbers][:k]

        return includes_stop, action_indices


class AggregatorResponseParser:
    """Parses LLM responses from the Aggregator."""

    @staticmethod
    def parse_answers(response: str) -> List[Tuple[str, str]]:
        """
        Parse aggregator response to extract answers and paths.

        Returns:
            List of (answer, reasoning_path) tuples
        """
        results = []
        lines = response.strip().split("\n")

        import re
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match pattern: # Answer: {entity}, #Reasoning Path: {path}
            match = re.match(
                r'#?\s*Answer:\s*([^,#]+),?\s*#?\s*Reasoning Path:\s*(.+)',
                line,
                re.IGNORECASE
            )
            if match:
                answer = match.group(1).strip()
                path = match.group(2).strip()
                results.append((answer, path))

        return results