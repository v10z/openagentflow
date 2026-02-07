"""Socratic Interrogation reasoning engine.

Reason by asking progressively deeper questions that expose hidden
assumptions, demand evidence, surface counterexamples, and probe
implications.  Based on the Socratic method: wisdom comes from knowing
what you do not know.

Phase overview::

    INITIAL RESPONSE -> Generate first-pass answer
    INTERROGATE      -> Ask probing questions across multiple categories
    RESPOND          -> Answer each question honestly
    REFINE           -> Update the answer based on exposed assumptions
    REPEAT           -> Until no new assumptions are exposed

Example::

    from openagentflow.reasoning.socratic import SocraticInterrogation

    engine = SocraticInterrogation(max_rounds=4)
    trace = await engine.reason(
        query="Should we rewrite our backend in Rust?",
        llm_provider=my_provider,
    )
    print(trace.final_output)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openagentflow.core.types import Message, ToolSpec
from openagentflow.llm.base import BaseLLMProvider
from openagentflow.reasoning.base import ReasoningEngine, ReasoningTrace

logger = logging.getLogger(__name__)

_DEFAULT_QUESTION_TYPES = [
    "assumption",
    "evidence",
    "counterexample",
    "implication",
    "perspective",
    "definition",
]


class SocraticInterrogation(ReasoningEngine):
    """Reason by asking progressively deeper questions.

    The engine implements a structured version of the Socratic method:

    1. **INITIAL RESPONSE** -- Generate a first-pass answer to the query,
       allowing the model to express its natural reasoning.
    2. **INTERROGATE** -- For each configured question type, generate one
       or more probing questions that challenge the initial answer:

       - *assumption*: "What are you assuming here that might not be true?"
       - *evidence*: "What evidence supports this claim?"
       - *counterexample*: "Can you think of a case where this fails?"
       - *implication*: "If this is true, what necessarily follows?"
       - *perspective*: "How would someone with a different viewpoint see this?"
       - *definition*: "Are you using any terms loosely or ambiguously?"

    3. **RESPOND** -- Answer each probing question honestly, surfacing
       weaknesses, gaps, and unjustified leaps.
    4. **REFINE** -- Revise the original answer to address the exposed
       assumptions and gaps.
    5. **REPEAT** -- Repeat INTERROGATE + RESPOND + REFINE until either
       no genuinely new assumptions are surfaced or ``max_rounds`` is
       reached.

    Attributes:
        name: ``"SocraticInterrogation"``
        description: Short human-readable summary.
        max_rounds: Maximum interrogation-refinement cycles.
        question_types: Categories of Socratic questions to use.
        questions_per_type: How many questions to generate per category
            in each round.
        novelty_threshold: If the fraction of "no new insight" responses
            exceeds this value, the loop terminates early.
    """

    name: str = "SocraticInterrogation"
    description: str = (
        "Progressive questioning to expose assumptions and deepen reasoning."
    )

    def __init__(
        self,
        max_rounds: int = 4,
        question_types: list[str] | None = None,
        questions_per_type: int = 1,
        novelty_threshold: float = 0.7,
    ) -> None:
        """Initialise the Socratic Interrogation engine.

        Args:
            max_rounds: Maximum interrogation-refinement rounds.
            question_types: List of question categories to use.  Defaults
                to all six standard types (assumption, evidence,
                counterexample, implication, perspective, definition).
            questions_per_type: Number of questions to generate per
                question type in each round.
            novelty_threshold: If this fraction or more of answers to
                probing questions indicate no new insight, the engine
                considers the answer converged and stops early.
        """
        self.max_rounds = max(1, max_rounds)
        self.question_types = question_types or list(_DEFAULT_QUESTION_TYPES)
        self.questions_per_type = max(1, questions_per_type)
        self.novelty_threshold = novelty_threshold

    # ------------------------------------------------------------------
    # Question type descriptions (used in prompts)
    # ------------------------------------------------------------------

    _QUESTION_DESCRIPTIONS: dict[str, str] = {
        "assumption": (
            "Challenge hidden assumptions. What is being taken for granted? "
            "What unstated premises is the answer relying on?"
        ),
        "evidence": (
            "Demand evidence. What data or reasoning supports the claims made? "
            "Are there unsupported assertions?"
        ),
        "counterexample": (
            "Find counterexamples. In what situations would this answer fail "
            "or be wrong? What edge cases break the logic?"
        ),
        "implication": (
            "Explore implications. If the answer is correct, what necessarily "
            "follows? Are there unexpected or undesirable consequences?"
        ),
        "perspective": (
            "Shift perspective. How would someone with a completely different "
            "background, role, or value system view this answer?"
        ),
        "definition": (
            "Clarify definitions. Are any key terms being used loosely or "
            "ambiguously? Would a different definition change the conclusion?"
        ),
    }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def reason(
        self,
        query: str,
        llm_provider: BaseLLMProvider,
        tools: list[ToolSpec] | None = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> ReasoningTrace:
        """Execute the full Socratic Interrogation reasoning strategy.

        Args:
            query: The user question or task to reason about.
            llm_provider: An LLM provider for generating answers and
                probing questions.
            tools: Optional tool specs (currently unused by this engine).
            max_iterations: Hard cap on rounds (combined with
                ``max_rounds``).
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` containing initial-response,
            interrogation, response, refinement, and convergence steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)
        effective_rounds = min(self.max_rounds, max_iterations)

        # Phase 1 -- INITIAL RESPONSE
        current_answer = await self._initial_response(query, llm_provider, trace)

        # Phases 2-5 -- iterative interrogation
        for rnd in range(effective_rounds):
            # Phase 2 -- INTERROGATE
            questions = await self._interrogate(
                query, current_answer, rnd, llm_provider, trace
            )

            if not questions:
                conv_step = self._make_step(
                    step_type="convergence",
                    content=f"No probing questions generated in round {rnd + 1}. Converged.",
                    score=1.0,
                    metadata={"phase": "converge", "round": rnd + 1},
                )
                trace.add_step(conv_step)
                break

            # Phase 3 -- RESPOND to each question
            responses = await self._respond_to_questions(
                query, current_answer, questions, rnd, llm_provider, trace
            )

            # Check for convergence (most responses yield no new insight)
            novelty_results = self._assess_novelty(responses)
            no_insight_fraction = novelty_results["no_insight_fraction"]

            if no_insight_fraction >= self.novelty_threshold:
                conv_step = self._make_step(
                    step_type="convergence",
                    content=(
                        f"Round {rnd + 1}: {no_insight_fraction:.0%} of responses "
                        f"yielded no new insight (threshold: "
                        f"{self.novelty_threshold:.0%}). Answer has converged."
                    ),
                    score=1.0,
                    metadata={
                        "phase": "converge",
                        "round": rnd + 1,
                        "no_insight_fraction": no_insight_fraction,
                    },
                )
                trace.add_step(conv_step)
                break

            # Phase 4 -- REFINE the answer
            current_answer = await self._refine_answer(
                query, current_answer, questions, responses, rnd, llm_provider, trace
            )

        # Final synthesis
        final_output = await self._final_synthesis(
            query, current_answer, llm_provider, trace
        )

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase 1 -- INITIAL RESPONSE
    # ------------------------------------------------------------------

    async def _initial_response(
        self,
        query: str,
        provider: BaseLLMProvider,
        trace: ReasoningTrace,
    ) -> str:
        """Generate the initial first-pass answer.

        Args:
            query: Original user query.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The initial answer text.
        """
        prompt = (
            f"Answer the following query thoroughly. Provide your best "
            f"initial response with reasoning.\n\n"
            f"Query: {query}"
        )

        answer = await self._call_llm(
            provider=provider,
            messages=[Message.user(prompt)],
            trace=trace,
            system_prompt="You are a knowledgeable, thoughtful analyst.",
        )

        step = self._make_step(
            step_type="initial_response",
            content=answer,
            score=0.5,
            metadata={"phase": "initial", "round": 0},
        )
        trace.add_step(step)
        return answer

    # ------------------------------------------------------------------
    # Phase 2 -- INTERROGATE
    # ------------------------------------------------------------------

    async def _interrogate(
        self,
        query: str,
        current_answer: str,
        round_num: int,
        provider: BaseLLMProvider,
        trace: ReasoningTrace,
    ) -> list[dict[str, str]]:
        """Generate probing questions for each question type.

        Returns a list of dicts with keys ``type``, ``description``, and
        ``question``.

        Args:
            query: Original user query.
            current_answer: The answer to interrogate.
            round_num: Current round number (0-indexed).
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            List of question dicts.
        """
        type_blocks = []
        for qt in self.question_types:
            desc = self._QUESTION_DESCRIPTIONS.get(qt, f"Probe the answer regarding: {qt}")
            type_blocks.append(f'  - "{qt}": {desc}')

        types_text = "\n".join(type_blocks)
        total_questions = len(self.question_types) * self.questions_per_type

        prompt = (
            f"You are a Socratic interrogator. Your job is to probe an answer "
            f"for weaknesses, hidden assumptions, and gaps.\n\n"
            f"Original query: {query}\n\n"
            f"Current answer (round {round_num + 1}):\n{current_answer}\n\n"
            f"Generate {self.questions_per_type} probing question(s) for EACH "
            f"of these question types:\n{types_text}\n\n"
            f"Return a JSON array of objects, each with 'type' and 'question' keys.\n"
            f"Example: [{{'type': 'assumption', 'question': 'What are you assuming about...?'}}]\n"
            f"Generate exactly {total_questions} questions total.\n"
            f"Return ONLY the JSON array."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[Message.user(prompt)],
            trace=trace,
            system_prompt="You are a rigorous Socratic questioner. Return valid JSON only.",
        )

        questions = self._parse_questions(raw, self.question_types, self.questions_per_type)

        # Record interrogation step
        step = self._make_step(
            step_type="interrogation",
            content="\n".join(f"[{q['type']}] {q['question']}" for q in questions),
            score=0.0,
            metadata={
                "phase": "interrogate",
                "round": round_num + 1,
                "question_count": len(questions),
                "question_types_used": list({q["type"] for q in questions}),
            },
        )
        trace.add_step(step)
        return questions

    # ------------------------------------------------------------------
    # Phase 3 -- RESPOND to questions
    # ------------------------------------------------------------------

    async def _respond_to_questions(
        self,
        query: str,
        current_answer: str,
        questions: list[dict[str, str]],
        round_num: int,
        provider: BaseLLMProvider,
        trace: ReasoningTrace,
    ) -> list[dict[str, str]]:
        """Honestly answer each probing question.

        The LLM is asked to answer from the perspective of a self-critical
        thinker who genuinely wants to improve the answer.

        Args:
            query: Original user query.
            current_answer: The current answer being examined.
            questions: The probing questions to answer.
            round_num: Current round number.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            List of dicts with keys ``type``, ``question``, and ``response``.
        """
        question_listing = "\n".join(
            f"{idx + 1}. [{q['type']}] {q['question']}" for idx, q in enumerate(questions)
        )

        prompt = (
            f"You previously gave this answer to a query, and a Socratic "
            f"interrogator has posed challenging questions. Answer each one "
            f"HONESTLY -- acknowledge weaknesses, gaps, and unjustified "
            f"assumptions where they exist.\n\n"
            f"Original query: {query}\n\n"
            f"Your answer:\n{current_answer}\n\n"
            f"Probing questions:\n{question_listing}\n\n"
            f"For each question, provide an honest response. If the question "
            f"exposes a genuine weakness, admit it. If the answer already "
            f"addresses the concern, explain how. If the question does not "
            f"reveal anything new, say 'no new insight' and briefly explain.\n\n"
            f"Return a JSON array of objects with 'question_index' (1-based) "
            f"and 'response' keys.\n"
            f"Example: [{{'question_index': 1, 'response': 'You are right, I assumed...'}}]\n"
            f"Return ONLY the JSON array."
        )

        raw = await self._call_llm(
            provider=provider,
            messages=[Message.user(prompt)],
            trace=trace,
            system_prompt=(
                "You are a rigorously honest self-critic. Acknowledge real "
                "weaknesses. Return valid JSON only."
            ),
        )

        responses = self._parse_responses(raw, questions)

        # Record response step
        response_summary = "\n".join(
            f"[{r['type']}] Q: {r['question'][:80]}... -> {r['response'][:100]}..."
            for r in responses
        )
        step = self._make_step(
            step_type="self_examination",
            content=response_summary,
            score=0.0,
            metadata={
                "phase": "respond",
                "round": round_num + 1,
                "response_count": len(responses),
            },
        )
        trace.add_step(step)
        return responses

    # ------------------------------------------------------------------
    # Phase 4 -- REFINE
    # ------------------------------------------------------------------

    async def _refine_answer(
        self,
        query: str,
        current_answer: str,
        questions: list[dict[str, str]],
        responses: list[dict[str, str]],
        round_num: int,
        provider: BaseLLMProvider,
        trace: ReasoningTrace,
    ) -> str:
        """Revise the answer based on insights from interrogation.

        Args:
            query: Original user query.
            current_answer: The answer before this round's refinement.
            questions: The probing questions that were asked.
            responses: The honest responses to those questions.
            round_num: Current round number.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The refined answer.
        """
        insights = "\n".join(
            f"- [{r['type']}] {r['question']}\n  Response: {r['response']}"
            for r in responses
            if "no new insight" not in r.get("response", "").lower()
        )

        if not insights:
            return current_answer

        prompt = (
            f"Revise your answer based on the insights gained from Socratic "
            f"questioning. Address exposed assumptions, fill gaps, and "
            f"strengthen weak points.\n\n"
            f"Original query: {query}\n\n"
            f"Previous answer:\n{current_answer}\n\n"
            f"Insights from interrogation:\n{insights}\n\n"
            f"Produce an improved answer that addresses these insights. "
            f"Be explicit about what changed and why."
        )

        refined = await self._call_llm(
            provider=provider,
            messages=[Message.user(prompt)],
            trace=trace,
            system_prompt="You are a rigorous thinker refining your position.",
        )

        step = self._make_step(
            step_type="refinement",
            content=refined,
            score=0.5 + (round_num + 1) * 0.1,
            metadata={
                "phase": "refine",
                "round": round_num + 1,
                "insights_incorporated": len([
                    r for r in responses
                    if "no new insight" not in r.get("response", "").lower()
                ]),
            },
        )
        trace.add_step(step)
        return refined

    # ------------------------------------------------------------------
    # Final synthesis
    # ------------------------------------------------------------------

    async def _final_synthesis(
        self,
        query: str,
        final_answer: str,
        provider: BaseLLMProvider,
        trace: ReasoningTrace,
    ) -> str:
        """Produce a polished final answer after all interrogation rounds.

        Args:
            query: Original user query.
            final_answer: The answer after all refinement rounds.
            provider: LLM provider.
            trace: Reasoning trace.

        Returns:
            The final polished answer.
        """
        prompt = (
            f"You are presenting a final answer that has been rigorously "
            f"tested through multiple rounds of Socratic interrogation. "
            f"Hidden assumptions have been exposed and addressed, evidence "
            f"has been demanded, counterexamples considered, and implications "
            f"explored.\n\n"
            f"Original query: {query}\n\n"
            f"Refined answer:\n{final_answer}\n\n"
            f"Present this as a clear, well-structured final answer. "
            f"Acknowledge any remaining uncertainties or limitations. "
            f"Be transparent about the assumptions that survived scrutiny."
        )

        final = await self._call_llm(
            provider=provider,
            messages=[Message.user(prompt)],
            trace=trace,
            system_prompt="You are a clear, honest communicator.",
        )

        step = self._make_step(
            step_type="final_synthesis",
            content=final,
            score=1.0,
            metadata={"phase": "finalise"},
        )
        trace.add_step(step)
        return final

    # ------------------------------------------------------------------
    # Novelty assessment
    # ------------------------------------------------------------------

    @staticmethod
    def _assess_novelty(responses: list[dict[str, str]]) -> dict[str, Any]:
        """Assess how many responses yielded new insights.

        Args:
            responses: The response dicts from ``_respond_to_questions``.

        Returns:
            Dict with ``no_insight_count``, ``total``, and
            ``no_insight_fraction``.
        """
        total = len(responses)
        if total == 0:
            return {"no_insight_count": 0, "total": 0, "no_insight_fraction": 1.0}

        no_insight = sum(
            1 for r in responses
            if "no new insight" in r.get("response", "").lower()
        )
        return {
            "no_insight_count": no_insight,
            "total": total,
            "no_insight_fraction": no_insight / total,
        }

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_questions(
        self,
        raw: str,
        question_types: list[str],
        per_type: int,
    ) -> list[dict[str, str]]:
        """Parse a JSON array of question objects from LLM output.

        Falls back to generating placeholder questions if parsing fails.

        Args:
            raw: Raw LLM output.
            question_types: Expected question type labels.
            per_type: Expected questions per type.

        Returns:
            List of dicts with ``type`` and ``question`` keys.
        """
        text = raw.strip()
        start = text.find("[")
        end = text.rfind("]")

        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, list):
                    questions = []
                    for item in parsed:
                        if isinstance(item, dict) and "question" in item:
                            q_type = str(item.get("type", "general"))
                            questions.append({
                                "type": q_type,
                                "question": str(item["question"]),
                            })
                    if questions:
                        return questions
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: try splitting by lines
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and "?" in ln]
        if lines:
            questions = []
            for idx, line in enumerate(lines):
                q_type = question_types[idx % len(question_types)]
                # Strip leading numbering
                cleaned = line.lstrip("0123456789.-) []")
                questions.append({"type": q_type, "question": cleaned})
            return questions

        # Last resort: generate minimal questions
        questions = []
        for qt in question_types:
            desc = self._QUESTION_DESCRIPTIONS.get(qt, qt)
            for _ in range(per_type):
                questions.append({
                    "type": qt,
                    "question": f"Regarding {qt}: {desc}",
                })
        return questions

    @staticmethod
    def _parse_responses(
        raw: str,
        questions: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Parse a JSON array of response objects from LLM output.

        Falls back to line-based splitting if JSON parsing fails.

        Args:
            raw: Raw LLM output.
            questions: The original questions (used to merge data).

        Returns:
            List of dicts with ``type``, ``question``, and ``response`` keys.
        """
        text = raw.strip()
        start = text.find("[")
        end = text.rfind("]")

        responses: list[dict[str, str]] = []

        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "response" in item:
                            idx = int(item.get("question_index", len(responses) + 1)) - 1
                            idx = max(0, min(idx, len(questions) - 1))
                            responses.append({
                                "type": questions[idx]["type"],
                                "question": questions[idx]["question"],
                                "response": str(item["response"]),
                            })
                    if responses:
                        return responses
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: split text into chunks and pair with questions
        paragraphs = [
            p.strip() for p in text.split("\n\n") if p.strip()
        ]
        if not paragraphs:
            paragraphs = [ln.strip() for ln in text.splitlines() if ln.strip()]

        for idx, q in enumerate(questions):
            resp_text = paragraphs[idx] if idx < len(paragraphs) else "no new insight"
            # Strip leading numbering from response
            cleaned = resp_text.lstrip("0123456789.-) ")
            responses.append({
                "type": q["type"],
                "question": q["question"],
                "response": cleaned,
            })

        return responses


__all__ = ["SocraticInterrogation"]
