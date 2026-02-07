"""Predictive Coding reasoning engine for OpenAgentFlow.

Based on Karl Friston's Free Energy Principle and the predictive processing
framework (Friston 2005, 2010; Clark 2013; Rao & Ballard 1999).

The brain is fundamentally a prediction machine.  Rather than passively
receiving sensory data, the cortex continuously generates top-down predictions
about expected inputs and compares them against bottom-up sensory evidence.
Mismatches produce *prediction errors* that propagate upward, driving the
internal model to update and minimise *surprise* (variational free energy).

Crucially, not all prediction errors are weighted equally.  Each error is
scaled by its *precision* -- the estimated reliability of the sensory channel.
High-precision errors (from reliable sources) force large model updates;
low-precision errors (from noisy or ambiguous sources) are attenuated.  This
precision-weighting mechanism explains phenomena from attention (increasing
precision on task-relevant channels) to psychosis (aberrant precision
assignment).

The engine implements this loop:

1. **Prediction Generation** -- The current internal model generates explicit
   predictions about what the answer should look like.
2. **Evidence Gathering** -- The system gathers evidence and observations
   relevant to the query.
3. **Prediction Error Computation** -- Predictions are compared to evidence;
   discrepancies are identified and scored for surprise.
4. **Precision Estimation** -- Each error channel is assessed for reliability;
   precision weights are assigned.
5. **Model Update** -- The internal model is revised to minimise
   precision-weighted prediction error.
6. **Iterate** -- The cycle repeats until free energy (total weighted
   prediction error) falls below a convergence threshold or max iterations
   are reached.

Example::

    from openagentflow.reasoning.predictive_coding import PredictiveCoding

    engine = PredictiveCoding(max_iterations=5, precision_threshold=0.15)
    trace = await engine.reason(
        query="What is the most effective approach to reducing technical debt?",
        llm_provider=my_provider,
    )
    print(trace.final_output)

    # Inspect precision-weighted errors across iterations
    for step in trace.get_steps_by_type("prediction_error"):
        it = step.metadata.get("iteration", "?")
        fe = step.metadata.get("free_energy", "?")
        print(f"Iteration {it}: free_energy={fe}")

Trace structure (DAG)::

    query
      +-- initial_model
      +-- prediction_iter_0
      |     +-- evidence_iter_0
      |           +-- prediction_error_iter_0
      |                 +-- precision_estimate_iter_0
      |                       +-- model_update_iter_0
      |                             +-- prediction_iter_1
      |                                   +-- ...
      +-- convergence_check
      +-- final_output
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openagentflow.reasoning.base import ReasoningEngine, ReasoningStep, ReasoningTrace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_INITIAL_MODEL_SYSTEM = (
    "You are building an internal predictive model.  Given a query, construct "
    "an initial mental model: what do you expect the answer to involve?  What "
    "are your prior beliefs, assumptions, and structural expectations?  Be "
    "explicit about what you predict and why.  Think of this as the brain's "
    "top-down generative model before encountering evidence."
)

_PREDICTION_SYSTEM = (
    "You are generating top-down predictions from your current internal model. "
    "Based on your model of the problem, generate specific, falsifiable "
    "predictions about what the evidence should show.  Each prediction should "
    "be concrete enough that it can be confirmed or disconfirmed.  Number your "
    "predictions clearly."
)

_EVIDENCE_SYSTEM = (
    "You are a careful evidence gatherer operating in bottom-up mode.  Set "
    "aside all prior expectations and examine the query from first principles. "
    "What does the raw evidence, data, and logical analysis actually show?  "
    "Focus on observations, not interpretations.  Be thorough and unbiased."
)

_ERROR_SYSTEM = (
    "You are a prediction error computation unit.  Compare the top-down "
    "predictions against the bottom-up evidence.  For each prediction, "
    "determine whether the evidence confirms or disconfirms it.  For each "
    "mismatch, describe the prediction error: what was expected vs. what was "
    "found.  Rate each error's surprise on a scale from 0.0 (perfectly "
    "predicted) to 1.0 (completely unexpected).\n\n"
    "Return a JSON object with this structure:\n"
    '{"errors": [{"prediction": "...", "evidence": "...", "surprise": 0.7, '
    '"description": "..."}], "overall_surprise": 0.5, "narrative": "..."}'
)

_PRECISION_SYSTEM = (
    "You are estimating the precision (reliability) of each prediction error "
    "channel.  For each error identified, assess how reliable and informative "
    "that error signal is.  High precision means the error is based on strong, "
    "unambiguous evidence and should force a model update.  Low precision "
    "means the error might stem from noise, ambiguity, or unreliable data "
    "and should be down-weighted.\n\n"
    "Return a JSON object:\n"
    '{"precisions": [{"error_index": 0, "precision": 0.8, '
    '"rationale": "..."}], "mean_precision": 0.6}'
)

_MODEL_UPDATE_SYSTEM = (
    "You are updating your internal model to minimise prediction error.  Given "
    "the precision-weighted prediction errors, revise your model of the "
    "problem.  Errors with high precision should cause large revisions; errors "
    "with low precision should cause small or no revisions.  Explain what "
    "changed in your model and why.  Present the updated model clearly."
)

_FINAL_SYNTHESIS_SYSTEM = (
    "You are producing a final answer after iterative predictive coding.  "
    "Your internal model has been refined through multiple cycles of "
    "prediction, evidence comparison, and precision-weighted updating.  "
    "Synthesize your final, refined model into a clear, comprehensive answer "
    "to the original query.  Your answer should reflect the accumulated "
    "insights from the entire predictive coding process."
)


# ---------------------------------------------------------------------------
# PredictiveCoding
# ---------------------------------------------------------------------------


class PredictiveCoding(ReasoningEngine):
    """Predictive Coding reasoning via iterative prediction-error minimisation.

    Implements Karl Friston's Free Energy Principle as a reasoning strategy.
    The engine maintains an evolving internal model that generates predictions,
    compares them against evidence, and updates itself to minimise
    precision-weighted surprise.

    At each iteration the engine:

    1. Generates top-down **predictions** from its current model.
    2. Gathers bottom-up **evidence** by analysing the query from first
       principles.
    3. Computes **prediction errors** -- mismatches between predictions and
       evidence, each scored for surprise (0.0--1.0).
    4. Estimates **precision** -- the reliability of each error channel.
    5. **Updates the model** proportionally to precision-weighted errors.

    The process converges when the *free energy* (mean precision-weighted
    surprise) drops below ``precision_threshold``, or after ``max_iterations``.

    Args:
        max_iterations: Maximum prediction-error-update cycles (default 4).
        precision_threshold: Free energy level below which the model is
            considered converged (default 0.15).
        initial_temperature: LLM temperature for the first iteration's
            evidence gathering (default 0.7).  Decreases across iterations.

    Attributes:
        name: ``"predictive_coding"``
        description: Short description of the engine.

    Example::

        engine = PredictiveCoding(max_iterations=5)
        trace = await engine.reason(
            query="How should we handle distributed transactions?",
            llm_provider=provider,
        )
        print(trace.final_output)
    """

    name: str = "predictive_coding"
    description: str = (
        "Iterative prediction-error minimisation inspired by Friston's "
        "Free Energy Principle.  Generates predictions, compares against "
        "evidence, and updates the model to minimise precision-weighted surprise."
    )

    def __init__(
        self,
        max_iterations: int = 4,
        precision_threshold: float = 0.15,
        initial_temperature: float = 0.7,
    ) -> None:
        self.max_iterations = max(1, max_iterations)
        self.precision_threshold = max(0.0, min(1.0, precision_threshold))
        self.initial_temperature = max(0.1, min(1.5, initial_temperature))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def reason(
        self,
        query: str,
        llm_provider: Any,
        tools: list[Any] | None = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> ReasoningTrace:
        """Execute the predictive coding reasoning loop.

        Args:
            query: The user question or task.
            llm_provider: A ``BaseLLMProvider`` instance.
            tools: Ignored by this engine.
            max_iterations: Ignored -- ``self.max_iterations`` controls depth.
            **kwargs: Reserved for future use.

        Returns:
            A :class:`ReasoningTrace` with prediction, evidence,
            prediction_error, precision_estimate, and model_update steps.
        """
        start = time.time()
        trace = ReasoningTrace(strategy_name=self.name)

        # Record the query as the root step
        query_step = self._make_step(
            step_type="query",
            content=query,
            metadata={"role": "user_query"},
        )
        trace.add_step(query_step)

        # Phase 1: Build initial internal model
        current_model = await self._build_initial_model(
            query, llm_provider, trace,
        )
        model_step = self._make_step(
            step_type="initial_model",
            content=current_model,
            metadata={"phase": "initialisation"},
            parent_step_id=query_step.step_id,
        )
        trace.add_step(model_step)

        # Phase 2: Iterative prediction-error minimisation
        last_step_id = model_step.step_id
        free_energy_history: list[float] = []
        converged = False

        for iteration in range(self.max_iterations):
            temperature = max(
                0.2,
                self.initial_temperature * (0.8 ** iteration),
            )

            # 2a: Generate predictions from current model
            predictions = await self._generate_predictions(
                query, current_model, iteration, llm_provider, trace,
            )
            pred_step = self._make_step(
                step_type="prediction",
                content=predictions,
                metadata={"iteration": iteration, "temperature": round(temperature, 3)},
                parent_step_id=last_step_id,
            )
            trace.add_step(pred_step)

            # 2b: Gather bottom-up evidence
            evidence = await self._gather_evidence(
                query, current_model, iteration, temperature, llm_provider, trace,
            )
            ev_step = self._make_step(
                step_type="evidence",
                content=evidence,
                metadata={"iteration": iteration},
                parent_step_id=pred_step.step_id,
            )
            trace.add_step(ev_step)

            # 2c: Compute prediction errors
            error_raw = await self._compute_prediction_errors(
                predictions, evidence, query, llm_provider, trace,
            )
            error_data = self._parse_error_json(error_raw)
            overall_surprise = error_data.get("overall_surprise", 0.5)

            err_step = self._make_step(
                step_type="prediction_error",
                content=error_data.get("narrative", error_raw),
                score=overall_surprise,
                metadata={
                    "iteration": iteration,
                    "overall_surprise": round(overall_surprise, 4),
                    "num_errors": len(error_data.get("errors", [])),
                },
                parent_step_id=ev_step.step_id,
            )
            trace.add_step(err_step)

            # 2d: Estimate precision of each error channel
            precision_raw = await self._estimate_precision(
                error_raw, query, llm_provider, trace,
            )
            precision_data = self._parse_precision_json(precision_raw)
            mean_precision = precision_data.get("mean_precision", 0.5)

            prec_step = self._make_step(
                step_type="precision_estimate",
                content=precision_raw,
                score=mean_precision,
                metadata={
                    "iteration": iteration,
                    "mean_precision": round(mean_precision, 4),
                },
                parent_step_id=err_step.step_id,
            )
            trace.add_step(prec_step)

            # 2e: Compute free energy (precision-weighted surprise)
            free_energy = self._compute_free_energy(error_data, precision_data)
            free_energy_history.append(free_energy)

            # 2f: Update the model
            current_model = await self._update_model(
                current_model, error_raw, precision_raw, query,
                free_energy, llm_provider, trace,
            )
            update_step = self._make_step(
                step_type="model_update",
                content=current_model,
                score=1.0 - free_energy,
                metadata={
                    "iteration": iteration,
                    "free_energy": round(free_energy, 4),
                    "overall_surprise": round(overall_surprise, 4),
                    "mean_precision": round(mean_precision, 4),
                },
                parent_step_id=prec_step.step_id,
            )
            trace.add_step(update_step)
            last_step_id = update_step.step_id

            # 2g: Convergence check
            conv_step = self._make_step(
                step_type="convergence_check",
                content=(
                    f"Iteration {iteration}: free_energy={free_energy:.4f} "
                    f"(threshold={self.precision_threshold})"
                ),
                score=free_energy,
                metadata={
                    "iteration": iteration,
                    "free_energy": round(free_energy, 4),
                    "threshold": self.precision_threshold,
                    "converged": free_energy <= self.precision_threshold,
                    "free_energy_history": [
                        round(fe, 4) for fe in free_energy_history
                    ],
                },
                parent_step_id=update_step.step_id,
            )
            trace.add_step(conv_step)

            if free_energy <= self.precision_threshold:
                converged = True
                break

        # Phase 3: Final synthesis
        final_output = await self._synthesize_final(
            query, current_model, free_energy_history,
            converged, llm_provider, trace,
        )

        final_step = self._make_step(
            step_type="final_output",
            content=final_output,
            score=1.0,
            metadata={
                "converged": converged,
                "total_iterations": len(free_energy_history),
                "final_free_energy": round(
                    free_energy_history[-1] if free_energy_history else 1.0, 4,
                ),
            },
            parent_step_id=last_step_id,
        )
        trace.add_step(final_step)

        trace.final_output = final_output
        trace.duration_ms = (time.time() - start) * 1000
        return trace

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    async def _build_initial_model(
        self,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Build the initial top-down generative model.

        Args:
            query: Original user query.
            provider: LLM provider.
            trace: Current trace for metric tracking.

        Returns:
            The initial internal model as text.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Build an initial internal model for reasoning about the "
                    f"following query.  State your prior beliefs, assumptions, "
                    f"expected structure of the answer, and key factors you "
                    f"anticipate will be important.\n\nQuery: {query}"
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_INITIAL_MODEL_SYSTEM,
            temperature=0.5,
        )

    async def _generate_predictions(
        self,
        query: str,
        current_model: str,
        iteration: int,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Generate top-down predictions from the current model.

        Args:
            query: Original user query.
            current_model: The current state of the internal model.
            iteration: Current iteration number.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Numbered list of specific predictions.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Based on the following internal model, generate specific, "
                    f"testable predictions about what the evidence should show "
                    f"for this query.\n\n"
                    f"Query: {query}\n\n"
                    f"Current Internal Model (iteration {iteration}):\n"
                    f"{current_model}\n\n"
                    f"Generate 4-6 specific predictions.  Each should be "
                    f"concrete enough to be confirmed or disconfirmed."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_PREDICTION_SYSTEM,
            temperature=0.4,
        )

    async def _gather_evidence(
        self,
        query: str,
        current_model: str,
        iteration: int,
        temperature: float,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Gather bottom-up evidence independently of the model.

        Args:
            query: Original user query.
            current_model: Current model (provided for context awareness).
            iteration: Current iteration number.
            temperature: LLM temperature for this call.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Evidence text gathered from first-principles analysis.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Analyse the following query purely from first principles "
                    f"and evidence.  Set aside any prior model or assumptions.  "
                    f"What does careful, unbiased analysis actually reveal?\n\n"
                    f"Query: {query}\n\n"
                    f"This is iteration {iteration} of an evidence-gathering "
                    f"process.  If previous iterations found certain things, "
                    f"focus on areas not yet explored or where understanding "
                    f"is weakest.  Be thorough and specific."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_EVIDENCE_SYSTEM,
            temperature=temperature,
        )

    async def _compute_prediction_errors(
        self,
        predictions: str,
        evidence: str,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Compare predictions against evidence and identify errors.

        Args:
            predictions: Top-down predictions.
            evidence: Bottom-up evidence.
            query: Original query.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Raw LLM output with structured error analysis (ideally JSON).
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Compare the following predictions against the evidence "
                    f"for this query.\n\n"
                    f"Query: {query}\n\n"
                    f"---PREDICTIONS---\n{predictions}\n---END PREDICTIONS---"
                    f"\n\n"
                    f"---EVIDENCE---\n{evidence}\n---END EVIDENCE---\n\n"
                    f"For each prediction, determine if the evidence confirms "
                    f"or disconfirms it.  Score each mismatch for surprise "
                    f"(0.0 = perfectly predicted, 1.0 = completely unexpected). "
                    f"Also compute an overall_surprise score.\n\n"
                    f"Return a JSON object with: errors (list of dicts with "
                    f"prediction, evidence, surprise, description), "
                    f"overall_surprise (float), and narrative (string summary)."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_ERROR_SYSTEM,
            temperature=0.3,
        )

    async def _estimate_precision(
        self,
        error_analysis: str,
        query: str,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Estimate precision (reliability) of each error channel.

        Args:
            error_analysis: The prediction error analysis from the prior step.
            query: Original query.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            Raw LLM output with precision estimates (ideally JSON).
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Assess the precision (reliability) of each prediction "
                    f"error channel for this query.\n\n"
                    f"Query: {query}\n\n"
                    f"---PREDICTION ERRORS---\n{error_analysis}\n"
                    f"---END PREDICTION ERRORS---\n\n"
                    f"For each error, rate its precision from 0.0 (unreliable, "
                    f"noisy, ambiguous -- should be down-weighted) to 1.0 "
                    f"(highly reliable, unambiguous -- should force model "
                    f"update).  Also compute a mean_precision.\n\n"
                    f"Return JSON: precisions (list with error_index, "
                    f"precision, rationale), mean_precision (float)."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_PRECISION_SYSTEM,
            temperature=0.2,
        )

    async def _update_model(
        self,
        current_model: str,
        error_analysis: str,
        precision_analysis: str,
        query: str,
        free_energy: float,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Update the internal model based on precision-weighted errors.

        Args:
            current_model: Current state of the internal model.
            error_analysis: Prediction error analysis.
            precision_analysis: Precision estimates.
            query: Original query.
            free_energy: Current free energy (for context).
            provider: LLM provider.
            trace: Current trace.

        Returns:
            The updated model text.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Update your internal model to minimise prediction error. "
                    f"High-precision errors should cause large revisions; "
                    f"low-precision errors should cause minimal change.\n\n"
                    f"Query: {query}\n\n"
                    f"---CURRENT MODEL---\n{current_model}\n---END MODEL---\n\n"
                    f"---PREDICTION ERRORS---\n{error_analysis}\n"
                    f"---END ERRORS---\n\n"
                    f"---PRECISION ESTIMATES---\n{precision_analysis}\n"
                    f"---END PRECISION---\n\n"
                    f"Current free energy: {free_energy:.4f}\n\n"
                    f"Produce the updated model.  Clearly state what changed "
                    f"and why each change was made (or not made) given the "
                    f"precision weights."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_MODEL_UPDATE_SYSTEM,
            temperature=0.4,
        )

    async def _synthesize_final(
        self,
        query: str,
        final_model: str,
        free_energy_history: list[float],
        converged: bool,
        provider: Any,
        trace: ReasoningTrace,
    ) -> str:
        """Synthesize the final answer from the refined model.

        Args:
            query: Original user query.
            final_model: The final state of the internal model.
            free_energy_history: Free energy values across iterations.
            converged: Whether the model converged below threshold.
            provider: LLM provider.
            trace: Current trace.

        Returns:
            The final answer text.
        """
        convergence_note = (
            f"The model converged after {len(free_energy_history)} iterations "
            f"(free energy: {' -> '.join(f'{fe:.3f}' for fe in free_energy_history)})."
            if converged
            else (
                f"The model completed {len(free_energy_history)} iterations "
                f"without full convergence "
                f"(free energy: {' -> '.join(f'{fe:.3f}' for fe in free_energy_history)}).  "
                f"Some uncertainty remains."
            )
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Produce a clear, comprehensive final answer to the "
                    f"original query based on your refined internal model.\n\n"
                    f"Query: {query}\n\n"
                    f"---REFINED MODEL---\n{final_model}\n---END MODEL---\n\n"
                    f"{convergence_note}\n\n"
                    f"Synthesize a thorough answer that reflects all the "
                    f"insights gained through iterative prediction and "
                    f"evidence comparison."
                ),
            },
        ]
        return await self._call_llm(
            provider, messages, trace,
            system=_FINAL_SYNTHESIS_SYSTEM,
            temperature=0.4,
        )

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_error_json(raw: str) -> dict[str, Any]:
        """Parse the prediction error JSON from LLM output.

        Tries ``json.loads`` on the first JSON object found, then falls
        back to regex extraction of the overall_surprise value.

        Args:
            raw: Raw LLM output.

        Returns:
            A dict with at least ``overall_surprise`` and ``errors`` keys.
        """
        text = raw.strip()

        # Try JSON parsing
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                parsed = json.loads(text[start_idx : end_idx + 1])
                if isinstance(parsed, dict):
                    parsed.setdefault("overall_surprise", 0.5)
                    parsed.setdefault("errors", [])
                    parsed.setdefault("narrative", raw)
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: extract overall_surprise via regex
        overall = 0.5
        match = re.search(r"overall[_\s]*surprise[\"'\s:]*([0-9]*\.?[0-9]+)", text, re.I)
        if match:
            try:
                overall = max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass

        return {
            "overall_surprise": overall,
            "errors": [],
            "narrative": raw,
        }

    @staticmethod
    def _parse_precision_json(raw: str) -> dict[str, Any]:
        """Parse precision estimates from LLM output.

        Args:
            raw: Raw LLM output.

        Returns:
            A dict with at least ``mean_precision`` and ``precisions`` keys.
        """
        text = raw.strip()

        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                parsed = json.loads(text[start_idx : end_idx + 1])
                if isinstance(parsed, dict):
                    parsed.setdefault("mean_precision", 0.5)
                    parsed.setdefault("precisions", [])
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback
        mean_prec = 0.5
        match = re.search(r"mean[_\s]*precision[\"'\s:]*([0-9]*\.?[0-9]+)", text, re.I)
        if match:
            try:
                mean_prec = max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass

        return {
            "mean_precision": mean_prec,
            "precisions": [],
        }

    # ------------------------------------------------------------------
    # Free energy computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_free_energy(
        error_data: dict[str, Any],
        precision_data: dict[str, Any],
    ) -> float:
        """Compute free energy as precision-weighted mean surprise.

        If individual error/precision pairs are available, computes:

            F = sum(precision_i * surprise_i) / sum(precision_i)

        Otherwise falls back to ``overall_surprise * mean_precision``.

        Args:
            error_data: Parsed prediction error data.
            precision_data: Parsed precision data.

        Returns:
            Free energy value in [0.0, 1.0].
        """
        errors = error_data.get("errors", [])
        precisions = precision_data.get("precisions", [])

        # Try element-wise computation
        if errors and precisions and len(errors) == len(precisions):
            weighted_sum = 0.0
            precision_sum = 0.0
            for err, prec in zip(errors, precisions):
                s = float(err.get("surprise", 0.5))
                p = float(prec.get("precision", 0.5))
                weighted_sum += p * s
                precision_sum += p
            if precision_sum > 0:
                return max(0.0, min(1.0, weighted_sum / precision_sum))

        # Fallback: overall_surprise * mean_precision
        overall_surprise = float(error_data.get("overall_surprise", 0.5))
        mean_precision = float(precision_data.get("mean_precision", 0.5))
        return max(0.0, min(1.0, overall_surprise * mean_precision))


__all__ = ["PredictiveCoding"]
