"""End-to-end evaluation runner: runs the orchestrator over the gold test set."""

import csv
import json
import logging
from pathlib import Path
from typing import Any

from agent.orchestrator import TriageOrchestrator
from config import settings
from evaluation.error_analysis import build_error_summary, find_misclassifications, write_error_report
from evaluation.llm_judge import LLMJudge
from evaluation.metrics import compute_intent_metrics
from evaluation.routing_eval import compute_routing_accuracy

logger = logging.getLogger(__name__)

_TEST_SET_DIR = Path(settings.knowledge_base_dir) / "test_set"
_GOLD_TEST_PATH = _TEST_SET_DIR / "gold_test.csv"
_ADVERSARIAL_PATHS = [
    _TEST_SET_DIR / "adversarial_cases.json",
    _TEST_SET_DIR / "hotel_adversarial_cases.json",
]


def _load_gold_test() -> list[dict[str, str]]:
    """Load the gold test set from CSV.

    Returns:
        list[dict[str, str]]: A list of test case dictionaries.

    Raises:
        FileNotFoundError: If `gold_test.csv` does not exist.
    """
    if not _GOLD_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Gold test set not found at {_GOLD_TEST_PATH}. Run scripts/sample_dataset.py first."
        )

    with open(_GOLD_TEST_PATH, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_adversarial_cases() -> list[dict[str, Any]]:
    """Load and merge all adversarial test case files.

    Returns:
        list[dict[str, Any]]: The combined adversarial cases from every
            file in `_ADVERSARIAL_PATHS` that exists on disk.
    """
    cases: list[dict[str, Any]] = []
    for path in _ADVERSARIAL_PATHS:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as handle:
            cases.extend(json.load(handle))
    return cases


def run_adversarial_eval(orchestrator: TriageOrchestrator) -> list[dict[str, Any]]:
    """Run the orchestrator over every adversarial case and record its output.

    These cases have no `expected_intent`/`expected_team` gold labels (only
    a free-text `expected_behavior` description), so this is a qualitative
    report rather than a scored metric -- intended for manual review.

    Args:
        orchestrator: The orchestrator instance to run cases through.

    Returns:
        list[dict[str, Any]]: One result per adversarial case, including the
            triage outputs and reasoning trace.
    """
    cases = _load_adversarial_cases()
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        logger.info("Evaluating adversarial case %d/%d (id=%s)", index, len(cases), case["id"])
        triage_result = orchestrator.run(message=case["message"], customer_id=f"ADV-{case['id']}")
        results.append({
            "id": case["id"],
            "message": case["message"],
            "description": case.get("description", ""),
            "expected_behavior": case.get("expected_behavior", ""),
            "intent": triage_result.intent,
            "urgency": triage_result.urgency,
            "sentiment": triage_result.sentiment,
            "dialect": triage_result.dialect,
            "requires_human": triage_result.requires_human,
            "routed_team": triage_result.routed_team,
            "draft_response_ar": triage_result.draft_response_ar,
        })

    output_dir = Path(settings.logs_dir) / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "adversarial_report.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    return results


def run_evaluation(use_llm_judge: bool = False) -> dict[str, Any]:
    """Run the full evaluation pipeline over the gold test set.

    Args:
        use_llm_judge: If `True`, also score each generated response with
            the LLM judge rubric.

    Returns:
        dict[str, Any]: A summary report containing classification
            metrics, routing accuracy, per-case results, and (optionally)
            judge scores.

    Raises:
        FileNotFoundError: If the gold test set is missing.
    """
    cases = _load_gold_test()
    orchestrator = TriageOrchestrator()
    judge = LLMJudge(llm_client=orchestrator.llm_client) if use_llm_judge else None

    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        logger.info("Evaluating gold case %d/%d (id=%s)", index, len(cases), case["id"])
        cost_before = orchestrator.llm_client.total_cost_usd
        triage_result = orchestrator.run(message=case["message"], customer_id=case["customer_id"])
        cost_after = orchestrator.llm_client.total_cost_usd

        case_result: dict[str, Any] = {
            "id": case["id"],
            "message": case["message"],
            "expected_intent": case["expected_intent"],
            "predicted_intent": triage_result.intent,
            "expected_team": case["expected_team"],
            "predicted_team": triage_result.routed_team,
            "expected_dialect": case["expected_dialect"],
            "predicted_dialect": triage_result.dialect,
            "draft_response_ar": triage_result.draft_response_ar,
            "latency_ms": triage_result.latency_ms,
            "est_cost_usd": cost_after - cost_before,
        }

        if judge is not None:
            case_result["judge_scores"] = judge.judge(
                message=case["message"],
                response=triage_result.draft_response_ar,
                expected_dialect=case["expected_dialect"],
            )

        results.append(case_result)

    intent_metrics = compute_intent_metrics(
        y_true=[r["expected_intent"] for r in results],
        y_pred=[r["predicted_intent"] for r in results],
    )
    routing_accuracy = compute_routing_accuracy(
        y_true=[r["expected_team"] for r in results],
        y_pred=[r["predicted_team"] for r in results],
    )
    dialect_accuracy = sum(
        1 for r in results if r["predicted_dialect"] == r["expected_dialect"]
    ) / len(results)

    errors = find_misclassifications(results)
    error_summary = build_error_summary(results)
    write_error_report(errors, summary=error_summary)

    adversarial_results = run_adversarial_eval(orchestrator)

    avg_latency_ms = sum(r["latency_ms"] for r in results) / len(results)
    avg_cost_usd = sum(r["est_cost_usd"] for r in results) / len(results)
    total_cost_usd = sum(r["est_cost_usd"] for r in results)

    report: dict[str, Any] = {
        "intent_metrics": intent_metrics,
        "routing_accuracy": routing_accuracy,
        "dialect_accuracy": dialect_accuracy,
        "num_cases": len(results),
        "num_errors": len(errors),
        "error_summary": error_summary,
        "num_adversarial_cases": len(adversarial_results),
        "avg_latency_ms": avg_latency_ms,
        "avg_cost_usd": avg_cost_usd,
        "total_cost_usd": total_cost_usd,
        "results": results,
        "adversarial_results": adversarial_results,
    }

    if judge is not None:
        rubric_fields = ("dialect_match", "correctness", "tone", "helpfulness")
        judge_scores = [r["judge_scores"] for r in results if "judge_scores" in r]
        report["judge_scores_avg"] = {
            field: sum(s.get(field, 0) for s in judge_scores) / len(judge_scores)
            for field in rubric_fields
        } if judge_scores else {}

    output_dir = Path(settings.logs_dir) / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "evaluation_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = run_evaluation(use_llm_judge=True)
    logger.info("Intent metrics: %s", summary["intent_metrics"])
    logger.info("Routing accuracy: %.2f", summary["routing_accuracy"])
    logger.info("Dialect accuracy: %.2f", summary["dialect_accuracy"])
    logger.info("Avg latency (ms): %.1f", summary["avg_latency_ms"])
    logger.info("Avg cost (USD): %.6f", summary["avg_cost_usd"])
    logger.info("Total cost (USD): %.6f", summary["total_cost_usd"])
    logger.info("Judge scores (avg): %s", summary.get("judge_scores_avg"))
    logger.info("Adversarial cases run: %d", summary["num_adversarial_cases"])
