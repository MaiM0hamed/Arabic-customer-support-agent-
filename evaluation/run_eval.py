"""End-to-end evaluation runner: runs the orchestrator over the gold test set."""

import csv
import json
import logging
from pathlib import Path
from typing import Any

from agent.orchestrator import TriageOrchestrator
from config import settings
from evaluation.error_analysis import find_misclassifications, write_error_report
from evaluation.llm_judge import LLMJudge
from evaluation.metrics import compute_intent_metrics
from evaluation.routing_eval import compute_routing_accuracy

logger = logging.getLogger(__name__)

_GOLD_TEST_PATH = Path(settings.knowledge_base_dir) / "test_set" / "gold_test.csv"


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
    for case in cases:
        triage_result = orchestrator.run(message=case["message"], customer_id=case["customer_id"])

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

    errors = find_misclassifications(results)
    write_error_report(errors)

    report = {
        "intent_metrics": intent_metrics,
        "routing_accuracy": routing_accuracy,
        "num_cases": len(results),
        "num_errors": len(errors),
        "results": results,
    }

    output_dir = Path(settings.logs_dir) / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "evaluation_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = run_evaluation(use_llm_judge=False)
    logger.info("Intent metrics: %s", summary["intent_metrics"])
    logger.info("Routing accuracy: %.2f", summary["routing_accuracy"])
