"""Run a small demo against a running instance of the FastAPI service."""

import json
import logging

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "http://127.0.0.1:8000"

_DEMO_MESSAGES = [
    {"customer_id": "CUST-001", "message": "وين طلبي رقم ORD-1001؟ ما وصلني لحد الحين"},
    {"customer_id": "CUST-004", "message": "وصلني الجهاز مكسور من الصندوق، أبغى استبداله فورًا"},
    {"customer_id": "CUST-005", "message": "نسيت كلمة المرور ومش قادر أدخل على حسابي"},
]


def run_demo() -> None:
    """Send a handful of sample requests to the `/triage` endpoint and print results.

    Raises:
        httpx.HTTPError: If the API is unreachable or returns an error
            status code for any request.
    """
    with httpx.Client(base_url=_BASE_URL, timeout=60) as client:
        health = client.get("/health")
        health.raise_for_status()
        logger.info("Health check: %s", health.json())

        for case in _DEMO_MESSAGES:
            response = client.post("/triage", json=case)
            response.raise_for_status()
            print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_demo()
