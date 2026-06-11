"""Seed the PostgreSQL database with sample order data."""

import json
import logging
from datetime import datetime
from pathlib import Path

from config import settings
from database.models import Order
from database.postgres import get_session, init_db

logger = logging.getLogger(__name__)

_ORDERS_PATH = Path(settings.data_dir) / "orders.json"


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse an ISO-8601 datetime string.

    Args:
        value: An ISO-8601 datetime string, or `None`.

    Returns:
        datetime | None: The parsed datetime, or `None` if `value` is `None`
            or cannot be parsed.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def seed_orders() -> int:
    """Load `data/orders.json` and upsert each order into the database.

    Returns:
        int: The number of orders inserted or updated.

    Raises:
        FileNotFoundError: If `data/orders.json` does not exist.
    """
    if not _ORDERS_PATH.exists():
        raise FileNotFoundError(f"Orders data file not found: {_ORDERS_PATH}")

    with open(_ORDERS_PATH, encoding="utf-8") as handle:
        orders = json.load(handle)

    count = 0
    with get_session() as session:
        for order_data in orders:
            existing = session.get(Order, order_data["order_id"])
            if existing is not None:
                session.delete(existing)
                session.flush()

            order = Order(
                order_id=order_data["order_id"],
                customer_id=order_data["customer_id"],
                status=order_data["status"],
                items=order_data["items"],
                total_amount=order_data["total_amount"],
                currency=order_data.get("currency", "SAR"),
                tracking_number=order_data.get("tracking_number"),
                created_at=_parse_datetime(order_data.get("created_at")) or datetime.utcnow(),
                expected_delivery_date=_parse_datetime(order_data.get("expected_delivery_date")),
            )
            session.add(order)
            count += 1

    return count


def main() -> None:
    """Initialize the database schema and seed sample orders."""
    logging.basicConfig(level=logging.INFO)
    init_db()
    count = seed_orders()
    logger.info("Seeded %d orders.", count)


if __name__ == "__main__":
    main()
