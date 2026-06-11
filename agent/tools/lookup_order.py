"""Order lookup tool backed by the PostgreSQL `orders` table."""

import logging
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from database.models import Order
from database.postgres import get_session

logger = logging.getLogger(__name__)


def lookup_order(order_id: str) -> dict[str, Any] | None:
    """Look up an order by its identifier.

    Args:
        order_id: The order identifier (e.g. `ORD-1001`).

    Returns:
        dict[str, Any] | None: A dictionary representation of the order if
            found, otherwise `None`. Returns `None` (and logs a warning)
            if the database query fails.
    """
    if not order_id:
        return None

    try:
        with get_session() as session:
            order = session.get(Order, order_id.upper())
            if order is None:
                return None
            return {
                "order_id": order.order_id,
                "customer_id": order.customer_id,
                "status": order.status,
                "items": order.items,
                "total_amount": order.total_amount,
                "currency": order.currency,
                "tracking_number": order.tracking_number,
                "created_at": order.created_at.isoformat() if order.created_at else None,
                "expected_delivery_date": (
                    order.expected_delivery_date.isoformat() if order.expected_delivery_date else None
                ),
            }
    except SQLAlchemyError as exc:
        logger.error("Order lookup failed for '%s': %s", order_id, exc)
        return None
