"""Routing tool: maps an intent and urgency level to a support team."""

import json
import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_TAXONOMY_PATH = Path(settings.knowledge_base_dir) / "routing_taxonomy.json"
_DEFAULT_TEAM = "customer_care"


def _load_routing_taxonomy() -> dict:
    """Load the routing taxonomy from disk.

    Returns:
        dict: A dictionary with `teams`, `intent_routing`, and
            `urgency_overrides` keys. Returns an empty dict if the file is
            missing or malformed.
    """
    try:
        with open(_TAXONOMY_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load routing taxonomy from %s: %s", _TAXONOMY_PATH, exc)
        return {}


_TAXONOMY = _load_routing_taxonomy()
_INTENT_ROUTING: dict[str, str] = _TAXONOMY.get("intent_routing", {})
_URGENCY_OVERRIDES: dict[str, str] = _TAXONOMY.get("urgency_overrides", {})
_VALID_TEAM_IDS = {team["id"] for team in _TAXONOMY.get("teams", [])} or {_DEFAULT_TEAM}


def route_team(intent: str, urgency: str) -> tuple[str, bool]:
    """Determine the routed team and whether human escalation is required.

    Args:
        intent: Classified intent identifier.
        urgency: Urgency level (`low`, `medium`, `high`, `critical`).

    Returns:
        tuple[str, bool]: The routed team id and a boolean indicating
            whether the case requires human intervention (`True` for
            `high` or `critical` urgency).
    """
    requires_human = urgency in ("high", "critical")

    if requires_human:
        team = _URGENCY_OVERRIDES.get(urgency, _DEFAULT_TEAM)
        if team in _VALID_TEAM_IDS:
            return team, requires_human

    team = _INTENT_ROUTING.get(intent, _DEFAULT_TEAM)
    if team not in _VALID_TEAM_IDS:
        team = _DEFAULT_TEAM

    return team, requires_human
