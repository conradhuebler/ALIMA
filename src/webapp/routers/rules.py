"""Read-only view of the personal indexing rules for the webapp.

Claude Generated. The rules are applied here like everywhere else — the webapp
run goes through the same ``WorkflowExecutor`` — so the operator needs to see
which ones are in force before reading a result.

Editing is deliberately absent: storing a rule requires a confirmation, and the
webapp session runs with ``AutoRejectGateway`` (``routers/agent.py``), which has
no channel to ask on. Rules are added in the Qt dialog or with ``alima rules``.
"""

import logging

from fastapi import APIRouter

from src.core.user_rules import RuleStore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/rules")
async def list_rules(enabled_only: bool = True):
    """The stored rules, active ones first by default."""
    store = RuleStore()
    rules = store.load()
    if enabled_only:
        rules = [r for r in rules if r.enabled]
    return {
        # Name only: the browser has no use for the server's directory layout.
        # - Claude Generated
        "file": store.path.name,
        "count": len(rules),
        "editable": False,
        "rules": [r.to_display_dict() for r in rules],
    }
