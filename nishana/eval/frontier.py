from __future__ import annotations

from datetime import UTC, datetime

FRONTIER_PROMPTS: dict[str, str] = {
    "v1": """Classify this customer support ticket and extract entities.

Ticket:
{ticket}

Reply with the intent (order_status, refund, account, or other) and any
entities you find.""",
    "v2": """You label Hinglish customer support tickets for an automated
router. Tickets mix Hindi (often written in Latin script), English, and
occasional Devanagari.

Read the ticket, then reply with exactly one line of JSON and nothing else:

{{"intent": "<order_status | refund | account | other>",
  "entities": [{{"type": "<order_id | amount | date | product | phone>",
                "value": "<text copied exactly from the ticket>"}}]}}

Rules:
- Copy entity values character for character from the ticket. Do not translate,
  reformat, or complete them.
- Amounts keep their currency form as written (Rs 1499 stays Rs 1499).
- If no entity of a type appears, omit it. Never invent one.
- When a ticket asks about delivery status AND mentions refund words
  (refund, paisa wapas, return money), label it refund.

Ticket:
{ticket}""",
}

PROMPT_LOG = """Frontier prompt iterations, kept in the open because a weak
baseline prompt is the most common way fine-tune-versus-frontier comparisons
lie.

v1 (2026-08-24): minimal classification instruction. Written first, on purpose.
It says what to produce but not how entities must be copied, which predicts two
failure modes on code-mixed text: translated or normalised amounts (Rs 1499
becoming 1499 rupees), and invented order ids filling a schema it was never
given.

v2 (2026-08-24): adds the exact output schema, a copy-exactly rule, a closed
entity-type list, and a tie-break rule toward refund when refund vocabulary
rides inside a status question. The tie-break exists because this corpus has
refunds asked about in shipping-status phrasing, and downstream routing treats
the two differently.

The baseline reported everywhere is v2. v1 stays scored alongside it rather
than deleted: if they land close together the task was prompt-insensitive; if
they diverge, that gap is part of what the fine-tune has to beat, and hiding it
would flatter the adapter."""

DEFAULT_VERSION = "v2"


def build_messages(version: str, ticket: str) -> list[dict[str, str]]:
    if version not in FRONTIER_PROMPTS:
        raise KeyError(f"unknown frontier prompt {version!r}, have {sorted(FRONTIER_PROMPTS)}")
    return [{"role": "user", "content": FRONTIER_PROMPTS[version].format(ticket=ticket)}]


def stamp(model: str, provider: str) -> dict[str, str]:
    """Model name plus date, attached to every row the baseline writes. A
    comparison without both is unreadable six weeks later."""
    return {
        "model": f"{provider}:{model}",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
