"""Memory firewall: provenance, trust tiers, quarantine, origin-bound authority.

Agent memory has a poisoning problem: content an agent read on the web ends up
stored next to things the user actually said, with equal standing, and later recall
treats both as truth. Systems whose ingest is an LLM interpreting the content have
a second, worse problem: the content can attack the extractor itself.

GENOME's write path has no LLM, so that extraction-time class of attack has nothing
to run against. This module closes the recall-time half:

- **Provenance**: every ``add()`` can declare where the content came from
  (``user``, ``system``, ``agent``, ``tool``, ``web``). Stored under a reserved
  metadata key; untagged records behave exactly as before.
- **Quarantine**: with a :class:`TrustPolicy` whose ``recall_min_trust`` is set,
  memories below the threshold are held out of ``search()`` results.
  ``search_quarantined()`` shows what was held back, so quarantine is inspectable
  rather than silent.
- **Origin-bound authority**: when conflict resolution is on, a lower-trust fact
  can never UPDATE or DELETE a higher-trust memory - even when the resolver LLM is
  fooled into asking for exactly that. Web content cannot overwrite what your user
  told you.

Honest scope: this mitigates recall-time poisoning by provenance, it does not
detect malicious *content*. A trusted user can still assert something false; that
is what the belief layer's provenance-linked timeline is for.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Reserved metadata key carrying {"source": str, "trust": int}.
PROVENANCE_KEY = "_provenance"

#: Default trust tiers, highest to lowest. ``system`` is the deployment itself,
#: ``user`` is the human, ``agent`` is the model's own conclusions, ``tool`` is
#: structured tool output, ``web`` is arbitrary retrieved content.
DEFAULT_TIERS: dict[str, int] = {
    "system": 4,
    "user": 3,
    "agent": 2,
    "tool": 1,
    "web": 0,
}


@dataclass(frozen=True)
class TrustPolicy:
    """How provenance translates to recall visibility and supersede authority."""

    tiers: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_TIERS))
    recall_min_trust: int | None = None
    """Memories below this trust are quarantined from search(). None = off."""
    supersede_requires_geq: bool = True
    """A conflict-resolution UPDATE/DELETE is refused when the incoming fact's
    trust is lower than the target memory's trust (origin-bound authority)."""
    untagged_trust: int = 3
    """Trust assigned to records with no provenance tag. Defaults to user-level
    so pre-firewall stores keep their existing behavior."""

    def tier_of(self, source: str) -> int:
        try:
            return self.tiers[source]
        except KeyError:
            raise ValueError(
                f"unknown provenance {source!r}; valid sources: "
                f"{', '.join(sorted(self.tiers))}"
            ) from None


def provenance_metadata(policy: TrustPolicy | None, source: str) -> dict[str, Any]:
    """The metadata entry for a tagged write. Validates the source name."""
    tiers = policy or TrustPolicy()
    return {"source": source, "trust": tiers.tier_of(source)}


def trust_of(record: Any, policy: TrustPolicy) -> int:
    """A record's effective trust: its tag, or the untagged default."""
    tag = (getattr(record, "metadata", None) or {}).get(PROVENANCE_KEY)
    if isinstance(tag, dict) and isinstance(tag.get("trust"), int):
        return tag["trust"]
    return policy.untagged_trust


def is_quarantined(record: Any, policy: TrustPolicy | None) -> bool:
    if policy is None or policy.recall_min_trust is None:
        return False
    return trust_of(record, policy) < policy.recall_min_trust
