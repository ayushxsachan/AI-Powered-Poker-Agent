"""Simple opponent range estimates from observed tendencies."""

from __future__ import annotations

from dataclasses import dataclass

from poker_ai.analytics.opponent_model import OpponentStats


@dataclass(frozen=True)
class RangeEstimate:
    label: str
    percentile: float
    description: str
    example_hands: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "percentile": self.percentile,
            "description": self.description,
            "example_hands": self.example_hands,
        }


class RangeModel:
    """Map coarse stats into an understandable preflop range bucket."""

    def estimate_preflop_range(
        self,
        stats: OpponentStats,
        *,
        position: str = "unknown",
        faced_raise: bool = False,
    ) -> RangeEstimate:
        vpip = stats.vpip_rate
        pfr = stats.pfr_rate
        aggression = stats.aggression_factor

        if stats.hands < 10:
            return RangeEstimate(
                label="unknown",
                percentile=0.35,
                description="Not enough hands yet; use a broad default range.",
                example_hands=["22+", "A2s+", "K9s+", "QTs+", "JTs", "ATo+", "KQo"],
            )

        if faced_raise:
            percentile = max(0.08, min(0.32, vpip * 0.55 + pfr * 0.35))
        elif aggression > 2.2:
            percentile = max(0.12, min(0.55, pfr * 1.4 + 0.08))
        else:
            percentile = max(0.12, min(0.65, vpip + 0.05))

        if position.lower() in {"button", "dealer", "cutoff"}:
            percentile = min(0.75, percentile + 0.08)
        elif position.lower() in {"small blind", "big blind", "early"}:
            percentile = max(0.06, percentile - 0.05)

        if percentile <= 0.12:
            return RangeEstimate(
                label="very tight",
                percentile=percentile,
                description="Likely premium-heavy: big pairs, strong broadways, strong suited aces.",
                example_hands=["77+", "ATs+", "KQs", "AQo+"],
            )
        if percentile <= 0.25:
            return RangeEstimate(
                label="tight",
                percentile=percentile,
                description="Likely strong pairs, suited aces, broadways, and a few suited connectors.",
                example_hands=["55+", "A8s+", "KTs+", "QJs", "ATo+", "KQo"],
            )
        if percentile <= 0.42:
            return RangeEstimate(
                label="balanced",
                percentile=percentile,
                description="Likely includes medium pairs, many suited aces, broadways, and connectors.",
                example_hands=["22+", "A2s+", "K9s+", "QTs+", "76s+", "A9o+", "KJo+"],
            )
        return RangeEstimate(
            label="loose",
            percentile=percentile,
            description="Likely entering many speculative hands and weaker offsuit broadways.",
            example_hands=["22+", "A2s+", "K2s+", "Q7s+", "54s+", "A2o+", "K9o+", "QTo+"],
        )

