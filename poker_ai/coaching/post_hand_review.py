"""Post-hand and practice coaching reports."""

from __future__ import annotations

from dataclasses import dataclass, field

from poker_ai.analysis.equity_calculator import EquityCalculator, EquityReport
from poker_ai.analysis.preflop_chart import PreflopAdvice, PreflopChart
from poker_ai.analysis.range_model import RangeEstimate, RangeModel
from poker_ai.analytics.opponent_model import OpponentModel
from poker_ai.engine.betting import Action
from poker_ai.engine.cards import Card


@dataclass(frozen=True)
class ReviewReport:
    equity: EquityReport
    pot_odds: float | None
    break_even_equity: float | None
    expected_value: float | None
    recommendation_tier: str
    recommendation: str
    position: str = "unknown"
    stack_depth_bb: float | None = None
    stack_depth_label: str | None = None
    preflop_advice: PreflopAdvice | None = None
    reasoning: list[str] = field(default_factory=list)
    opponent_ranges: dict[str, RangeEstimate] = field(default_factory=dict)
    safety_note: str = (
        "Use this for practice, private/consented games, or post-session review; "
        "do not use it as live assistance on third-party poker sites."
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "equity": self.equity.as_dict(),
            "pot_odds": self.pot_odds,
            "break_even_equity": self.break_even_equity,
            "expected_value": self.expected_value,
            "recommendation_tier": self.recommendation_tier,
            "position": self.position,
            "stack_depth_bb": self.stack_depth_bb,
            "stack_depth_label": self.stack_depth_label,
            "preflop_advice": self.preflop_advice.as_dict() if self.preflop_advice else None,
            "recommendation": self.recommendation,
            "reasoning": self.reasoning,
            "opponent_ranges": {
                name: estimate.as_dict()
                for name, estimate in self.opponent_ranges.items()
            },
            "safety_note": self.safety_note,
        }


class PostHandCoach:
    def __init__(
        self,
        equity_calculator: EquityCalculator | None = None,
        range_model: RangeModel | None = None,
        preflop_chart: PreflopChart | None = None,
    ) -> None:
        self.equity_calculator = equity_calculator or EquityCalculator()
        self.range_model = range_model or RangeModel()
        self.preflop_chart = preflop_chart or PreflopChart()

    def review_spot(
        self,
        *,
        hero_cards: list[Card],
        board_cards: list[Card],
        opponent_count: int,
        pot: int,
        call_amount: int,
        action: Action | None = None,
        opponent_model: OpponentModel | None = None,
        opponent_names: list[str] | None = None,
        simulations: int = 10_000,
        position: str = "unknown",
        hero_stack: int | None = None,
        effective_stack: int | None = None,
        big_blind: int = 10,
        facing_raise: bool = False,
    ) -> ReviewReport:
        equity = self.equity_calculator.estimate(
            hero_cards,
            board_cards,
            opponent_count=opponent_count,
            simulations=simulations,
        )
        pot_odds = call_amount / (pot + call_amount) if call_amount > 0 else None
        break_even = pot_odds
        expected_value = self._call_ev(equity.equity, pot, call_amount)
        stack_depth_bb = self._stack_depth(hero_stack, effective_stack, big_blind)
        stack_depth_label = self._stack_label(stack_depth_bb)
        preflop_advice = self.preflop_chart.advise(
            hero_cards,
            position=position,
            facing_raise=facing_raise,
            stack_depth_bb=stack_depth_bb,
        ) if not board_cards else None

        reasoning = [
            f"Estimated equity is {equity.equity:.1%} against {opponent_count} opponent(s).",
        ]

        if pot_odds is not None:
            reasoning.append(f"Calling needs about {pot_odds:.1%} equity before future betting.")
            if expected_value is not None:
                sign = "+" if expected_value >= 0 else ""
                reasoning.append(f"Estimated call EV is {sign}{expected_value:.1f} chips.")
        else:
            reasoning.append("No call price was supplied, so this is a check/bet planning spot.")

        if stack_depth_bb is not None and stack_depth_label is not None:
            reasoning.append(f"Effective stack depth is {stack_depth_bb:.1f} BB ({stack_depth_label}).")
        if preflop_advice is not None:
            reasoning.append(
                f"Preflop chart: {preflop_advice.hand_code} is {preflop_advice.strength_bucket}; "
                f"{preflop_advice.recommendation}."
            )

        recommendation = self._recommend(equity.equity, pot_odds, action)
        tier = self._tier(equity.equity, pot_odds, expected_value)
        opponent_ranges = self._opponent_ranges(opponent_model, opponent_names)
        if opponent_ranges:
            for name, estimate in opponent_ranges.items():
                stats = opponent_model.stats_for(name) if opponent_model else None
                extra = ""
                if stats and stats.fold_to_raise_rate > 0.55:
                    extra = " This player has folded often to raises in the sample."
                elif stats and stats.aggression_factor > 2.5:
                    extra = " This player has shown aggressive tendencies."
                reasoning.append(
                    f"{name} range estimate: {estimate.label} ({estimate.percentile:.0%}).{extra}"
                )

        return ReviewReport(
            equity=equity,
            pot_odds=pot_odds,
            break_even_equity=break_even,
            expected_value=expected_value,
            recommendation_tier=tier,
            position=position,
            stack_depth_bb=stack_depth_bb,
            stack_depth_label=stack_depth_label,
            preflop_advice=preflop_advice,
            recommendation=recommendation,
            reasoning=reasoning,
            opponent_ranges=opponent_ranges,
        )

    @staticmethod
    def _recommend(equity: float, pot_odds: float | None, action: Action | None) -> str:
        if pot_odds is None:
            if equity >= 0.68:
                base = "value bet or raise"
            elif equity >= 0.45:
                base = "check or make a cautious semi-bluff"
            else:
                base = "check and avoid building a large pot"
        elif equity + 0.04 >= pot_odds:
            base = "call is mathematically reasonable"
        else:
            base = "fold is preferred unless you have strong implied odds or reads"

        if action is None:
            return base

        action_label = action.label.replace("_", " ")
        if action == Action.FOLD and pot_odds is not None and equity > pot_odds + 0.08:
            return f"{base}; your fold may have been too tight"
        if action in {Action.CALL, Action.CHECK} and equity >= 0.7:
            return f"{base}; your passive action may have missed value"
        if action == Action.ALL_IN and equity < 0.55:
            return f"{base}; the all-in looks high variance or reckless"
        return f"{base}; reviewed action was {action_label}"

    @staticmethod
    def _call_ev(equity: float, pot: int, call_amount: int) -> float | None:
        if call_amount <= 0:
            return None
        return equity * (pot + call_amount) - call_amount

    @staticmethod
    def _stack_depth(
        hero_stack: int | None,
        effective_stack: int | None,
        big_blind: int,
    ) -> float | None:
        stack = effective_stack if effective_stack is not None else hero_stack
        if stack is None or big_blind <= 0:
            return None
        return stack / big_blind

    @staticmethod
    def _stack_label(stack_depth_bb: float | None) -> str | None:
        if stack_depth_bb is None:
            return None
        if stack_depth_bb <= 25:
            return "shallow"
        if stack_depth_bb <= 80:
            return "medium"
        return "deep"

    @staticmethod
    def _tier(
        equity: float,
        pot_odds: float | None,
        expected_value: float | None,
    ) -> str:
        if expected_value is not None:
            if expected_value >= 25:
                return "strong"
            if expected_value >= 0:
                return "good"
            if expected_value >= -10:
                return "caution"
            return "fold"
        if equity >= 0.68:
            return "strong"
        if equity >= 0.45:
            return "good"
        if pot_odds is not None and equity + 0.04 >= pot_odds:
            return "caution"
        return "fold"

    def _opponent_ranges(
        self,
        opponent_model: OpponentModel | None,
        opponent_names: list[str] | None,
    ) -> dict[str, RangeEstimate]:
        if opponent_model is None or not opponent_names:
            return {}
        return {
            name: self.range_model.estimate_preflop_range(opponent_model.stats_for(name))
            for name in opponent_names
        }
