"""Command-line post-session coach and equity calculator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_ai.analysis.hand_history_parser import HandHistoryParser
from poker_ai.analytics.opponent_model import OpponentModel
from poker_ai.coaching.post_hand_review import PostHandCoach
from poker_ai.engine.betting import Action
from poker_ai.engine.cards import Card

DEFAULT_PROFILE_PATH = Path("poker_ai/logs/opponent_profiles.json")


def parse_cards(values: list[str] | None) -> list[Card]:
    return [Card.from_str(value) for value in values or []]


def action_from_label(value: str | None) -> Action | None:
    if not value:
        return None
    return Action[value.upper()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay-safe Poker AI coach")
    subparsers = parser.add_subparsers(dest="command", required=True)

    equity = subparsers.add_parser("equity", help="Estimate win/tie/loss probabilities")
    equity.add_argument("--hero", nargs=2, required=True, help="Hero cards, e.g. As Ks")
    equity.add_argument("--board", nargs="*", default=[], help="Board cards, e.g. Ah 7d 2c")
    equity.add_argument("--opponents", type=int, default=1)
    equity.add_argument("--pot", type=int, default=0)
    equity.add_argument("--call", type=int, default=0)
    equity.add_argument("--action", choices=[action.label for action in Action], default=None)
    equity.add_argument("--simulations", type=int, default=10_000)
    equity.add_argument("--position", default="unknown")
    equity.add_argument("--hero-stack", type=int, default=None)
    equity.add_argument("--effective-stack", type=int, default=None)
    equity.add_argument("--big-blind", type=int, default=10)
    equity.add_argument("--facing-raise", action="store_true")

    history = subparsers.add_parser("history", help="Analyze exported hand-history text")
    history.add_argument("path", type=Path)
    history.add_argument("--json", action="store_true")

    importer = subparsers.add_parser("import-history", help="Learn opponent profiles from exported hand history")
    importer.add_argument("path", type=Path)
    importer.add_argument("--output", type=Path, default=DEFAULT_PROFILE_PATH)
    importer.add_argument("--replace", action="store_true", help="Replace existing profile file instead of merging")

    next_move = subparsers.add_parser("next-move", help="Review a practice/replay spot using saved profiles")
    next_move.add_argument("--hero", nargs=2, required=True)
    next_move.add_argument("--board", nargs="*", default=[])
    next_move.add_argument("--opponents", nargs="+", required=True, help="Opponent names from the profile file")
    next_move.add_argument("--pot", type=int, required=True)
    next_move.add_argument("--call", type=int, default=0)
    next_move.add_argument("--action", choices=[action.label for action in Action], default=None)
    next_move.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    next_move.add_argument("--simulations", type=int, default=10_000)
    next_move.add_argument("--position", default="unknown")
    next_move.add_argument("--hero-stack", type=int, default=None)
    next_move.add_argument("--effective-stack", type=int, default=None)
    next_move.add_argument("--big-blind", type=int, default=10)
    next_move.add_argument("--facing-raise", action="store_true")
    next_move.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.command == "equity":
        action = action_from_label(args.action)
        report = PostHandCoach().review_spot(
            hero_cards=parse_cards(args.hero),
            board_cards=parse_cards(args.board),
            opponent_count=args.opponents,
            pot=args.pot,
            call_amount=args.call,
            action=action,
            simulations=args.simulations,
            position=args.position,
            hero_stack=args.hero_stack,
            effective_stack=args.effective_stack,
            big_blind=args.big_blind,
            facing_raise=args.facing_raise,
        )
        print(json.dumps(report.as_dict(), indent=2))
    elif args.command == "history":
        report = HandHistoryParser().parse_file(args.path)
        if args.json:
            print(json.dumps(report.summary(), indent=2))
        else:
            print(f"Parsed hands: {report.hand_count}")
            for player, stats in report.summary().items():
                print(
                    f"{player}: hands={stats['hands']:.0f} "
                    f"VPIP={stats['vpip']:.1%} PFR={stats['pfr']:.1%} "
                    f"AF={stats['aggression_factor']:.2f} "
                    f"fold-to-raise={stats['fold_to_raise']:.1%}"
                )
    elif args.command == "import-history":
        parsed = HandHistoryParser().parse_file(args.path)
        model = parsed.opponent_model
        if args.output.exists() and not args.replace:
            existing = load_profiles(args.output)
            existing.merge(model)
            model = existing
        save_profiles(model, args.output)
        print(f"Imported {parsed.hand_count} hands into {args.output}")
        for player, stats in model.players.items():
            rates = stats.as_dict()
            print(
                f"{player}: hands={rates['hands']:.0f} "
                f"VPIP={rates['vpip']:.1%} PFR={rates['pfr']:.1%} "
                f"AF={rates['aggression_factor']:.2f}"
            )
    elif args.command == "next-move":
        model = load_profiles(args.profiles) if args.profiles.exists() else OpponentModel()
        report = PostHandCoach().review_spot(
            hero_cards=parse_cards(args.hero),
            board_cards=parse_cards(args.board),
            opponent_count=len(args.opponents),
            pot=args.pot,
            call_amount=args.call,
            action=action_from_label(args.action),
            opponent_model=model,
            opponent_names=args.opponents,
            simulations=args.simulations,
            position=args.position,
            hero_stack=args.hero_stack,
            effective_stack=args.effective_stack,
            big_blind=args.big_blind,
            facing_raise=args.facing_raise,
        )
        if args.json:
            print(json.dumps(report.as_dict(), indent=2))
        else:
            print(f"Recommendation: {report.recommendation}")
            print(f"Equity: {report.equity.equity:.1%}")
            if report.pot_odds is not None:
                print(f"Pot odds: {report.pot_odds:.1%}")
            for line in report.reasoning:
                print(f"- {line}")
            print(report.safety_note)


def save_profiles(model: OpponentModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "players": model.to_counts_dict(),
        "note": "Post-session/practice profiles. Do not use for live assistance on third-party poker sites.",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_profiles(path: Path) -> OpponentModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return OpponentModel.from_counts_dict(payload.get("players", {}))


if __name__ == "__main__":
    main()
