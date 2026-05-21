"""Command-line human-vs-AI poker game."""

from __future__ import annotations

import argparse
import random

from poker_ai.agents.rule_based_agent import RuleBasedAgent
from poker_ai.engine.betting import Action
from poker_ai.engine.game_manager import TexasHoldemGame


ACTION_ALIASES = {
    "f": Action.FOLD,
    "fold": Action.FOLD,
    "x": Action.CHECK,
    "check": Action.CHECK,
    "c": Action.CALL,
    "call": Action.CALL,
    "s": Action.SMALL_RAISE,
    "small": Action.SMALL_RAISE,
    "m": Action.MEDIUM_RAISE,
    "medium": Action.MEDIUM_RAISE,
    "l": Action.LARGE_RAISE,
    "large": Action.LARGE_RAISE,
    "a": Action.ALL_IN,
    "allin": Action.ALL_IN,
    "all-in": Action.ALL_IN,
}


def render(game: TexasHoldemGame, hero_index: int = 0) -> None:
    state = game.public_state(reveal_hole_cards=False)
    hero = game.players[hero_index]
    print("\n" + "=" * 72)
    print(f"Hand {game.hand_number} | {state['phase']} | pot={state['pot']} | board={_cards(game.community_cards)}")
    print(f"Your cards: {_cards(hero.hole_cards)} | stack={hero.stack} | bet={hero.current_bet}")
    for index, player in enumerate(game.players):
        if index == hero_index:
            continue
        status = "folded" if player.folded else "all-in" if player.all_in else "active"
        print(f"{player.name}: stack={player.stack} bet={player.current_bet} {status}")


def prompt_action(game: TexasHoldemGame) -> Action:
    legal = game.legal_actions()
    labels = ", ".join(action.label for action in legal)
    while True:
        raw = input(f"Action ({labels}): ").strip().lower()
        action = ACTION_ALIASES.get(raw)
        if action in legal:
            return action
        print("That action is not legal here.")


def _cards(cards) -> str:  # type: ignore[no-untyped-def]
    return " ".join(card.pretty for card in cards) if cards else "-"


def play(starting_stack: int = 1_000, seed: int | None = None) -> None:
    rng = random.Random(seed)
    game = TexasHoldemGame(["You", "TAG Bot"], starting_stack=starting_stack, rng=rng)
    bot = RuleBasedAgent(rng=rng)

    while all(player.stack > 0 for player in game.players):
        game.reset_hand()
        while not game.hand_over:
            if game.current_player_index == 0:
                render(game)
                game.step(prompt_action(game))
            else:
                action = bot.act(game, 1)
                print(f"TAG Bot chooses {action.label}")
                game.step(action)

        final_state = game.public_state(reveal_hole_cards=True)
        print("\nShowdown")
        print(f"Board: {_cards(game.community_cards)}")
        for player in game.players:
            print(f"{player.name}: {_cards(player.hole_cards)} stack={player.stack}")
        print(final_state["last_result"]["summary"])

        again = input("Play next hand? [Y/n] ").strip().lower()
        if again == "n":
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    play(starting_stack=args.stack, seed=args.seed)


if __name__ == "__main__":
    main()

