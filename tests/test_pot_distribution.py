from poker_ai.engine.cards import cards_from_strings
from poker_ai.engine.game_manager import GamePhase, TexasHoldemGame


def test_all_in_side_pots_are_split_by_eligibility() -> None:
    game = TexasHoldemGame(["short", "middle", "deep"], starting_stack=0)
    game.phase = GamePhase.SHOWDOWN
    game.community_cards = cards_from_strings(["2c", "3d", "4h", "5s", "9c"])

    game.players[0].hole_cards = cards_from_strings(["As", "Ah"])
    game.players[1].hole_cards = cards_from_strings(["Ks", "Kh"])
    game.players[2].hole_cards = cards_from_strings(["Qs", "Qh"])

    commitments = [50, 100, 100]
    for player, committed in zip(game.players, commitments):
        player.stack = 0
        player.committed = committed
        player.current_bet = committed
        player.folded = False
        player.all_in = True

    game._showdown()

    assert game.last_result is not None
    assert game.last_result.payouts[0] == 150
    assert game.last_result.payouts[1] == 100
    assert 2 not in game.last_result.payouts

