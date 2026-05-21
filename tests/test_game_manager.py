import random

from poker_ai.engine.betting import Action
from poker_ai.engine.cards import cards_from_strings
from poker_ai.engine.game_manager import GamePhase, TexasHoldemGame


def test_heads_up_blinds_and_fold_award():
    game = TexasHoldemGame(["Alice", "Bob"], starting_stack=100, small_blind=5, big_blind=10, rng=random.Random(1))
    game.reset_hand()

    assert game.phase == GamePhase.PREFLOP
    assert game.players[0].current_bet == 5
    assert game.players[1].current_bet == 10
    assert Action.CALL in game.legal_actions()

    game.step(Action.FOLD)

    assert game.hand_over
    assert game.last_result is not None
    assert game.last_result.winners == [1]
    assert game.players[1].stack == 105


def test_side_pot_distribution():
    game = TexasHoldemGame(["Short", "Middle", "Big"], starting_stack=100)
    game.reset_hand()
    game.community_cards = cards_from_strings(["2c", "3d", "4h", "5s", "9c"])
    game.phase = GamePhase.SHOWDOWN

    for player in game.players:
        player.folded = False
        player.all_in = True
        player.stack = 0
        player.current_bet = 0

    game.players[0].hole_cards = cards_from_strings(["As", "Ah"])
    game.players[1].hole_cards = cards_from_strings(["Kd", "Ks"])
    game.players[2].hole_cards = cards_from_strings(["Qd", "Qs"])
    game.players[0].committed = 50
    game.players[1].committed = 100
    game.players[2].committed = 100

    game._showdown()

    assert game.last_result is not None
    assert game.last_result.payouts[0] == 150
    assert game.last_result.payouts[1] == 100
    assert game.players[0].stack == 150
    assert game.players[1].stack == 100

