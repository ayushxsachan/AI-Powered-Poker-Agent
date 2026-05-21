from poker_ai.engine.cards import cards_from_strings
from poker_ai.engine.hand_evaluator import HandCategory, HandEvaluator


def category(card_texts):
    return HandEvaluator.evaluate(cards_from_strings(card_texts)).category


def test_all_hand_categories():
    cases = [
        (["As", "Kd", "Qh", "9c", "7d", "4s", "2h"], HandCategory.HIGH_CARD),
        (["As", "Ad", "Qh", "9c", "7d", "4s", "2h"], HandCategory.PAIR),
        (["As", "Ad", "Qh", "Qc", "7d", "4s", "2h"], HandCategory.TWO_PAIR),
        (["As", "Ad", "Ah", "Qc", "7d", "4s", "2h"], HandCategory.THREE_OF_A_KIND),
        (["As", "2d", "3h", "4c", "5d", "Ks", "Qh"], HandCategory.STRAIGHT),
        (["As", "Qs", "9s", "6s", "2s", "Kd", "3h"], HandCategory.FLUSH),
        (["As", "Ad", "Ah", "Qc", "Qd", "4s", "2h"], HandCategory.FULL_HOUSE),
        (["As", "Ad", "Ah", "Ac", "Qd", "4s", "2h"], HandCategory.FOUR_OF_A_KIND),
        (["9s", "8s", "7s", "6s", "5s", "Ad", "2h"], HandCategory.STRAIGHT_FLUSH),
        (["As", "Ks", "Qs", "Js", "Ts", "2d", "3h"], HandCategory.ROYAL_FLUSH),
    ]
    for cards, expected in cases:
        assert category(cards) == expected


def test_wheel_straight_uses_five_high_tiebreaker():
    evaluation = HandEvaluator.evaluate(cards_from_strings(["As", "2d", "3h", "4c", "5d", "Ks", "Qh"]))
    assert evaluation.category == HandCategory.STRAIGHT
    assert evaluation.tiebreakers == (5,)


def test_pair_kicker_breaks_ties():
    first = HandEvaluator.evaluate(cards_from_strings(["As", "Ad", "Kh", "8c", "7d", "4s", "2h"]))
    second = HandEvaluator.evaluate(cards_from_strings(["Ac", "Ah", "Qh", "8d", "7c", "4d", "2s"]))
    assert first.score > second.score

