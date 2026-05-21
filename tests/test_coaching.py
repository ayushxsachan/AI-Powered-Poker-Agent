from poker_ai.analysis.equity_calculator import EquityCalculator
from poker_ai.analysis.hand_history_parser import HandHistoryParser
from poker_ai.analysis.preflop_chart import PreflopChart
from poker_ai.coaching.coach_cli import load_profiles, save_profiles
from poker_ai.coaching.post_hand_review import PostHandCoach
from poker_ai.engine.betting import Action
from poker_ai.engine.cards import cards_from_strings


def test_equity_calculator_with_known_complete_board():
    report = EquityCalculator().estimate(
        cards_from_strings(["As", "Ah"]),
        cards_from_strings(["2c", "3d", "4h", "5s", "9c"]),
        opponent_count=1,
        known_opponent_cards=[cards_from_strings(["Kd", "Ks"])],
        simulations=20,
    )

    assert report.wins == 20
    assert report.losses == 0
    assert report.equity == 1.0


def test_post_hand_coach_recommends_fold_when_equity_is_too_low():
    report = PostHandCoach().review_spot(
        hero_cards=cards_from_strings(["2s", "7d"]),
        board_cards=cards_from_strings(["As", "Kd", "Qh", "9c", "4s"]),
        opponent_count=1,
        pot=100,
        call_amount=100,
        action=Action.CALL,
        simulations=200,
    )

    assert report.pot_odds == 0.5
    assert report.expected_value is not None
    assert report.expected_value < 0
    assert report.recommendation_tier in {"caution", "fold"}
    assert "fold" in report.recommendation.lower()


def test_preflop_chart_and_stack_depth_are_reported():
    report = PostHandCoach().review_spot(
        hero_cards=cards_from_strings(["As", "Ks"]),
        board_cards=[],
        opponent_count=1,
        pot=15,
        call_amount=0,
        position="btn",
        hero_stack=1000,
        big_blind=10,
        simulations=50,
    )

    assert report.preflop_advice is not None
    assert report.preflop_advice.hand_code == "AKs"
    assert report.stack_depth_bb == 100.0
    assert report.stack_depth_label == "deep"


def test_preflop_chart_tightens_speculative_hands_when_shallow():
    advice = PreflopChart().advise(
        cards_from_strings(["7s", "6s"]),
        position="btn",
        stack_depth_bb=12,
    )

    assert advice is not None
    assert advice.recommendation == "tighten up"


def test_hand_history_parser_builds_opponent_stats():
    text = """
PokerStars Hand #1
*** HOLE CARDS ***
Alice: raises 30
Bob: calls 30
*** FLOP *** [Ah 7d 2c]
Alice: bets 40
Bob: folds
Alice collected 100 from pot

PokerStars Hand #2
*** HOLE CARDS ***
Alice: calls 10
Bob: raises 40
Alice: folds
Bob collected 60 from pot
"""
    report = HandHistoryParser().parse_text(text)
    summary = report.summary()

    assert report.hand_count == 2
    assert summary["Alice"]["hands"] == 2
    assert summary["Alice"]["vpip"] == 1.0
    assert summary["Bob"]["pfr"] == 0.5
    assert summary["Alice"]["fold_to_raise"] == 1.0


def test_profile_save_load_roundtrip(tmp_path):
    text = """
PokerStars Hand #1
*** HOLE CARDS ***
Alice: raises 30
Bob: calls 30
*** FLOP *** [Ah 7d 2c]
Alice: bets 40
Bob: folds
Alice collected 100 from pot
"""
    parsed = HandHistoryParser().parse_text(text)
    output = tmp_path / "profiles.json"

    save_profiles(parsed.opponent_model, output)
    loaded = load_profiles(output)

    assert loaded.stats_for("Alice").hands == 1
    assert loaded.stats_for("Alice").pfr_rate == 1.0
    assert loaded.stats_for("Bob").vpip_rate == 1.0


def test_web_coach_endpoint():
    import asyncio

    from poker_ai.ui.web_ui.server import app

    endpoint = next(route.endpoint for route in app.routes if route.path == "/coach/equity")
    response = asyncio.run(
        endpoint(
            {
            "hero_cards": ["As", "Ah"],
            "board_cards": ["2c", "3d", "4h", "5s", "9c"],
            "opponent_count": 1,
            "pot": 100,
            "call_amount": 20,
            "simulations": 20,
            "position": "btn",
            "hero_stack": 1000,
            "big_blind": 10,
            }
        )
    )

    assert response["equity"]["simulations"] == 20
    assert response["expected_value"] is not None
    assert response["recommendation_tier"] in {"strong", "good", "caution", "fold"}
