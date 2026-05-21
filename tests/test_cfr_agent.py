from poker_ai.agents.cfr_agent import CFRPokerAgent, KuhnCFRTrainer, LeducCFRTrainer


def test_kuhn_cfr_trains_strategy_table():
    trainer = KuhnCFRTrainer()
    trainer.train(100)
    strategy = trainer.average_strategy()

    assert strategy
    assert all(abs(sum(actions.values()) - 1.0) < 1e-6 for actions in strategy.values())


def test_leduc_cfr_trains_strategy_table():
    trainer = LeducCFRTrainer()
    trainer.train(100)
    strategy = trainer.average_strategy()

    assert strategy
    assert all(abs(sum(actions.values()) - 1.0) < 1e-6 for actions in strategy.values())


def test_cfr_agent_wrapper():
    agent = CFRPokerAgent(game="kuhn", iterations=50)
    assert agent.strategy()

