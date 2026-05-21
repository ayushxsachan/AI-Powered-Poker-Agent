# Poker AI Research Platform

A modular Python platform for Texas Hold'em gameplay, poker AI experiments, self-play training, opponent modeling, and real-time interfaces.

## What Is Included

- Texas Hold'em engine with blinds, betting rounds, folds, calls, raises, all-ins, side pots, community cards, winner detection, and showdown logic.
- Seven-card hand evaluator covering high card through royal flush.
- Gymnasium-compatible `PokerEnv` with action masking and shaped rewards.
- Agents: random, tight-aggressive rule-based, local PyTorch DQN, local PyTorch PPO, and tabular CFR for Kuhn plus a compact Leduc abstraction.
- Opponent analytics: VPIP, PFR, aggression factor, bluff frequency, fold-to-raise, and showdown win rate.
- Bluff strategy and suspicious-aggression scoring.
- Self-play tournament runner with ELO updates.
- CLI game, PyGame table, and FastAPI/WebSocket web room server.
- SQLite hand-record persistence.
- Pytest coverage for evaluator, betting/pots, environment, and CFR.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On CPU-only machines, install the PyTorch build recommended for your platform from the PyTorch site if the default `torch` wheel is not suitable.

## Run

Human vs AI in the terminal:

```bash
python -m poker_ai.ui.cli_game
```

WebSocket web table:

```bash
uvicorn poker_ai.ui.web_ui.server:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

PyGame table:

```bash
python -m poker_ai.ui.pygame_ui
```

## Practice Coach And Post-Session Review

The coach is for practice, private/consented games, and post-session study. It is not intended for live assistance on third-party poker sites.

Estimate a replay/practice spot:

```bash
python -m poker_ai.coaching.coach_cli equity --hero As Ks --board Ah 7d 2c --opponents 1 --pot 100 --call 25 --position btn --hero-stack 1000 --big-blind 10 --simulations 5000
```

Analyze an exported hand-history text file:

```bash
python -m poker_ai.coaching.coach_cli history path\to\hand_history.txt
```

Import hand histories into persistent opponent profiles:

```bash
python -m poker_ai.coaching.coach_cli import-history path\to\hand_history.txt --output poker_ai\logs\opponent_profiles.json
```

Use those saved profiles to review a difficult spot:

```bash
python -m poker_ai.coaching.coach_cli next-move --hero As Ks --board Ah 7d 2c --opponents Alice Bob --pot 100 --call 25 --position btn --effective-stack 1000 --big-blind 10 --profiles poker_ai\logs\opponent_profiles.json
```

The web UI also includes a Practice Coach panel at:

```text
http://127.0.0.1:8000
```

API endpoints:

```text
POST /coach/equity
POST /coach/history
GET  /coach/profiles
POST /coach/import-history
```

The coach now reports:

- Monte Carlo simulation count
- Win/tie/loss probabilities
- Pot odds and call EV
- Color-coded recommendation tier
- Stack depth in big blinds
- Position-aware preflop chart hints
- Opponent profile and range estimates from imported hand histories
- Outcome and hand-distribution bars in the browser UI

## Train

Local DQN:

```bash
python -m poker_ai.training.train_dqn --steps 50000 --backend local
```

Stable-Baselines3 DQN:

```bash
python -m poker_ai.training.train_dqn --steps 50000 --backend sb3
```

Local PPO:

```bash
python -m poker_ai.training.train_ppo --steps 50000 --backend local
```

Stable-Baselines3 PPO:

```bash
python -m poker_ai.training.train_ppo --steps 50000 --backend sb3
```

## Test

```bash
pytest
```

## Architecture

```text
poker_ai/
  agents/      random, rule-based, DQN, PPO, CFR
  engine/      cards, deck, evaluator, betting, Hold'em manager, Gym env
  training/    reward shaping, self-play, DQN/PPO scripts
  analytics/   opponent stats, bluffing, persistence, charts
  ui/          CLI, PyGame, FastAPI/WebSocket UI
  models/      checkpoints
  logs/        databases, TensorBoard logs, analytics exports
```

## Roadmap

1. Add richer multi-player table rules and configurable betting abstractions.
2. Add vectorized self-play workers and replay review.
3. Train action-masked PPO with league self-play.
4. Expand CFR into Deep CFR with Hold'em abstractions.
5. Add coaching explanations and OpenCV-based card/table recognition.
