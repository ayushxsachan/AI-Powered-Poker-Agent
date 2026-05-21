"""FastAPI WebSocket poker room server."""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poker_ai.agents.rule_based_agent import RuleBasedAgent
from poker_ai.analysis.hand_history_parser import HandHistoryParser
from poker_ai.analytics.opponent_model import OpponentModel
from poker_ai.coaching.post_hand_review import PostHandCoach
from poker_ai.engine.betting import Action
from poker_ai.engine.cards import Card
from poker_ai.engine.game_manager import TexasHoldemGame

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
except ImportError:  # pragma: no cover - optional dependency
    FastAPI = None
    WebSocket = None
    WebSocketDisconnect = Exception
    HTMLResponse = None


ACTION_BY_NAME = {action.label: action for action in Action}
PROFILE_PATH = Path("poker_ai/logs/opponent_profiles.json")


@dataclass
class Room:
    id: str
    game: TexasHoldemGame
    bots: dict[int, RuleBasedAgent] = field(default_factory=dict)


rooms: dict[str, Room] = {}


def create_app():
    if FastAPI is None:
        raise ImportError("fastapi and uvicorn are required for the web server")

    app = FastAPI(title="Poker AI", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    async def index():
        html = Path(__file__).with_name("index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.post("/rooms")
    async def create_room() -> dict[str, str]:
        room = _new_room()
        return {"room_id": room.id}

    @app.get("/rooms/{room_id}")
    async def room_state(room_id: str) -> dict[str, Any]:
        return _room(room_id).game.public_state(reveal_hole_cards=False)

    @app.post("/coach/equity")
    async def coach_equity(payload: dict[str, Any]) -> dict[str, Any]:
        action = _parse_action(payload.get("action"))
        opponent_names = _parse_names(payload.get("opponent_names", []))
        opponent_model = _load_profiles() if opponent_names else None
        report = PostHandCoach().review_spot(
            hero_cards=_parse_cards(payload.get("hero_cards", [])),
            board_cards=_parse_cards(payload.get("board_cards", [])),
            opponent_count=int(payload.get("opponent_count", 1)),
            pot=int(payload.get("pot", 0)),
            call_amount=int(payload.get("call_amount", 0)),
            action=action,
            opponent_model=opponent_model,
            opponent_names=opponent_names,
            simulations=int(payload.get("simulations", 5_000)),
            position=str(payload.get("position", "unknown")),
            hero_stack=_optional_int(payload.get("hero_stack")),
            effective_stack=_optional_int(payload.get("effective_stack")),
            big_blind=int(payload.get("big_blind", 10)),
            facing_raise=bool(payload.get("facing_raise", False)),
        )
        return report.as_dict()

    @app.post("/coach/history")
    async def coach_history(payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text", ""))
        report = HandHistoryParser().parse_text(text)
        return {
            "hand_count": report.hand_count,
            "players": report.summary(),
            "safety_note": (
                "This endpoint is for exported hand histories and post-session review, "
                "not live monitoring of third-party poker sites."
            ),
        }

    @app.get("/coach/profiles")
    async def coach_profiles() -> dict[str, Any]:
        model = _load_profiles()
        return {
            "profile_path": str(PROFILE_PATH),
            "players": {
                name: stats.as_dict()
                for name, stats in sorted(model.players.items())
            },
        }

    @app.post("/coach/import-history")
    async def coach_import_history(payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text", ""))
        replace = bool(payload.get("replace", False))
        parsed = HandHistoryParser().parse_text(text)
        model = parsed.opponent_model
        if PROFILE_PATH.exists() and not replace:
            existing = _load_profiles()
            existing.merge(model)
            model = existing
        _save_profiles(model)
        return {
            "hand_count": parsed.hand_count,
            "profile_path": str(PROFILE_PATH),
            "players": {
                name: stats.as_dict()
                for name, stats in sorted(model.players.items())
            },
            "safety_note": (
                "Profiles are for practice, private/consented games, and post-session review."
            ),
        }

    @app.websocket("/ws/{room_id}")
    async def websocket_endpoint(websocket: WebSocket, room_id: str) -> None:
        await websocket.accept()
        room = rooms.get(room_id) or _new_room(room_id)
        await websocket.send_json(_state_payload(room))
        try:
            while True:
                message = await websocket.receive_text()
                payload = json.loads(message)
                action = _parse_action(payload.get("action"))
                if action is not None and not room.game.hand_over:
                    room.game.step(action)
                    _play_bots(room)
                elif payload.get("command") == "new_hand":
                    room.game.reset_hand()
                    _play_bots(room)
                await websocket.send_json(_state_payload(room))
        except WebSocketDisconnect:
            return

    return app


def _new_room(room_id: str | None = None) -> Room:
    rng = random.Random()
    game = TexasHoldemGame(["Human", "AI"], rng=rng)
    game.reset_hand()
    room = Room(id=room_id or uuid.uuid4().hex[:8], game=game, bots={1: RuleBasedAgent(rng=rng)})
    _play_bots(room)
    rooms[room.id] = room
    return room


def _room(room_id: str) -> Room:
    if room_id not in rooms:
        raise KeyError(f"Unknown room {room_id}")
    return rooms[room_id]


def _play_bots(room: Room) -> None:
    while not room.game.hand_over and room.game.current_player_index in room.bots:
        index = room.game.current_player_index
        room.game.step(room.bots[index].act(room.game, index))


def _parse_action(value: Any) -> Action | None:
    if value is None:
        return None
    if isinstance(value, int):
        return Action(value)
    return ACTION_BY_NAME.get(str(value).lower())


def _parse_cards(values: Any) -> list[Card]:
    if isinstance(values, str):
        values = [item for item in values.replace(",", " ").split(" ") if item]
    return [Card.from_str(str(value)) for value in values]


def _parse_names(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [item for item in values.replace(",", " ").split(",") if item]
    return [str(value).strip() for value in values if str(value).strip()]


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _load_profiles() -> OpponentModel:
    if not PROFILE_PATH.exists():
        return OpponentModel()
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    return OpponentModel.from_counts_dict(payload.get("players", {}))


def _save_profiles(model: OpponentModel) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "players": model.to_counts_dict(),
        "note": "Practice/post-session profiles. Not for live third-party poker assistance.",
    }
    PROFILE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _state_payload(room: Room) -> dict[str, Any]:
    state = room.game.public_state(reveal_hole_cards=room.game.hand_over)
    state["room_id"] = room.id
    state["legal_actions"] = [action.label for action in room.game.legal_actions()]
    return state


app = create_app() if FastAPI is not None else None
