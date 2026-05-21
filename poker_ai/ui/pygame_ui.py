"""Minimal PyGame table for human-vs-AI play."""

from __future__ import annotations

import random

from poker_ai.agents.rule_based_agent import RuleBasedAgent
from poker_ai.engine.betting import Action
from poker_ai.engine.game_manager import TexasHoldemGame


def run() -> None:
    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pygame is required for the graphical UI") from exc

    pygame.init()
    screen = pygame.display.set_mode((1_000, 680))
    pygame.display.set_caption("Poker AI")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24)
    small_font = pygame.font.SysFont("arial", 18)

    rng = random.Random()
    game = TexasHoldemGame(["You", "TAG Bot"], rng=rng)
    bot = RuleBasedAgent(rng=rng)
    game.reset_hand()

    button_rects = _button_rects()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not game.hand_over:
                for action, rect in button_rects.items():
                    if rect.collidepoint(event.pos) and action in game.legal_actions():
                        game.step(action)
                        _play_bot_until_human(game, bot)
            elif event.type == pygame.KEYDOWN and game.hand_over:
                if event.key == pygame.K_SPACE:
                    game.reset_hand()
                    _play_bot_until_human(game, bot)

        screen.fill((20, 92, 62))
        _draw_table(screen, font, small_font, game, button_rects)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def _play_bot_until_human(game: TexasHoldemGame, bot: RuleBasedAgent) -> None:
    while not game.hand_over and game.current_player_index != 0:
        game.step(bot.act(game, game.current_player_index))


def _button_rects():  # type: ignore[no-untyped-def]
    import pygame

    actions = [Action.FOLD, Action.CHECK, Action.CALL, Action.SMALL_RAISE, Action.MEDIUM_RAISE, Action.ALL_IN]
    return {
        action: pygame.Rect(80 + i * 145, 585, 130, 48)
        for i, action in enumerate(actions)
    }


def _draw_table(screen, font, small_font, game: TexasHoldemGame, button_rects) -> None:  # type: ignore[no-untyped-def]
    import pygame

    pygame.draw.ellipse(screen, (24, 122, 78), pygame.Rect(120, 80, 760, 420))
    pygame.draw.ellipse(screen, (225, 184, 92), pygame.Rect(120, 80, 760, 420), 6)

    _text(screen, font, f"Pot: {game.pot}", (445, 120))
    _text(screen, font, f"Board: {_cards(game.community_cards)}", (345, 205))

    for i, player in enumerate(game.players):
        y = 470 if i == 0 else 55
        cards = _cards(player.hole_cards) if i == 0 or game.hand_over else "?? ??"
        status = " folded" if player.folded else " all-in" if player.all_in else ""
        _text(screen, font, f"{player.name}: {cards}", (390, y))
        _text(screen, small_font, f"stack={player.stack} bet={player.current_bet}{status}", (390, y + 32))

    if game.hand_over and game.last_result:
        _text(screen, font, game.last_result.summary, (280, 540))
        _text(screen, small_font, "Press SPACE for next hand", (395, 570))
    else:
        legal = set(game.legal_actions())
        for action, rect in button_rects.items():
            enabled = action in legal
            color = (238, 238, 238) if enabled else (112, 112, 112)
            pygame.draw.rect(screen, color, rect, border_radius=6)
            pygame.draw.rect(screen, (26, 26, 26), rect, 2, border_radius=6)
            _text(screen, small_font, action.label.replace("_", " "), (rect.x + 15, rect.y + 14), (20, 20, 20))


def _text(screen, font, value: str, pos: tuple[int, int], color: tuple[int, int, int] = (255, 255, 255)) -> None:  # type: ignore[no-untyped-def]
    screen.blit(font.render(value, True, color), pos)


def _cards(cards) -> str:  # type: ignore[no-untyped-def]
    return " ".join(card.pretty for card in cards) if cards else "-"


if __name__ == "__main__":
    run()

