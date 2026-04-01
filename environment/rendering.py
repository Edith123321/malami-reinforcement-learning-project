"""
Malami - Pygame Visualisation
==============================
Rich 2D GUI showing:
  - Per-topic mastery progress bars with colour gradient
  - Engagement meter
  - Recent quiz scores sparkline
  - Agent action history
  - Step counter and topic tracker
  - Live reward display
"""

import sys
import os
import math
from typing import Dict, List, Optional

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Colour palette
BG_DARK      = (10,  14,  26)
BG_PANEL     = (18,  24,  44)
BG_CARD      = (26,  34,  60)
ACCENT_BLUE  = (64, 156, 255)
ACCENT_GREEN = (52, 211, 153)
ACCENT_RED   = (251,  99,  64)
ACCENT_GOLD  = (251, 191,  36)
ACCENT_PURP  = (167, 139, 250)
TEXT_WHITE   = (230, 235, 255)
TEXT_GREY    = (120, 130, 160)
TEXT_DIM     = ( 60,  70, 100)

ACTION_NAMES = [
    "Video Lesson", "Practice Quiz", "Problem Solving", "Review Summary",
    "Adaptive Hint", "New Topic", "Remediation", "Peer Discussion", "Worked Example"
]
ACTION_ICONS = ["▶", "?", "⚙", "📋", "💡", "→", "↩", "👥", "📖"]
ACTION_COLORS = [
    ACCENT_BLUE, ACCENT_GOLD, ACCENT_PURP, TEXT_GREY, ACCENT_GREEN,
    ACCENT_GREEN, ACCENT_RED, ACCENT_BLUE, ACCENT_PURP,
]

TOPIC_COLORS = [
    (52, 211, 153),   # Biology   – green
    (96, 165, 250),   # Chemistry – blue
    (251, 146, 60),   # Physics   – orange
    (167, 139, 250),  # Maths     – purple
    (34, 211, 238),   # CS        – cyan
    (251, 191, 36),   # AI & ML   – gold
]

W, H = 960, 680


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def mastery_color(v: float):
    if v < 0.5:
        return lerp_color(ACCENT_RED, ACCENT_GOLD, v * 2)
    return lerp_color(ACCENT_GOLD, ACCENT_GREEN, (v - 0.5) * 2)


class MalamiRenderer:
    def __init__(self):
        if not PYGAME_AVAILABLE:
            print("[Renderer] pygame not installed – skipping GUI.")
            return
        pygame.init()
        pygame.display.set_caption("Malami – Adaptive Learning Tutor")
        self.screen  = pygame.display.set_mode((W, H))
        self.clock   = pygame.time.Clock()
        self._load_fonts()
        self.action_history: List[int] = []
        self.reward_history: List[float] = []
        self.frame = 0

    def _load_fonts(self):
        self.font_title  = pygame.font.SysFont("DejaVuSans",  28, bold=True)
        self.font_head   = pygame.font.SysFont("DejaVuSans",  16, bold=True)
        self.font_body   = pygame.font.SysFont("DejaVuSans",  13)
        self.font_small  = pygame.font.SysFont("DejaVuSans",  11)
        self.font_icon   = pygame.font.SysFont("DejaVuSans",  18)

    def render(self, state: Dict, action: Optional[int] = None, reward: Optional[float] = None):
        if not PYGAME_AVAILABLE:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

        if action is not None:
            self.action_history.append(action)
            if len(self.action_history) > 12:
                self.action_history.pop(0)
        if reward is not None:
            self.reward_history.append(reward)
            if len(self.reward_history) > 40:
                self.reward_history.pop(0)

        self.screen.fill(BG_DARK)
        self._draw_header(state)
        self._draw_mastery_panel(state)
        self._draw_engagement_panel(state)
        self._draw_action_panel(state, action)
        self._draw_reward_chart()
        self._draw_topic_indicator(state)

        pygame.display.flip()
        self.clock.tick(30)
        self.frame += 1

    def _draw_header(self, state: Dict):
        # Title bar
        pygame.draw.rect(self.screen, BG_PANEL, (0, 0, W, 56))
        pygame.draw.line(self.screen, ACCENT_BLUE, (0, 56), (W, 56), 1)
        t = self.font_title.render("MALAMI  –  Adaptive Learning Tutor", True, TEXT_WHITE)
        self.screen.blit(t, (20, 14))
        step_txt = self.font_head.render(f"Step {state.get('step', 0):>4}", True, ACCENT_GOLD)
        self.screen.blit(step_txt, (W - 120, 18))

    def _draw_mastery_panel(self, state: Dict):
        x0, y0, pw, ph = 20, 72, 420, 380
        pygame.draw.rect(self.screen, BG_PANEL, (x0, y0, pw, ph), border_radius=10)
        pygame.draw.rect(self.screen, ACCENT_BLUE, (x0, y0, pw, ph), 1, border_radius=10)
        lbl = self.font_head.render("TOPIC MASTERY", True, ACCENT_BLUE)
        self.screen.blit(lbl, (x0 + 14, y0 + 12))

        masteries = state.get("topic_masteries", [0.0] * 6)
        names     = state.get("topic_names", [f"T{i}" for i in range(6)])
        current   = state.get("current_topic", 0)

        bar_h   = 36
        bar_gap = 18
        bx      = x0 + 14
        by_start = y0 + 44

        for i, (m, name) in enumerate(zip(masteries, names)):
            by = by_start + i * (bar_h + bar_gap)
            # Background track
            pygame.draw.rect(self.screen, BG_CARD, (bx, by, pw - 28, bar_h), border_radius=6)
            # Fill
            fill_w = int((pw - 28) * m)
            if fill_w > 0:
                col = TOPIC_COLORS[i % len(TOPIC_COLORS)]
                pygame.draw.rect(self.screen, col, (bx, by, fill_w, bar_h), border_radius=6)
            # Highlight active topic
            border_col = ACCENT_GOLD if i == current else TEXT_DIM
            border_w   = 2 if i == current else 1
            pygame.draw.rect(self.screen, border_col, (bx, by, pw - 28, bar_h), border_w, border_radius=6)
            # Label
            label = self.font_body.render(name, True, TEXT_WHITE if i == current else TEXT_GREY)
            self.screen.blit(label, (bx + 8, by + 10))
            pct = self.font_body.render(f"{m*100:.1f}%", True, TEXT_WHITE)
            self.screen.blit(pct, (bx + pw - 68, by + 10))
            # Mastery threshold line
            thresh_x = bx + int((pw - 28) * 0.85)
            pygame.draw.line(self.screen, TEXT_DIM, (thresh_x, by), (thresh_x, by + bar_h), 1)

    def _draw_engagement_panel(self, state: Dict):
        x0, y0, pw, ph = 460, 72, 480, 130
        pygame.draw.rect(self.screen, BG_PANEL, (x0, y0, pw, ph), border_radius=10)
        pygame.draw.rect(self.screen, ACCENT_PURP, (x0, y0, pw, ph), 1, border_radius=10)
        lbl = self.font_head.render("ENGAGEMENT", True, ACCENT_PURP)
        self.screen.blit(lbl, (x0 + 14, y0 + 12))
        eng = state.get("engagement", 1.0)
        bx, by = x0 + 14, y0 + 42
        bw, bh = pw - 28, 32
        pygame.draw.rect(self.screen, BG_CARD, (bx, by, bw, bh), border_radius=8)
        fw = int(bw * eng)
        if fw > 0:
            col = lerp_color(ACCENT_RED, ACCENT_GREEN, eng)
            pygame.draw.rect(self.screen, col, (bx, by, fw, bh), border_radius=8)
        pygame.draw.rect(self.screen, TEXT_DIM, (bx, by, bw, bh), 1, border_radius=8)
        pct = self.font_head.render(f"{eng*100:.1f}%", True, TEXT_WHITE)
        self.screen.blit(pct, (bx + bw // 2 - 22, by + 7))
        # Recent quiz scores
        scores = state.get("recent_scores", [0, 0, 0])
        sq_lbl = self.font_small.render("Recent Quiz Scores:", True, TEXT_GREY)
        self.screen.blit(sq_lbl, (x0 + 14, y0 + 88))
        for j, sc in enumerate(scores):
            cx = x0 + 150 + j * 70
            col = mastery_color(sc)
            pygame.draw.rect(self.screen, BG_CARD,  (cx, y0 + 85, 56, 22), border_radius=4)
            pygame.draw.rect(self.screen, col, (cx, y0 + 85, int(56 * sc), 22), border_radius=4)
            pygame.draw.rect(self.screen, TEXT_DIM, (cx, y0 + 85, 56, 22), 1, border_radius=4)
            st = self.font_small.render(f"{sc*100:.0f}", True, TEXT_WHITE)
            self.screen.blit(st, (cx + 16, y0 + 89))

    def _draw_action_panel(self, state: Dict, last_action: Optional[int]):
        x0, y0, pw, ph = 460, 218, 480, 234
        pygame.draw.rect(self.screen, BG_PANEL, (x0, y0, pw, ph), border_radius=10)
        pygame.draw.rect(self.screen, ACCENT_GOLD, (x0, y0, pw, ph), 1, border_radius=10)
        lbl = self.font_head.render("ACTION SPACE", True, ACCENT_GOLD)
        self.screen.blit(lbl, (x0 + 14, y0 + 12))
        cols, rows = 3, 3
        bw = (pw - 28 - (cols - 1) * 6) // cols
        bh = (ph - 44 - (rows - 1) * 6) // rows
        for i in range(9):
            c, r  = i % cols, i // cols
            bx = x0 + 14 + c * (bw + 6)
            by = y0 + 38 + r * (bh + 6)
            active = (last_action == i)
            bg_col = ACTION_COLORS[i] if active else BG_CARD
            pygame.draw.rect(self.screen, bg_col, (bx, by, bw, bh), border_radius=6)
            border = TEXT_WHITE if active else TEXT_DIM
            pygame.draw.rect(self.screen, border, (bx, by, bw, bh), 1 if not active else 2, border_radius=6)
            name = self.font_small.render(ACTION_NAMES[i], True, TEXT_WHITE if active else TEXT_GREY)
            self.screen.blit(name, (bx + 4, by + bh // 2 - 6))

    def _draw_reward_chart(self):
        x0, y0, pw, ph = 20, 468, 920, 180
        pygame.draw.rect(self.screen, BG_PANEL, (x0, y0, pw, ph), border_radius=10)
        pygame.draw.rect(self.screen, ACCENT_GREEN, (x0, y0, pw, ph), 1, border_radius=10)
        lbl = self.font_head.render("CUMULATIVE REWARD", True, ACCENT_GREEN)
        self.screen.blit(lbl, (x0 + 14, y0 + 10))
        if len(self.reward_history) < 2:
            return
        cx0, cy0 = x0 + 14, y0 + 34
        cw, ch   = pw - 28, ph - 50
        # Zero line
        pygame.draw.line(self.screen, TEXT_DIM, (cx0, cy0 + ch // 2), (cx0 + cw, cy0 + ch // 2), 1)
        mn = min(self.reward_history); mx = max(self.reward_history)
        span = max(mx - mn, 1e-6)
        pts = []
        for k, r in enumerate(self.reward_history):
            rx = cx0 + int(k * cw / max(len(self.reward_history) - 1, 1))
            ry = cy0 + ch - int((r - mn) / span * ch)
            pts.append((rx, ry))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, ACCENT_GREEN, False, pts, 2)
        # Dots
        for pt in pts[-3:]:
            pygame.draw.circle(self.screen, ACCENT_GREEN, pt, 3)

    def _draw_topic_indicator(self, state: Dict):
        x0, y0 = 460, 464
        pw, ph  = 480, 185
        pygame.draw.rect(self.screen, BG_PANEL, (x0, y0, pw, ph), border_radius=10)
        pygame.draw.rect(self.screen, ACCENT_PURP, (x0, y0, pw, ph), 1, border_radius=10)
        lbl = self.font_head.render("CURRICULUM PROGRESS", True, ACCENT_PURP)
        self.screen.blit(lbl, (x0 + 14, y0 + 10))
        masteries = state.get("topic_masteries", [0.0]*6)
        names     = state.get("topic_names", [])
        current   = state.get("current_topic", 0)
        node_r = 18
        spacing = pw // (len(names) + 1)
        for i, (name, m) in enumerate(zip(names, masteries)):
            cx = x0 + spacing * (i + 1)
            cy = y0 + 90
            col = TOPIC_COLORS[i % len(TOPIC_COLORS)]
            pygame.draw.circle(self.screen, BG_CARD, (cx, cy), node_r)
            # Arc showing mastery
            arc_rect = pygame.Rect(cx - node_r, cy - node_r, node_r * 2, node_r * 2)
            end_angle = -math.pi / 2 + 2 * math.pi * m
            if m > 0:
                pygame.draw.arc(self.screen, col, arc_rect, -math.pi / 2, end_angle, 4)
            # Connector
            if i < len(names) - 1:
                nx = x0 + spacing * (i + 2)
                pygame.draw.line(self.screen, TEXT_DIM, (cx + node_r, cy), (nx - node_r, cy), 1)
            border = ACCENT_GOLD if i == current else TEXT_DIM
            pygame.draw.circle(self.screen, border, (cx, cy), node_r, 2)
            short = name[:3]
            t = self.font_small.render(short, True, TEXT_WHITE if i == current else TEXT_GREY)
            self.screen.blit(t, (cx - t.get_width() // 2, cy - t.get_height() // 2))
            pct = self.font_small.render(f"{m*100:.0f}%", True, col)
            self.screen.blit(pct, (cx - pct.get_width() // 2, cy + node_r + 4))

        # Consecutive fails warning
        fails = state.get("consecutive_fails", 0)
        if fails >= 2:
            warn = self.font_body.render(f"⚠ Fail streak: {fails}", True, ACCENT_RED)
            self.screen.blit(warn, (x0 + 14, y0 + 156))

    def close(self):
        if PYGAME_AVAILABLE:
            pygame.quit()
