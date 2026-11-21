"""Piano-roll preview of a transcription: one canvas with a pitched panel (88 keys) and a drum panel (GM keys 27-87)."""

from __future__ import annotations

import numpy as np

PITCH_TOP, PITCH_BOTTOM = 108, 21           # G7 .. A0, the 88 piano keys
DRUM_TOP, DRUM_BOTTOM = 87, 27              # top .. bottom GM drum key shown
N_PITCH_ROWS = PITCH_TOP - PITCH_BOTTOM + 1
N_DRUM_ROWS = DRUM_TOP - DRUM_BOTTOM + 1

#: One colour per track, in track order (the grouper lists tracks largest first).
PALETTE = [
    (0.30, 0.77, 0.98),   # blue
    (0.99, 0.60, 0.24),   # orange
    (0.42, 0.87, 0.54),   # green
    (0.95, 0.40, 0.72),   # pink
    (0.70, 0.60, 0.97),   # purple
    (0.99, 0.85, 0.35),   # yellow
    (0.45, 0.90, 0.85),   # cyan
    (0.96, 0.48, 0.44),   # red
    (0.66, 0.82, 0.43),   # olive
    (0.72, 0.61, 0.53),   # taupe
    (0.51, 0.54, 0.90),   # indigo
    (0.90, 0.70, 0.45),   # tan
]
DRUM_COLOR = (0.94, 0.33, 0.32)

BG_PITCH = (0.055, 0.055, 0.070)
BG_C_ROW = (0.100, 0.100, 0.120)          # C rows mark the piano key boundaries
BG_DRUM = (0.048, 0.060, 0.090)
SEPARATOR = (0.30, 0.30, 0.36)
TICK = (0.16, 0.16, 0.20)
TICK_STRONG = (0.26, 0.26, 0.32)


def _track_get(track, key, default=None):
    """Tracks arrive as dettato Track dataclasses (dicts are accepted too)."""
    if isinstance(track, dict):
        return track.get(key, default)
    return getattr(track, key, default)


def _row_for(track, pitch: float) -> int | None:
    """Canvas row of a note, or None when it falls outside the drawn panels."""
    p = int(round(pitch))
    if _track_get(track, "is_drum"):
        if not (DRUM_BOTTOM <= p <= DRUM_TOP):
            return None
        return N_PITCH_ROWS + (DRUM_TOP - p)
    if not (PITCH_BOTTOM <= p <= PITCH_TOP):
        return None
    return PITCH_TOP - p


def render_piano_roll(
    tracks: list[dict],
    seconds: float,
    px_per_ms: float = 0.0,
    max_seconds: float = 0.0,
    show_drum_panel: bool = True,
    row_px: int = 2,
) -> np.ndarray:
    """-> (H, W, 3) float32 RGB in 0..1. Notes are rectangles, brightness scaled by velocity."""
    duration = max(float(seconds), 0.05)
    if max_seconds > 0:
        duration = min(duration, float(max_seconds))
    if px_per_ms > 0:
        width = int(round(duration * 1000.0 * px_per_ms))
    else:
        width = int(round(min(2.0, 1400.0 / (duration * 1000.0)) * duration * 1000.0))
    width = max(64, min(4096, width))
    px = width / (duration * 1000.0)

    n_drum = N_DRUM_ROWS if show_drum_panel else 0
    height = (N_PITCH_ROWS + n_drum + 1) * row_px
    canvas = np.zeros((height, width, 3), np.float32)

    # Background: C rows shaded, drum panel its own colour, one separator row.
    for row in range(N_PITCH_ROWS):
        pitch = PITCH_TOP - row
        colour = BG_C_ROW if pitch % 12 == 0 else BG_PITCH
        canvas[row * row_px:(row + 1) * row_px] = colour
    if n_drum:
        canvas[N_PITCH_ROWS * row_px:(N_PITCH_ROWS + n_drum) * row_px] = BG_DRUM
    canvas[(N_PITCH_ROWS + n_drum) * row_px:] = SEPARATOR

    # Second grid: a line per second, stronger every fifth.
    for s in range(int(duration)):
        x = int(round((s + 1) * 1000.0 * px))
        if 0 < x < width:
            canvas[:, x] = TICK_STRONG if (s + 1) % 5 == 0 else TICK

    for i, track in enumerate(tracks):
        is_drum = bool(_track_get(track, "is_drum"))
        colour = DRUM_COLOR if is_drum else PALETTE[i % len(PALETTE)]
        notes = _track_get(track, "notes")
        if notes is None or len(notes) == 0:
            continue
        if is_drum and not show_drum_panel:
            continue
        for on, off, pitch, vel in np.asarray(notes, np.float64).reshape(-1, 4):
            row = _row_for(track, pitch)
            if row is None or off <= 0 or on > duration:
                continue
            x0 = max(0, int(round(on * 1000.0 * px)))
            x1 = max(x0 + 1, min(width, int(round(off * 1000.0 * px))))
            y0, y1 = row * row_px, (row + 1) * row_px
            bright = 0.5 + 0.5 * float(np.clip(vel, 1, 127)) / 127.0
            canvas[y0:y1, x0:x1] = np.clip(np.array(colour) * bright, 0.0, 1.0)
    return canvas
