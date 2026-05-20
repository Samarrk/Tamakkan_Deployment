"""
src/tamakkan/models/light_classifier.py

Traffic-light color classifier using HSV color space.

Takes a BGR crop of a traffic_light bbox from TamakkanTracker and decides
whether the bulb is red, green, or unknown (off / amber / ambiguous).

Design notes
------------
- Stateless: each crop is classified independently. No state across calls.
- CPU-only: pure NumPy/OpenCV. Runs in <1ms per crop; no GPU needed.
- Brightness-gate first: if no part of the crop is bright enough to be a
  lit bulb, return unknown immediately. Catches off lights, dark housings,
  bbox false-positives. This is the most important early filter.

Amber/yellow handling
---------------------
Amber lights fall in hue ≈ 20-30, which is inside the red range (0-35).
This is INTENTIONAL: Tamakkan treats amber as red for alerting purposes.
The amber phase is a "be ready to stop" signal, so flagging it as red is
the safe behavior. If you ever want explicit amber detection, narrow
HUE_RED_LOW to (0, 15) and add HUE_AMBER = (16, 35).

Saudi-specific tuning (TODO: measure on Jetson)
-----------------------------------------------
Saudi LED green lights can lean cyan/teal under heavy bloom — bright-center
pixels may drift toward hue 95-100. Current HUE_GREEN is (60, 95), tightened
to exclude olive/teal housings. We're keeping these values until we have
real YOLO crops from Jetson footage; if green-light detection misses bloomed
LEDs, widen to (55, 100).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class LightClassification:
    """
    Result of classifying one traffic_light crop.

    color: "red" | "green" | "unknown"
    confidence: dominance of the winning color in [0, 1]. Roughly:
        0.5 = ambiguous (will be rejected before reaching here)
        0.7 = decent
        0.9+ = strongly dominant
    debug: per-frame scoring detail, useful for tuning thresholds.
    """
    color: str
    confidence: float
    debug: dict = field(default_factory=dict)


class LightClassifier:
    """Classify a traffic_light crop as red / green / unknown via HSV."""

    # ── HSV thresholds ────────────────────────────────────────────────────────
    # OpenCV hue is in [0, 180]. Red wraps around both ends of the circle.
    HUE_RED_LOW  = (0,   35)   # pure red + amber + orange-red LEDs
    HUE_RED_HIGH = (165, 180)  # wrap-around from magenta-red
    HUE_GREEN    = (60,  95)   # TODO: measure on Jetson, may widen to (55, 100)
                               # if bloomed Saudi LED greens are missed

    # A pixel needs either decent saturation+value, OR very high brightness
    # with at least some saturation (to catch bloomed bulb centers near-white).
    SAT_MIN = 40
    VAL_MIN = 60
    BRIGHT_VAL_MIN = 200
    BRIGHT_SAT_MIN = 15

    # Final classification gates
    MIN_PIXEL_PCT = 0.015      # >=1.5% of crop must match the winning color
    MARGIN_PCT    = 0.2        # winner must beat runner-up by ≥20% relatively

    # Brightness gate — reject crops with no lit bulb at all
    MIN_BRIGHT_FRACTION = 0.015   # 1.5% of pixels must be V >= BRIGHT_GATE_VAL
    BRIGHT_GATE_VAL     = 180

    def classify(self, bgr_crop: Optional[np.ndarray]) -> LightClassification:
        """
        Args:
            bgr_crop: BGR uint8 array (H, W, 3) — typically a YOLO traffic_light
                      bbox crop. Any size, but must be >= 5x5 to bother.

        Returns:
            LightClassification with color, confidence, and debug info.
        """
        # ── Sanity guards ─────────────────────────────────────────────────────
        if bgr_crop is None or bgr_crop.size == 0:
            return self._unknown(0.0, reason="empty_crop")
        if bgr_crop.shape[0] < 5 or bgr_crop.shape[1] < 5:
            return self._unknown(0.0, reason="crop_too_small")

        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        total = bgr_crop.shape[0] * bgr_crop.shape[1]

        # ── Brightness gate ───────────────────────────────────────────────────
        # If nothing is bright enough to be a lit bulb, return unknown fast.
        # This catches: off lights, dark housings, night poles, junk crops.
        bright_fraction = float(np.sum(v >= self.BRIGHT_GATE_VAL)) / total
        if bright_fraction < self.MIN_BRIGHT_FRACTION:
            return self._unknown(
                0.0,
                reason="no_bright_region",
                bright_fraction=round(bright_fraction, 4),
            )

        # ── Color scoring ─────────────────────────────────────────────────────
        # Valid pixels: either decent saturation+value, or very bright with
        # a little saturation (catches bloomed bulb centers near-white).
        valid_standard = (s >= self.SAT_MIN) & (v >= self.VAL_MIN)
        valid_bright   = (v >= self.BRIGHT_VAL_MIN) & (s >= self.BRIGHT_SAT_MIN)
        valid = valid_standard | valid_bright

        red_mask = (
            ((h >= self.HUE_RED_LOW[0])  & (h <= self.HUE_RED_LOW[1])) |
            ((h >= self.HUE_RED_HIGH[0]) & (h <= self.HUE_RED_HIGH[1]))
        )
        green_mask = (h >= self.HUE_GREEN[0]) & (h <= self.HUE_GREEN[1])

        red_pct   = float(np.sum(valid & red_mask))   / total
        green_pct = float(np.sum(valid & green_mask)) / total

        scores        = {"red": red_pct, "green": green_pct}
        winner        = max(scores, key=scores.get)
        winner_pct    = scores[winner]
        runner_up_pct = scores["green" if winner == "red" else "red"]

        debug = {
            "red_pct":         round(red_pct,         4),
            "green_pct":       round(green_pct,       4),
            "bright_fraction": round(bright_fraction, 4),
        }

        # ── Gate 1: minimum pixel coverage ────────────────────────────────────
        if winner_pct < self.MIN_PIXEL_PCT:
            return LightClassification(
                color="unknown",
                confidence=0.0,
                debug={**debug, "reason": "below_min_pixel_pct"},
            )

        # ── Gate 2: margin between winner and runner-up ───────────────────────
        if runner_up_pct > 0 and \
           (winner_pct - runner_up_pct) / winner_pct < self.MARGIN_PCT:
            return LightClassification(
                color="unknown",
                confidence=0.0,
                debug={**debug, "reason": "ambiguous_margin"},
            )

        # ── Compute relative confidence ───────────────────────────────────────
        # Winner's share of total colored area. 1.0 = pure red/green, 0.5 = tied.
        total_colored = red_pct + green_pct
        real_conf = winner_pct / (total_colored + 1e-6) if total_colored > 0 else 0.0

        return LightClassification(
            color=winner,
            confidence=float(real_conf),
            debug=debug,
        )

    @staticmethod
    def _unknown(confidence: float = 0.0, **debug_kv) -> LightClassification:
        return LightClassification(
            color="unknown",
            confidence=float(confidence),
            debug=debug_kv,
        )


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    classifier = LightClassifier()

    test_cases = [
        ("Solid red",        np.full((40, 40, 3), (0,   0,   255), dtype=np.uint8)),
        ("Amber/orange-red", np.full((40, 40, 3), (0,   140, 255), dtype=np.uint8)),
        ("Solid green",      np.full((40, 40, 3), (0,   255, 0),   dtype=np.uint8)),
        ("Solid gray",       np.full((40, 40, 3), 128,             dtype=np.uint8)),
        ("Dark housing",     np.full((40, 40, 3), 30,              dtype=np.uint8)),
    ]

    for label, img in test_cases:
        result = classifier.classify(img)
        print(f"{label:20s} → {result.color:8s}  conf={result.confidence:.3f}  {result.debug}")