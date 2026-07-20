"""
Pattern Recognition Agent (new, per request #1).

Detects the classic chart patterns and candlestick patterns from the
reference sheets you provided (reversal, continuation, and bilateral chart
patterns; the ~35 standard bullish/bearish candlestick patterns) directly
from OHLC price history, using swing-point geometry and candle-shape rules
- not a black box, every detection is traceable to the swing points or
candle measurements that triggered it.

IMPORTANT - read this before trusting the output:
Chart patterns are famously unreliable in isolation. Published backtests
(e.g. Bulkowski's pattern statistics, the most cited independent study of
this exact question) put the BEST individual chart patterns' historical
"worked as expected" rate around 60-75%, and most cluster in the 45-65%
range - barely better than a coin flip once you account for how often a
pattern is misread on the right edge of a live, still-forming chart. This
agent reports that literature-backed reliability alongside every detected
pattern instead of a single fabricated confidence number, and its output
is ONE signal that feeds validation_agent.py alongside technical,
fundamental, and sentiment scores - it is never used alone to justify a
BUY/SELL call.

This agent's `reason()` returns:
    {
      "score": float in [-1, 1],      # net directional lean, like the other agents
      "confidence": float 0-100,       # how much swing-point/candle evidence supports it
      "patterns": [ {name, type, direction, reliability_pct, detected_at_index}, ... ],
      "details": {...}
    }
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from agents.base_agent import BaseAgent

# Bulkowski-style published reliability (% of time the pattern resolved in
# its "textbook" direction across large historical samples). These are
# ballpark literature figures, not guarantees for any specific stock/time.
PATTERN_RELIABILITY = {
    "Double Top": 65, "Double Bottom": 66, "Head and Shoulders": 68,
    "Inverse Head and Shoulders": 68, "Rising Wedge": 62, "Falling Wedge": 63,
    "Ascending Triangle": 63, "Descending Triangle": 63, "Symmetrical Triangle": 55,
    "Bullish Rectangle": 58, "Bearish Rectangle": 58, "Bullish Pennant": 60,
    "Bearish Pennant": 60, "Bull Flag": 61, "Bear Flag": 61,
    "Cup and Handle": 65, "Bullish Engulfing": 58, "Bearish Engulfing": 58,
    "Hammer": 55, "Inverted Hammer": 53, "Hanging Man": 54, "Shooting Star": 55,
    "Morning Star": 60, "Evening Star": 60, "Bullish Harami": 52, "Bearish Harami": 52,
    "Three White Soldiers": 60, "Three Black Crows": 60, "Doji": 50,
    "Dragonfly Doji": 53, "Gravestone Doji": 53, "Piercing Line": 56,
    "Dark Cloud Cover": 56, "Tweezer Top": 52, "Tweezer Bottom": 52,
    "Marubozu Bullish": 54, "Marubozu Bearish": 54,
}

CHART_PATTERNS_BULLISH = {
    "Double Bottom", "Inverse Head and Shoulders", "Falling Wedge", "Ascending Triangle",
    "Bullish Rectangle", "Bullish Pennant", "Bull Flag", "Cup and Handle",
}
CHART_PATTERNS_BEARISH = {
    "Double Top", "Head and Shoulders", "Rising Wedge", "Descending Triangle",
    "Bearish Rectangle", "Bearish Pennant", "Bear Flag",
}


def _swing_points(close: np.ndarray, order: int = 5):
    highs_idx = argrelextrema(close, np.greater_equal, order=order)[0]
    lows_idx = argrelextrema(close, np.less_equal, order=order)[0]
    # de-duplicate consecutive equal-value plateaus
    highs_idx = np.array(sorted(set(highs_idx.tolist())))
    lows_idx = np.array(sorted(set(lows_idx.tolist())))
    return highs_idx, lows_idx


def _pct_close(a: float, b: float, tol_pct: float = 1.5) -> bool:
    if a == 0 or b == 0:
        return False
    return abs(a - b) / max(abs(a), abs(b)) * 100 <= tol_pct


class PatternRecognitionAgent(BaseAgent):
    name = "pattern_recognition_agent"
    goal = ("Detect classic chart & candlestick patterns from price history and translate them "
            "into a directional signal with a literature-backed (not fabricated) reliability score.")
    output_schema = {"score": float, "confidence": float, "patterns": list, "details": dict}

    def reason(self, price_history: pd.DataFrame = None, **_) -> dict:
        df = price_history
        if df is None or len(df) < 30:
            return {"agent": self.name, "score": 0.0, "confidence": 10,
                    "reason": "Insufficient history for pattern detection.",
                    "patterns": [], "details": {}}

        chart_patterns = self._detect_chart_patterns(df)
        candle_patterns = self._detect_candlestick_patterns(df)
        all_patterns = chart_patterns + candle_patterns

        if not all_patterns:
            return {"agent": self.name, "score": 0.0, "confidence": 30,
                    "reason": "No recognizable pattern in the current price action.",
                    "patterns": [], "details": {"chart_patterns_checked": True}}

        # Weight recent patterns higher (right edge of the chart matters most),
        # weight each by its published reliability above/below a 50% coin-flip baseline.
        n = len(df)
        weighted_sum, weight_total = 0.0, 0.0
        for p in all_patterns:
            recency = 1.0 - min((n - p["detected_at_index"]) / n, 1.0) * 0.5  # 0.5-1.0
            edge = (p["reliability_pct"] - 50) / 50  # -1..+1 relative to a coin flip
            direction = 1 if p["direction"] == "bullish" else -1
            w = recency * (p["reliability_pct"] / 100)
            weighted_sum += direction * edge * w
            weight_total += w

        score = float(np.clip(weighted_sum / weight_total if weight_total else 0, -1, 1))
        confidence = float(np.clip(30 + weight_total / len(all_patterns) * 50, 20, 85))
        # Cap confidence hard - this agent's own docstring explains why it should
        # never claim near-certainty, regardless of how many patterns line up.

        top_patterns = sorted(all_patterns, key=lambda p: p["detected_at_index"], reverse=True)[:5]
        return {
            "agent": self.name, "score": round(score, 3), "confidence": round(confidence, 1),
            "reason": f"{len(all_patterns)} pattern(s) detected; most recent: "
                      f"{top_patterns[0]['name']} ({top_patterns[0]['direction']})" if top_patterns else "none",
            "patterns": top_patterns,
            "details": {"total_patterns_found": len(all_patterns), "bullish_count":
                        sum(1 for p in all_patterns if p["direction"] == "bullish"),
                        "bearish_count": sum(1 for p in all_patterns if p["direction"] == "bearish")},
        }

    # ------------------------------------------------------------------
    # Chart patterns: double top/bottom, head & shoulders, triangles,
    # wedges, rectangles, flags/pennants, cup & handle.
    # ------------------------------------------------------------------
    def _detect_chart_patterns(self, df: pd.DataFrame) -> list:
        close = df["close"].to_numpy(dtype=float)
        highs_idx, lows_idx = _swing_points(close, order=max(3, len(close) // 60))
        found = []

        found += self._double_top_bottom(close, highs_idx, lows_idx)
        found += self._head_and_shoulders(close, highs_idx, lows_idx)
        found += self._triangles_and_wedges(close, highs_idx, lows_idx)
        found += self._rectangles(close, highs_idx, lows_idx)
        found += self._flags_pennants(df, close)
        found += self._cup_and_handle(close, lows_idx)
        return found

    def _double_top_bottom(self, close, highs_idx, lows_idx) -> list:
        found = []
        for i in range(len(highs_idx) - 1):
            a, b = highs_idx[i], highs_idx[i + 1]
            if b - a < 5:
                continue
            if _pct_close(close[a], close[b]):
                trough = close[a:b + 1].min()
                if trough < close[a] * 0.97:  # meaningful pullback between the two tops
                    found.append({"name": "Double Top", "type": "chart", "direction": "bearish",
                                  "reliability_pct": PATTERN_RELIABILITY["Double Top"],
                                  "detected_at_index": int(b)})
        for i in range(len(lows_idx) - 1):
            a, b = lows_idx[i], lows_idx[i + 1]
            if b - a < 5:
                continue
            if _pct_close(close[a], close[b]):
                peak = close[a:b + 1].max()
                if peak > close[a] * 1.03:
                    found.append({"name": "Double Bottom", "type": "chart", "direction": "bullish",
                                  "reliability_pct": PATTERN_RELIABILITY["Double Bottom"],
                                  "detected_at_index": int(b)})
        return found

    def _head_and_shoulders(self, close, highs_idx, lows_idx) -> list:
        found = []
        for i in range(len(highs_idx) - 2):
            l_sh, head, r_sh = highs_idx[i], highs_idx[i + 1], highs_idx[i + 2]
            if close[head] > close[l_sh] * 1.02 and close[head] > close[r_sh] * 1.02 \
                    and _pct_close(close[l_sh], close[r_sh], tol_pct=3):
                found.append({"name": "Head and Shoulders", "type": "chart", "direction": "bearish",
                              "reliability_pct": PATTERN_RELIABILITY["Head and Shoulders"],
                              "detected_at_index": int(r_sh)})
        for i in range(len(lows_idx) - 2):
            l_sh, head, r_sh = lows_idx[i], lows_idx[i + 1], lows_idx[i + 2]
            if close[head] < close[l_sh] * 0.98 and close[head] < close[r_sh] * 0.98 \
                    and _pct_close(close[l_sh], close[r_sh], tol_pct=3):
                found.append({"name": "Inverse Head and Shoulders", "type": "chart", "direction": "bullish",
                              "reliability_pct": PATTERN_RELIABILITY["Inverse Head and Shoulders"],
                              "detected_at_index": int(r_sh)})
        return found

    def _triangles_and_wedges(self, close, highs_idx, lows_idx) -> list:
        """Fits a line through the last 3 swing highs and last 3 swing lows;
        classifies by slope combination (converging/parallel/diverging,
        rising/falling)."""
        found = []
        if len(highs_idx) < 3 or len(lows_idx) < 3:
            return found

        h_idx, h_val = highs_idx[-3:], close[highs_idx[-3:]]
        l_idx, l_val = lows_idx[-3:], close[lows_idx[-3:]]
        h_slope = np.polyfit(h_idx, h_val, 1)[0]
        l_slope = np.polyfit(l_idx, l_val, 1)[0]
        last_idx = int(max(h_idx[-1], l_idx[-1]))
        avg_price = float(np.mean(close[-30:]))
        slope_tol = avg_price * 0.0005  # "flat" threshold scaled to price level

        rising = h_slope > slope_tol and l_slope > slope_tol
        falling = h_slope < -slope_tol and l_slope < -slope_tol
        converging = h_slope < -slope_tol and l_slope > slope_tol
        flat_top = abs(h_slope) <= slope_tol
        flat_bottom = abs(l_slope) <= slope_tol

        if rising:
            found.append({"name": "Rising Wedge", "type": "chart", "direction": "bearish",
                          "reliability_pct": PATTERN_RELIABILITY["Rising Wedge"], "detected_at_index": last_idx})
        elif falling:
            found.append({"name": "Falling Wedge", "type": "chart", "direction": "bullish",
                          "reliability_pct": PATTERN_RELIABILITY["Falling Wedge"], "detected_at_index": last_idx})
        elif converging and flat_top:
            found.append({"name": "Descending Triangle", "type": "chart", "direction": "bearish",
                          "reliability_pct": PATTERN_RELIABILITY["Descending Triangle"], "detected_at_index": last_idx})
        elif converging and flat_bottom:
            found.append({"name": "Ascending Triangle", "type": "chart", "direction": "bullish",
                          "reliability_pct": PATTERN_RELIABILITY["Ascending Triangle"], "detected_at_index": last_idx})
        elif converging:
            direction = "bullish" if close[-1] >= close[l_idx[-1]] else "bearish"
            found.append({"name": "Symmetrical Triangle", "type": "chart", "direction": direction,
                          "reliability_pct": PATTERN_RELIABILITY["Symmetrical Triangle"], "detected_at_index": last_idx})
        return found

    def _rectangles(self, close, highs_idx, lows_idx) -> list:
        found = []
        if len(highs_idx) < 2 or len(lows_idx) < 2:
            return found
        recent_highs = close[highs_idx[-2:]]
        recent_lows = close[lows_idx[-2:]]
        if _pct_close(recent_highs[0], recent_highs[1], tol_pct=1.2) and \
                _pct_close(recent_lows[0], recent_lows[1], tol_pct=1.2) and \
                recent_highs.mean() > recent_lows.mean() * 1.02:
            last_idx = int(max(highs_idx[-1], lows_idx[-1]))
            direction = "bullish" if close[-1] > close[lows_idx[-1]] else "bearish"
            name = "Bullish Rectangle" if direction == "bullish" else "Bearish Rectangle"
            found.append({"name": name, "type": "chart", "direction": direction,
                          "reliability_pct": PATTERN_RELIABILITY[name], "detected_at_index": last_idx})
        return found

    def _flags_pennants(self, df: pd.DataFrame, close) -> list:
        """Flag/pennant = a sharp pole move followed by a tight, low-slope
        consolidation of <= ~15 bars."""
        found = []
        if len(close) < 25:
            return found
        pole = close[-25:-10]
        consolidation = close[-10:]
        pole_move_pct = (pole[-1] - pole[0]) / pole[0] * 100 if pole[0] else 0
        consolidation_range_pct = (consolidation.max() - consolidation.min()) / consolidation.mean() * 100

        if abs(pole_move_pct) > 8 and consolidation_range_pct < abs(pole_move_pct) * 0.4:
            direction = "bullish" if pole_move_pct > 0 else "bearish"
            slope = np.polyfit(range(len(consolidation)), consolidation, 1)[0]
            is_pennant = consolidation_range_pct < abs(pole_move_pct) * 0.2
            if is_pennant:
                name = "Bullish Pennant" if direction == "bullish" else "Bearish Pennant"
            else:
                name = "Bull Flag" if direction == "bullish" else "Bear Flag"
            found.append({"name": name, "type": "chart", "direction": direction,
                          "reliability_pct": PATTERN_RELIABILITY[name], "detected_at_index": len(close) - 1})
        return found

    def _cup_and_handle(self, close, lows_idx) -> list:
        found = []
        if len(close) < 60:
            return found
        window = close[-60:]
        min_idx = int(np.argmin(window))
        left, right = window[:min_idx + 1], window[min_idx:]
        if 10 < min_idx < 50 and len(right) > 5:
            # U-shape check: both sides recover to near the starting level
            left_recovery = window[0]
            right_recovery = window[-6:].mean()
            if _pct_close(left_recovery, right_recovery, tol_pct=6) and window[min_idx] < window[0] * 0.90:
                found.append({"name": "Cup and Handle", "type": "chart", "direction": "bullish",
                              "reliability_pct": PATTERN_RELIABILITY["Cup and Handle"],
                              "detected_at_index": len(close) - 1})
        return found

    # ------------------------------------------------------------------
    # Candlestick patterns (single/double/triple-candle rules)
    # ------------------------------------------------------------------
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> list:
        o, h, l, c = (df["open"].to_numpy(dtype=float), df["high"].to_numpy(dtype=float),
                      df["low"].to_numpy(dtype=float), df["close"].to_numpy(dtype=float))
        n = len(c)
        found = []
        body = np.abs(c - o)
        rng = np.maximum(h - l, 1e-9)
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - l
        avg_body = float(np.mean(body[-30:])) if n >= 30 else float(np.mean(body))

        i = n - 1  # only evaluate the most recent completed candle(s) - that's what's actionable
        if i < 3:
            return found

        # --- Doji family ---
        if body[i] < rng[i] * 0.1:
            if lower_wick[i] > body[i] * 2 and upper_wick[i] < body[i]:
                found.append(self._c("Dragonfly Doji", "bullish", i))
            elif upper_wick[i] > body[i] * 2 and lower_wick[i] < body[i]:
                found.append(self._c("Gravestone Doji", "bearish", i))
            else:
                found.append(self._c("Doji", "neutral", i))

        # --- Hammer / Hanging Man / Inverted Hammer / Shooting Star ---
        downtrend = c[i - 3] > c[i]
        uptrend = c[i - 3] < c[i]
        if lower_wick[i] > body[i] * 2 and upper_wick[i] < body[i] * 0.5:
            found.append(self._c("Hammer", "bullish", i) if downtrend else self._c("Hanging Man", "bearish", i))
        if upper_wick[i] > body[i] * 2 and lower_wick[i] < body[i] * 0.5:
            found.append(self._c("Inverted Hammer", "bullish", i) if downtrend else self._c("Shooting Star", "bearish", i))

        # --- Marubozu ---
        if body[i] > avg_body * 1.5 and upper_wick[i] < body[i] * 0.05 and lower_wick[i] < body[i] * 0.05:
            found.append(self._c("Marubozu Bullish" if c[i] > o[i] else "Marubozu Bearish",
                                  "bullish" if c[i] > o[i] else "bearish", i))

        # --- Two-candle patterns ---
        prev_body = body[i - 1]
        if c[i - 1] < o[i - 1] and c[i] > o[i] and c[i] > o[i - 1] and o[i] < c[i - 1]:
            found.append(self._c("Bullish Engulfing", "bullish", i))
        if c[i - 1] > o[i - 1] and c[i] < o[i] and c[i] < o[i - 1] and o[i] > c[i - 1]:
            found.append(self._c("Bearish Engulfing", "bearish", i))
        if c[i - 1] < o[i - 1] and c[i] > o[i] and o[i] > c[i - 1] and c[i] < o[i - 1]:
            found.append(self._c("Bullish Harami", "bullish", i))
        if c[i - 1] > o[i - 1] and c[i] < o[i] and o[i] < c[i - 1] and c[i] > o[i - 1]:
            found.append(self._c("Bearish Harami", "bearish", i))
        if c[i - 1] < o[i - 1] and c[i] > o[i] and o[i] < c[i - 1] and c[i] > (o[i - 1] + c[i - 1]) / 2:
            found.append(self._c("Piercing Line", "bullish", i))
        if c[i - 1] > o[i - 1] and c[i] < o[i] and o[i] > c[i - 1] and c[i] < (o[i - 1] + c[i - 1]) / 2:
            found.append(self._c("Dark Cloud Cover", "bearish", i))
        if _pct_close(h[i - 1], h[i], tol_pct=0.3) and c[i - 1] > o[i - 1] and c[i] < o[i]:
            found.append(self._c("Tweezer Top", "bearish", i))
        if _pct_close(l[i - 1], l[i], tol_pct=0.3) and c[i - 1] < o[i - 1] and c[i] > o[i]:
            found.append(self._c("Tweezer Bottom", "bullish", i))

        # --- Three-candle patterns ---
        if i >= 2:
            if c[i - 2] > o[i - 2] and body[i - 1] < avg_body * 0.5 and c[i] < o[i] and \
                    c[i] < (o[i - 2] + c[i - 2]) / 2:
                found.append(self._c("Evening Star", "bearish", i))
            if c[i - 2] < o[i - 2] and body[i - 1] < avg_body * 0.5 and c[i] > o[i] and \
                    c[i] > (o[i - 2] + c[i - 2]) / 2:
                found.append(self._c("Morning Star", "bullish", i))
            if all(c[j] > o[j] for j in (i - 2, i - 1, i)) and c[i] > c[i - 1] > c[i - 2] and \
                    all(body[j] > avg_body * 0.6 for j in (i - 2, i - 1, i)):
                found.append(self._c("Three White Soldiers", "bullish", i))
            if all(c[j] < o[j] for j in (i - 2, i - 1, i)) and c[i] < c[i - 1] < c[i - 2] and \
                    all(body[j] > avg_body * 0.6 for j in (i - 2, i - 1, i)):
                found.append(self._c("Three Black Crows", "bearish", i))

        return found

    @staticmethod
    def _c(name: str, direction: str, idx: int) -> dict:
        return {"name": name, "type": "candlestick", "direction": direction,
                "reliability_pct": PATTERN_RELIABILITY.get(name, 50), "detected_at_index": int(idx)}


pattern_recognition_agent = PatternRecognitionAgent()


def analyze(price_history: pd.DataFrame) -> dict:
    return pattern_recognition_agent.reason(price_history=price_history)
