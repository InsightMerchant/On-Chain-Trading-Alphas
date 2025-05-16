from utils.interfaces import LogicInterface

class FastReverseShortLogic(LogicInterface):
    LOGIC_CONFIG = {
        "long_threshold": {
            "default": 0.0,
            "type": float,
            "description": "positive threshold"
        }
    }

    def apply(self, signal, long_threshold, short_threshold, position):
        # —— Flat: look for overbought entry ——
        if position == 0:
            return -1 if signal >= long_threshold else 0

        # —— Short: exit on mean‑reversion ——
        if position == -1:
            return 0 if signal <= long_threshold else -1

        # Safety default
        return 0
