from utils.interfaces import LogicInterface

class FastReverseLongLogic(LogicInterface):
    LOGIC_CONFIG = {
        "short_threshold": {
            "default": 0.0,
            "type": float,
            "description": "negative threshold"
        }
    }

    def apply(self, signal, long_threshold, short_threshold, position):
        # —— Flat: look for oversold entry ——
        if position == 0:
            return 1 if signal <= short_threshold else 0

        # —— Long: exit on mean‑reversion ——
        if position == 1:
            return 0 if signal >= short_threshold else 1

        # Safety default
        return 0
