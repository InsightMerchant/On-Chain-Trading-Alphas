from utils.interfaces import LogicInterface

class FastReverseLogic(LogicInterface):
    LOGIC_CONFIG = {
        "long_threshold": {
            "default": 0.0,
            "type": float,
            "description": "positive threshold"
        },
        "short_threshold": {
            "default": 0.0,
            "type": float,
            "description": "negative threshold"
        }
    }

    def apply(self, signal, long_threshold, short_threshold, position):
        # ——— Flat: look for contrarian entry ———
        if position == 0:
            if signal >= long_threshold:
                return -1  # open short (overbought)
            if signal <= short_threshold:
                return 1   # open long (oversold)
            return 0       # stay flat

        # ——— Long: exit when mean‑reverted to ≥ 0 ———
        if position == 1:
            return 0 if signal >= short_threshold else 1

        # ——— Short: exit when mean‑reverted to ≤ 0 ———
        if position == -1:
            return 0 if signal <= long_threshold else -1

        # Safety default
        return 0
