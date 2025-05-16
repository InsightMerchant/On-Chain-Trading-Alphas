from utils.interfaces import LogicInterface

class FastLogic(LogicInterface):
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
        # ——— Flat: look for entry signals ———
        if position == 0:
            if signal >= long_threshold:
                return 1  # open long
            if signal <= short_threshold:
                return -1  # open short
            return 0  # stay flat

        # ——— Currently long: check for exit condition ———
        if position == 1:
            return 0 if signal <= long_threshold else 1

        # ——— Currently short: check for exit condition ———
        if position == -1:
            return 0 if signal >= short_threshold else -1

        # Safety default
        return 0
