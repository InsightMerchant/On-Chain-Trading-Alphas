from utils.interfaces import LogicInterface

class MrLongLogic(LogicInterface):
    LOGIC_CONFIG = {
        "long_threshold": {
            "default": 0.0,
            "type": float,
            "description": "positive threshold"
        }
    }

    def apply(self, signal, long_threshold, short_threshold, position):
        # If flat, open a long when the signal is strong enough.
        if position == 0:
            return 1 if signal >= long_threshold else 0

        # If already long, exit when the signal drops to or below zero.
        if position == 1:
            return 0 if signal <= 0 else 1

        # Any other position code: stay flat for safety.
        return 0
