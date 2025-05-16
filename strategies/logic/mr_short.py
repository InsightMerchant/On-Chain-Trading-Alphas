from utils.interfaces import LogicInterface

class MrShortLogic(LogicInterface):
    LOGIC_CONFIG = {
        "short_threshold": {
            "default": 0.0,
            "type": float,
            "description": "negative threshold"
        }
    }

    def apply(self, signal, long_threshold, short_threshold, position):

        # If flat, open a short when the signal is weak enough.
        if position == 0:
            return -1 if signal <= short_threshold else 0

        # If already short, exit when the signal rises back to or above zero.
        if position == -1:
            return 0 if signal >= 0 else -1

        # Any other position code: stay flat for safety.
        return 0
