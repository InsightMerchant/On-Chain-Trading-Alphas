from utils.interfaces import LogicInterface

class MrReverseShortLogic(LogicInterface):
    LOGIC_CONFIG = {
        "long_threshold": {
            "default": 0.0,
            "type": float,
            "description": "positive threshold"
        }
    }

    def apply(self, signal, long_threshold, short_threshold, position):
        # —— Entry rule (currently flat) ——
        if position == 0:
            return -1 if signal >= long_threshold else 0

        # —— Exit rule (currently short) ——
        if position == -1:
            return 0 if signal <= 0 else -1

        # Safety default for any unexpected position value
        return 0
