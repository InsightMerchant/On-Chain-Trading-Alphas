from utils.interfaces import LogicInterface

class MrReverseLongLogic(LogicInterface):
    LOGIC_CONFIG = {
        "short_threshold": {
            "default": 0.0,
            "type": float,
            "description": "negative threshold"
        }
    }

    def apply(self, signal, long_threshold, short_threshold, position):
        if position == 0:
            return 1 if signal <= short_threshold else 0

        # —— Exit rule (currently long) ——
        if position == 1:
            return 0 if signal >= 0 else 1

        # Safety default for any unexpected position value
        return 0
