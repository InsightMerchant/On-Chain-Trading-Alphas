from utils.interfaces import LogicInterface

class MrReverseLogic(LogicInterface):
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
        if signal >= long_threshold:
            return -1
        elif signal <= short_threshold:
            return 1
        elif signal <= 0 and position == -1:
            return 0
        elif signal >= 0 and position == 1:
            return 0
        else:
            return position
