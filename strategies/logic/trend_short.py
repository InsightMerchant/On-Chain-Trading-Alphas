from utils.interfaces import LogicInterface

class TrendShortLogic(LogicInterface):
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
        if signal <= short_threshold:
            return -1
        elif signal >= long_threshold:
            return 0
        else:
            return position
