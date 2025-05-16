from utils.interfaces import LogicInterface
class FastLongLogic(LogicInterface):
    LOGIC_CONFIG = {
        "long_threshold": {
            "default": 0.0,
            "type": float,
            "description": "positive threshold"
        }
    }

    def apply(self, signal, long_threshold, short_threshold, position):
        """Return desired position (0 or 1)."""
        # —— Flat: look to open a long ——
        if position == 0:
            return 1 if signal >= long_threshold else 0

        # —— Long: look to exit ——
        if position == 1:
            return 0 if signal <= long_threshold else 1

