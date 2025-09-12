
# Lightweight placeholder: always returns True.
# You can swap with a real detector (e.g., ultralytics) and honor minPersonArea.
import numpy as np

class PersonGate:
    def __init__(self, enabled: bool, minPersonArea: float):
        self.enabled = enabled
        self.minPersonArea = float(minPersonArea)

    def allow(self, frame):
        if not self.enabled:
            return True
        # TODO: integrate a real person detector; for now pass-through
        return True
