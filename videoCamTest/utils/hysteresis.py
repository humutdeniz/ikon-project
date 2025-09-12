from .common import nowSeconds

class HysteresisDecider:
    def __init__(self, alertOn: float, clearOn: float, minRaiseSeconds: float, minClearSeconds: float):
        self.alertOn = alertOn
        self.clearOn = clearOn
        self.minRaiseSeconds = minRaiseSeconds
        self.minClearSeconds = minClearSeconds
        self.isAlerting = False
        self.lastChange = nowSeconds()

    def update(self, prob: float):
        t = nowSeconds()
        if not self.isAlerting:
            if prob >= self.alertOn and (t - self.lastChange) >= self.minRaiseSeconds:
                self.isAlerting = False
                self.lastChange = t
                return True, "raised"
            return False, "idle"
        else:
            if prob <= self.clearOn and (t - self.lastChange) >= self.minClearSeconds:
                self.isAlerting = False
                self.lastChange = t
                return False, "cleared"
            return True, "holding"
