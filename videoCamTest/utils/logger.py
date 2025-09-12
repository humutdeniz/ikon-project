import os, json
from .common import ensureDir

class Logger:
    def __init__(self, baseDir: str, level: str = "info"):
        self.baseDir = baseDir
        self.level = level.lower()
        ensureDir(self.baseDir)

    def _log(self, tag: str, msg: str, data=None):
        line = f"[{tag.upper()}] {msg}"
        print(line)
        if data is not None:
            with open(os.path.join(self.baseDir, f"{tag}.jsonl"), "a") as f:
                f.write(json.dumps(data) + "\n")

    def info(self, msg: str, data=None):
        if self.level in ("debug","info"):
            self._log("info", msg, data)

    def debug(self, msg: str, data=None):
        if self.level == "debug":
            self._log("debug", msg, data)

    def warn(self, msg: str, data=None):
        if self.level in ("debug","info","warning","warn"):
            self._log("warn", msg, data)

    def error(self, msg: str, data=None):
        self._log("error", msg, data)
