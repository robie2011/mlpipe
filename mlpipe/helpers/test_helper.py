import time
from typing import Any, Optional
import numpy as np


class Timer:
    def __init__(self):
        self._start = time.time()
        self._last = time.time()

    def tock(self, action_name: Optional[str] = None):
        out = (time.time() - np.array([self._last, self._start])) / 1000
        self._last = time.time()

        if action_name:
            print(f"🕓 {out[0]:.5f} ms (Total time: {out[1]:.5f} ms) - Executed action [{action_name}]")
        return out
