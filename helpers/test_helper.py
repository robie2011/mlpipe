import time
from collections import namedtuple
from typing import Any


class TimerResult(namedtuple):
    timeMs: float
    result: Any


def measure_time(f):
    ts = time.time()
    return TimerResult(
        result=f(),
        timeMs=(time.time() - ts) * 1000
    )
