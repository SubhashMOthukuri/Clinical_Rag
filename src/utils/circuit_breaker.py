import time

""" Helper funtion for fda_client.py & rxnorm_client.py"""
class _CircuitBreaker:
    def __init__(self, threshold, cooldown_s):
        self._threshold = threshold
        self._cooldown = cooldown_s
        self._failures = 0
        self._open_until = 0.0

    def is_open(self):
        return time.monotonic() < self._open_until

    def record_success(self):
        self._failures = 0

    def record_failure(self):
        self._failures += 1
        if self._failures >= self._threshold:
            self._open_until = time.monotonic() + self._cooldown