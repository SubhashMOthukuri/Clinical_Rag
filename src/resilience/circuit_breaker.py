import time

# CircuitBreaker usage & TODOs
# Current (Phase 1): manual API — callers must check `is_open()`, perform
# the protected operation, then call `record_success()` or `record_failure()`
# depending on outcome. This keeps the implementation simple but requires
# repetitive boilerplate at each call site (see fda_client.py / rxnorm_client.py).
#
# Phase 2 (TODO): add convenience interfaces to reduce boilerplate:
# - a context manager (``with CircuitBreaker(...):``) that automatically
#   checks `is_open()` and records success/failure based on exceptions
# - a decorator (`@circuit_breaker.wrap`) to wrap callables
# Implementing these will make call sites cleaner and safer.

""" Helper funtion for fda_client.py & rxnorm_client.py"""
class CircuitBreaker:
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