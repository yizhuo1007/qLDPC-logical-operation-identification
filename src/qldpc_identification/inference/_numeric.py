"""Directed floating-point arithmetic and GF(2) helpers for count-aware inference."""

from __future__ import annotations
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
import numpy as np

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    _get_current_process = ctypes.WinDLL("kernel32", use_last_error=True).GetCurrentProcess
    _get_current_process.restype = wintypes.HANDLE
    _get_process_memory_info = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    _get_process_memory_info.argtypes = (
        wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD
    )
    _get_process_memory_info.restype = wintypes.BOOL
else:
    import resource

TIME_LIMIT = 3600
MEMORY_LIMIT = 12 * 2**30
EPS = np.finfo(np.float64).eps
TINY = np.finfo(np.float64).tiny

class ResourceStop(RuntimeError):
    pass


class NumericalStop(RuntimeError):
    pass


class ModelStop(RuntimeError):
    pass


def peak_rss():
    if sys.platform == "win32":
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not _get_process_memory_info(_get_current_process(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


@dataclass
class Resources:
    started: float

    def check(self, expanded_states=0):
        if time.perf_counter()-self.started >= TIME_LIMIT:
            raise ResourceStop("scorer time limit reached")
        if peak_rss()+512*expanded_states+8*2**20 >= MEMORY_LIMIT:
            raise ResourceStop("scorer memory limit reached")


def insert(rows, value):
    while value:
        p = value.bit_length()-1
        if rows[p]:
            value ^= rows[p]
        else:
            rows[p] = value
            return


def remainder(rows, value):
    for p in range(40, -1, -1):
        if value >> p & 1 and rows[p]:
            value ^= rows[p]
    return value


def span(basis):
    values = [0]
    for v in basis:
        values += [u ^ v for u in values]
    return tuple(values)


def down(value):
    a = np.asarray(value, dtype=np.float64)
    result = np.zeros_like(a)
    positive = a > 0
    result[positive] = np.nextafter(a[positive], -np.inf)
    return result


def up(value):
    a = np.asarray(value, dtype=np.float64)
    result = np.zeros_like(a)
    positive = a > 0
    result[positive] = np.nextafter(a[positive], np.inf)
    return result


def checked_positive(values, label):
    a = np.asarray(values)
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise NumericalStop(label+": negative or nonfinite probability")
    if np.any((a > 0) & (a < TINY)):
        raise NumericalStop(label+": subnormal probability; stop before underflow")


def sum_upper(values):
    if not len(values):
        return 0.0
    # For n<=65536, 2*n*eps exceeds gamma_(n-1)/(1-gamma_(n-1)).
    # Positive summation, rounded multiplication, and outward nextafter give
    # an enclosure independent of NumPy's choice of reduction tree.
    return float(up(np.sum(values, dtype=np.float64)*(1+2*len(values)*EPS)))


def local_float(p):
    value = float(p)
    exact_float = Fraction.from_float(value)
    lo = float(down(value)) if exact_float > p else value
    hi = float(up(value)) if exact_float < p else value
    if p > 0 and (value <= 0 or lo <= 0):
        raise NumericalStop("Exact positive local factor cannot be represented safely")
    checked_positive([value, lo, hi], "local conversion")
    return value, lo, hi


def reachable(states, target, lookup):
    if lookup is None:
        return np.ones(len(states), dtype=np.bool_)
    value = states ^ np.uint64(target)
    residual = np.zeros(len(states), dtype=np.uint64)
    for chunk in range(6):
        residual ^= lookup[chunk, ((value >> np.uint64(8*chunk)) & np.uint64(255)).astype(np.intp)]
    return residual == 0
