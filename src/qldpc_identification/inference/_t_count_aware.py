"""Dimension-adapted count-aware recurrence for one magic-state joint measurement."""
from __future__ import annotations
import math
import time
from collections import Counter
from fractions import Fraction
import numpy as np
from . import _numeric as num
from ._pm_count_aware import down, up, sum_lower, sum_upper, check_interval, add_interval, fwht_integer

def exact_group(key, multiplicity, cap):
    """Count DP in the actual <=4-dimensional local span, in exact integers.

    DP[d] is the d-fold XOR convolution of the nonzero local numerator.
    Choosing d identical factors supplies binomial(m,d)*p0**(m-d).
    No physical subsets are enumerated. All results have denominator D**m.
    Full count-marginal convolution uses the exact integer XOR transform.
    """
    local = {v: Fraction(n, d) for v, n, d in key}
    if any(p < 0 for p in local.values()) or sum(local.values()) != 1:
        raise num.ModelStop('Exact local group distribution fails normalization')
    rows = [ZERO]*DIM
    for v, p in local.items():
        if p:
            insert(rows, v)
    basis = tuple(v for v in rows if v != ZERO)
    support = span(basis)
    n = len(support)
    lookup = {v: i for i, v in enumerate(support)}
    denominator = math.lcm(*(p.denominator for p in local.values()))
    local_int = [0]*n
    for v, p in local.items():
        local_int[lookup[v]] = p.numerator*(denominator//p.denominator)
    total_den = denominator**multiplicity
    spectral = [x**multiplicity for x in fwht_integer(local_int)]
    inverse = fwht_integer(spectral)
    if any(x % n for x in inverse):
        raise num.ModelStop('Nonintegral exact local convolution')
    full = [x//n for x in inverse]
    zero = local_int[0]
    events = [(i, x) for i, x in enumerate(local_int) if i and x]
    dp = [1]+[0]*(n-1)
    counts = []
    for d in range(cap+1):
        if d <= multiplicity:
            coefficient = math.comb(multiplicity, d)*zero**(multiplicity-d)
            counts.append([coefficient*x for x in dp])
        else:
            counts.append([0]*n)
        if d < min(cap, multiplicity):
            nxt = [0]*n
            for i, x in enumerate(dp):
                if x:
                    for j, y in events:
                        nxt[i ^ j] += x*y
            dp = nxt
    overflow = [full[i]-sum(row[i] for row in counts) for i in range(n)]
    if (sum(full) != total_den or any(x < 0 for x in full+overflow)
            or sum(map(sum, counts))+sum(overflow) != total_den):
        raise num.ModelStop('Exact grouped count/full distributions fail normalization')
    # Conversion follows exact duplicate aggregation in the complete local span.
    def convert(row):
        return np.asarray([num.local_float(Fraction(x, total_den)) for x in row])
    full_float = convert(full)
    count_float = np.asarray([convert(row) for row in counts])
    # Conditional count overflow thresholds, including d>cap, are exact positive
    # local sums; subtraction takes place only on exact integer numerators.
    exceed = np.asarray([convert([full[i]-sum(counts[d][i] for d in range(cap-k+1))
                                  for i in range(n)]) for k in range(cap+1)])
    d, v = np.nonzero(count_float[:, :, 0])
    sig = np.asarray(support, dtype=np.uint64)
    check_words(sig)
    projected = np.stack([projection(sig, s) for s in SHIFTS])
    xor = projected[:, :, None] ^ np.arange(256, dtype=np.intp)
    return {'basis': basis, 'support': sig, 'multiplicity': multiplicity,
            'full': full_float, 'counts': count_float, 'exceed': exceed,
            'xor': xor, 'd': d.astype(np.uint8), 'v': sig[v],
            'weights': count_float[d, v]}


def conv_interval(lo, hi, weights, xor):
    """Positive XOR convolution, batched over optional count and 21 projections."""
    active = np.flatnonzero(weights[:, 2])
    if not len(active):
        return np.zeros_like(lo), np.zeros_like(hi)
    w = weights[active]
    idx = xor[:, active]
    # Last dimensions are (projection, signature); expanded local support is
    # inserted immediately before the final byte coordinate.
    js = np.arange(NPROJ)[:, None, None]
    products_lo = down(lo[..., js, idx]*w[:, 1, None])
    products_hi = up(hi[..., js, idx]*w[:, 2, None])
    result_lo = sum_lower(products_lo, axis=-2)
    result_hi = sum_upper(products_hi, axis=-2)
    check_interval(result_lo, result_hi, 'projected convolution')
    return result_lo, result_hi


def build_candidate(ci, data, exact, terms, resources):
    started = time.perf_counter()
    groups = []
    for stage in range(10):
        a, b = map(int, data['stage_location_offsets'][ci, stage:stage+2])
        # Group locations by their full rational fault distributions in input order.
        counts = Counter(exact[int(fid)] for fid in data['location_factor_id'][a:b])
        for position, (key, multiplicity) in enumerate(counts.items()):
            resources.check()
            group = exact_group(key, multiplicity, CPLUS[stage])
            group.update(stage=stage, stage_end=position == len(counts)-1)
            groups.append(group)
    suffix_rows = [None]*(len(groups)+1)
    rows = [ZERO]*DIM
    suffix_rows[-1] = tuple(rows)
    for t in range(len(groups)-1, -1, -1):
        for v in groups[t]['basis']:
            insert(rows, v)
        suffix_rows[t] = tuple(rows)
    cache, suffix = {}, []
    for rows in suffix_rows:
        if rows not in cache:
            if sum(v != ZERO for v in rows) == DIM:
                cache[rows] = None
            else:
                lut = np.zeros((21, 256, 3), dtype=np.uint64)
                for chunk in range(21):
                    units = [remainder(rows, unit_word(8*chunk+j))
                             if 8*chunk+j < DIM else ZERO for j in range(8)]
                    lut[chunk] = span(units)
                cache[rows] = lut
        suffix.append(cache[rows])
    shape = (len(groups)+1, NPROJ, 256)
    resources.check(reserved_bytes=2*math.prod(shape)*8)
    pl, pu = np.zeros(shape), np.zeros(shape)
    pl[-1, :, 0] = pu[-1, :, 0] = 1.0
    if (np.any(pl[-1, :, 1:]) or not np.all(pl[-1, :, 0] == 1)
            or not np.array_equal(pl[-1], pu[-1])):
        raise num.ModelStop('Projected terminal suffix is not delta_0')
    for t in range(len(groups)-1, -1, -1):
        resources.check()
        g = groups[t]
        pl[t], pu[t] = conv_interval(pl[t+1], pu[t+1], g['full'], g['xor'])
        pu[t] = np.minimum(1.0, pu[t])
    # Small projected block DP. high is an absorbing exceeded-any-stage flag;
    # its count is irrelevant. good retains every current count 0..CPLUS.
    block_lo, block_hi = np.zeros((NPROJ, 256)), np.zeros((NPROJ, 256))
    block_lo[:, 0] = block_hi[:, 0] = 1.0
    high_lo, high_hi = np.zeros((NPROJ, 256)), np.zeros((NPROJ, 256))
    previous_stage = -1
    for g in groups:
        resources.check()
        stage, cap = g['stage'], CPLUS[g['stage']]
        if stage != previous_stage:
            good_lo, good_hi = np.zeros((cap+1, NPROJ, 256)), np.zeros((cap+1, NPROJ, 256))
            good_lo[0], good_hi[0] = block_lo, block_hi
            previous_stage = stage
        next_lo, next_hi = np.zeros_like(good_lo), np.zeros_like(good_hi)
        next_high_lo, next_high_hi = conv_interval(high_lo, high_hi, g['full'], g['xor'])
        for d in range(cap+1):
            lo, hi = conv_interval(good_lo[:cap+1-d], good_hi[:cap+1-d],
                                   g['counts'][d], g['xor'])
            next_lo[d:], next_hi[d:] = add_interval(next_lo[d:], next_hi[d:], lo, hi)
        for k in range(cap+1):
            lo, hi = conv_interval(good_lo[k], good_hi[k], g['exceed'][k], g['xor'])
            next_high_lo, next_high_hi = add_interval(next_high_lo, next_high_hi, lo, hi)
        good_lo, good_hi = next_lo, next_hi
        high_lo, high_hi = next_high_lo, next_high_hi
        check_interval(high_lo, high_hi, 'projected high-order tail')
        if g['stage_end']:
            block_lo, block_hi = sum_lower(good_lo, 0), sum_upper(good_hi, 0)
    # The stage-count audit uses 110 decimal digits. Enclose its numerical
    # saved omitted tail with 1e-90 absolute slack before outward float conversion.
    tail = Fraction(str(data['low_tail_CPLUS'][ci]))
    tau_hi = num.local_float(tail+Fraction(1, 10**90))[2]
    return {'candidate': CANDIDATES[ci], 'groups': groups, 'suffix': suffix,
            'projected_upper': pu, 'tail_lower': high_lo, 'tail_upper': high_hi,
            'tau_upper': tau_hi, 'term_indices': terms,
            'precompute_seconds': time.perf_counter()-started,
            'exact_group_count': len(groups)}
