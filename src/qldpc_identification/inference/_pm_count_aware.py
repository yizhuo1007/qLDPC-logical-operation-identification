"""Count-aware bounded likelihood for a [[18,4,2]] Pauli measurement."""

from __future__ import annotations
import math
import time
from collections import Counter
from fractions import Fraction
import numpy as np
from . import _numeric as num

CANDIDATES = tuple("XI YI ZI IX IY IZ XX XY XZ YX YY YZ ZX ZY ZZ".split())
C = (1,) * 10
CPLUS = (2,) * 10
SHIFTS = (0, 8, 16, 24, 32, 33)
B = 4096
LIKELIHOOD_FACTOR = 2.0**-7
MASK = np.uint64((1 << 41)-1)
E_BIT = np.uint64(1 << 44)

def down(value):
    num.checked_positive(value, 'lower arithmetic input')
    result = num.down(value)
    num.checked_positive(result, 'lower arithmetic result')
    return result


def up(value):
    num.checked_positive(value, 'upper arithmetic input')
    result = num.up(value)
    num.checked_positive(result, 'upper arithmetic result')
    return result


def check_interval(lo, hi, label):
    num.checked_positive(lo, label+' lower')
    num.checked_positive(hi, label+' upper')
    if np.any(np.asarray(lo) > np.asarray(hi)):
        raise num.NumericalStop(label+': upper below lower')


def sum_lower(a, axis=None):
    n = a.size if axis is None else a.shape[axis]
    return down(np.sum(a, axis=axis)*(1-2*n*num.EPS))


def sum_upper(a, axis=None):
    n = a.size if axis is None else a.shape[axis]
    return up(np.sum(a, axis=axis)*(1+2*n*num.EPS))


class Resources(num.Resources):
    def check(self, expanded_states=0):
        super().check(expanded_states)


def fwht_integer(a):
    a = list(a)
    h = 1
    while h < len(a):
        for i in range(0, len(a), 2*h):
            for j in range(i, i+h):
                x, y = a[j], a[j+h]
                a[j], a[j+h] = x+y, x-y
        h *= 2
    return a


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
    rows = [0]*41
    for v, p in local.items():
        if p:
            num.insert(rows, v)
    basis = tuple(v for v in rows if v)
    support = num.span(basis)
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
    projected = np.stack([((sig >> np.uint64(s)) & np.uint64(255)).astype(np.intp)
                          for s in SHIFTS])
    xor = projected[:, :, None] ^ np.arange(256, dtype=np.intp)
    return {'basis': basis, 'support': sig, 'multiplicity': multiplicity,
            'full': full_float, 'counts': count_float, 'exceed': exceed,
            'xor': xor, 'd': d.astype(np.uint64), 'v': sig[v],
            'weights': count_float[d, v]}


def conv_interval(lo, hi, weights, xor):
    """Positive XOR convolution, batched over optional count and six projections."""
    active = np.flatnonzero(weights[:, 2])
    if not len(active):
        return np.zeros_like(lo), np.zeros_like(hi)
    w = weights[active]
    idx = xor[:, active]
    # Last dimensions are (projection, signature); expanded local support is
    # inserted immediately before the final byte coordinate.
    js = np.arange(6)[:, None, None]
    products_lo = down(lo[..., js, idx]*w[:, 1, None])
    products_hi = up(hi[..., js, idx]*w[:, 2, None])
    result_lo = sum_lower(products_lo, axis=-2)
    result_hi = sum_upper(products_hi, axis=-2)
    check_interval(result_lo, result_hi, 'projected convolution')
    return result_lo, result_hi


def add_interval(lo, hi, addlo, addhi):
    return down(lo+addlo), up(hi+addhi)


def build_candidate(ci, data, exact, terms, resources):
    started = time.perf_counter()
    groups = []
    for stage in range(len(C)):
        a, b = map(int, data['stage_location_offsets'][ci, stage:stage+2])
        # Group locations by their full rational fault distributions in input order.
        counts = Counter(exact[int(fid)] for fid in data['location_factor_id'][a:b])
        for position, (key, multiplicity) in enumerate(counts.items()):
            resources.check()
            group = exact_group(key, multiplicity, CPLUS[stage])
            group.update(stage=stage, stage_end=position == len(counts)-1)
            groups.append(group)
    suffix_rows = [None]*(len(groups)+1)
    rows = [0]*41
    suffix_rows[-1] = tuple(rows)
    for t in range(len(groups)-1, -1, -1):
        for v in groups[t]['basis']:
            num.insert(rows, v)
        suffix_rows[t] = tuple(rows)
    cache, suffix = {}, []
    for rows in suffix_rows:
        if rows not in cache:
            if sum(bool(v) for v in rows) == 41:
                cache[rows] = None
            else:
                lut = np.zeros((6, 256), dtype=np.uint64)
                for chunk in range(6):
                    units = [num.remainder(rows, 1 << (8*chunk+j))
                             if 8*chunk+j < 41 else 0 for j in range(8)]
                    lut[chunk] = num.span(units)
                cache[rows] = lut
        suffix.append(cache[rows])
    shape = (len(groups)+1, 6, 256)
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
    block_lo, block_hi = np.zeros((6, 256)), np.zeros((6, 256))
    block_lo[:, 0] = block_hi[:, 0] = 1.0
    high_lo, high_hi = np.zeros((6, 256)), np.zeros((6, 256))
    previous_stage = -1
    for g in groups:
        resources.check()
        stage, cap = g['stage'], CPLUS[g['stage']]
        if stage != previous_stage:
            good_lo, good_hi = np.zeros((cap+1, 6, 256)), np.zeros((cap+1, 6, 256))
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
    # retained mass with 1e-90 absolute slack before outward float conversion.
    retained = Fraction(str(data['candidate_quantile_retained_mass'][ci, 1, 4]))
    tau_hi = num.local_float(1-retained+Fraction(1, 10**90))[2]
    return {'candidate': CANDIDATES[ci], 'groups': groups, 'suffix': suffix,
            'projected_upper': pu, 'tail_lower': high_lo, 'tail_upper': high_hi,
            'tau_upper': tau_hi, 'term_indices': terms,
            'precompute_seconds': time.perf_counter()-started,
            'exact_group_count': len(groups)}


def aggregate(keys, mass, lower, upper):
    if not len(keys):
        return keys, mass, lower, upper
    order = np.argsort(keys, kind='stable')
    keys = keys[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1])+1]
    counts = np.diff(np.r_[starts, len(keys)])
    mass = np.add.reduceat(mass[order], starts)
    lower = down(np.add.reduceat(lower[order], starts)*(1-2*counts*num.EPS))
    upper = up(np.add.reduceat(upper[order], starts)*(1+2*counts*num.EPS))
    check_interval(lower, upper, 'aggregated forward state')
    return keys[starts], mass, lower, upper


def score(model, public_boundary, resources):
    """Evaluate a candidate likelihood from the observed measurement boundary."""
    started = time.perf_counter()
    target = sum((sum(int(public_boundary[i]) for i in term) & 1) << j
                 for j, term in enumerate(model['term_indices']))
    keys = np.array([0], dtype=np.uint64)
    mass, lower, upper = np.ones(1), np.ones(1), np.ones(1)
    scale_lo = scale_hi = 1.0
    log_scale = 0.0
    error_c = error_cp = 0.0
    max_retained = max_aggregated = max_feasible = 1
    prune_events = exact_removed = 0
    for t, g in enumerate(model['groups'], 1):
        resources.check(len(keys)*len(g['d']))
        stage, cap = g['stage'], CPLUS[g['stage']]
        count = ((keys >> np.uint64(41)) & np.uint64(7))[:, None]+g['d']
        parent, transition = np.nonzero(count <= cap)
        k = count[parent, transition]
        e = ((keys[parent] & E_BIT) != 0) | (k > C[stage])
        sig = (keys[parent] & MASK) ^ g['v'][transition]
        # Stage-boundary reset is exact; aggregate counts that become identical
        # before applying the one budget truncation at this factor cut.
        new_keys = sig | (e.astype(np.uint64) << np.uint64(44))
        if not g['stage_end']:
            new_keys |= k << np.uint64(41)
        weights = g['weights'][transition]
        new_mass = mass[parent]*weights[:, 0]
        new_lo = down(lower[parent]*weights[:, 1])
        new_hi = up(upper[parent]*weights[:, 2])
        num.checked_positive(new_mass, 'forward point mass')
        if np.any(new_mass <= 0) or np.any(new_lo <= 0):
            raise num.NumericalStop('Positive forward transition lost to underflow')
        keys, mass, lower, upper = aggregate(new_keys, new_mass, new_lo, new_hi)
        max_aggregated = max(max_aggregated, len(keys))
        feasible = num.reachable(keys & MASK, target, model['suffix'][t])
        exact_removed += len(keys)-int(feasible.sum())
        keys, mass, lower, upper = keys[feasible], mass[feasible], lower[feasible], upper[feasible]
        max_feasible = max(max_feasible, len(keys))
        if not len(keys):
            scale_lo = scale_hi = 0.0
            log_scale = -math.inf
            break
        if len(keys) > B:
            residual = (keys & MASK) ^ np.uint64(target)
            u = np.ones(len(keys))
            for j, shift in enumerate(SHIFTS):
                index = ((residual >> np.uint64(shift)) & np.uint64(255)).astype(np.intp)
                u = np.minimum(u, model['projected_upper'][t, j, index])
            if np.any(u <= 0):
                raise num.NumericalStop('Feasible residual has zero projected completion upper')
            priority = up(upper*u)
            sector = ((keys & E_BIT) != 0)
            current_k = ((keys >> np.uint64(41)) & np.uint64(7))
            order = np.lexsort((keys & MASK, current_k, sector, -mass, -priority))
            keep, drop = order[:B], order[B:]
            delta_cp = float(up(scale_hi*sum_upper(priority[drop])))
            c_drop = drop[(~sector[drop]) & (current_k[drop] <= C[stage])]
            delta_c = float(up(scale_hi*sum_upper(priority[c_drop])))
            error_c = float(up(error_c+delta_c))
            error_cp = float(up(error_cp+delta_cp))
            keys, mass, lower, upper = keys[keep], mass[keep], lower[keep], upper[keep]
            prune_events += 1
        max_retained = max(max_retained, len(keys))
        normalization = float(mass.sum())
        if normalization <= 0 or not math.isfinite(normalization):
            raise num.NumericalStop('Invalid positive common scaling')
        mass = mass/normalization
        lower, upper = down(lower/normalization), up(upper/normalization)
        scale_lo, scale_hi = float(down(scale_lo*normalization)), float(up(scale_hi*normalization))
        log_scale += math.log(normalization)
        check_interval(lower, upper, 'normalized forward')
        check_interval(scale_lo, scale_hi, 'common scale')
        if scale_lo <= 0 or np.any(lower <= 0):
            raise num.NumericalStop('Positive retained scale/state lost')
    sector_mass = []
    for sector in (0, 1):
        chosen = ((keys & MASK) == np.uint64(target)) & (((keys & E_BIT) != 0) == sector)
        lo = float(down(scale_lo*sum_lower(lower[chosen])))
        hi = float(up(scale_hi*sum_upper(upper[chosen])))
        point = math.exp(log_scale)*float(mass[chosen].sum()) if len(keys) else 0.0
        sector_mass.append((lo, hi, point))
    (cl, cu, cp), (pl, pu, pp) = sector_mass
    cpl, cpu = float(down(cl+pl)), float(up(cu+pu))
    cpp = cp+pp
    qcu, qcpu = float(up(cu+error_c)), float(up(cpu+error_cp))
    high = min(float(model['tail_upper'][j, (target >> shift) & 255])
               for j, shift in enumerate(SHIFTS))
    # Outward recurrences can widen the scalar tail by relative roundoff;
    # this check allows 1e-9 relative, far above their O(groups*eps) bound.
    if high > model['tau_upper']*(1+1e-9)+1e-18:
        raise num.ModelStop('HIGH_ORDER_BOUND_INVALID')
    qfu = min(1.0, float(up(qcpu+high)))
    for label, lo, hi in (('C', cl, qcu), ('CPLUS', cpl, qcpu), ('FULL', cpl, qfu)):
        check_interval(lo, hi, label+' completed query')
    result = {'candidate': model['candidate'],
        'qC_lower': cl, 'qC_upper': qcu, 'qPlus1_retained': pp,
        'qCplus_lower': cpl, 'qCplus_upper': qcpu,
        'plus1_fraction': pp/cpp if cpp else math.nan,
        'beam_bound_C': error_c, 'beam_bound_CPLUS': error_cp,
        'projected_high_order_bound': high, 'qFull_lower': cpl, 'qFull_upper': qfu,
        'max_retained_states': max_retained, 'query_seconds': time.perf_counter()-started,
        'qC_retained': cp, 'qCplus_retained': cpp, 'qC_ret_lower': cl, 'qC_ret_upper': cu,
        'qPlus1_ret_lower': pl, 'qPlus1_ret_upper': pu,
        'qCplus_ret_lower': cpl, 'qCplus_ret_upper': cpu,
        'likelihood_factor': LIKELIHOOD_FACTOR, 'tau_CPLUS_upper': model['tau_upper'],
        'max_aggregated_states': max_aggregated, 'max_feasible_states': max_feasible,
        'prune_events': prune_events, 'exact_reachability_removed': exact_removed,
        'scale_lower': scale_lo, 'scale_upper': scale_hi, 'log_point_scale': log_scale}
    for label, lo, hi in (('C', cl, qcu), ('CPLUS', cpl, qcpu), ('FULL', cpl, qfu)):
        # Multiplication by a power of two is exact here (normal float range).
        result[f'L_{label}_lower'] = lo*LIKELIHOOD_FACTOR
        result[f'L_{label}_upper'] = hi*LIKELIHOOD_FACTOR
    return result
