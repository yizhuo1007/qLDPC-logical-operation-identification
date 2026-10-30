"""MP-JP state ordering and forward likelihood recurrence."""
from __future__ import annotations
import math
import time
import numpy as np
from . import _numeric as num
from ._pm_count_aware import down, up, sum_lower, sum_upper, check_interval

def log_nonnegative(values):
    a = np.asarray(values,dtype=np.float64)
    if np.any(a < 0) or not np.all(np.isfinite(a)):
        raise num.NumericalStop('Invalid nonnegative heuristic input')
    result = np.full(a.shape,-np.inf)
    np.log(a,out=result,where=a>0)
    return result


def score(model, public_boundary, resources):
    """Public-only candidate scorer: no truth, record ID or evaluator metadata."""
    started = time.perf_counter()
    target = np.zeros(3, dtype=np.uint64)
    for j, term in enumerate(model['term_indices']):
        if sum(int(public_boundary[i]) for i in term) & 1:
            target[j//64] |= np.uint64(1) << np.uint64(j % 64)
    check_words(target)
    keys = np.zeros(1, dtype=STATE_DTYPE)
    mass, lower, upper = np.ones(1), np.ones(1), np.ones(1)
    scale_lo = scale_hi = 1.0
    log_scale = 0.0
    error_c = error_cp = 0.0
    max_retained = max_aggregated = max_feasible = 1
    prune_events = exact_removed = 0
    for t, g in enumerate(model['groups'], 1):
        resources.check(len(keys)*len(g['d']))
        stage, cap = g['stage'], CPLUS[g['stage']]
        count = keys['k'][:, None]+g['d']
        parent, transition = np.nonzero(count <= cap)
        k = count[parent, transition]
        e = (keys['e'][parent] != 0) | (k > C[stage])
        sig = keys['sig'][parent] ^ g['v'][transition]
        # Stage-boundary reset is exact; aggregate counts that become identical
        # before applying the one budget truncation at this factor cut.
        new_keys = np.zeros(len(parent), dtype=STATE_DTYPE)
        new_keys['sig'], new_keys['e'] = sig, e
        if not g['stage_end']:
            new_keys['k'] = k
        weights = g['weights'][transition]
        new_mass = mass[parent]*weights[:, 0]
        new_lo = down(lower[parent]*weights[:, 1])
        new_hi = up(upper[parent]*weights[:, 2])
        num.checked_positive(new_mass, 'forward point mass')
        if np.any(new_mass <= 0) or np.any(new_lo <= 0):
            raise num.NumericalStop('Positive forward transition lost to underflow')
        keys, mass, lower, upper = aggregate(new_keys, new_mass, new_lo, new_hi)
        max_aggregated = max(max_aggregated, len(keys))
        feasible = reachable(keys['sig'], target, model['suffix'][t])
        exact_removed += len(keys)-int(feasible.sum())
        keys, mass, lower, upper = keys[feasible], mass[feasible], lower[feasible], upper[feasible]
        max_feasible = max(max_feasible, len(keys))
        if not len(keys):
            scale_lo = scale_hi = 0.0
            log_scale = -math.inf
            break
        if len(keys) > B:
            residual = keys['sig'] ^ target
            u = np.ones(len(keys))
            for j, shift in enumerate(SHIFTS):
                index = projection(residual, shift)
                u = np.minimum(u, model['projected_upper'][t, j, index])
            if np.any(u <= 0):
                raise num.NumericalStop('Feasible residual has zero projected completion upper')
            priority = up(upper*u)
            sector = keys['e'] != 0
            current_k = keys['k']
            sig = keys['sig']
            order = retention_order(model, t, upper, residual, mass, sector, current_k, sig)
            keep, drop = order[:B], order[B:]
            delta_cp = float(up(scale_hi*sum_upper(priority[drop])))
            c_drop = drop[(~sector[drop]) & (current_k[drop] <= C[stage])]
            delta_c = float(up(scale_hi*sum_upper(priority[c_drop])))
            error_c = float(up(error_c+delta_c))
            error_cp = float(up(error_cp+delta_cp))
            keys, mass, lower, upper = keys[keep], mass[keep], lower[keep], upper[keep]
            prune_events += 1
        max_retained = max(max_retained, len(keys))
        resources.peak_retained = max(resources.peak_retained, max_retained)
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
        chosen = np.all(keys['sig'] == target, axis=1) & (keys['e'] == sector)
        lo = float(down(scale_lo*sum_lower(lower[chosen])))
        hi = float(up(scale_hi*sum_upper(upper[chosen])))
        point = math.exp(log_scale)*float(mass[chosen].sum()) if len(keys) else 0.0
        sector_mass.append((lo, hi, point))
    (cl, cu, cp), (pl, pu, pp) = sector_mass
    cpl, cpu = float(down(cl+pl)), float(up(cu+pu))
    cpp = cp+pp
    qcu, qcpu = float(up(cu+error_c)), float(up(cpu+error_cp))
    high = min(float(model['tail_upper'][j, projection(target, shift)])
               for j, shift in enumerate(SHIFTS))
    # Outward recurrences can widen the scalar tail by relative roundoff;
    # this check allows 1e-9 relative, far above their O(groups*eps) bound.
    if high > model['tau_upper']*(1+1e-9)+1e-18:
        raise num.ModelStop('[[72,12,6]]_HIGH_ORDER_BOUND_INVALID')
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
