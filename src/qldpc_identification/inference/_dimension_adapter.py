"""Quotient-dimension and stage-count adapter for scalable inference."""
from __future__ import annotations
import inspect
import time
from types import FunctionType
import numpy as np
from . import _t_count_aware as baseline
from . import _t_mpjp as mpjp

class Resources:
    def __init__(self, seconds=43200, gib=8):
        self.started=time.perf_counter(); self.seconds=seconds; self.gib=gib; self.peak_retained=0
    def check(self,expanded_states=0,reserved_bytes=0):
        if time.perf_counter()-self.started>self.seconds:
            raise baseline.num.ResourceStop('scorer time limit reached')
        if baseline.num.peak_rss()+1024*expanded_states+reserved_bytes+8*2**20>self.gib*2**30:
            raise baseline.num.ResourceStop('scorer memory limit reached')


def adapt_function(function, namespace, replacements=()):
    if not replacements:
        return FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    source=inspect.getsource(function)
    for old,new in replacements:
        if old not in source: raise RuntimeError('dimension adapter source mismatch: '+old)
        source=source.replace(old,new)
    exec(compile(source,'<dimension_adapter>','exec'),namespace)
    return namespace[function.__name__]


def engine(dimension,ideal_dimension,caps,stages):
    words=(dimension+63)//64; chunks=(dimension+7)//8
    shifts=tuple(range(0,dimension-7,8))
    if dimension%8: shifts+=(dimension-8,)
    zero=(0,)*words
    ns=dict(baseline.__dict__)
    ns.update(DIM=dimension,NWORDS=words,NCHUNKS=chunks,NSTAGES=stages,ZERO=zero,
        NPROJ=len(shifts),SHIFTS=shifts,C=tuple(caps['C']),CPLUS=tuple(caps['CPLUS']),B=4096,
        LIKELIHOOD_FACTOR=2.0**-ideal_dimension,
        STATE_DTYPE=np.dtype([('sig',np.uint64,(words,)),('k',np.uint8),('e',np.uint8)]))
    def check_words(a):
        if a.dtype!=np.uint64 or a.shape[-1]!=words or (dimension%64 and np.any(a[...,-1]>>np.uint64(dimension%64))):
            raise baseline.num.ModelStop('E3 signature storage exceeds derived quotient dimension')
    def xor_word(a,b): return tuple(x^y for x,y in zip(a,b))
    def insert(rows,value):
        value=tuple(map(int,value))
        while value!=zero:
            w=next(i for i in range(words-1,-1,-1) if value[i])
            p=w*64+value[w].bit_length()-1
            if rows[p]==zero: rows[p]=value; return
            value=xor_word(value,rows[p])
    def remainder(rows,value):
        for p in range(dimension-1,-1,-1):
            if (value[p//64]>>(p%64))&1 and rows[p]!=zero: value=xor_word(value,rows[p])
        return value
    def span(basis):
        result=[zero]
        for row in basis: result += [xor_word(v,row) for v in result]
        return tuple(result)
    def unit_word(p): return tuple((1<<(p%64)) if i==p//64 else 0 for i in range(words))
    def projection(a,shift):
        result=a[...,shift//64]>>np.uint64(shift%64)
        if shift%64>56 and shift//64+1<words:
            result=result | (a[...,shift//64+1]<<np.uint64(64-shift%64))
        return (result&np.uint64(255)).astype(np.intp)
    def reachable(signatures,target,lookup):
        if lookup is None: return np.ones(len(signatures),dtype=bool)
        residual=signatures^target; rem=np.zeros_like(residual)
        for j in range(chunks): rem^=lookup[j,projection(residual,8*j)]
        return ~np.any(rem,axis=1)
    def aggregate(keys,mass,lower,upper):
        if not len(keys): return keys,mass,lower,upper
        sig=keys['sig']; order=np.lexsort(tuple(sig[:,i] for i in range(words))+(keys['k'],keys['e']))
        keys=keys[order]
        changed=np.any(keys['sig'][1:]!=keys['sig'][:-1],axis=1)|(keys['k'][1:]!=keys['k'][:-1])|(keys['e'][1:]!=keys['e'][:-1])
        starts=np.r_[0,np.flatnonzero(changed)+1]; counts=np.diff(np.r_[starts,len(keys)])
        mass=np.add.reduceat(mass[order],starts)
        lower=baseline.down(np.add.reduceat(lower[order],starts)*(1-2*counts*baseline.num.EPS))
        upper=baseline.up(np.add.reduceat(upper[order],starts)*(1+2*counts*baseline.num.EPS))
        baseline.check_interval(lower,upper,'E3 aggregated forward state'); check_words(keys['sig'])
        if np.any(keys['k']>max(caps['CPLUS'])): raise baseline.num.ModelStop('Count overflow')
        return keys[starts],mass,lower,upper
    def log_priority(model,cut,upper,residual):
        result=mpjp.log_nonnegative(upper)
        full=dimension//8
        for j in range(full): result=result+model['mpjp_log'][cut,j,projection(residual,8*j)]
        if dimension%8:
            index=projection(residual,dimension-8)>>(8-dimension%8)
            result=result+model['mpjp_last'][cut,index]
        return result
    def retention_order(model,cut,upper,residual,mass,sector,current_k,sig):
        priority=log_priority(model,cut,upper,residual)
        return np.lexsort(tuple(sig[:,i] for i in range(words))+(current_k,sector,-mass,-priority))
    ns.update(check_words=check_words,insert=insert,remainder=remainder,span=span,unit_word=unit_word,
        projection=projection,reachable=reachable,aggregate=aggregate,retention_order=retention_order)
    for name in ('exact_group','conv_interval'):
        ns[name]=adapt_function(getattr(baseline,name),ns)
    ns['build_candidate']=adapt_function(baseline.build_candidate,ns,(
        ('range(10)','range(NSTAGES)'),('np.zeros((21, 256, 3)','np.zeros((NCHUNKS, 256, NWORDS)'),
        ('range(21)','range(NCHUNKS)')))
    ns['score']=adapt_function(mpjp.score,ns,(('np.zeros(3, dtype=np.uint64)','np.zeros(NWORDS, dtype=np.uint64)'),))
    return ns
