# Direct timing of store/recall at larger N (NOT part of CI; timings are machine-dependent).
# Usage: python experiments/measure_large_n.py   (N=4096 needs ~0.5 GB RAM; do not go to 16384 without ~12 GB free)
import sys,time,numpy as np
sys.path.insert(0, '.')
from cseq_wave.encode import bits_to_wave
from cseq_wave.memory import HolographicMemory
rng=np.random.default_rng(45)
for n in (128,256,512,1024,2048,4096):
    reps=[]
    for _ in range(5 if n<=1024 else 3):
        m=HolographicMemory(n); p=bits_to_wave(rng.integers(0,2,size=n))
        t=time.perf_counter(); m.store(p); ts=time.perf_counter()-t
        m.finalize()
        t=time.perf_counter(); m.recall(p,steps=25); reps.append(time.perf_counter()-t)
    print(n, 'store %.4f'%ts, 'recall25 min %.4f median %.4f'%(min(reps),sorted(reps)[len(reps)//2]), 'MB', n*n*16/2**20, flush=True)
