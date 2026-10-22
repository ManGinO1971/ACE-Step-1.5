"""Background-Chor (NEU, 23. Okt 2026): Melodie für den Vocoder ruhiger machen.
In jedem gehaltenen Ton (>= 120 ms) bleibt vom langsamen Wandern nur HALT übrig,
das Vibrato (4-8,5 Hz) wird mit VIB multipliziert. Hörtest 22. Okt 2026:
Frau 1 = 0.3 / 1.4 (Vocoder C), Mann = 0.3 / 1.0 (Vocoder B), Frau 2 = ohne (Vocoder A).
Aufruf: python f0_halt_vib.py <f0_ein.npy> <f0_aus.npy> <HALT> <VIB>"""
import sys, numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt
f=np.load(sys.argv[1]); halt=float(sys.argv[3]); vibf=float(sys.argv[4]); n=len(f); v=f>1
m=np.where(v,69+12*np.log2(np.maximum(f,1e-6)/440),np.nan); toene=[];i=0
while i<n:
    if v[i]:
        j=i
        while j<n and v[j]: j+=1
        q=np.round(median_filter(m[i:j],7,mode='nearest'));k=0
        while k<len(q):
            l=k
            while l<len(q) and q[l]==q[k]: l+=1
            if l-k>=12: toene.append((i+k,i+l))
            k=l
        i=j
    else: i+=1
sv=butter(2,[4,8.5],btype='band',fs=100,output='sos'); neu=m.copy()
for a,b in toene:
    x=m[a:b]; c=np.median(x); vib=sosfiltfilt(sv,x) if b-a>15 else np.zeros_like(x)
    neu[a:b]=c+vibf*vib+halt*(x-c-vib)
np.save(sys.argv[2],np.where(v,440*2**((neu-69)/12),0.0)); print(f'{len(toene)} Töne: Wandern x{halt}, Vibrato x{vibf}')
