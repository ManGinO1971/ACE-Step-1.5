"""Spielt Schläge mit dem Reggae-Kit aus den Loops. Jeder Schlag klingt VOLL aus (Ausklänge überlappen),
Stärke-Stufe nach Lautstärke, Wechsel zwischen Varianten (kein Maschinengewehr-Effekt)."""
import os,glob,numpy as np,soundfile as sf
SR=44100
# Klang-Ordner: auf dem Mac neben dem Skript; auf dem Server (RunPod) liegt das Klang-Paket auf dem Volume.
SAMPLES=os.environ.get('HIELTECH_DRUM_SAMPLES') or os.path.join(os.path.dirname(os.path.abspath(__file__)),'samples')
K=os.path.join(SAMPLES,'kit')+'/'
ART={'kick':'01_Kick','snare':'02_Snare','rimshot':'03_Rimshot','ghost':'04_Snare_Ghost',
     'hh_zu':'05_HiHat_zu','hh_halb':'06_HiHat_halboffen','hh_offen':'07_HiHat_offen'}
# 13. Nov 2026: Roots-Reggae-Kit = feste Auswahl nach Gehör (samples/kit_roots.json). HIELTECH_DRUM_KIT=alt -> altes Kit.
import json as _json
def _roots_laden():
    p=os.path.join(SAMPLES,'kit_roots.json')
    if os.environ.get('HIELTECH_DRUM_KIT','').strip().lower()=='alt' or not os.path.isfile(p): return None
    try:
        with open(p,encoding='utf-8') as f: return _json.load(f)
    except (OSError,ValueError): return None
ROOTS=_roots_laden()
def _sample(p):
    x,_=sf.read(p,always_2d=True,dtype='float32') if not p.lower().endswith(('.mp3','.m4a')) else (None,None)
    if x is None:
        y,_=_lb.load(p,sr=SR,mono=False); x=(np.atleast_2d(y).T if y.ndim==1 else y.T).astype(np.float32)
    if x.shape[1]==1: x=np.repeat(x,2,1)
    a=np.abs(x).max(1); st=max(0,int(np.argmax(a>a.max()*0.05))-88); x=x[st:]; a=np.abs(x).max(1)  # Stille vorn weg
    return {'x':x,'spitze':int(np.argmax(a)),'einsatz':int(np.argmax(a>a.max()*0.1)),'wert':float(np.abs(x.mean(1)).max()),'name':os.path.basename(p)}
def lade_kit(k=K,roots=True):
    if roots and ROOTS:
        return {art:{st:[_sample(os.path.join(SAMPLES,p))] for st,p in ROOTS[art].items()} for art in ART if art in ROOTS}
    kit={}
    for art,ordner in ART.items():
        kit[art]={}
        for st in ('leise','mittel','laut'):
            L=[]
            for p in sorted(glob.glob(f'{k}{ordner}/{st}/*.wav')+glob.glob(f'{k}{ordner}/{st}/*.flac')):
                x,_=sf.read(p,always_2d=True,dtype='float32')
                a=np.abs(x).max(1); L.append({'x':x,'spitze':int(np.argmax(a)),'einsatz':int(np.argmax(a>a.max()*0.1)),'wert':float(np.abs(x.mean(1)).max()),'name':os.path.basename(p)})
            if L: kit[art][st]=L
    return kit
def einsatz_setzen(x,hits,fenster=0.06):
    """Anschlag-Zeitpunkt je Schlag: erste Stelle über 10 % der Spitze im Fenster ab Onset."""
    a=np.abs(x).max(1) if x.ndim>1 else np.abs(x)
    for h in hits:
        w=a[h['start']:h['start']+int(fenster*SR)]
        if len(w): h['einsatz']=int(h['start']+np.argmax(w>w.max()*0.1))
def stufe_von(pegel,alle):
    lo,hi=np.percentile(alle,[33,66])
    return 'leise' if pegel<lo else ('laut' if pegel>hi else 'mittel')
def spielen(laenge,hits,kit,choke_offen=True,seed=0):
    """hits: Liste von dict(art, spitze (Sample), pegel (dBFS der Spitze))"""
    rng=np.random.default_rng(seed)
    y=np.zeros((laenge,2),np.float32); zaehler={}; benutzt=[]
    pegel_je_art={}
    for h in hits: pegel_je_art.setdefault(h['art'],[]).append(h['pegel'])
    letzte_offen=None
    for h in sorted(hits,key=lambda h:h['spitze']):
        art=h['art']
        if art=='snare' and 'ghost' in kit and h['pegel']<np.median(pegel_je_art['snare'])-12: art='ghost'
        stufen=kit[art]; st=stufe_von(h['pegel'],pegel_je_art[h['art']]) if len(pegel_je_art[h['art']])>=6 else 'mittel'
        if st not in stufen: st=min(stufen,key=lambda s:abs(['leise','mittel','laut'].index(s)-['leise','mittel','laut'].index(st)))
        L=stufen[st]; k=zaehler.get((art,st),0); zaehler[(art,st)]=k+1
        # Variante: der Reihe nach, mit kleinem Zufall, nie zweimal dieselbe hintereinander
        i=(k+int(rng.integers(0,max(1,min(3,len(L))))))%len(L)
        s=L[i]
        g=10**(h['pegel']/20)/max(s['wert'],1e-6)
        if len(L)==1: g*=10**(float(rng.normal(0,0.4))/20)    # ein Klang je Stufe: winzige Lautstärke-Schwankung (lebendig)
        a=((h['einsatz']-s['einsatz']) if 'einsatz' in h else (h['spitze']-s['spitze']))+h.get('schub',0)
        if art.startswith('hh') and choke_offen and letzte_offen is not None and a>letzte_offen[0]:
            # offene Hi-Hat wird vom nächsten Hi-Hat-Schlag natürlich abgedämpft (Pedal), 30 ms weich
            a0,n0,g0,s0=letzte_offen; f=int(0.03*SR)
            cut=a-a0
            if 0<cut<n0:
                seg=s0['x'][cut:n0]*g0; fade=np.ones(len(seg),np.float32); fade[:f]=np.linspace(1,0,min(f,len(seg)))[:len(fade[:f])]; fade[f:]=0
                st0=max(0,a); e0=min(laenge,a0+n0)
                if e0>st0: y[st0:e0]-=(seg*(1-fade)[:,None])[st0-a:e0-a] if a>=0 else 0
            letzte_offen=None
        q0=max(0,-a); a0=max(0,a); n=min(len(s['x'])-q0,laenge-a0)
        if n>0: y[a0:a0+n]+=g*s['x'][q0:q0+n]
        if art in ('hh_offen','hh_halb'): letzte_offen=(a,len(s['x']),g,s)
        benutzt.append((h['spitze'],art,st,s['name']))
    return y,benutzt

# ---------- Anpassung an einen bestimmten Song (Original-Spur als Vorbild) ----------
import librosa as _lb
def _fp(m,dauer=0.15):
    n=int(dauer*SR); m=np.pad(m[:n],(0,max(0,n-len(m))))
    S=_lb.feature.melspectrogram(y=m,sr=SR,n_mels=40,n_fft=2048,hop_length=512); v=10*np.log10(S+1e-10); return (v-v.max()).ravel()
def _huelle(m,dauer=0.5,ms=5):
    h=int(SR*ms/1000); n=int(dauer*SR)//h; m=np.pad(m[:n*h],(0,max(0,n*h-len(m))))
    e=20*np.log10(np.sqrt((m.reshape(n,h)**2).mean(1))+1e-9); return e-e.max()
def einzelne(hits,laenge,abstand=0.3):
    t=sorted(h['einsatz'] for h in hits); out=[]
    for h in hits:
        nach=[x for x in t if x>h['einsatz']]
        if (nach[0] if nach else laenge)-h['einsatz']>abstand*SR: out.append(h)
    return out
def passend_machen(kit,arten,x,hits,k=4,huelle=True):
    """Für die Arten (z.B. ['snare','ghost']) nur die k ähnlichsten Varianten je Stufe behalten und
    den Ausklang an die Original-Spur angleichen. Gibt Bericht zurück."""
    m=x.mean(1); ber={}
    for art in arten:
        hs=[h for h in hits if h.get('art',art)==art] or hits
        frei=einzelne(hs,len(m)) or hs
        if art not in kit: continue
        ref=np.median([_fp(m[h['einsatz']-88:]) for h in frei],0)
        alle=[]
        for st,L in kit[art].items():
            for s in L: s['abstand']=float(np.linalg.norm(_fp(s['x'].mean(1))-ref)/40)
            L.sort(key=lambda s:s['abstand']); kit[art][st]=L[:k]; alle+=L[:k]
        b={'behalten_je_stufe':k,'abstand_mittel':round(float(np.mean([s['abstand'] for s in alle])),2)}
        if huelle and len(frei)>=3:
            Eo=np.median([_huelle(m[h['einsatz']-88:]) for h in frei],0); Ek=np.median([_huelle(s['x'].mean(1)) for s in alle],0)
            D=Eo-Ek; D[:3]=0                                    # Anschlag bleibt unverändert
            D=np.convolve(np.r_[np.zeros(4),D,np.full(4,D[-1])],np.ones(5)/5,'same')[4:-4]
            D=np.clip(np.minimum.accumulate(np.minimum(D,2.0)) if D.mean()<0 else D,-30,2)   # nur weich verkürzen
            g=10**(D/20); h=int(SR*0.005)
            gk=np.interp(np.arange(int(12*SR)),np.arange(len(g))*h,g,right=g[-1])
            for st,L in kit[art].items():
                for s in L:
                    n=len(s['x']); s['x']=(s['x']*gk[:n,None]).astype(np.float32)
                    a=np.abs(s['x']).max(1); s['wert']=float(np.abs(s['x'].mean(1)).max()); s['spitze']=int(np.argmax(a))
            b['ausklang_angleich_db_bei_100_250_500ms']=[round(float(D[min(len(D)-1,i)]),1) for i in (20,50,99)]
        ber[art]=b
    return ber

# ---------- Toms: passender Shot je Tonhöhen-Gruppe, höchstens ±2 Halbtöne umgestimmt ----------
def _grund(seg):
    from scipy.signal import find_peaks
    n=len(seg); S=np.abs(np.fft.rfft(seg*np.hanning(n),1<<16)); f=np.fft.rfftfreq(1<<16,1/SR)
    b=(f>50)&(f<500); p,_=find_peaks(S[b],height=S[b].max()*0.3,distance=20); return float(f[b][p[0]]) if len(p) else 120.0
def tom_bibliothek(wurzel=None):
    T=os.path.join(SAMPLES,'toms')
    tr={}
    if ROOTS and ROOTS.get('toms'):
        tr={name:[os.path.join(SAMPLES,p) for p in P] for name,P in ROOTS['toms'].items()}
        tr={n:[p for p in P if os.path.isfile(p)] for n,P in tr.items()}; tr={n:P for n,P in tr.items() if P}
    for name in (('Tom1','Tom2','Tom3','Tom4') if not tr else ()):
        P=sorted(glob.glob(os.path.join(T,'Muldjord '+name,'*.flac'))+glob.glob(os.path.join(T,'Muldjord '+name,'*.wav')))
        if P: tr['Muldjord '+name]=P
    for p in (sorted(glob.glob(os.path.join(T,'shots','*'))) if not (ROOTS and ROOTS.get('toms')) else []):
        tr[os.path.basename(p)]=[p]
    lib={}
    for name,P in tr.items():
        L=[]
        for p in P:
            y,_=_lb.load(p,sr=SR,mono=False); y=np.atleast_2d(y).T if y.ndim==1 else y.T
            if y.shape[1]==1: y=np.repeat(y,2,1)
            a=np.abs(y).max(1); st=max(0,int(np.argmax(a>a.max()*0.1))-88); y=y[st:].astype(np.float32); a=np.abs(y).max(1)
            L.append({'x':y,'einsatz':int(np.argmax(a>a.max()*0.1)),'wert':float(np.abs(y.mean(1)).max()),'name':os.path.basename(p)})
        L.sort(key=lambda s:s['wert']); g=float(np.median([_grund(s['x'].mean(1)[int(.02*SR):int(.3*SR)]) for s in L]))
        lib[name]={'stufen':L,'ton':g}
    return lib
def _stimmen(s,halbton):
    if abs(halbton)<0.05: return s
    r=2**(halbton/12); y=_lb.resample(s['x'].T,orig_sr=SR*r,target_sr=SR).T.astype(np.float32); a=np.abs(y).max(1)
    return {'x':y,'einsatz':int(np.argmax(a>a.max()*0.1)),'wert':float(np.abs(y.mean(1)).max()),'name':s['name']}
def toms_spielen(laenge,x,hits,lib,max_halbton=2.0):
    m=x.mean(1)
    for h in hits: h['ton2']=_grund(m[h['einsatz']+int(.02*SR):h['einsatz']+int(.3*SR)])
    # Gruppen nach Tonhöhe (1,5 Halbtöne)
    # Gruppen = ein Tom: nur trennen, wo zwischen benachbarten Tonhöhen eine Lücke > 1,2 Halbtöne ist
    reihen=[]
    for h in sorted(hits,key=lambda h:h['ton2']):
        if reihen and 12*np.log2(h['ton2']/reihen[-1][-1]['ton2'])<=1.2: reihen[-1].append(h)
        else: reihen.append([h])
    y=np.zeros((laenge,2),np.float32); ber={}
    pegel=np.array([h['pegel'] for h in hits]); lo,hi=np.percentile(pegel,5),np.percentile(pegel,95)
    for k,rr in enumerate(reihen):
        ziel=float(np.median([h['ton2'] for h in rr]))
        ref=np.median([_fp(m[h['einsatz']-88:],0.3) for h in rr],0)
        best=None
        for name,d in lib.items():
            st=float(np.clip(12*np.log2(ziel/d['ton']),-max_halbton,max_halbton))
            mid=d['stufen'][len(d['stufen'])//2]; s=_stimmen(mid,st)
            dist=float(np.linalg.norm(_fp(s['x'].mean(1),0.3)-ref)/40)
            if best is None or dist<best[0]: best=(dist,name,st)
        dist,name,st=best; L=[_stimmen(s,st) for s in lib[name]['stufen']]
        # KEIN Kürzen: Toms klingen immer natürlich ganz aus (ein Schlagzeuger dämpft nicht ab)
        ber[f'tom_{k+1}']={'schlaege':len(rr),'ton_original_hz':round(ziel),'zeiten':[f"{int(h['start']/SR//60)}:{h['start']/SR%60:05.2f}" for h in rr],'shot':name,'umgestimmt_halbtoene':round(st,1),'klang_abstand':round(dist,2)}
        for j,h in enumerate(rr):
            rang=float(np.clip((h['pegel']-lo)/(hi-lo+1e-9),0,1)); i=int(round(rang*(len(L)-1)))
            if len(L)>1 and j%2: i=min(len(L)-1,i+1) if i==0 else i-1
            s=L[i]; g=10**(h['pegel']/20)/max(s['wert'],1e-6); a=h['einsatz']-s['einsatz']+h.get('schub',0)
            q0=max(0,-a); a0=max(0,a); n=min(len(s['x'])-q0,laenge-a0)
            if n>0: y[a0:a0+n]+=g*s['x'][q0:q0+n]
    return y,ber

# ---------- Becken: Einsatz aus der Becken-Spur, Lautstärke über den Ausklang (robust gegen Trennfehler) ----------
from scipy.signal import butter as _bt, sosfiltfilt as _sff
_HP5=_bt(4,5000,'high',fs=SR,output='sos')
def _hf(m): return _sff(_HP5,m)
def becken_bibliothek(wurzel=None,ordner=('CrashL',),pfade=None):
    mj=os.path.join(SAMPLES,'becken')+'/'
    L=[]
    quellen=[(os.path.basename(os.path.dirname(p)),[p]) for p in pfade] if pfade else \
            [(o,sorted(glob.glob(mj+o+'/*.wav')+glob.glob(mj+o+'/*.flac'))) for o in ordner]
    for o,P in quellen:
        for p in P:
            if p.lower().endswith(('.mp3','.m4a')):
                y0,_=_lb.load(p,sr=SR,mono=False); y=(np.atleast_2d(y0).T if y0.ndim==1 else y0.T).astype(np.float32)
            else:
                y,_=sf.read(p,always_2d=True,dtype='float32')
            if y.shape[1]==1: y=np.repeat(y,2,1)
            a=np.abs(y).max(1); st=max(0,int(np.argmax(a>a.max()*0.1))-88); y=y[st:]; a=np.abs(y).max(1)
            h=_hf(y.mean(1)); e=int(np.argmax(a>a.max()*0.1))
            L.append({'x':y,'einsatz':e,'hf':float(np.sqrt(np.mean(h[e+int(.02*SR):e+int(.3*SR)]**2))),'name':o+'/'+os.path.basename(p)})
    L.sort(key=lambda s:s['hf']); return L
def becken_karte(xb,abstand=0.3,unter_max=30.0,anstieg=10.0):
    m=xb.mean(1); h=_hf(m); k=int(.01*SR); n=len(h)//k
    e=20*np.log10(np.sqrt((h[:n*k].reshape(n,k)**2).mean(1))+1e-9); top=e.max()
    hits=[]; i=5
    while i<n-1:
        if e[i]>top-unter_max and e[i]-e[i-5:i].min()>=anstieg and e[i]>=e[i+1]-1:
            a=(i-3)*k; w=np.abs(h[a:a+int(.06*SR)]); ein=a+int(np.argmax(w>w.max()*0.3))
            ziel=float(np.sqrt(np.mean(h[ein+int(.02*SR):ein+int(.3*SR)]**2)))
            if not hits or ein-hits[-1]['einsatz']>abstand*SR: hits.append({'einsatz':ein,'start':ein,'ziel_hf':ziel,'pegel':20*np.log10(ziel+1e-9)})
            i+=int(abstand*100)
        else: i+=1
    return hits
def becken_spielen(laenge,hits,lib):
    y=np.zeros((laenge,2),np.float32)
    if not hits: return y,[]
    p=np.array([h['pegel'] for h in hits]); lo,hi=np.percentile(p,10),np.percentile(p,90); ben=[]
    for j,h in enumerate(hits):
        r=float(np.clip((h['pegel']-lo)/(hi-lo+1e-9),0,1)); i=int(round(r*(len(lib)-1)))
        if len(lib)>1 and j%2: i=min(len(lib)-1,i+1) if i==0 else i-1
        s=lib[i]; g=h['ziel_hf']/max(s['hf'],1e-9); a=h['einsatz']-s['einsatz']
        q0=max(0,-a); a0=max(0,a); n=min(len(s['x'])-q0,laenge-a0)
        if n>0: y[a0:a0+n]+=g*s['x'][q0:q0+n]
        ben.append((round(h['einsatz']/SR,2),s['name'],round(20*np.log10(h['ziel_hf']+1e-9),1)))
    return y,ben
def hh_uebersprechen_weg(xh,xb,hits,abstand_db=3.0):
    """Hi-Hat-Schläge, die nur Becken-Übersprechen sind, entfernen: HH-Spur muss in den Höhen
    deutlich lauter sein als die Becken-Spur im selben Moment."""
    hh=_hf(xh.mean(1)); hb=_hf(xb.mean(1)); raus=[]; gut=[]
    for h in hits:
        a=h['start']; w=int(.03*SR)
        lh=20*np.log10(np.abs(hh[a:a+w]).max()+1e-9); lb=20*np.log10(np.sqrt(np.mean(hb[a:a+w]**2))*1.414+1e-9)
        (gut if lh>=lb+abstand_db else raus).append(h)
    return gut,raus
def _hf_huelle(m,dauer=1.5,ms=10):
    k=int(SR*ms/1000); n=int(dauer*SR)//k; m=np.pad(m[:n*k],(0,max(0,n*k-len(m)))); return 20*np.log10(np.sqrt((m.reshape(n,k)**2).mean(1))+1e-9)
def becken_spielen2(laenge,xb,hits,lib):
    """wie becken_spielen, aber je Schlag die Variante, deren Ausklang-Verlauf (Höhen, 1,5 s) am besten passt"""
    y=np.zeros((laenge,2),np.float32); hb=_hf(xb.mean(1)); ben=[]; vorher=None
    for s in lib: s['huelle']=_hf_huelle(_hf(s['x'].mean(1))[s['einsatz']:])
    for j,h in enumerate(hits):
        nxt=hits[j+1]['einsatz'] if j+1<len(hits) else laenge
        ziel=_hf_huelle(hb[h['einsatz']:]); n=max(10,min(len(ziel),int((nxt-h['einsatz'])/(SR*0.01))))
        best=None
        for s in lib:
            z=ziel[:n]; ok=z>z.max()-40; ok[:5]=False
            g0=h['ziel_hf']/max(s['hf'],1e-9); gdb=float(np.median(z[ok]-s['huelle'][:n][ok])) if ok.sum()>5 else 20*np.log10(g0)
            gdb=float(np.clip(gdb,20*np.log10(g0)-6,20*np.log10(g0)+6)); g=10**(gdb/20); e=s['huelle'][:n]+gdb
            dist=float(np.mean(np.abs(e[ok]-z[ok]))) + (1.0 if s is vorher else 0.0)
            if best is None or dist<best[0]: best=(dist,s,g)
        dist,s,g=best; vorher=s; a=h['einsatz']-s['einsatz']+h.get('schub',0)
        q0=max(0,-a); a0=max(0,a); m=min(len(s['x'])-q0,laenge-a0)
        if m>0: y[a0:a0+m]+=g*s['x'][q0:q0+m]
        ben.append((round(h['einsatz']/SR,2),s['name'],round(20*np.log10(h['ziel_hf']+1e-9),1),round(dist,1)))
    return y,ben
