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
# 14. Nov 2026: Bibliothek je Element (viele Klänge, die App wählt je Song die passendsten). Hat Vorrang vor dem
# festen Roots-Kit; die Favoriten des Inhabers stecken darin (bevorzugt, nicht erzwungen). HIELTECH_DRUM_KIT=alt|roots
BIB_DIR=os.path.join(SAMPLES,'bibliothek')
def _bib_laden():
    if os.environ.get('HIELTECH_DRUM_KIT','').strip().lower() in ('alt','roots'): return None
    p=os.path.join(BIB_DIR,'bibliothek.json')
    if not os.path.isfile(p): return None
    try:
        with open(p,encoding='utf-8') as f: d=_json.load(f)['klaenge']
    except (OSError,ValueError,KeyError): return None
    d=[e for e in d if os.path.isfile(os.path.join(BIB_DIR,e['datei']))]
    return d or None
BIB=_bib_laden()
if BIB: ROOTS=None
POOL={'kick':['Kick'],'snare':['Snare','Snare_hoch','Snare_tief','Snare_Rand_Fell'],'rimshot':['Rimshot'],
      'ghost':['Ghost','Snare','Snare_hoch'],'hh_zu':['HiHat_zu'],'hh_halb':['HiHat_halb_offen'],
      'hh_offen':['HiHat_offen']}
_BIB_CACHE={}
def _bib_sample(e):
    p=os.path.join(BIB_DIR,e['datei'])
    if p not in _BIB_CACHE:
        s=_sample(p); s['favorit']=bool(e.get('favorit')); s['element']=e['element']; s['name']=e['datei']; _BIB_CACHE[p]=s
    return _BIB_CACHE[p]
def bib_pfade(elemente):
    return [os.path.join(BIB_DIR,e['datei']) for e in (BIB or []) if e['element'] in elemente]
def lade_kit_bib():
    kit={}
    for art,els in POOL.items():
        L=[_bib_sample(e) for e in BIB if e['element'] in els]
        if L: kit[art]={'alle':L}
    return kit
def passend_bib(kit,arten,x,hits,k=3):
    """Je Art und Stärke (leise/mittel/laut des Originals) die k Bibliotheks-Klänge, die den Original-Schlägen
    dieser Stärke am ähnlichsten klingen (Favoriten 10 % bevorzugt). Kein Kürzen des Ausklangs."""
    m=x.mean(1); ber={}
    for art in arten:
        if art not in kit or 'alle' not in kit[art]: continue
        pool=kit[art]['alle']
        for s in pool:
            if 'fp' not in s: s['fp']=_fp(s['x'].mean(1))
        hs=[h for h in hits if h.get('art',art)==art]
        if not hs:
            kit[art]={'mittel':pool[:k]}; continue
        frei=einzelne(hs,len(m)) or hs
        pg=[h['pegel'] for h in hs]
        neu,b={},{}
        for st in ('leise','mittel','laut'):
            sel=[h for h in frei if stufe_von(h['pegel'],pg)==st] if len(hs)>=6 else frei
            if len(sel)<3: sel=frei
            ref=np.median([_fp(m[max(0,h['einsatz']-88):]) for h in sel],0)
            ab=sorted(((float(np.linalg.norm(s['fp']-ref)/40)*(0.9 if s.get('favorit') else 1.0),i) for i,s in enumerate(pool)))
            neu[st]=[pool[i] for _,i in ab[:k]]
            b[st]={'klaenge':[pool[i]['name'] for _,i in ab[:k]],'abstand':round(ab[0][0],2)}
        kit[art]=neu; ber[art]=b
    return ber
def _ton(seg):
    """Grundton des Fells (120–600 Hz) aus 93 ms ab 5 ms nach dem Anschlag"""
    seg=seg[200:200+4096]
    if len(seg)<1024: return 0.0
    # tiefster kräftiger Teilton = Grundton. Der stärkste Teilton springt bei derselben Snare zwischen Grundton und
    # 1,6-fachem Oberton (Fell-Moden) – das hat eine Snare fälschlich in „normal“ und „hoch“ geteilt.
    from scipy.signal import find_peaks
    S=np.abs(np.fft.rfft(seg*np.hanning(len(seg)),1<<15)); f=np.fft.rfftfreq(1<<15,1/SR); b=(f>120)&(f<600)
    p,_=find_peaks(S[b],height=S[b].max()*0.3,distance=30)
    return float(f[b][p[0]]) if len(p) else float(f[b][np.argmax(S[b])])
def _farbe(seg,dauer=0.25):
    S=_lb.feature.melspectrogram(y=np.ascontiguousarray(seg[:int(dauer*SR)],dtype=np.float32),sr=SR,n_mels=40,fmin=60)
    v=_lb.power_to_db(S.mean(1)+1e-12); return v-v.mean()
def snare_gruppen(kit,x,hits):
    """Snare-Trommeln des Originals an den LAUTEN Schlägen nach Tonhöhe trennen (eine oder zwei, z. B. hohe Intro-Snare
    und normale Snare). Je Trommel genau EIN Klang aus dem passenden Ordner (hoch -> Snare_hoch, sonst Snare), gewählt
    nach Tonhöhe und Klangfarbe. Leise Schläge (Ghost) und Übersprechen bekommen den Klang der Trommel, deren lauter
    Schlag zeitlich am nächsten liegt – so wechselt der Klang nie willkürlich zwischen zwei Snares."""
    m=x.mean(1); sn=[h for h in hits if h.get('art','snare') in ('snare','ghost')]
    if not sn: return {}
    pg=np.array([h['pegel'] for h in sn]); laut=[h for h in sn if h['pegel']>=np.percentile(pg,90)-8] or sn   # nur echte, kräftige Schläge bestimmen die Trommeln
    ton=np.array([_ton(m[h['einsatz']:]) for h in laut]); lg=np.log2(np.maximum(ton,60))
    gruppen=[list(range(len(laut)))]
    o=np.argsort(lg); sp=np.diff(lg[o])
    if len(sp):
        i=int(np.argmax(sp)); lo,hi=list(o[:i+1]),list(o[i+1:])
        if sp[i]*12>=4 and min(len(lo),len(hi))>=max(4,0.05*len(laut)): gruppen=[lo,hi]
    bib={el:[_bib_sample(e) for e in BIB if e['element']==el] for el in ('Snare','Snare_hoch','Snare_tief','Snare_Rand_Fell')}
    for L in bib.values():
        for s in L:
            if 'ton' not in s: s['ton']=_ton(s['x'].mean(1)[s['einsatz']:]); s['farbe']=_farbe(s['x'].mean(1)[s['einsatz']:])
    ber={}; zeit=[]; 
    for gi,G in enumerate(gruppen):
        t=float(np.median(ton[G])); ref=np.median([_farbe(m[laut[j]['einsatz']:]) for j in G],0)
        hoch=(len(gruppen)==2 and gi==1) or (len(gruppen)==1 and t>380)
        pool=(bib['Snare_hoch'] if hoch else bib['Snare']+bib['Snare_tief']) or [s for L in bib.values() for s in L]
        if not pool: return {}
        ab=sorted((0.5*abs(12*np.log2(max(s['ton'],60)/t))+float(np.linalg.norm(s['farbe']-ref)/np.sqrt(40)),k) for k,s in enumerate(pool))
        s=pool[ab[0][1]]; art=f'snare_{gi+1}'; kit[art]={'leise':[s],'mittel':[s],'laut':[s]}
        ber[art]={'ton_original_hz':round(t),'schlaege_laut':len(G),'klang':s['name'],'abstand':round(ab[0][0],2)}
        zeit+= [(laut[j]['einsatz'],art) for j in G]
    zeit.sort(); zt=np.array([z for z,_ in zeit])
    for h in sn:
        i=int(np.argmin(np.abs(zt-h['einsatz']))); h['art']=zeit[i][1]
    for a_ in ber: ber[a_]['schlaege_gesamt']=sum(1 for h in sn if h['art']==a_)
    return ber
def lade_kit(k=K,roots=True):
    if roots and BIB:
        return lade_kit_bib()
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
    if BIB:                                       # je Trommel (Element + Herkunft) ein Eintrag mit ihren Stärken
        for e in BIB:
            if e['element'] in ('Tom_hoch','Tom_mittel','Tom_tief','Standtom','Timbales_RotoToms'):
                tr.setdefault(e['element']+': '+os.path.basename(os.path.dirname(e.get('quelle',''))),[]).append(os.path.join(BIB_DIR,e['datei']))
    if ROOTS and ROOTS.get('toms'):
        tr={name:[os.path.join(SAMPLES,p) for p in P] for name,P in ROOTS['toms'].items()}
        tr={n:[p for p in P if os.path.isfile(p)] for n,P in tr.items()}; tr={n:P for n,P in tr.items() if P}
    for name in (('Tom1','Tom2','Tom3','Tom4') if not tr else ()):  # noqa: E501
        P=sorted(glob.glob(os.path.join(T,'Muldjord '+name,'*.flac'))+glob.glob(os.path.join(T,'Muldjord '+name,'*.wav')))
        if P: tr['Muldjord '+name]=P
    for p in (sorted(glob.glob(os.path.join(T,'shots','*'))) if not (ROOTS and ROOTS.get('toms')) and not BIB else []):
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
        lib[name]={'stufen':L,'ton':g,'roto':('roto' in name.lower() or 'timbales' in name.lower())}
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
        # 14. Nov 2026: Tonhöhe zählt – tiefe Tom bleibt tief. Roto-Toms/Timbales nur, wenn das Original höher ist als
        # die höchste normale Tom der Bibliothek (+4 Halbtöne).
        normal=[d['ton'] for d in lib.values() if not d.get('roto')]
        roto_ok=not normal or ziel>max(normal)*2**(4/12)
        for name,d in lib.items():
            if d.get('roto') and not roto_ok: continue
            roh_st=12*np.log2(ziel/d['ton'])
            st=float(np.clip(roh_st,-max_halbton,max_halbton))
            mid=d['stufen'][len(d['stufen'])//2]; s=_stimmen(mid,st)
            dist=float(np.linalg.norm(_fp(s['x'].mean(1),0.3)-ref)/40)+1.5*max(0.0,abs(roh_st)-max_halbton)
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
            # Splash/China nur, wenn sie klar besser passen (getrennte Crashs klingen oft kürzer als echte -> Splash-Falle)
            malus={'Splash':1.25,'China':1.6}.get(o,1.0)
            L.append({'x':y,'einsatz':e,'hf':float(np.sqrt(np.mean(h[e+int(.02*SR):e+int(.3*SR)]**2))),'name':o+'/'+os.path.basename(p),'malus':malus})
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
    for s in lib: s['huelle']=_hf_huelle(_hf(s['x'].mean(1))[s['einsatz']:]); s['farbe']=_farbe(s['x'].mean(1)[s['einsatz']:],0.5)
    mb=xb.mean(1)
    for j,h in enumerate(hits):
        nxt=hits[j+1]['einsatz'] if j+1<len(hits) else laenge
        ziel=_hf_huelle(hb[h['einsatz']:]); n=max(10,min(len(ziel),int((nxt-h['einsatz'])/(SR*0.01))))
        best=None
        for s in lib:
            z=ziel[:n]; ok=z>z.max()-40; ok[:5]=False
            g0=h['ziel_hf']/max(s['hf'],1e-9); gdb=float(np.median(z[ok]-s['huelle'][:n][ok])) if ok.sum()>5 else 20*np.log10(g0)
            gdb=float(np.clip(gdb,20*np.log10(g0)-6,20*np.log10(g0)+6)); g=10**(gdb/20); e=s['huelle'][:n]+gdb
            # 14. Nov 2026: Klangfarbe des Anschlags zählt mit (sonst gewinnt ein Ride-artiges Becken mit passendem
            # Ausklang); derselbe Klang wie beim letzten Schlag wird leicht bevorzugt (ein Becken bleibt ein Becken)
            fz=_farbe(mb[h['einsatz']:],0.5); farbe=float(np.linalg.norm(s['farbe']-fz)/np.sqrt(40))
            dist=float(np.mean(np.abs(e[ok]-z[ok])))*s.get('malus',1.0)+0.6*farbe-(0.5 if s is vorher else 0.0)
            if best is None or dist<best[0]: best=(dist,s,g)
        dist,s,g=best; vorher=s; a=h['einsatz']-s['einsatz']+h.get('schub',0)
        # 14. Nov 2026: nie lauter als das Original an dieser Stelle (+1 dB) – sonst wird ein leiser Rest zum lauten Becken
        o=float(np.abs(xb[h['einsatz']:h['einsatz']+int(0.06*SR)]).max()) if h['einsatz']<len(xb) else 0.0
        sp=float(np.abs(s['x'][s['einsatz']:s['einsatz']+int(0.06*SR)]).max()) if len(s['x'])>s['einsatz'] else 1.0
        if o>0 and g*sp>1.12*o: g=1.12*o/max(sp,1e-9)
        q0=max(0,-a); a0=max(0,a); m=min(len(s['x'])-q0,laenge-a0)
        if m>0: y[a0:a0+m]+=g*s['x'][q0:q0+m]
        ben.append((round(h['einsatz']/SR,2),s['name'],round(20*np.log10(h['ziel_hf']+1e-9),1),round(dist,1)))
    return y,ben
