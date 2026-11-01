#!/usr/bin/env python3
"""
HIELTECH Stems – Lead fertig machen (Stand "Lead E-H1", 1. Nov 2026) – per Hörtest Schritt für Schritt freigegeben
(Mac-Tests ohne GPU, Rezept: ~/HELTECH_Stems/stems_test/STAND-LEAD-NK/STAND-LEAD-NK2.md). Gilt für JEDEN Song.

Eingang: vor_vocoder.wav = neu gesungener Lead (beste Variante, Zeitmaske + Text-Maske + Gate, wie Vocals V2)
         f0_neu.npy      = RMVPE-Melodie davon, f0_orig.npy = RMVPE-Melodie des getrennten Original-Leads
Schritte:
  1  Melodielinie (bauen): Oktav-Fehler je Stück -> Lock nur bei eigenen Kippern (<=150 ms) -> Folgen -> Ausreißer/Glätten
     -> weiche Tonwechsel -> "wie Mensch" (Wellen nur wie im Original, Ton-Enden nicht hoch) -> Leier-Bremse
     -> gehaltene Töne -> Verzierungen (nur wenn das Original sie singt) -> Melodie-Treue
  2  Vocoder C (nsf_vocoder_step.py, ohne Referenz) mit dieser Melodielinie
  3  Harmonie-Kamm weich (6 % / -10 dB)
  4  Ende-Gate B (Vocoder-Reste in Ausklängen, Maßstab = natürliche Stimme, erlaubt 3 dB, leise -20 dB)
  5  Pausen-Gate D (Reste zwischen Wörtern, -16 dB / ab 40 ms / 20 ms vor, 50 ms nach dem Wort offen, -40 dB)
  6  Stimm-Entrauschung voll (denoise_mel_band_roformer_aufr33, "dry")
  7  kleiner Hall nur für vollständige Wortenden (0,45 s, 8 %)
Aufruf: python stems_lead_fein.py <vor_vocoder.wav> <f0_neu.npy> <f0_orig.npy> <ausgabe.wav>
Letzte stdout-Zeile: JSON-Bericht.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import world_resynth_step as w  # noqa: E402

st = lambda x: 12 * np.log2(np.maximum(x, 1e-6) / 440) + 69  # noqa: E731
hz = lambda s: 440 * 2 ** ((s - 69) / 12)  # noqa: E731

# ============================================================ 1. Melodielinie
SOS=butter(2,3.0,'lowpass',fs=100,output='sos')   # Tonhöhen-Verlauf ohne Vibrato (Vibrato 4-7 Hz raus)
def stuecke(v):
    i,N=0,len(v)
    while i<N:
        if v[i]:
            j=i
            while j<N and v[j]: j+=1
            yield i,j; i=j
        else: i+=1
def trend(s):
    if len(s)>=12: return sosfiltfilt(SOS,s)
    return np.full(len(s),np.median(s))
def mensch(f, ref, vib_max=0.7, eigen_vib=0.5, eigen_max=0.5, ende_ms=120, ende_fall=0.3, start_ms=60):
    """Regeln wie ein menschlicher Sänger (31.10. Hörtest):
       1) Wellen (Vibrato) nur dort, wo das Original sie hat – Tiefe höchstens ±vib_max HT
       2) Ton-Enden: kein Hochziehen am Ende, statt dessen leicht abfallen (ende_fall HT über ende_ms)
       3) Ton-Anfänge: kein Überschießen über den Ton
       Tonwechsel (Bögen) kommen aus dem Verlauf und bleiben."""
    f=f.copy(); n=min(len(f),len(ref)); v=f>0; s=np.where(v,st(f),0.0)
    r=np.where(ref>0,st(ref),np.nan)
    for a,b in stuecke(v):
        seg=s[a:b]; T=trend(seg); fl=seg-T
        rr=r[a:min(b,n)]; rf=np.full(b-a,np.nan)
        ok=~np.isnan(rr)
        if ok.sum()>=12:
            # Original-Verlauf in der Oktave der neuen Stimme
            idx=np.flatnonzero(ok); ri=np.interp(np.arange(len(rr)),idx,rr[idx])
            Tr=trend(ri); rfl=ri-Tr
            gleich=ok&(np.abs(((Tr-T[:len(rr)])+6)%12-6)<1.0)&(np.abs(rfl)<1.5)   # gleicher Ton, Original sauber
            rf[:len(rr)][gleich]=np.clip(rfl[gleich],-vib_max,vib_max)
        neu=np.where(np.isnan(rf), np.clip(fl*eigen_vib,-eigen_max,eigen_max), rf)
        out=T+neu
        L=len(out); e=min(int(ende_ms/10),L//3)
        if e>=3:
            ziel=T[L-e-1] if L-e-1>=0 else T[0]
            rampe=np.linspace(0,-ende_fall,e)
            out[L-e:]=np.minimum(out[L-e:], ziel+0.15)+rampe*(out[L-e:]>ziel-1)
        sa=min(int(start_ms/10),L//3)
        if sa>=2:
            out[:sa]=np.minimum(out[:sa], T[:sa]+0.2)
        s[a:b]=out
    f[v]=hz(s[v]); return f


def gleiten(f, basis_ms=60, je_ht_ms=15, min_ms=80, max_ms=200, schwelle=1.5):
    """Tonwechsel (Sprung > schwelle HT in <= 60 ms) durch einen weichen Bogen (S-Kurve) ersetzen.
    Bogenlänge wächst mit dem Intervall: großer Sprung nach oben = längerer, langsamer Bogen."""
    f=f.copy(); v=f>0; s=np.where(v, st(f), np.nan); N=len(f)
    # geglättete Linie (100 ms Median) zum Erkennen der Töne – Vibrato soll nicht als Tonwechsel zählen
    k=11; sm=np.full(N,np.nan)
    for i in range(N):
        if v[i]:
            w=s[max(0,i-k//2):i+k//2+1]; w=w[~np.isnan(w)]; sm[i]=np.median(w)
    n_ch=0; i=6
    while i<N-6:
        if v[i-6] and v[i+6] and not np.isnan(sm[i-6]) and not np.isnan(sm[i+6]):
            iv=sm[i+6]-sm[i-6]
            if abs(iv)>schwelle and abs(iv)<13:
                # Mitte des Wechsels = größte Steigung der geglätteten Linie in [i-6,i+6]
                seg=np.abs(np.diff(sm[i-6:i+7]))
                if np.all(np.isnan(seg)): i+=1; continue
                m=i-6+int(np.nanargmax(seg))
                d=int(np.clip(basis_ms+je_ht_ms*abs(iv), min_ms, max_ms)/10)
                a,b=max(0,m-d//2), min(N-1,m+d//2+1)
                if v[a:b+1].all():
                    pa=np.nanmedian(s[max(0,a-5):a+1]); pb=np.nanmedian(s[b:b+6])
                    x=np.linspace(0,1,b-a+1); kurve=pa+(pb-pa)*(1-np.cos(np.pi*x))/2
                    f[a:b+1]=hz(kurve); n_ch+=1; i=b+6; continue
        i+=1
    return f,n_ch

def oktav_ausreisser(f, k=15):
    """Einzelne Frames, die eine Oktave neben ihrer Umgebung liegen (typisches Pfeifen/Kippen), zurücksetzen."""
    f=f.copy(); v=f>0; s=np.where(v, st(f), np.nan); N=len(f); n=0
    for i in np.flatnonzero(v):
        w=s[max(0,i-k//2):i+k//2+1]; w=w[~np.isnan(w)]
        if len(w)<5: continue
        m=np.median(w); d=s[i]-m
        if abs(d)>6:
            s[i]-=12*np.round(d/12); n+=1
    f[v]=hz(s[v]); return f,n

def oktav_je_ton(f, schwelle=9.0, max_ms=300):
    """Erkennungsfehler 'Oktave daneben' innerhalb EINES zusammenhängend gesungenen Stücks: Frames, die mehr als
    schwelle Halbtöne vom Median des ganzen Stücks entfernt liegen, um ganze Oktaven zurückfalten (z. B. +14,8 -> +2,8).
    Echte große Sprünge innerhalb eines gebundenen Stücks sind selten größer als 9 Halbtöne."""
    f=f.copy(); v=f>0; s=np.where(v, st(f), 0.0); n=0
    i,N=0,len(f)
    while i<N:
        if v[i]:
            j=i
            while j<N and v[j]: j+=1
            if j-i>=10:
                m=np.median(s[i:j]); d=s[i:j]-m; idx=np.abs(d)>schwelle
                # nur KURZE Ausreißer-Stücke (<= max_ms) – lange Töne in anderer Lage sind echt gesungen
                k=0; L=j-i
                while k<L:
                    if idx[k]:
                        e=k
                        while e<L and idx[e]: e+=1
                        if (e-k)*10<=max_ms:
                            s[i+k:i+e]-=12*np.round(d[k:e]/12); n+=e-k
                        k=e
                    else: k+=1
            i=j
        else: i+=1
    f[v]=hz(s[v]); return f,n


def kette(f0, ref, lock_tol=1.5, fol_tol=0.8, fol_ms=200, lock_ms=None):
    t=np.arange(f0.size)*0.01
    f0,_=w.octave_correct_f0(f0); f0,_=w.limit_f0_slew_rate(f0,t,500.0); f0,_=w.repair_f0_dropouts(f0,t,120.0,2.0,80.0)
    n=min(len(ref),len(f0)); both=(f0[:n]>0)&(ref[:n]>0)
    d=np.zeros(n); d[both]=12*np.log2(f0[:n][both]/ref[:n][both]); k=np.round(d/12)
    kb=float(np.median(k[both])); kb=float(np.round(kb)) if abs(kb)<=2 else 0
    kr=k-kb; fix=both&(kr!=0)&(np.abs(d-12*k)<lock_tol)&(np.abs(kr)<=2)
    if lock_ms is not None:
        # 1.11.: nur wenn die NEUE Stimme selbst kippt (Oktave weg von ihrer eigenen Umgebung) – kippt nur das
        # Original (z. B. Background am Tonanfang), bleibt die neue Stimme, wie sie ist
        sn=np.where(f0[:n]>0,12*np.log2(np.maximum(f0[:n],1e-6)/440),np.nan); eigen=np.zeros(n,bool)
        for t in np.flatnonzero(fix):
            ww=sn[max(0,t-15):t+16]; ww=ww[~np.isnan(ww)]
            if len(ww)>=5 and abs(sn[t]-np.median(ww))>6: eigen[t]=True
        fix=fix&eigen
    if lock_ms is not None:
        mr=int(lock_ms/10); i=0; keep=np.zeros(n,bool)
        while i<n:
            if fix[i]:
                j=i
                while j<n and fix[j]: j+=1
                if j-i<=mr: keep[i:j]=True
                i=j
            else: i+=1
        fix=keep
    f0[:n][fix]/=np.power(2.0,kr[fix])
    d2=np.zeros(n); d2[both]=12*np.log2(f0[:n][both]/ref[:n][both]); oc=np.round(d2/12); rest=d2-12*oc
    off=both&(np.abs(rest)>fol_tol); mr=int(fol_ms/10); i=0
    while i<n:
        if off[i]:
            j=i
            while j<n and off[j]: j+=1
            if j-i<=mr: f0[i:j]=ref[i:j]*np.power(2.0,oc[i:j])
            i=j
        else: i+=1
    return f0


def spitzen_weg(f, max_ms=60, thr=1.5):
    """kurze Ausreißer (<= max_ms) die > thr HT von beiden Nachbarn abweichen -> zwischen den Nachbarn interpolieren"""
    f=f.copy(); s=np.where(f>0, st(f), np.nan); N=len(f); w=max_ms//10
    i=1; fixed=0
    while i<N-1:
        if f[i]>0 and f[i-1]>0:
            # Start eines Sprungs
            if abs(s[i]-s[i-1])>thr:
                j=i
                while j<N and j-i<=w and f[j]>0 and abs(s[j]-s[i-1])>thr: j+=1
                if j<N and j-i<=w and f[j]>0 and abs(s[j]-s[i-1])<=thr:
                    a,b=s[i-1],s[j]
                    for k in range(i,j): s[k]=a+(b-a)*(k-i+1)/(j-i+1)
                    f[i:j]=440*2**((s[i:j]-69)/12); fixed+=j-i; i=j; continue
        i+=1
    return f, fixed
def glatt(f, k=5):
    """leichtes Glätten (Median 50 ms) nur innerhalb stimmhafter Stücke – Vibrato bleibt (Periode ~200 ms)"""
    out=f.copy(); v=f>0; s=np.where(v, st(f), 0); N=len(f); i=0
    while i<N:
        if v[i]:
            j=i
            while j<N and v[j]: j+=1
            seg=s[i:j]
            if j-i>=k:
                p=np.pad(seg,k//2,mode='edge'); m=np.array([np.median(p[t:t+k]) for t in range(j-i)])
                out[i:j]=440*2**((m-69)/12)
            i=j
        else: i+=1
    return out


def leier_bremse(f, fenster_ms=600, behalten=0.2, min_ht=0.8, min_wechsel=2):
    """'Leiern' = langsames HIN-UND-HER (ca. 1,5–4 Hz) in einem gehaltenen Ton, deutlich über ±min_ht Halbton.
       Erkannt wird es nur, wenn die langsame Bewegung um die Ton-Linie (Median 600 ms) im 600-ms-Fenster
       mindestens min_wechsel-mal die Richtung wechselt (echtes Schwanken). Echte Tonwechsel (gehen nur in eine
       Richtung) und feine Wellen (Vibrato > 4 Hz) bleiben unverändert. Danach Tonwechsel wieder weich (gleiten)."""
    f=f.copy(); v=f>0; s=np.where(v,st(f),0.0); k=int(fenster_ms/10)|1; n_fix=0
    for a,b in stuecke(v):
        seg=s[a:b]; L=len(seg)
        if L<k//2: continue
        kk=min(k,(L//2)*2-1) if L<k else k
        p=np.pad(seg,kk//2,mode='edge'); P=np.array([np.median(p[t:t+kk]) for t in range(L)])
        T=trend(seg); fein=seg-T; langsam=T-P
        vz=np.sign(langsam); wechsel=np.r_[0,np.abs(np.diff(vz))>0].astype(float)
        w=np.convolve(wechsel,np.ones(kk),mode='same')
        gross=np.convolve((np.abs(langsam)>min_ht).astype(float),np.ones(kk),mode='same')>0
        leier=(w>=min_wechsel)&gross
        if leier.any():
            m=np.convolve(leier.astype(float),np.ones(9)/9,mode='same')   # weich ein-/ausblenden
            fak=1-(1-behalten)*np.clip(m,0,1)
            s[a:b]=P+fak*langsam+fein; n_fix+=int(leier.sum())
    f[v]=hz(s[v]); g,_=gleiten(f); return g, n_fix


def plateaus(s, v, tol=0.35, min_ms=100):
    """Gehaltene Töne in der gesungenen Melodie: Stücke, in denen die Tonhöhe mind. min_ms innerhalb ±tol HT bleibt."""
    N=len(s); out=[]; i=0; k=int(min_ms/10)
    while i<N:
        if not v[i]: i+=1; continue
        j=i; lo=hi=s[i]
        while j+1<N and v[j+1] and max(hi,s[j+1])-min(lo,s[j+1])<=2*tol:
            j+=1; lo=min(lo,s[j]); hi=max(hi,s[j])
        if j-i+1>=k: out.append((i,j+1)); i=j+1
        else: i+=1
    return out
def noten_erhalten(f, roh, tol=0.3, min_ms=140, rand=2):
    """Wo die NEUE Stimme einen Ton gerade hält (Verzierungen/Melisma: Stufe für Stufe), wird genau diese Stufe
       übernommen – gerade, in der Lage der Melodielinie f. Übergänge dazwischen bleiben aus f (weiche Bögen)."""
    f=f.copy(); vr=roh>0; sr=np.where(vr,st(roh),0.0); vf=f>0; sf=np.where(vf,st(f),0.0); n=0
    for a,b in plateaus(sr,vr,tol,min_ms):
        a2,b2=a+rand,b-rand
        if b2-a2<3 or not vf[a2:b2].all(): continue
        lvl=np.median(sr[a2:b2]); okt=12*np.round((np.median(sf[a2:b2])-lvl)/12)
        sf[a2:b2]=lvl+okt; n+=b2-a2
    f[vf]=hz(sf[vf]); return f,n

def verzierungen_erhalten(f, roh, ref, tol=0.5, min_ms=80, passt_ht=0.7, anteil=0.8, rand=1):
    """Kurze Verzierungs-Töne (ab 80 ms, ±0,5 HT) der neuen Stimme übernehmen – aber NUR, wenn der Original-Lead an
       dieser Stelle denselben Ton singt (mind. 80 % der Frames innerhalb 0,7 HT). So bleiben echte Verzierungen
       (z. B. Chorus 'hoch-runter-hoch'), während Leiern (das das Original so nicht hat) korrigiert bleibt."""
    f=f.copy(); vr=roh>0; sr=np.where(vr,st(roh),0.0); vf=f>0; sf=np.where(vf,st(f),0.0)
    n=min(len(ref),len(f)); so=np.full(len(f),np.nan); so[:n]=np.where(ref[:n]>0,st(np.maximum(ref[:n],1e-6)),np.nan); k=0
    for a,b in plateaus(sr,vr,tol,min_ms):
        a2,b2=a+rand,b-rand
        if b2-a2<3 or not vf[a2:b2].all(): continue
        d=so[a2:b2]-sr[a2:b2]; d=d-12*np.round(np.nanmedian(d)/12) if np.isfinite(d).any() else d
        if np.mean(np.abs(np.nan_to_num(d,nan=99))<passt_ht)<anteil: continue
        lvl=np.median(sr[a2:b2]); okt=12*np.round((np.median(sf[a2:b2])-lvl)/12)
        sf[a2:b2]=lvl+okt; k+=b2-a2
    f[vf]=hz(sf[vf]); return f,k

def melodie_treue(f, roh, ref, gross_ht=1.5, passt_ht=0.7, min_ms=80):
    """Wo neue Stimme und Original-Lead DENSELBEN Ton singen (±0,7 HT, mind. 80 ms), die Melodielinie aber mehr als
       1,5 HT danebenliegt (z. B. tiefer Ton auf halbem Weg verschmiert), wird der gesungene Ton übernommen
       (leicht geglättet). Kleine Unterschiede (Leier-Korrektur, gerade Töne) bleiben unberührt."""
    f=f.copy(); n=min(len(f),len(roh),len(ref)); vf=f>0; sf=np.where(vf,st(f),0.0)
    sr=np.where(roh[:n]>0,st(np.maximum(roh[:n],1e-6)),np.nan); so=np.where(ref[:n]>0,st(np.maximum(ref[:n],1e-6)),np.nan)
    d_ro=so-sr; d_ro=d_ro-12*np.round(d_ro/12)
    einig=np.abs(np.nan_to_num(d_ro,nan=99))<passt_ht
    d_fr=sf[:n]-sr; okt=12*np.round(d_fr/12); weg=vf[:n]&np.isfinite(sr)&(np.abs(d_fr-okt)>gross_ht)
    sel=einig&weg; k=0; i=0; mr=int(min_ms/10)
    # nur zusammenhängende Einigkeit >= 80 ms zählt
    ein_lang=np.zeros(n,bool)
    while i<n:
        if einig[i]:
            j=i
            while j<n and einig[j]: j+=1
            if j-i>=mr: ein_lang[i:j]=True
            i=j
        else: i+=1
    sel=sel&ein_lang
    glatt=np.copy(sr)
    for t in np.flatnonzero(sel):
        w=sr[max(0,t-1):t+2]; glatt[t]=np.nanmedian(w)
    sf[:n][sel]=glatt[sel]+okt[sel]; k=int(sel.sum())
    f[vf]=hz(sf[vf]); return f,k


"""Ganze Melodie-Kette für den Lead (Stand 1. Nov 2026) – aus f0_neu (neue Stimme) und f0_lead_original."""
def bauen(f0_neu, f0_orig, treue=True):
    n,_=oktav_je_ton(f0_neu); o,_=oktav_je_ton(f0_orig)
    A=kette(n.copy(),o,lock_ms=150); C,_=spitzen_weg(A); C=glatt(C); C,_=oktav_ausreisser(C); H,_=gleiten(C)
    N=mensch(H,o,vib_max=0.35,eigen_vib=0.0,eigen_max=0.0,ende_ms=150,ende_fall=0.5); O,_=leier_bremse(N)
    R,_=noten_erhalten(O,n); S,_=verzierungen_erhalten(R,n,o)
    if not treue: return S
    T,_=melodie_treue(S,n,o); return T


# ============================================================ 3. Harmonie-Kamm
def kamm(x, sr, f0, breite=0.06, min_hz=18.0, boden_db=-10.0, bis_hz=9000.0, hop_s=0.01):
    """Pro 10 ms nur Grundton + Obertöne behalten, dazwischen absenken; stimmlose Stellen unverändert."""
    import librosa
    from scipy.ndimage import uniform_filter1d
    nfft = 4096
    hop = int(sr * hop_s)
    out = np.zeros_like(x)
    fr = librosa.fft_frequencies(sr=sr, n_fft=nfft)
    boden = 10 ** (boden_db / 20)
    for c in range(x.shape[1]):
        X = librosa.stft(x[:, c], n_fft=nfft, hop_length=hop)
        T = X.shape[1]
        M = np.ones((len(fr), T), np.float32)
        ff = np.interp(np.arange(T), np.arange(len(f0)), f0) if len(f0) != T else f0
        vz = np.interp(np.arange(T), np.arange(len(f0)), (f0 > 0).astype(float)) if len(f0) != T else (f0 > 0).astype(float)
        sel = fr < bis_hz
        fs = fr[sel][:, None].astype(np.float32)
        aktiv = (vz >= 0.5) & (ff > 0)
        f = np.where(aktiv, ff, 1.0).astype(np.float32)[None, :]
        k = np.maximum(1, np.round(fs / f))
        d = np.abs(fs - k * f)
        wd = np.maximum(min_hz, breite * k * f)
        g = np.exp(-0.5 * (d / wd) ** 2)
        Msel = boden + (1 - boden) * g
        Msel[:, ~aktiv] = 1.0
        M[sel, :] = Msel
        M = uniform_filter1d(M, 3, axis=1)
        out[:, c] = librosa.istft(X * M, hop_length=hop, length=len(x))
    return out


# ============================================================ 4./5. Gates (Maßstab = natürliche Stimme)
HOP_S = 0.01


def _pegel_db(x, sr):
    h = int(sr * HOP_S)
    n = len(x) // h
    m = x[:n * h].reshape(n, h, -1).mean(2)
    return 10 * np.log10((m ** 2).mean(1) + 1e-12)


def _phrase(pn, s=2.0):
    k = int(s / HOP_S)
    pad = np.pad(pn, (k // 2, k // 2), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(pad, k)[:len(pn)].max(1)


def _glaetten(g, auf_ms=5, zu_ms=40):
    a_auf, a_zu = np.exp(-HOP_S * 1000 / auf_ms), np.exp(-HOP_S * 1000 / zu_ms)
    out = np.empty_like(g)
    cur = 1.0
    for i, x in enumerate(g):
        a = a_auf if x > cur else a_zu
        cur = a * cur + (1 - a) * x
        out[i] = cur
    return out


def ende_gate(voc, sr, nat, sr_n, erlaubt_db=3.0, leise_db=-20.0):
    pv, pn = _pegel_db(voc, sr), _pegel_db(nat, sr_n)
    n = min(len(pv), len(pn))
    pv, pn = pv[:n], pn[:n]
    phr = _phrase(pn)
    rest = (pn < phr + leise_db) & (pv > pn + erlaubt_db)
    g = _glaetten(10 ** (np.clip(np.where(rest, (pn + erlaubt_db) - pv, 0.0), -60, 0) / 20))
    h = int(sr * HOP_S)
    gs = np.interp(np.arange(len(voc)), np.arange(n) * h + h / 2, g)
    return voc * gs[:, None], float(rest.mean())


def pausen_gate(y, sr, nat, sr_n, f0, leise_db=-16.0, min_ms=40, vor_ms=20, nach_ms=50, tief_db=-40.0):
    pn = _pegel_db(nat, sr_n)
    n = min(len(pn), len(f0), len(y) // int(sr * HOP_S))
    pn, ton = pn[:n], f0[:n] > 0
    luecke = (~ton) & (pn < _phrase(pn) + leise_db)
    m = np.zeros(n, bool)
    i = 0
    while i < n:
        if luecke[i]:
            j = i
            while j < n and luecke[j]:
                j += 1
            if (j - i) * 10 >= min_ms:
                a, b = i + int(nach_ms / 10), j - int(vor_ms / 10)
                if b > a:
                    m[a:b] = True
            i = j
        else:
            i += 1
    g = np.where(m, 10 ** (tief_db / 20), 1.0)
    g = np.convolve(np.pad(g, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    h = int(sr * HOP_S)
    gs = np.interp(np.arange(len(y)), np.arange(n) * h + h / 2, g)
    return y * gs[:, None], float(m.mean())


# ============================================================ 6. Stimm-Entrauschung
DENOISE_MODEL = os.environ.get("HIELTECH_LEAD_DENOISE", "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt")


def entrauschen(pfad, work):
    from backing_chor_prepare_step import run_model   # dieselbe Funktion wie im Background-Chor
    return run_model(pfad, DENOISE_MODEL, os.path.join(work, "denoise"), "dry")


# ============================================================ 7. kleiner Hall für die Wortenden
def ende_hall(x, sr, dauer_s=0.45, anteil=0.08, seed=7):
    rng = np.random.default_rng(seed)
    n = int(sr * dauer_s)
    t = np.arange(n) / sr
    ir = rng.standard_normal((n, 2)) * np.exp(-6.9 * t / dauer_s)[:, None]
    ir = sosfilt(butter(2, [200, 6000], "bandpass", fs=sr, output="sos"), ir, axis=0)
    ir = np.vstack([np.zeros((int(0.012 * sr), 2)), ir])
    ir = ir / np.sqrt((ir ** 2).sum(0, keepdims=True))
    if x.shape[1] == 1:
        x = np.repeat(x, 2, 1)
    nass = np.stack([fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)
    y = x + anteil * nass
    return y * min(1.0, 0.98 / max(float(np.abs(y).max()), 1e-6))


# ============================================================ Ablauf
def main():
    vor, f0n_p, f0o_p, ziel = sys.argv[1:5]
    t0 = time.time()
    work = tempfile.mkdtemp(prefix="lead_fein_")
    bericht = {"ok": True, "schritte": [], "zeiten": {}}
    f0n, f0o = np.load(f0n_p), np.load(f0o_p)
    T = bauen(f0n, f0o)
    np.save(os.path.join(work, "melodie.npy"), T)
    bericht["schritte"].append("melodie")
    # 2 Vocoder C
    voc_p = os.path.join(work, "voc.wav")
    t1 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, "nsf_vocoder_step.py"), vor, os.path.join(work, "melodie.npy"), voc_p],
                       capture_output=True, text=True, cwd=HERE, timeout=1800)
    if r.returncode != 0 or not os.path.isfile(voc_p):
        print(json.dumps({"ok": False, "fehler": "vocoder", "log": (r.stderr or "")[-800:]}, ensure_ascii=False))
        sys.exit(1)
    bericht["zeiten"]["vocoder"] = round(time.time() - t1, 1)
    bericht["schritte"].append("vocoder_c")
    voc, sr = sf.read(voc_p, always_2d=True, dtype="float32")
    nat, sr_n = sf.read(vor, always_2d=True, dtype="float32")
    # 3 Kamm weich
    y = kamm(voc, sr, T)
    bericht["schritte"].append("kamm_weich")
    # 4/5 Gates
    y, a1 = ende_gate(y, sr, nat, sr_n)
    y, a2 = pausen_gate(y, sr, nat, sr_n, T)
    bericht["schritte"] += ["ende_gate_B", "pausen_gate_D"]
    bericht["gate_anteil"] = {"ende": round(a1, 3), "pausen": round(a2, 3)}
    # 6 Entrauschen (bei Fehler weiter ohne – der Lead fällt nie weg)
    g_p = os.path.join(work, "gated.wav")
    sf.write(g_p, y, sr, subtype="FLOAT")
    try:
        t1 = time.time()
        d_p = entrauschen(g_p, work)
        e, sr_e = sf.read(d_p, always_2d=True, dtype="float32")
        if sr_e != sr:
            import librosa
            e = np.stack([librosa.resample(e[:, c], orig_sr=sr_e, target_sr=sr) for c in range(e.shape[1])], 1)
        n = min(len(e), len(y))
        y = np.pad(e[:n], ((0, len(y) - n), (0, 0)))[:, :y.shape[1]] if e.shape[1] >= y.shape[1] else np.repeat(e[:n].mean(1, keepdims=True), y.shape[1], 1)
        bericht["schritte"].append("entrauscht")
        bericht["zeiten"]["entrauschen"] = round(time.time() - t1, 1)
    except Exception as ex:  # noqa: BLE001
        bericht["warnung"] = f"Entrauschen übersprungen: {ex}"[:300]
    # 7 Hall für Wortenden
    y = ende_hall(y, sr)
    bericht["schritte"].append("hall_wortenden")
    sf.write(ziel, y.astype(np.float32), sr, subtype="PCM_24")
    bericht["zeiten"]["gesamt"] = round(time.time() - t0, 1)
    print(json.dumps(bericht, ensure_ascii=False))


if __name__ == "__main__":
    main()
