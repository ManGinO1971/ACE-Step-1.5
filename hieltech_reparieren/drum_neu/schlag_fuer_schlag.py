"""HIELTECH Schlagzeug – „Schlag für Schlag“ + Hi-Hat-Hörmodell (22. Nov 2026, freigegeben: Rise Up Test 6 + Test 8).

Läuft NACH dem Feinschliff (Stand G). Muster, Zeitpunkte und Lautstärke je Schlag kommen aus G; die Klänge aus der
Bibliothek „Acoustic Drums.2026“ (+ alte Bibliothek); die Hi-Hat kommt aus dem Original (Hörmodell).
  Kick       : EIN Klang je Song (dem Original am nächsten, sanft angeglichen)
  Haupt-Snare: EINE Snare; hohe Snare nur an den Stellen, an denen G sie spielt
  Toms       : nach eigenem Ton (max. ±2 Halbtöne gestimmt), klingen voll aus
  Becken     : (vorerst) das Becken, das G an der Stelle hat – klingt voll aus, Abstoppen nur bei echter Song-Pause,
               Lautstärke über den Ausklang. Automatische Becken-Wahl kommt später.
  Hi-Hat     : Hörmodell – gleiche Viertel übereinanderlegen, Übersprechen/Percussion weg, Klang über den Ausklang
               bis zum nächsten Schlag (eigene Hi-Hats gleich getrennt), Lautstärke je Stelle und Zählzeit, Timing je
               Stelle, nächster Schlag schließt die Hi-Hat, Selbstprüfung (Ausschnitt getrennt, Pegel je Stelle).
Alles gesichert in ~/HELTECH_Stems/METHODE_Schlag_fuer_Schlag_2026-11-22/ (Test-Fassung + Beschreibung)."""
import os, json, numpy as np, soundfile as sf, librosa
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.signal import firwin2, fftconvolve

SR = 44100
# 28. Nov 2026: wie weit die Hi-Hat dem getrennten Original-Schlag folgt (Test-Schalter; getrennte Spitzen wackeln)
HH_FOLGE_MS = float(os.environ.get('HH_FOLGE_MS', '8'))
HH_FOLGE_ANTEIL = float(os.environ.get('HH_FOLGE_ANTEIL', '0.5'))


def _lade(p):
    x, s = sf.read(p, always_2d=True, dtype='float32')
    if s != SR:
        x = librosa.resample(x.T, orig_sr=s, target_sr=SR).T
    return np.repeat(x, 2, 1) if x.shape[1] == 1 else x[:, :2]


def _mono(p, n=None):
    x = _lade(p).mean(1)
    if n is not None:
        x = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
    return x


def _merkmale(m, s=0):
    w = m[s:s + int(.12 * SR)]; w = np.pad(w, (0, int(.12 * SR) - len(w)))
    S = librosa.feature.melspectrogram(y=w, sr=SR, n_fft=2048, hop_length=512, n_mels=40, fmin=40, fmax=16000).mean(1)
    d = 10 * np.log10(S + 1e-12); d -= d.max()
    g = m[s + int(.01 * SR):s + int(.25 * SR)]; g = np.pad(g, (0, max(0, 4096 - len(g))))
    F = np.abs(np.fft.rfft(g * np.hanning(len(g)), 1 << 15)); f = np.fft.rfftfreq(1 << 15, 1 / SR)
    b = (f > 45) & (f < 1200); ton = float(f[b][np.argmax(F[b])])
    hell = float(librosa.feature.spectral_centroid(y=w[:int(.05 * SR)] + 1e-9, sr=SR).mean())
    return np.maximum(d, -60), ton, hell


def _abst(a, b, tonw=0.5):
    return float(np.sqrt(np.mean((a[0] - b[0]) ** 2))) + tonw * abs(12 * np.log2(max(a[1], 40) / max(b[1], 40)))


def _schlaege(x, delta=0.07, unter=30):
    on = librosa.onset.onset_detect(y=x, sr=SR, hop_length=128, units='samples', backtrack=True, delta=delta)
    pk = 20 * np.log10(np.abs(x).max() + 1e-12); H = []
    for s in on:
        lv = 20 * np.log10(np.abs(x[s:s + int(.05 * SR)]).max() + 1e-9)
        if lv > pk - unter:
            H.append([int(s), lv])
    return H


# ------------------------------------------------------------------------------------------------ Bibliothek
_ALT = {'Kick': 'Kick/alt', 'Snare': 'Snare/1_Normal_alt', 'Snare_hoch': 'Snare/4_Hohe_alt',
        'HiHat_zu': 'Hi-Hat/1_Zu_alt', 'HiHat_halb_offen': 'Hi-Hat/2_Halb_alt', 'HiHat_offen': 'Hi-Hat/4_Offen_alt',
        'Tom_hoch': 'Toms/3_Tom_hoch_alt', 'Standtom': 'Toms/1_Stand-Tom_alt'}


def bibliothek(samples):
    """Klänge: neue Bibliothek (samples/sfs) + alte Bibliothek (samples/bibliothek). Merkmale je Klang."""
    L = []
    for e in json.load(open(os.path.join(samples, 'sfs', 'sfs_bibliothek.json'), encoding='utf-8'))['klaenge']:
        L.append(dict(kat=e['kat'], quelle=os.path.join(samples, e['datei']), name=e['name']))
    ab = os.path.join(samples, 'bibliothek')
    for e in json.load(open(os.path.join(ab, 'bibliothek.json'), encoding='utf-8'))['klaenge']:
        if e['element'] in _ALT and 'SnareHi_SP' not in e['datei'] and 'SnareDry_SP' not in e['datei']:
            L.append(dict(kat=_ALT[e['element']], quelle=os.path.join(ab, e['datei']), name='alt: ' + os.path.basename(e['datei'])))
    L = [s for s in L if os.path.isfile(s['quelle'])]
    try:   # 26. Nov 2026: vom Inhaber gesperrte Klänge (alle Teile, z. B. Doppelschläge) – sfs/gesperrt.json
        gs = json.load(open(os.path.join(samples, 'sfs', 'gesperrt.json'), encoding='utf-8')).get('gesperrt', [])
    except (OSError, ValueError):
        gs = []
    L = [s for s in L if not any(g in s['name'] or g in s['quelle'] for g in gs)]
    for s in L:
        x = _lade(s['quelle']); a = np.abs(x).max(1); st = max(0, int(np.argmax(a > a.max() * 0.1)) - 44)
        x = x[st:]; s['x'] = x / (np.abs(x).max() + 1e-9); s['m'] = _merkmale(s['x'].mean(1))
    return L


FAM = {'kick': ['Kick/'], 'snare': ['Snare/', 'Timbale/', 'Rim/1_'], 'toms': ['Toms/', 'Timbale/'], 'hh': ['Hi-Hat/']}


def _pool(LIB, fam, nicht=(), nur=None):
    return [s for s in LIB if any(s['kat'].startswith(p) for p in (nur or FAM[fam])) and not any(n in s['kat'] for n in nicht)]


def _gruppen_wahl(LIB, ber, name, fam, hits, orig, nicht=(), k_trommel=6, max_gr=3, tonw=0.5, haupt=False,
                  stimmen=False, eq=0.5, nur=None, einzeln=False):
    # Klangfarbe am Anschlag im ORIGINAL messen (an dessen eigenem Einsatz, ±30 ms) – unabhängig davon, wie genau
    # die Schläge von G liegen (sonst wechselt z. B. der Kick schon bei wenigen ms Versatz)
    on = librosa.onset.onset_detect(y=orig, sr=SR, hop_length=128, units='samples', backtrack=True, delta=0.07)
    if len(on):
        hits_o = []
        for s, l in hits:
            j = int(np.argmin(np.abs(on - s))); hits_o.append([int(on[j]) if abs(on[j] - s) < 0.03 * SR else int(s), l])
    else:
        hits_o = hits
    M = [_merkmale(orig, s) for s, _ in hits_o]
    P = _pool(LIB, fam, nicht, nur)
    if not P or not hits:
        return [None] * len(hits)
    if len(hits) >= 6:
        X = np.array([np.r_[m[0] / 10, 3 * np.log2(max(m[1], 40) / 100)] for m in M])
        lab = fcluster(linkage(X, 'ward'), t=max_gr, criterion='maxclust')
        gr = {g: np.where(lab == g)[0] for g in set(lab)}
        gross = [g for g in gr if len(gr[g]) >= 4] or list(gr)
        for g in list(gr):
            if g not in gross:
                c = X[gr[g]].mean(0); z = min(gross, key=lambda h: np.linalg.norm(X[gr[h]].mean(0) - c)); lab[gr[g]] = z
    else:
        lab = np.ones(len(hits), int)
    wahl = [None] * len(hits); info = {}
    for g in sorted(set(lab)):
        idx = np.where(lab == g)[0]
        ref = (np.median([M[i][0] for i in idx], 0), float(np.median([M[i][1] for i in idx])), 0)
        ab = sorted(P, key=lambda s: _abst(s['m'], ref, tonw))
        kd = {}
        for s_ in ab:
            kd.setdefault(s_['kat'], []).append(_abst(s_['m'], ref, tonw))
        kd = {c: float(np.mean(v[:3])) for c, v in kd.items()}
        if haupt and len(idx) == max(np.bincount(lab)):
            kd = {c: v for c, v in kd.items() if c.startswith('Snare/')}
        kat = min(kd, key=kd.get)
        if einzeln:                                          # eine Trommel = die eine passendste Aufnahme (Toms)
            kat = min((s_ for s_ in ab if s_['kat'] in kd), key=lambda s_: _abst(s_['m'], ref, tonw))['kat']
        tr = [s for s in ab if s['kat'] == kat][:k_trommel]
        if stimmen:
            tr2 = []
            for s_ in tr:
                ht = float(np.clip(12 * np.log2(max(ref[1], 40) / max(s_['m'][1], 40)), -2, 2))
                y_ = np.stack([librosa.effects.pitch_shift(s_['x'][:, c].copy(), sr=SR, n_steps=ht) for c in (0, 1)], 1) if abs(ht) > 0.05 else s_['x']
                d_ = dict(s_); d_['x'] = y_; d_['name'] = s_['name'] + f' ({ht:+.1f} Halbtöne)'; tr2.append(d_)
            tr = tr2
        if eq:                                               # sanft an das Original angleichen
            N = 16384; f = np.fft.rfftfreq(N, 1 / SR); fb = np.geomspace(20, 20000, 60)
            Pw = lambda w: np.abs(np.fft.rfft(np.pad(w, (0, max(0, N - len(w))))[:N] * np.hanning(N))) ** 2
            Gb = lambda p: np.array([p[(f > q / 1.12) & (f < q * 1.12)].mean() + 1e-20 for q in fb])
            lv_ = np.array([hits[i][1] for i in idx]); laut = [hits_o[i][0] for i in idx if hits[i][1] > lv_.max() - 8]
            Po = np.mean([Pw(orig[i:i + int(0.3 * SR)]) for i in laut], 0)
            Ps = np.mean([Pw(s_['x'].mean(1)[:int(0.3 * SR)]) for s_ in tr], 0)
            dd = 10 * np.log10(Gb(Po) / Gb(Ps)); dd -= np.median(dd[(fb > 80) & (fb < 8000)])
            dd = np.clip(eq * dd, -6, 3)
            fr = np.clip(np.concatenate([[0], fb / (SR / 2), [1]]), 0, 1); gg = 10 ** (np.concatenate([[dd[0]], dd, [dd[-1]]]) / 20)
            h = firwin2(2049, fr, gg); tr3 = []
            for s_ in tr:
                d_ = dict(s_); x_ = s_['x']
                d_['x'] = np.stack([fftconvolve(x_[:, c], h)[1024:1024 + len(x_)] for c in (0, 1)], 1).astype(np.float32)
                d_['x'] /= np.abs(d_['x']).max() + 1e-9; tr3.append(d_)
            tr = tr3
        tr.sort(key=lambda s: s['m'][2])                     # weich (dunkel) -> hart (hell)
        lv = np.array([hits[i][1] for i in idx]); rk = lv.argsort().argsort() / max(1, len(idx) - 1)
        letzte = None
        for j, i in enumerate(idx):
            q = int(round(rk[j] * (len(tr) - 1)))
            c = [v for v in (q, q + 1, q - 1) if 0 <= v < len(tr)]
            v = c[0] if tr[c[0]]['name'] != letzte or len(c) == 1 else c[1]   # nie zweimal dieselbe Datei hintereinander
            wahl[i] = tr[v]; letzte = tr[v]['name']
        info[f'gruppe_{g}'] = dict(schlaege=len(idx), art=kat, ton_original_hz=round(ref[1]), dateien=[s['name'] for s in tr],
                                   abstand=round(_abst(tr[0]['m'], ref, tonw), 2))
    ber[name] = info
    return wahl



def _tom_ton(m, s):
    """Grundton einer Tom: pyin über den Ausklang (30–300 ms) und stärkste Spitze im Spektrum (45–450 Hz). Stimmen
    beide (±1,5 Halbtöne) überein, gilt pyin, sonst die Spitze (pyin springt bei Toms gern auf Obertöne)."""
    w = m[s + int(.03 * SR):s + int(.30 * SR)]
    if len(w) < 2048:
        return max(_merkmale(m, s)[1], 40.0)
    S = np.abs(np.fft.rfft(w * np.hanning(len(w)), 1 << 16)); fr = np.fft.rfftfreq(1 << 16, 1 / SR)
    band = (fr > 45) & (fr < 450); spitze = float(fr[band][np.argmax(S[band])])
    try:
        f0, v, _ = librosa.pyin(w, fmin=40, fmax=450, sr=SR, frame_length=2048, hop_length=256)
        f0 = f0[v & np.isfinite(f0)]
        if len(f0) >= 3:
            p_ = float(np.median(f0))
            if p_ >= 46 and abs(12 * np.log2(p_ / spitze)) < 1.5:
                return p_
    except Exception:
        pass
    return spitze



def _abkling_dbs(m, s, hz, bis=0.25):
    """Wie schnell klingt der Grundton ab (dB/s, negativ), gemessen 60 ms bis `bis` nach dem Schlag."""
    import scipy.signal as sg
    w = m[s:s + int(0.6 * SR)]
    if len(w) < int(0.2 * SR):
        return None
    if not np.isfinite(hz) or hz <= 20:
        return None
    y = np.nan_to_num(sg.sosfiltfilt(sg.butter(2, [hz / 1.15, min(hz * 1.15, 0.45 * SR)], 'band', fs=SR, output='sos'), w))
    h = int(.01 * SR); e = 20 * np.log10(np.array([np.abs(y[i:i + h]).max() for i in range(0, len(y) - h, h)]) + 1e-9)
    t = np.arange(len(e)) * 0.01; msk = (t > 0.06) & (t < bis) & (e > e.max() - 50)
    msk &= np.isfinite(e)
    if msk.sum() <= 4:
        return None
    try:                                                # 24. Nov: nie an einer Messung abstürzen (SVD)
        k = float(np.polyfit(t[msk], e[msk], 1)[0])
        return k if np.isfinite(k) else None
    except (np.linalg.LinAlgError, ValueError):
        return None


def _tom_abklingen_song(orig, starts, hz, alle_starts):
    v = []
    for s in starts:
        nach = [a for a in alle_starts if a > s + int(.02 * SR)]
        if nach and nach[0] - s < int(0.3 * SR):
            continue                                    # nächster Schlag zu nah – Ausklang nicht messbar
        k = _abkling_dbs(orig, s, hz)
        if k is not None and k < -5:                    # steigt der Pegel, war es Übersprechen – nicht zählen
            v.append(k)
    return float(np.median(v)) if v else None


def _tom_abklingen_lib(s_):
    if 'abkl' not in s_:
        x = s_['x'].mean(1); hz = 100 * 2 ** (s_['st'] / 12)
        s_['abkl'] = _abkling_dbs(x, 0, hz) or -80.0
    return s_['abkl']

def _tom_satz(name):
    """Aufnahme-Sitzung einer Tom-Datei (= dieselbe Trommel): z. B. Break_2148, Neu_0825, Nr0442, RotoTom, alt."""
    import re
    m = re.search(r'_(Break|Neu)_(\d{4})-', name) or re.search(r'(Nr\d+)-', name)
    if m:
        return '_'.join(g for g in m.groups() if g)
    return 'RotoTom' if 'RotoTom' in name else ('alt' if 'alt' in name.lower() else name)


def _toms_wahl(LIB, ber, hits, orig, sprung=2.5, max_stimmen=3.0):
    """22. Nov 2026: so viele Toms, wie der Song hat. Ton je Schlag (im Original, am eigenen Einsatz); eine neue Tom
    beginnt, wo der Ton um mehr als `sprung` Halbtöne springt. Jede Tom bekommt die Aufnahme mit dem nächsten eigenen
    Ton (keine zwei Toms dieselbe, Reihenfolge tief->hoch bleibt), max. ±3 Halbtöne gestimmt, Stärken derselben Trommel
    (±1 Halbton, gleiche Art) als Lagen. Fehlt eine passende Tom, steht das im Bericht."""
    on = librosa.onset.onset_detect(y=orig, sr=SR, hop_length=128, units='samples', backtrack=True, delta=0.07)
    hits_o = []
    for s, l in hits:
        j = int(np.argmin(np.abs(on - s))) if len(on) else 0
        hits_o.append(int(on[j]) if len(on) and abs(on[j] - s) < 0.03 * SR else int(s))
    ton = np.array([_tom_ton(orig, s) for s in hits_o])
    st = 12 * np.log2(ton / 100.0)
    o = np.argsort(st); grp = np.zeros(len(st), int); g = 0
    for a, b in zip(o[:-1], o[1:]):
        if st[b] - st[a] > sprung:
            g += 1
        grp[b] = g
    # Einzel-Ausreißer (1 Schlag) zur nächsten Tom
    for g_ in set(grp):
        idx = np.where(grp == g_)[0]
        if len(idx) == 1 and len(set(grp)) > 1:
            andere = [h for h in set(grp) if h != g_]
            grp[idx] = min(andere, key=lambda h: abs(np.median(st[grp == h]) - st[idx[0]]))
    gruppen = sorted(set(grp), key=lambda g_: np.median(st[grp == g_]))
    P = [s_ for s_ in LIB if s_['kat'].startswith('Toms/')]
    for s_ in P:
        if 'st' not in s_:                              # Ton aus dem Dateinamen (sorgfältig gemessen), sonst messen
            import re as _re
            m_hz = _re.search(r'_(\d{2,3})Hz', s_['name'])
            s_['st'] = 12 * np.log2((float(m_hz.group(1)) if m_hz else _tom_ton(s_['x'].mean(1), 0)) / 100.0)
    vergeben = set(); wahl = [None] * len(hits); info = {}; fehlt = []
    for nr, g_ in enumerate(gruppen):
        idx = np.where(grp == g_)[0]; ziel = float(np.median(st[idx])); hz = 100 * 2 ** (ziel / 12)
        frei = [s_ for s_ in P if s_['name'] not in vergeben] or P
        # Ton zuerst: alle Aufnahmen, die sich höchstens ±3 Halbtöne auf den Ton stimmen lassen; darunter entscheidet
        # die Klangfarbe am Anschlag (wie bisher bei Rise Up). Sonst die mit dem nächsten Ton.
        nah = [s_ for s_ in frei if abs(s_['st'] - ziel) <= max_stimmen]
        mo = [_merkmale(orig, hits_o[i]) for i in idx]
        ab_o = None
        ref_ton = float(np.median([m_[1] for m_ in mo]))
        if nah:
            # 24. Nov (Inhaber): genau die richtige Tom – Ton, Klangfarbe UND wie lange sie klingt (Abklingen wie im
            # Original, gemessen am Grundton, nur an Schlägen ohne direkt folgenden Schlag)
            ab_o = _tom_abklingen_song(orig, [hits_o[i] for i in idx], 100 * 2 ** (ziel / 12), [h_[0] for h_ in hits])
            def wert(s_):
                k = np.mean([_abst(m_, s_['m'], tonw=3.0) for m_ in mo])
                # (Abklingen wird gemessen und angezeigt, zählt aber vorerst nicht – 24. Nov: Stand 8a bleibt)
                return (k, s_['name'])
            haupt = min(nah, key=wert)
        else:
            haupt = min(frei, key=lambda s_: (abs(s_['st'] - ziel), s_['name']))
        # Stärken derselben Trommel: gleiche Aufnahme-Sitzung (gleiche Trommel), ±1 Halbton, gleicher Ausklang (kurz/lang)
        satz = _tom_satz(haupt['name'])
        lagen = [s_ for s_ in P if _tom_satz(s_['name']) == satz and abs(s_['st'] - haupt['st']) <= 1.0
                 and ('_kurz_' in s_['name']) == ('_kurz_' in haupt['name']) and s_['name'] not in vergeben][:4] or [haupt]
        if haupt not in lagen:
            lagen = [haupt] + lagen[:3]
        for s_ in lagen:
            vergeben.add(s_['name'])
        if abs(haupt['st'] - ziel) > max_stimmen and 48 <= hz <= 450:
            fehlt.append(round(hz))
        tr = []
        for s_ in lagen:
            # Stimmen wie bisher (Spektrum-Spitze Original gegen Aufnahme); passt die nicht zum Tom-Ton, nach dem Tom-Ton
            ht = 12 * np.log2(max(ref_ton, 40) / max(s_['m'][1], 40))
            if abs(ht - (ziel - s_['st'])) > 2.0:
                ht = ziel - s_['st']
            ht = float(np.clip(ht, -max_stimmen, max_stimmen))
            y_ = np.stack([librosa.effects.pitch_shift(s_['x'][:, c].copy(), sr=SR, n_steps=ht) for c in (0, 1)], 1) if abs(ht) > 0.05 else s_['x']
            d_ = dict(s_); d_['x'] = y_; d_['name'] = s_['name'] + f' ({ht:+.1f} Halbtöne)'; d_['tom_hz'] = hz; tr.append(d_)
        tr.sort(key=lambda s_: s_['m'][2])
        lv = np.array([hits[i][1] for i in idx]); rk = lv.argsort().argsort() / max(1, len(idx) - 1); letzte = None
        for j, i in enumerate(idx):
            q = int(round(rk[j] * (len(tr) - 1))); c = [v for v in (q, q + 1, q - 1) if 0 <= v < len(tr)]
            v = c[0] if tr[c[0]]['name'] != letzte or len(c) == 1 else c[1]; wahl[i] = tr[v]; letzte = tr[v]['name']
        info[f'tom_{nr + 1}'] = dict(schlaege=len(idx), ton_original_hz=round(hz), art=haupt['kat'], dateien=[t['name'] for t in tr],
                                     abklingen_original_db_s=None if ab_o is None else round(ab_o), abklingen_gewaehlt_db_s=round(_tom_abklingen_lib(haupt)))
    ber['toms'] = info
    if fehlt:
        ber['toms_fehlt_in_bibliothek_hz'] = fehlt
    return wahl

# ------------------------------------------------------------------------------------------------ Hi-Hat-Hörmodell
_NF, _HOP = 1024, 128
_fr = np.fft.rfftfreq(_NF, 1 / SR)
_fb = 1000 * 2 ** (np.arange(0, 16) / 4); _fb = _fb[_fb < 17000]
_BI = [(_fr > q / 1.09) & (_fr < q * 1.09) for q in _fb]


def _hh_merk(x, s, W, p=None):
    w = x[s:s + int(.2 * SR)]; w = np.pad(w, (0, int(.2 * SR) - len(w)))
    M = np.abs(librosa.stft(w, n_fft=_NF, hop_length=_HOP, center=False))
    if p is not None:
        q = p[s:s + int(.2 * SR)]; q = np.pad(q, (0, int(.2 * SR) - len(q)))
        M = np.maximum(M - np.abs(librosa.stft(q, n_fft=_NF, hop_length=_HOP, center=False)), 0)
        v = x[max(0, s - int(.03 * SR)):s - int(.003 * SR)]
        if len(v) >= _NF:
            M = np.maximum(M - np.abs(librosa.stft(v, n_fft=_NF, hop_length=_HOP, center=False)).mean(1, keepdims=True), 0)
    t = np.arange(M.shape[1]) * _HOP / SR
    sl = [(0, .02), (.02, min(.06, W)), (min(.06, W), W)]
    B = [np.array([M[b][:, (t >= a) & (t < e)].mean() + 1e-12 for b in _BI]) for a, e in sl]
    L = [10 * np.log10(x_.sum()) for x_ in B]; form = [10 * np.log10(x_ / x_.max()) for x_ in B]
    return np.concatenate([form[0], form[1] * 0.7, form[2] * 0.5, np.array([L[1] - L[0], L[2] - L[0]]) * 2.0])



def _raster_pruefen(HH, beats, vorsprung=0.10):
    """22. Nov 2026 (abends, freigegeben): halbes/doppeltes Tempo prüfen. Drei Raster (erkannt, halb, doppelt); gewählt wird das,
    bei dem die Hi-Hat-Schläge auf 16tel-Stellen liegen (Anteil daneben) und jedes Viertel dasselbe Muster hat
    (Wechsel gerade/ungerade Viertel). Gewechselt wird nur, wenn das andere Raster klar besser ist (≥ vorsprung)."""
    on = librosa.onset.onset_detect(y=HH, sr=SR, hop_length=128, units='samples', backtrack=False, delta=0.05)
    if len(on) < 16 or len(beats) < 8:
        return beats, None
    pk = np.array([20 * np.log10(np.abs(HH[max(0, o - 200):o + 800]).max() + 1e-9) for o in on]); on = on[pk > np.percentile(pk, 99) - 24]
    kand = {'erkannt': beats, 'halb': beats[::2], 'doppelt': np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) // 2]))}
    pkt = {}
    for k, b in kand.items():
        q = np.diff(b); i = np.searchsorted(b, on) - 1; ok = (i >= 0) & (i < len(q)); i = i[ok]
        if len(i) < 8:
            pkt[k] = 9.0; continue
        ph = (on[ok] - b[i]) / q[i] * 4; daneben = float(np.mean(np.abs(ph - np.round(ph)) > 0.25))
        occ = np.zeros((len(q), 4))
        for ii, p_ in zip(i, np.round(ph).astype(int) % 4):
            occ[ii, p_] = 1
        akt = np.where(occ.sum(1) > 0)[0]; o = occ[akt]
        wechsel = float(np.abs(o[akt % 2 == 0].mean(0) - o[akt % 2 == 1].mean(0)).mean()) if len(akt) > 4 else 1.0
        v = daneben + wechsel
        pkt[k] = round(v, 3) if np.isfinite(v) else 9.0
    best = min(pkt, key=pkt.get)
    wahl = best if pkt['erkannt'] - pkt[best] >= vorsprung else 'erkannt'
    return np.array(kand[wahl]), dict(punkte=pkt, wahl=wahl)

def hihat_hoermodell(song, hh53, hhds, per, andere, samples, melde=print, tamb=None, kick=None):
    """Muster der Hi-Hat aus dem ganzen Song. Rückgabe dict(muster, plan, tempo) wie im Test (plan2.json).
    25. Nov 2026 (Money Tumble Down): `tamb` = 53er-Tamburin. Die DrumSep-Hi-Hat enthält ein Tamburin mit, dann
    klingelt es auf den schwachen Sechzehnteln lang und das Hörmodell spielte dort halboffene Hi-Hats. Ist ein
    Tamburin da, wird Klang/Länge nur aus der 53er-Hi-Hat gehört, und Stellen, an denen das Tamburin klar lauter
    ist als die 53er-Hi-Hat, gelten nicht als Hi-Hat. Ohne Tamburin bleibt alles wie vorher."""
    n = min(map(len, (song, hh53, hhds, per)))
    song, S53, DS, PE = song[:n], hh53[:n], hhds[:n], per[:n]
    TB = None
    if tamb is not None and len(tamb):
        TB = np.zeros(n, np.float32); TB[:min(n, len(tamb))] = tamb[:n]
        if np.abs(TB).max() < 0.1 * (np.abs(S53).max() + 1e-9):      # unter −20 dB der Hi-Hat: kein echtes Tamburin
            TB = None
    if TB is None:
        HH = 0.5 * (S53 / (np.abs(S53).max() + 1e-9) + DS / (np.abs(DS).max() + 1e-9))
    else:
        HH = S53 / (np.abs(S53).max() + 1e-9)                       # Tamburin im Song: nur die saubere 53er-Hi-Hat
    hh_luecken = None
    if TB is not None:
        # 26. Nov 2026 (Zion Callin): das 53er verliert die Hi-Hat abschnittsweise (ab 0:30 −76 dB, sie landet dann im
        # „Tamburin“), DrumSep hört sie durchgehend. Wo die 53er-Hi-Hat fehlt, aber DrumSep klar Hi-Hat hat → DrumSep nehmen.
        DSn = DS / (np.abs(DS).max() + 1e-9)
        w_ = int(SR * 0.5); k_ = np.ones(w_) / w_
        eS = np.sqrt(np.convolve(HH.astype(np.float64) ** 2, k_, 'same')) + 1e-12
        eD = np.sqrt(np.convolve(DSn.astype(np.float64) ** 2, k_, 'same')) + 1e-12
        pS, pD = np.percentile(eS, 90), np.percentile(eD, 90)
        rS, rD = 20 * np.log10(eS / pS), 20 * np.log10(eD / pD)
        luecke = (rS < rD - 12) & (rD > -20)          # 53er hört dort deutlich weniger Hi-Hat als DrumSep
        if luecke.mean() > 0.02:
            wl = np.convolve(luecke.astype(np.float64), np.ones(int(SR * .2)) / int(SR * .2), 'same')
            HH = ((1 - wl) * HH + wl * DSn * (pS / pD)).astype(np.float32)
            gr = np.flatnonzero(np.diff(np.r_[0, luecke.astype(np.int8), 0]))
            hh_luecken = [[round(gr[i] / SR, 1), round(gr[i + 1] / SR, 1)] for i in range(0, len(gr), 2)
                          if gr[i + 1] - gr[i] > SR][:30]
            melde(f"… Hi-Hat: 53er-Hi-Hat fehlt in {100 * luecke.mean():.0f} % des Songs – dort DrumSep-Hi-Hat")
    PEn = PE / (np.abs(S53).max() + 1e-9) * 0.5
    AND = np.array(andere)
    tempo, beats = librosa.beat.beat_track(y=song, sr=SR, units='samples', start_bpm=85); beats = np.array(beats)
    env = np.abs(HH)
    raster_info = None
    if os.environ.get('HH_RASTER', '1') == '1':
        beats, raster_info = _raster_pruefen(HH, beats)
        tempo = 60 * SR / float(np.median(np.diff(beats)))
    if len(beats) > 12 and os.environ.get('HH_RASTER_GLATT', '1') == '1':
        beats = _beats_glatt(beats)
    zaehlung = None
    if kick is not None and len(kick) >= 8 and len(beats) > 8:
        # 26. Nov 2026 (Plan Stufe 0): die Kick fällt bei Reggae auf einen Schlag. Liegt sie meistens auf Stelle 2,
        # zählt die App eine Achtel verschoben (Money Tumble Down: 98 %) – dann Zählung um eine Achtel verschieben.
        kk = np.array(sorted(kick)); q_ = np.diff(beats); i_ = np.searchsorted(beats, kk) - 1
        ok_ = (i_ >= 0) & (i_ < len(q_))
        st_ = np.round((kk[ok_] - beats[i_[ok_]]) / q_[i_[ok_]] * 4).astype(int) % 4
        if len(st_) >= 8:
            a0, a2 = float(np.mean(st_ == 0)), float(np.mean(st_ == 2))
            zaehlung = dict(kick_auf_stelle_0=round(a0, 2), kick_auf_stelle_2=round(a2, 2), verschoben=False)
            if a2 >= 0.6 and a0 < 0.2:
                beats = ((beats[:-1] + beats[1:]) // 2).astype(beats.dtype)
                zaehlung['verschoben'] = True
                melde('… Hi-Hat: Grundschlag eine Achtel verschoben gezählt – korrigiert (Kick)')

    def spitze(s):
        a = max(0, s - int(.05 * SR)); w = env[a:s + int(.06 * SR)]; return a + int(np.argmax(w)), 20 * np.log10(w.max() + 1e-9)
    R = [[spitze(int(beats[i] + (beats[i + 1] - beats[i]) * k / 4)) for k in range(4)] for i in range(len(beats) - 1)]
    lv = np.array([[r[1] for r in row] for row in R]); gmax = np.percentile(lv, 99)
    aktiv = lv.max(1) > gmax - 24

    def stoert(s):
        return len(AND) and np.min(np.abs(AND - s)) < 0.06 * SR
    # 22. Nov (Mehr-Song-Test): wie laut hört die 53er-Hi-Hat jede Stelle? (sie trennt Hi-Hat am saubersten)
    e53 = np.abs(S53)
    p53 = np.array([[20 * np.log10(e53[max(0, int(beats[i] + (beats[i + 1] - beats[i]) * k / 4) - int(.05 * SR)):
                                          int(beats[i] + (beats[i + 1] - beats[i]) * k / 4) + int(.06 * SR)].max() + 1e-9)
                     for k in range(4)] for i in range(len(beats) - 1)])
    s53_abs = np.median(p53[aktiv], 0)
    s53 = s53_abs - s53_abs.max()
    tb_db = None
    if TB is not None:                                      # Tamburin-Pegel je Stelle (gleiche Messung, gleiches Modell)
        eT = np.abs(TB)
        pT = np.array([[20 * np.log10(eT[max(0, int(beats[i] + (beats[i + 1] - beats[i]) * k / 4) - int(.05 * SR)):
                                         int(beats[i] + (beats[i + 1] - beats[i]) * k / 4) + int(.06 * SR)].max() + 1e-9)
                        for k in range(4)] for i in range(len(beats) - 1)])
        tb_db = np.median(pT[aktiv], 0) - s53_abs           # > 0: Tamburin lauter als die Hi-Hat an dieser Stelle
    # 28. Nov 2026 (Money, gesäuberter Stem): ist das Tamburin an ALLEN Stellen viel lauter als die Hi-Hat (Median ≥ 12 dB),
    # kann der Vergleich nicht mehr sagen, wo die Hi-Hat fehlt – dann entscheidet nur die 53er-Hi-Hat (≤ −18 dB = keine).
    # Vorher kippte Stelle 0 bei −9,1 statt −9,0 dB weg, und der halbe Hi-Hat-Grundschlag fehlte (klang wie falsches Timing).
    tb_ueberall = tb_db is not None and float(np.median(tb_db)) >= 12
    stellen = {}
    tamb_stellen = []
    for k in range(4):
        idx = [i for i in range(len(R)) if aktiv[i] and lv[i, k] > gmax - 30]
        if len(idx) < 0.5 * aktiv.sum():
            continue
        if s53[k] <= -18:
            continue                                        # die 53er-Hi-Hat hört hier (fast) nichts – keine Hi-Hat
        if np.mean([stoert(R[i][k][0]) for i in idx]) > 0.6 and s53[k] <= -12:
            continue                                        # Übersprechen von Kick/Snare – keine Hi-Hat
        if tb_db is not None and tb_db[k] >= 6 and s53[k] <= -9 and not tb_ueberall:   # 28. Nov: s. tb_ueberall
            tamb_stellen.append(k)
            continue                                        # 25. Nov: hier spielt das Tamburin, nicht die Hi-Hat
        stellen[k] = idx
    # ---- 25. Nov 2026: Katalog aller Reggae-Hi-Hat-Figuren (hihat_figuren.py, Plan mit dem Inhaber) ----------------
    # Je Stelle wird gemessen, wie laut und wie lang die Hi-Hat klingt (bei Tamburin nur aus der 53er-Hi-Hat), daraus die
    # Spielweise (zu / halb offen vom Fuß / offen) – und die ähnlichste Figur des Katalogs gewählt. Die Figur bestimmt,
    # WELCHE Stellen gespielt werden und mit welcher Spielweise; Lautstärke, Timing und Tempo kommen weiter aus dem Song.
    figur, fig_art, fig_klaenge, fig_info, fig_len = None, {}, {}, None, {}
    try:
        import hihat_figuren as HF
        envH = np.abs(HH)
        mess = []
        for k in range(4):
            lens = []
            for i in np.where(aktiv)[0]:
                p_ = R[i][k][0]; h_ = int(.01 * SR)
                e_ = np.array([np.sqrt(np.mean(HH[p_ + q * h_:p_ + (q + 1) * h_] ** 2)) for q in range(50)]) + 1e-12
                if TB is not None:
                    # 26. Nov 2026 (Money): der Ausklang der Hi-Hat folgt dem Tamburin (Korrelation ~0,8) – dessen
                    # Anteil abziehen, sonst wirkt jeder leise Schlag 300–400 ms lang statt der echten 160–190 ms
                    eT = np.array([np.sqrt(np.mean(TB[p_ + q * h_:p_ + (q + 1) * h_] ** 2)) for q in range(50)]) + 1e-12
                    g_ = float(np.min(e_[3:30] / eT[3:30])) * 0.9
                    e_ = np.maximum(e_ - g_ * eT, 1e-12)
                e_ = 20 * np.log10(e_ / (e_[0] + 1e-12) + 1e-9)
                lens.append(int(np.argmax(e_ < -15)) * 10 if (e_ < -15).any() else 500)
            mess.append(float(np.median(lens)) if lens else 0.0)
        pegel_k = (s53 if TB is not None else np.median(lv[aktiv], 0) - np.median(lv[aktiv], 0).max())
        # Übertönt das Tamburin die Hi-Hat klar, ist die Länge unsicher (Rest-Tamburin) – „halb offen“ dann erst ab 250 ms
        # (26. Nov, Money: 150/190 ms gemessen, der Inhaber hört aber den Rise-Up-Stil = alles zu)
        b_ab = 250 if (tb_db is not None and float(np.median(tb_db)) >= 12) else 130
        art_k = ['A' if L_ < b_ab else 'B' if L_ <= 450 else 'D' for L_ in mess]
        # gespielt: was das Hörmodell ohnehin als Hi-Hat-Stelle erkannt hat – dazu leise, aber LANG klingende Stellen
        # (halb offen/offen sind von Natur aus leise Zischer; leise kurze „Ticks“ sind dagegen oft nur Übersprechen)
        # dazu die schwachen Sechzehntel (Stelle 1/3), wenn sie klar zu hören sind (> −20 dB) – sie sind der Groove
        da_k = [bool(k in stellen or (art_k[k] in 'BCD' and pegel_k[k] > -24) or (k in (1, 3) and pegel_k[k] > -20))
                for k in range(4)]
        beste = None
        for fn, f in HF.FIGUREN.items():
            if f.get('triolen'):
                continue
            t4 = f['takt'][:4]
            pkt = 0.0
            for k in range(4):
                fd = t4[k] is not None
                pkt += 2.0 if fd == da_k[k] else 0.0
                if fd and da_k[k]:
                    fa = t4[k][0]
                    pkt += 1.0 if fa == art_k[k] or (fa in 'BC' and art_k[k] in 'BC') else (0.5 if fa in 'BCD' and art_k[k] in 'BCD' else 0.0)
            if beste is None or pkt > beste[0]:
                beste = (pkt, fn)
        figur = beste[1]
        if os.environ.get('HIELTECH_HH_FIGUR') in HF.FIGUREN:        # nur für Tests: Figur vorgeben
            figur = os.environ['HIELTECH_HH_FIGUR']
            if os.environ.get('HIELTECH_HH_ALLE_STELLEN'):
                da_k = [bool(HF.FIGUREN[figur]['takt'][k] is not None and pegel_k[k] > -24) for k in range(4)]
        t4 = HF.FIGUREN[figur]['takt'][:4]
        # Gespielt wird, was im Song gehört wurde; die Figur legt die Spielweise je Stelle fest (wo sie nichts sagt: gemessen)
        fig_art = {k: (t4[k][0] if t4[k] is not None else art_k[k]) for k in range(4) if da_k[k]}
        fig_len = {k: mess[k] for k in range(4)}
        auswahl = json.load(open(os.path.join(samples, 'sfs', 'hihat_plan_auswahl.json'), encoding='utf-8'))['gruppen']
        for g_, liste in auswahl.items():
            fig_klaenge[g_[0]] = [os.path.basename(p_).split('_')[-1].split('.')[0] for p_ in liste]   # z. B. Nr0426-287
        neu_st = {}
        for k in fig_art:
            neu_st[k] = stellen.get(k) or [i for i in range(len(R)) if aktiv[i] and lv[i, k] > gmax - 40]
        stellen = {k: v for k, v in neu_st.items() if v}
        fig_info = dict(figur=figur, name=HF.FIGUREN[figur]['name'], punkte=beste[0], von=12,
                        gemessen=[dict(stelle=k, pegel_db=round(float(pegel_k[k]), 1), lang_ms=int(mess[k]), spielweise=art_k[k],
                                       gespielt=da_k[k]) for k in range(4)])
        melde(f"… Hi-Hat-Katalog: Figur „{HF.FIGUREN[figur]['name']}“ ({beste[0]:.0f}/12 Punkte)")
    except Exception as e_:  # noqa: BLE001 – ohne Katalog wie bisher
        figur = None
        fig_info = dict(fehler=str(e_)[:200])
    ks = sorted(stellen)
    kette = json.load(open(os.path.join(samples, 'sfs', 'hihat_kette.json'), encoding='utf-8'))
    KH = _mono(os.path.join(samples, 'sfs', 'hihat_kette_getrennt_hh.flac'))
    for p in kette:
        s = p['start']; p['pk'] = s + int(np.argmax(np.abs(KH[s:s + int(.03 * SR)])))
        p['quelle'] = os.path.join(samples, p['datei'])
    # 22. Nov abends (Inhaber): einzelne Hi-Hat-Aufnahmen passen nicht zu dieser Aufgabe (z. B. Nr0426-277: zu, aber
    # nicht für Reggae) – sie stehen in sfs/hihat_gesperrt.json und werden vom Hörmodell nie gewählt.
    try:
        gesperrt = json.load(open(os.path.join(samples, 'sfs', 'hihat_gesperrt.json'), encoding='utf-8')).get('gesperrt', [])
    except (OSError, ValueError):
        gesperrt = []
    kette = [p for p in kette if os.path.isfile(p['quelle']) and not any(g in p['name'] for g in gesperrt)]
    muster = {}
    for j, k in enumerate(ks):
        idx = stellen[k]; nxt = ks[(j + 1) % len(ks)]
        gaps = [(R[i][nxt][0] if nxt > k else R[i + 1][nxt][0] if i + 1 < len(R) else 0) - R[i][k][0] for i in idx]
        gap = float(np.median([g for g in gaps if g > 0]))
        sech = np.median(np.diff(beats)) / 4
        bis16 = (sech - np.median([R[i][k][0] - (beats[i] + (beats[i + 1] - beats[i]) * k / 4) for i in idx])) / SR
        W = float(min(.16, gap / SR - .015, bis16 - .025))
        sauber = [i for i in idx if not stoert(R[i][k][0])] or idx
        F = np.median([_hh_merk(HH, max(0, R[i][k][0] - int(.004 * SR)), W, PEn) for i in sauber], 0)
        nh = max(3, int(W / .01))

        def hull(x, s_):
            h = int(.01 * SR); e = np.array([20 * np.log10(np.sqrt(np.mean(x[s_ + q * h:s_ + (q + 1) * h] ** 2)) + 1e-9) for q in range(nh)]); return e - e[0]
        Hk = np.median([hull(HH, R[i][k][0]) for i in sauber], 0)

        def ab(p):
            return float(np.sqrt(np.mean((hull(KH, p['pk']) - Hk) ** 2))) + 0.3 * float(np.sqrt(np.mean((_hh_merk(KH, max(0, p['pk'] - int(.004 * SR)), W) - F) ** 2)))
        kette_k = kette
        if figur is not None and k in fig_art:
            # 25. Nov 2026 (Katalog-Plan): nur die vom Inhaber ausgewählten Klänge der erkannten Spielweise
            ids = fig_klaenge.get(fig_art[k], [])
            kette_k = [p for p in kette if any(i_ in p['name'] for i_ in ids)] or kette
        kand = sorted(((ab(p), p['name'], p['kat'], p['quelle']) for p in kette_k))
        ziel = np.array([beats[i] + (beats[i + 1] - beats[i]) * k / 4 for i in idx]); ist = np.array([R[i][k][0] for i in idx])
        takt = {}
        for b in range(4):
            v = [lv[i, k] for i in idx if i % 4 == b and not stoert(R[i][k][0])] or [lv[i, k] for i in idx if i % 4 == b]
            if v:
                takt[b] = float(np.median(v) - gmax)
        muster[k] = dict(ausklang_db=[round(float(v), 1) for v in Hk], fenster_s=round(W, 3),
                         klang=[dict(name=c[1], kat=c[2], quelle=c[3], abstand=round(c[0], 2)) for c in kand[:6]],
                         pegel_je_zaehlzeit=takt, pegel=float(np.median(list(takt.values()))),
                         timing_ms=float(np.median(ist - ziel) / SR * 1000), streuung_ms=float(np.std(ist - ziel) / SR * 1000),
                         anzahl=len(idx))
        if figur is not None and fig_art.get(k) in ('B', 'C'):
            muster[k]['fuss_ms'] = int(max(80, fig_len[k]))     # 26. Nov: der Fuß stoppt nach der gemessenen Länge
    # 26. Nov 2026: feste Klänge + Reihenfolge je Stil (hihat_figuren.FESTE_KLAENGE, vom Inhaber an Rise Up bestätigt)
    try:
        import hihat_figuren as HF
        fest = HF.FESTE_KLAENGE.get(figur) if figur is not None else None
        if fest and muster:
            rang = sorted(muster, key=lambda k_: -muster[k_]['pegel'])
            for r_, k_ in enumerate(rang, 1):
                ids = fest.get(r_) or fest[max(fest)]
                neu = []
                for i_ in ids:
                    p_ = next((p for p in kette if i_ in p['name']), None)
                    if p_ is not None:
                        neu.append(dict(name=p_['name'], kat=p_['kat'], quelle=p_['quelle'], abstand=0.0))
                if neu:
                    muster[k_]['klang'] = neu
                    muster[k_]['fest_rang'] = r_
            if fig_info is not None:
                fig_info['feste_klaenge'] = True
    except Exception:  # noqa: BLE001
        pass
    plan = []
    for i in range(len(R)):
        if not aktiv[i]:
            continue
        for k in ks:
            m = muster[k]; ziel = beats[i] + (beats[i + 1] - beats[i]) * k / 4 + m['timing_ms'] / 1000 * SR
            eig = float(np.clip(R[i][k][0] - ziel, -HH_FOLGE_MS / 1000 * SR, HH_FOLGE_MS / 1000 * SR)) if lv[i, k] > gmax - 30 and not stoert(R[i][k][0]) else 0.0
            pz = m['pegel_je_zaehlzeit'].get(i % 4, m['pegel'])
            dl = float(np.clip(lv[i, k] - gmax - pz, -1.5, 1.5)) if lv[i, k] > gmax - 30 and not stoert(R[i][k][0]) else 0.0
            plan.append(dict(t=int(ziel + HH_FOLGE_ANTEIL * eig), stelle=int(k), pegel=pz + dl))
    return dict(muster={str(k): v for k, v in muster.items()}, plan=plan, tempo=float(np.atleast_1d(tempo)[0]),
                raster=raster_info, beats=[int(b) for b in beats], stellen_53er_db=[round(float(v), 1) for v in s53],
                katalog=fig_info, zaehlung=zaehlung, hh_53er_luecken_s=hh_luecken,
                tamburin=None if TB is None else dict(stellen_tamburin_minus_hihat_db=[round(float(v), 1) for v in tb_db],
                                                      als_tamburin_weggelassen=tamb_stellen))


def _hh_spielen(HP, ref_pegel, n, korr=None):
    """Spielplan -> Hi-Hat-Spur. ref_pegel: Pegel der lauten Hi-Hat-Schläge (90 %), korr: dB je Stelle."""
    MU = HP['muster']; korr = korr or {}
    proben = {}
    for k_, m_ in MU.items():
        best = m_['klang'][0]['abstand']; L_ = [c for c in m_['klang'] if c['abstand'] <= best * 1.2][:3]
        P_ = []
        for c in L_:
            x_ = _lade(c['quelle']); a2 = np.abs(x_).max(1); x_ = x_[max(0, int(np.argmax(a2 > a2.max() * 0.1)) - 44):]; x_ /= np.abs(x_).max()
            P_.append(dict(x=x_, name=c['name'], spitze=int(np.argmax(np.abs(x_).max(1)))))
        proben[k_] = P_
    plan_ = sorted(HP['plan'], key=lambda p: p['t'])
    if not plan_:
        return np.zeros((n, 2), np.float32), {}
    off = float(ref_pegel - np.percentile([p['pegel'] for p in plan_], 90))
    y = np.zeros((n + 5 * SR, 2), np.float32); zaehl = {}; schlaege = []
    for j_, p in enumerate(plan_):
        P_ = proben[str(p['stelle'])]; z = zaehl.get(p['stelle'], 0); zaehl[p['stelle']] = z + 1
        smp = P_[z % len(P_)]; st_ = max(0, p['t'] - smp['spitze']); g = 10 ** ((p['pegel'] + off + korr.get(str(p['stelle']), 0.0)) / 20)
        ende = (plan_[j_ + 1]['t'] - st_ + int(.01 * SR)) if j_ + 1 < len(plan_) else len(smp['x'])   # nächster Schlag schließt
        fm = MU[str(p['stelle'])].get('fuss_ms')
        if fm:                                                  # 26. Nov: halb offen – der Fuß stoppt nach gemessener Länge
            ende = min(ende, smp['spitze'] + int(fm / 1000 * SR))
        x_ = smp['x'][:max(1, ende)].copy() * g; f_ = min(len(x_), int(.012 * SR)); x_[-f_:] *= np.linspace(1, 0, f_)[:, None]
        y[st_:st_ + len(x_)] += x_
        schlaege.append((int(st_), float(20 * np.log10(g + 1e-12)), smp['name'],
                         float(MU[str(p['stelle'])]['pegel']) - max(float(m_['pegel']) for m_ in MU.values())))
    info = {k_: dict(pegel=round(m_['pegel'], 1), timing_ms=round(m_['timing_ms'], 1), klaenge=[p['name'] for p in proben[k_]])
            for k_, m_ in MU.items()}
    info['_schlaege'] = schlaege
    return y[:n], info


def _hh_bild(X, beats, AND):
    """Pegel je Stelle (e, &, a) relativ zur lautesten Stelle – wie die Selbstprüfung im Test."""
    X = X / (np.abs(X).max() + 1e-9); env = np.abs(X); out = {}
    for k in (1, 2, 3):
        P = []
        for i in range(len(beats) - 1):
            z = int(beats[i] + (beats[i + 1] - beats[i]) * k / 4); a = z - int(.05 * SR)
            if a < 0 or z + int(.06 * SR) > len(X):
                continue
            w = env[a:z + int(.06 * SR)]; s = a + int(np.argmax(w))
            if len(AND) and np.min(np.abs(AND - s)) < 0.06 * SR:
                continue
            P.append(20 * np.log10(w.max() + 1e-9))
        if P:
            out[k] = float(np.median(P))
    return out


# ------------------------------------------------------------------------------------------------ Hauptschritt

def _an_original(hits, wahl, orig, teil, ber, min_ms=10, max_ms=45):
    """23. Nov 2026 (freigegeben): jeden Schlag an den Original-Schlag legen. Verglichen wird der Anschlag (50 % der Spitze der
    geglätteten Hülle) von Aufnahme und Original; beim Kick nur der Tiefbass (< 150 Hz), weil dort nur der Kick klingt.
    Verschoben wird nur, wenn der Schlag 10–45 ms daneben liegt (App-Fehler, nicht das Spiel des Schlagzeugers)."""
    from scipy.signal import butter, sosfiltfilt
    def huelle(x):
        if teil == 'kick':
            x = sosfiltfilt(butter(4, 150 / (SR / 2), output='sos'), x)
        h = int(.002 * SR); return np.convolve(np.abs(x), np.ones(h) / h, mode='same')
    E = huelle(orig); gmax = np.percentile(E, 99.9) + 1e-12
    anschlag = {}
    neu = []; geschoben = []
    for (s, lv), smp in zip(hits, wahl):
        if smp is None:
            neu.append([s, lv]); continue
        nm = smp['name']
        if nm not in anschlag:
            e = huelle(smp['x'].mean(1)[:int(.15 * SR)]); anschlag[nm] = int(np.argmax(e > e.max() * 0.5))
        a0 = max(0, s - int(max_ms / 1000 * SR)); a1 = min(len(E), s + int(max_ms / 1000 * SR) + anschlag[nm])
        w = E[a0:a1]
        if not len(w) or w.max() < gmax * 10 ** (-30 / 20):
            neu.append([s, lv]); continue
        ziel = a0 + int(np.argmax(w > w.max() * 0.5)) - anschlag[nm]
        d = (ziel - s) / SR * 1000
        if min_ms < abs(d) <= max_ms:
            neu.append([int(ziel), lv]); geschoben.append(round(d, 1))
        else:
            neu.append([s, lv])
    ber.setdefault('an_original', {})[teil] = dict(geschoben=len(geschoben), von=len(hits),
                                                   median_ms=float(np.median(geschoben)) if geschoben else 0.0)
    return neu


def _beats_glatt(beats, halb=4):
    """28. Nov 2026: die Beat-Erkennung wackelt von Schlag zu Schlag (Money: ±15 ms), der Schlagzeuger nicht.
    Jeder Beat wird auf eine gerade Linie durch seine 9 Nachbarn gelegt – Tempo-Änderungen bleiben, das Wackeln geht."""
    b = np.asarray(beats, float)
    # 28. Nov 2026 (Money, Inhaber: Hi-Hat am Anfang nicht gleichmäßig): am Song-Anfang/-Ende verliert die Beat-Erkennung
    # das Tempo (Money 0:00–0:06: 78 statt 72,8 BPM). Beats, deren Abstand mehr als 4 %
    # vom Tempo des Songs abweicht (kürzer als 24 Beats), sind unsicher; sie werden aus den sicheren Beats weitergezählt.
    d = np.diff(b)
    if len(d) >= 12:
        q_ = float(np.median(d)); ab_ = np.abs(d / q_ - 1) > 0.04
        # nur kurze Abweichungen (< 24 Beats) sind Fehler der Erkennung; längere sind ein echter Tempowechsel
        gr_ = np.flatnonzero(np.diff(np.r_[0, ab_.astype(np.int8), 0])); gut_d = np.ones(len(d), bool)
        for a_, e_ in zip(gr_[::2], gr_[1::2]):
            if e_ - a_ < 24:
                gut_d[a_:e_] = False
        # Abschnitte mit sicheren Abständen; ein kurzer Abschnitt (< 16 Beats) zwischen/neben Fehlern ist meist um einen
        # halben Beat verrutscht (Money: 639 + 592 ms statt 2 × 824) – er wird mit dem langen Abschnitt weitergezählt
        gr_ = np.flatnonzero(np.diff(np.r_[0, gut_d.astype(np.int8), 0]))
        lang = np.zeros(len(b), bool)
        for a_, e_ in zip(gr_[::2], gr_[1::2]):         # Intervalle a_..e_-1 → Beats a_..e_
            if e_ - a_ >= 16:
                lang[a_:e_ + 1] = True
        g = np.flatnonzero(lang)
        if len(g) >= 8 and len(g) < len(b):
            for j in np.flatnonzero(~lang):
                nb = g[np.argsort(np.abs(g - j))[:8]]           # die 8 nächsten sicheren Beats
                k_, c_ = np.polyfit(nb, b[nb], 1)
                b[j] = k_ * j + c_
    out = b.copy(); i_ = np.arange(len(b))
    for j in range(len(b)):
        a, e = max(0, j - halb), min(len(b), j + halb + 1)
        k, d = np.polyfit(i_[a:e], b[a:e], 1)
        out[j] = k * j + d
    out = out[out >= 0]                                  # weitergezählte Beats vor dem Song-Anfang weglassen
    return np.round(out).astype(np.asarray(beats).dtype)


def _anschlag_versatz(neu, orig, lo, hi):
    """Anschlag (50 % der Hülle) der neuen Spur minus Original, nur an klaren Schlägen beider Spuren."""
    import scipy.signal as sg
    n = min(len(neu), len(orig))
    def E(x):
        x = sg.sosfiltfilt(sg.butter(4, [lo, hi], 'band', fs=SR, output='sos'), x[:n]); h = int(.001 * SR)
        return np.convolve(np.abs(x), np.ones(h) / h, 'same')
    en, eo = E(neu), E(orig)
    pk, _ = sg.find_peaks(en, height=en.max() * 0.05, distance=int(.08 * SR))
    w = int(.03 * SR); d = []
    for p in pk:
        a = max(0, p - w); so, sn = eo[a:p + w], en[a:p + w]
        if not len(so) or so.max() < eo.max() * 0.05:
            continue
        d.append((int(np.argmax(sn > sn.max() * .5)) - int(np.argmax(so > so.max() * .5))) / SR * 1000)
    if len(d) < 8:
        return None
    d = np.array(d)
    return dict(median_ms=round(float(np.median(d)), 1), streuung_ms=round(float(np.std(d)), 1), schlaege=len(d))


def _kick_grundmuster(kh, kick_orig, congas, song, ber, anteil=0.35, laut_db=-6.0, congas_db=-30.0):
    """28. Nov 2026 (Plan Stufe 2, Money): ein Reggae-Schlagzeuger wiederholt sein Kick-Muster. Die Kicks werden auf
    8 Stellen je 4 Schläge des Songs gelegt; Stellen mit mindestens 35 % der Kicks der stärksten Stelle sind das Grundmuster.
    Eine Kick außerhalb des Musters bleibt nur, wenn sie laut ist (höchstens 6 dB unter den lauten Kicks) UND dort keine
    Hand-Trommel tief klingt (Congas/Djembe mindestens 30 dB leiser) – z. B. die Kick am Anfang einer Phrase.
    Leise Kicks außerhalb des Musters mit Djembe/Congas daneben sind Trennreste und werden nicht gespielt."""
    if song is None or len(kh) < 16:
        return kh
    try:
        import scipy.signal as sg
        _, bt = librosa.beat.beat_track(y=song, sr=SR, units='samples', hop_length=256)
        bt = np.asarray(bt, float)
        if len(bt) < 16:
            return kh
        lp = lambda x: sg.sosfiltfilt(sg.butter(4, [40, 150], 'band', fs=SR, output='sos'), x)
        kl = lp(kick_orig); cl = lp(congas) if congas is not None else None
        sp = lambda x, s: 20 * np.log10(np.abs(x[max(0, s - 200):s + int(.05 * SR)]).max() + 1e-9)
        st = np.array([h[0] for h in kh], float)
        ph = np.round((np.interp(st, bt, np.arange(len(bt))) % 4) * 2).astype(int) % 8
        zahl = np.bincount(ph, minlength=8)
        muster = [int(i) for i in np.where(zahl >= anteil * zahl.max())[0]]
        if not muster:
            return kh
        lv = np.array([sp(kl, int(s)) for s in st]); ref = float(np.percentile(lv, 90))
        neu, raus = [], []
        for h, p_, l_ in zip(kh, ph, lv):
            if p_ in muster:
                neu.append(h); continue
            c_ = sp(cl, int(h[0])) - l_ if cl is not None else -99.0
            if l_ >= ref + laut_db and c_ <= congas_db:
                neu.append(h)
            else:
                raus.append(round(h[0] / SR, 2))
        ber['kick_grundmuster'] = dict(stellen=muster, je_stelle=[int(z) for z in zahl], raus=len(raus), zeiten_raus=raus)
        return neu
    except Exception as e:  # noqa: BLE001
        ber['kick_grundmuster'] = dict(fehler=str(e)[:200])
        return kh


def _wackeln_reparieren(hits, beats, teil, ber, grenze_ms=15, min_je_stelle=4):
    """23. Nov 2026 (freigegeben, Checkbox „Timing korrigieren“): nur Wackeln reparieren, Gefühl behalten.
    Raster = Viertel des Songs (nach Tempo-Prüfung) in 4 Stellen (1, e, &, a). Je Stelle der eigene typische Versatz
    dieses Elements (Median = das Gefühl, z. B. e immer +30 ms). Ein Schlag, der mehr als 15 ms von diesem Gefühl
    abweicht, wird genau auf Raster + Gefühl gesetzt; alle anderen bleiben unberührt."""
    if len(beats) < 8 or not hits:
        return hits
    st = np.array([h[0] for h in hits]); i = np.searchsorted(beats, st) - 1
    ok = (i >= 0) & (i < len(beats) - 1); i_ = np.clip(i, 0, len(beats) - 2)
    q = beats[i_ + 1] - beats[i_]; ph = (st - beats[i_]) / q * 4; k = np.round(ph).astype(int)
    rast = beats[i_] + q * k / 4; ab = (st - rast) / SR * 1000; pos = k % 4
    gef = {p: float(np.median(ab[ok & (pos == p)])) for p in range(4) if (ok & (pos == p)).sum() >= min_je_stelle}
    neu = []; n_ = 0
    for j, (s_, l_) in enumerate(hits):
        if ok[j] and pos[j] in gef and abs(ab[j] - gef[pos[j]]) > grenze_ms and abs(ph[j] - k[j]) < 0.4:
            neu.append([int(rast[j] + gef[pos[j]] / 1000 * SR), l_]); n_ += 1
        else:
            neu.append([s_, l_])
    ber.setdefault('timing_wackeln', {})[teil] = dict(repariert=n_, von=len(hits), gefuehl_ms={str(p): round(v, 1) for p, v in gef.items()})
    return neu

def ausfuehren(G, teile_roh, s53, song_pfad, eingabe, rep_ber, fb, samples, trennen_fn=None, melde=print, timing=False):
    """G: Teile von Stand G (kick/snare/toms/becken/hihat, Stereo, gleiche Länge). teile_roh: Ordner der Original-
    Trennung (kick/snare/hihat/toms/crash/ride .flac). Rückgabe (neue Teile, Bericht)."""
    n = min(len(v) for v in G.values())
    gm = {k: v[:n].mean(1) for k, v in G.items()}
    ost = lambda t: _mono(os.path.join(teile_roh, t + '.flac'), n)
    ber = {}
    melde('… Schlag für Schlag: lade die Schlagzeug-Bibliothek')
    LIB = bibliothek(samples)
    spuren = {}

    gespielt = {}
    def setze(hits, wahl, stopp=None, orig=None, teil=None):
        y = np.zeros((n + 40 * SR, 2), np.float32)
        if orig is not None and os.environ.get('SFS_AN_ORIGINAL', '1') == '1':
            hits = _an_original(hits, wahl, orig, teil, ber)
        if teil in ('kick', 'snare') and timing_raster is not None:     # Toms nicht (Fills, Triolen, Wirbel)
            hits = _wackeln_reparieren(hits, timing_raster, teil, ber)
        if teil == 'snare' and gespielt.get('kick'):
            # 26. Nov 2026 (Zion Callin, Inhaber): Kick + Rim/Snare auf DEMSELBEN Schlag (One Drop) liegen im Original genau
            # zusammen; durch Trennung/Erzeugung lagen sie 5–10 ms auseinander → klang wie ein Doppelschlag. Liegt eine
            # Snare näher als 25 ms an einer gespielten Kick, kommt sie genau auf die Kick.
            kz = np.array(sorted(k_[0] for k_ in gespielt['kick']))
            neu_h, zus = [], 0
            for (s_, lv_) in hits:
                j_ = int(np.argmin(np.abs(kz - s_))) if len(kz) else -1
                if j_ >= 0 and 0 < abs(kz[j_] - s_) < 0.025 * SR:
                    s_ = int(kz[j_]); zus += 1
                neu_h.append((s_, lv_))
            hits = neu_h
            ber['snare_auf_kick'] = zus
        for i, (s, lv) in enumerate(hits):
            smp = wahl[i]
            if smp is None or s < 0 or s >= len(y):            # 26. Nov (mix0): Schlag hinter dem Song-Ende → überspringen
                continue
            if teil:
                gespielt.setdefault(teil, []).append((int(s), float(lv), smp))   # 23. Nov: fürs MIDI (genau was klingt)
            x_ = smp['x'] * 10 ** (lv / 20)
            if teil == 'toms' and smp.get('tom_hz') is not None:
                # 23. Nov (Inhaber): eine Tom klingt bis zum nächsten Schlag auf DIESELBE Tom (der stoppt sie),
                # sonst voll bis zum Ende – nie mit der Hand gestoppt
                nx = [h_[0] for j_, h_ in enumerate(hits) if j_ > i and wahl[j_] is not None
                      and wahl[j_].get('tom_hz') == smp['tom_hz']]
                if nx and nx[0] > s:
                    f_ = int(.015 * SR); L = min(len(x_), nx[0] - s + f_); x_ = x_[:L].copy()
                    x_[max(0, L - f_):] *= np.linspace(1, 0, min(f_, L))[:, None]
            if stopp and stopp[i]:
                L = min(len(x_), stopp[i]); f_ = int(.06 * SR); x_ = x_[:L].copy()
                x_[max(0, L - f_):] *= np.linspace(1, 0, min(f_, L))[:, None]
            y[s:s + len(x_)] += x_[:len(y) - s]
        return y[:n]
    # Kick
    melde('… Schlag für Schlag: Kick, Snare, Toms')
    timing_raster = None
    if timing and song_pfad and os.path.isfile(song_pfad):        # Checkbox „Timing korrigieren“
        try:
            song_t = _mono(song_pfad, n)
            _, bt_ = librosa.beat.beat_track(y=song_t, sr=SR, units='samples', start_bpm=85)
            timing_raster, _ = _raster_pruefen(gm['hihat'], np.array(bt_))
        except Exception:
            timing_raster = None
    kh = _schlaege(gm['kick'])
    kh = _kick_grundmuster(kh, ost('kick'), _mono(s53['congas'], n) if s53.get('congas') and os.path.isfile(s53['congas']) else None,
                           _mono(song_pfad, n) if song_pfad and os.path.isfile(song_pfad) else None, ber)
    spuren['kick'] = setze(kh, _gruppen_wahl(LIB, ber, 'kick', 'kick', kh, ost('kick'), max_gr=1, tonw=1.0, k_trommel=1), orig=ost('kick'), teil='kick')
    # Snare: EINE Haupt-Snare, hohe Snare nur an G-Stellen
    sh = _schlaege(gm['snare'])
    # 26. Nov 2026 (Prüf-Sammlung, Plan Stufe 2): One Drop selbst erkennen. Liegt in der Snare-Spur an den Kick-Schlägen
    # ein klarer Anschlag (1–5 kHz höchstens 25 dB unter dem Kick-Bauch; Money −17, Zion −11, Songs ohne: −45…−52 dB),
    # spielt der Schlagzeuger Rim/Snare GENAU mit der Kick. Dann bekommt jeder solche Kick-Schlag seinen Rim/Snare-Schlag,
    # auch wenn die Trennung ihn verloren oder als Trommel aussortiert hat.
    try:
        import scipy.signal as sg
        sn_r = ost('snare'); kk_r = ost('kick')
        bnd = lambda x_, lo, hi: sg.sosfiltfilt(sg.butter(4, [lo, hi], 'band', fs=SR, output='sos'), x_)
        s_hi, k_lo = bnd(sn_r, 1000, 5000), bnd(kk_r, 40, 120)
        def _sp(x_, i_):
            w_ = x_[max(0, i_ - 200):i_ + int(.03 * SR)]
            return 20 * np.log10(np.abs(w_).max() + 1e-9) if len(w_) else -120.0
        verh = [(s_, _sp(s_hi, s_) - _sp(k_lo, s_)) for s_, _ in kh]
        med = float(np.median([v_ for _, v_ in verh])) if verh else -99.0
        od = dict(snare_an_kick_db=round(med, 1), one_drop=bool(med > -25))
        if od['one_drop']:
            st_ = np.array([x_[0] for x_ in sh]) if sh else np.zeros(0)
            neu_ = 0
            ref_s = float(np.percentile([l_ for _, l_ in sh], 90)) if sh else -20.0
            for s_, v_ in verh:
                if v_ > -25 and (not len(st_) or np.min(np.abs(st_ - s_)) > 0.025 * SR):
                    lv_ = _sp(sn_r, s_)                     # gleiche Einheit wie _schlaege (dB der Spur)
                    sh.append([int(s_), float(min(ref_s, lv_))]); neu_ += 1
            sh.sort(key=lambda h_: h_[0])
            od['dazu'] = neu_
        ber['one_drop'] = od
    except Exception as e_:  # noqa: BLE001
        ber['one_drop'] = dict(fehler=str(e_)[:200])
    hz = np.array(fb.get('snare_hoch_zeiten') or []) * SR
    ist_hoch = [bool(len(hz)) and np.min(np.abs(hz - s)) < 0.04 * SR for s, _ in sh]
    wahl_s = [None] * len(sh)
    # 26. Nov 2026 (Inhaber, Zion/Money): für den durchgehenden Grundschlag nur echte Einzelschläge (per Gehör bestätigt,
    # sfs/standard_klaenge.json) – oder Rim/Cross-Stick. Was dem Original-Schlag im Klang näher ist, gewinnt.
    try:
        std_ = json.load(open(os.path.join(samples, 'sfs', 'standard_klaenge.json'), encoding='utf-8'))
    except (OSError, ValueError):
        std_ = None
    LIB_g = LIB
    nur_g = ['Snare/1_Normal']
    if std_:
        wl_ = std_.get('snare_grundschlag') or []
        rl_ = std_.get('rim_grundschlag')
        LIB_g = [x_ for x_ in LIB if any(w_ in x_['name'] or w_ in x_['quelle'] for w_ in wl_)
                 or (x_['kat'].startswith('Rim/') and (rl_ is None or any(w_ in x_['name'] for w_ in rl_)))]
        nur_g = ['Snare/', 'Rim/']
    for name_, sel, nur_, lib_ in (('snare_haupt', [i for i in range(len(sh)) if not ist_hoch[i]], nur_g, LIB_g),
                                   ('snare_hoch', [i for i in range(len(sh)) if ist_hoch[i]], ['Snare/4_Hohe', 'Snare/2_Hell'], LIB)):
        if not sel:
            continue
        w_ = _gruppen_wahl(lib_, ber, name_, 'snare', [sh[i] for i in sel], ost('snare'), max_gr=1, k_trommel=1, nur=nur_)
        if name_ == 'snare_haupt' and (ber.get('one_drop') or {}).get('one_drop'):
            # 26. Nov 2026: beim One Drop (Kick + Schlag genau zusammen) ist es meist der Rim (Money, Zion) – Rim gewinnt,
            # wenn er im Klang höchstens 25 % weiter weg ist als die beste Snare
            try:
                b_tmp = {}
                w_r = _gruppen_wahl([x_ for x_ in lib_ if x_['kat'].startswith('Rim/')], b_tmp, name_, 'snare',
                                    [sh[i] for i in sel], ost('snare'), max_gr=1, k_trommel=1, nur=['Rim/'])
                a_s = (ber.get(name_) or {}).get('gruppe_1', {}).get('abstand')
                a_r = (b_tmp.get(name_) or {}).get('gruppe_1', {}).get('abstand')
                ber['one_drop']['abstand_snare_rim'] = [a_s, a_r]
                if a_s is not None and a_r is not None and a_r <= 1.25 * a_s and any(w_r):
                    w_ = w_r; ber[name_] = b_tmp[name_]; ber['one_drop']['rim_gewaehlt'] = True
            except Exception as e_:  # noqa: BLE001
                ber['one_drop']['rim_fehler'] = str(e_)[:200]
        for i, x_ in zip(sel, w_):
            wahl_s[i] = x_
    spuren['snare'] = setze(sh, wahl_s, orig=ost('snare'), teil='snare')
    # Toms
    th = _schlaege(gm['toms'])
    # 26. Nov 2026 (Plan Stufe 1, Prüf-Sammlung): ein Tom-Schlag zählt nur, wenn BEIDE Trennungen ihn klar hören
    # (DrumSep-Tom ≥ −20 dB und 53er-Tom ≥ −30 dB unter der Schlagzeug-Spitze) und die Hand-Trommeln dort nicht
    # deutlich lauter sind. Echte Toms (Rise Up, Zion): −5…−10 dB; Reste (Stand The Ghetto, mix0): −45…−70 dB.
    if th and s53.get('toms') and os.path.isfile(s53['toms']):
        try:
            dr_ = _mono(eingabe, n); t5_ = _mono(s53['toms'], n); ds_ = ost('toms')
            tr_ = np.zeros(n, np.float32)
            for k_ in ('congas', 'percussion'):
                if s53.get(k_) and os.path.isfile(s53[k_]):
                    tr_ = tr_ + _mono(s53[k_], n)
            def _pdb(x_, i_):
                w_ = x_[max(0, i_ - int(.01 * SR)):i_ + int(.08 * SR)]
                return 20 * np.log10(np.sqrt(np.mean(w_ ** 2)) + 1e-9) if len(w_) else -120.0
            ref_ = float(np.percentile([_pdb(dr_, i_) for i_ in range(0, max(1, n - SR), SR // 8)], 99))
            behalten, raus_ = [], []
            for (s_, lv_) in th:
                a_, b_, c_ = _pdb(ds_, s_) - ref_, _pdb(t5_, s_) - ref_, _pdb(tr_, s_) - ref_
                if a_ >= -20 and b_ >= -30 and c_ - max(a_, b_) < 6:
                    behalten.append((s_, lv_))
                else:
                    raus_.append(round(s_ / SR, 2))
            ber['toms_bestaetigt'] = dict(vorher=len(th), behalten=len(behalten), raus_s=raus_[:60])
            th = behalten
        except Exception as e_:  # noqa: BLE001
            ber['toms_bestaetigt'] = dict(fehler=str(e_)[:200])
    spuren['toms'] = setze(th, _toms_wahl(LIB, ber, th, ost('toms')), orig=ost('toms'), teil='toms') if th else np.zeros((n, 2), np.float32)
    # Becken (vorerst wie G: Art aus G, aber voll ausklingend, Lautstärke über den Ausklang)
    melde('… Schlag für Schlag: Becken (voll ausklingend)')
    bx = gm['becken']; bt = fb.get('becken_gespielt') or []
    shots = {round(z[0], 1): z[1] for z in (rep_ber.get('becken') or {}).get('schlaege(zeit,shot,hoehen_db,abstand)', [])}
    song = _mono(song_pfad, n) if song_pfad and os.path.isfile(song_pfad) else None
    bh, wb = [], []
    cache = {}
    for t in bt:
        s = int(t * SR); s0 = max(0, s - int(.03 * SR)); s1 = s0 + int(.08 * SR)
        seg = np.abs(bx[s0:s1])
        if not len(seg) or seg.max() <= 0:
            continue
        j = s0 + int(np.argmax(seg > seg.max() * 0.3))
        nm = shots[min(shots, key=lambda k: abs(k - t))] if shots else None
        p_ = os.path.join(samples, 'bibliothek', 'Akustik', nm) if nm else None
        if not p_ or not os.path.isfile(p_):
            continue
        if nm not in cache:
            x_ = _lade(p_); a_ = np.abs(x_).max(1); x_ = x_[max(0, int(np.argmax(a_ > a_.max() * 0.1)) - 44):]; x_ /= np.abs(x_).max()
            cache[nm] = dict(x=x_, name=nm)
        rms = lambda a: 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-12)
        a0, a1 = int(.3 * SR), int(.8 * SR)
        bh.append([j, rms(bx[j + a0:j + a1]) - rms(cache[nm]['x'].mean(1)[a0:a1])]); wb.append(cache[nm])
    # 22. Nov abends (Inhaber): abgestoppt wird nur, was im ORIGINAL wirklich aufhört. Entscheidend ist das getrennte
    # Original-Becken (crash+ride) selbst – Gesang oder Percussion, die weiterlaufen, spielen keine Rolle.
    #  a) mit der Hand gehalten: kurz nach dem Schlag (bis 0,6 s) fällt das Original-Becken steil (≥ 15 dB in 100 ms)
    #     und bleibt ≥ 20 dB unter seiner Spitze, bis es 0,3 s still ist oder der nächste Schlag kommt
    #  b) ganze Song-Pause (wie bisher) – aber nur, wenn auch das Original-Becken dort ≥ 20 dB unter der Spitze ist
    stopp = [0] * len(bh); stopp_grund = [None] * len(bh)
    ob = None
    try:
        teile_b = [t for t in ('crash', 'ride') if os.path.isfile(os.path.join(teile_roh, t + '.flac'))]
        ob = sum(ost(t) for t in teile_b) if teile_b else None
    except Exception:
        ob = None
    if ob is not None and bh:
        H_ = int(.01 * SR)
        oenv = 20 * np.log10(np.array([np.sqrt(np.mean(ob[q:q + H_] ** 2)) + 1e-9 for q in range(0, len(ob) - H_, H_)]))
        senv = None
        if song is not None:
            senv = 20 * np.log10(np.array([np.sqrt(np.mean(song[q:q + H_] ** 2)) + 1e-9 for q in range(0, len(song) - H_, H_)]))
        for i, (s_, _) in enumerate(bh):
            a = s_ // H_; nx = (bh[i + 1][0] // H_) if i + 1 < len(bh) else min(len(oenv), a + 1000)
            nx = min(nx, len(oenv))
            if nx - a < 15:
                continue
            spitze = float(oenv[a:a + 5].max())
            for q in range(a + 3, min(a + 60, nx - 10)):            # a) Abstoppen mit der Hand
                if oenv[q] - oenv[q + 10] >= 15 and (oenv[q + 10:min(nx, q + 40)] < spitze - 20).all():
                    stopp[i] = (q + 3 - a) * H_; stopp_grund[i] = 'mit der Hand gehalten'; break
            if stopp[i] or senv is None:
                continue
            vor = np.median(senv[max(0, a - 200):a]) if a > 20 else senv[min(a, len(senv) - 1)]
            for q in range(a + 15, min(len(senv) - 40, nx - 40, a + 1000)):   # b) ganze Song-Pause
                if (senv[q:q + 40] < vor - 25).all() and (oenv[q:q + 40] < spitze - 20).all():
                    stopp[i] = (q - a) * H_; stopp_grund[i] = 'Song-Pause'; break
    spuren['becken'] = setze(bh, wb, stopp, teil='becken') if bh else np.zeros((n, 2), np.float32)
    ber['becken'] = dict(schlaege=len(bh), klaenge=sorted(cache), abgestoppt_s=[round(v / SR, 2) if v else None for v in stopp], abgestoppt_grund=stopp_grund)
    # Hi-Hat-Hörmodell
    melde('… Hi-Hat-Hörmodell: höre den ganzen Song (Muster, Gefühl, Timing)')
    hh_g = _schlaege(gm['hihat'], delta=0.05, unter=35)
    ref90 = float(np.percentile([l for _, l in hh_g], 90)) if hh_g else -20.0
    andere = [s for s, _ in kh] + [s for s, _ in sh]
    song_hm = song if song is not None else _mono(eingabe, n)
    q53 = _mono(s53['hh'], n) if s53.get('hh') and os.path.isfile(s53['hh']) else ost('hihat')
    per = _mono(s53['percussion'], n) if s53.get('percussion') and os.path.isfile(s53['percussion']) else np.zeros(n, np.float32)
    tamb = _mono(s53['tambourine'], n) if s53.get('tambourine') and os.path.isfile(s53['tambourine']) and \
        s53.get('hh') and os.path.isfile(s53['hh']) else None       # 25. Nov: Tamburin nur mit echter 53er-Hi-Hat
    HP = hihat_hoermodell(song_hm, q53, ost('hihat'), per, andere, samples, melde, tamb=tamb, kick=[s for s, _ in kh])
    hh_y, hh_info = _hh_spielen(HP, ref90, n)
    korr = {}
    if trennen_fn is not None and HP['plan']:             # Selbstprüfung: Ausschnitt trennen, Pegel je Stelle angleichen
        try:
            melde('… Hi-Hat-Selbstprüfung (Ausschnitt trennen und nachhören)')
            beats = np.array(HP['beats']); L = min(n, 30 * SR); a0 = max(0, (n - L) // 2)
            mix = (sum(v for k, v in spuren.items()) + hh_y)[a0:a0 + L]
            mix = mix * 0.98 / (np.abs(mix).max() + 1e-9)
            hh_neu = trennen_fn(mix)                       # -> getrennte Hi-Hat (mono) des Ausschnitts
            bb = beats[(beats >= a0) & (beats < a0 + L)] - a0
            AND = np.array([s - a0 for s in andere if a0 <= s < a0 + L])
            B_o = _hh_bild(ost('hihat')[a0:a0 + L], bb, AND); B_n = _hh_bild(hh_neu[:L], bb, AND)
            lauteste = max(B_o, key=B_o.get) if B_o else None
            if lauteste is not None and lauteste in B_n:
                for k in B_o:
                    if k != lauteste and k in B_n:
                        korr[str(k)] = float(np.clip((B_o[k] - B_o[lauteste]) - (B_n[k] - B_n[lauteste]), -8, 8))
                hh_y, hh_info = _hh_spielen(HP, ref90, n, korr)
        except Exception as e:  # noqa: BLE001 – ohne Selbstprüfung weiter
            melde(f'… Hi-Hat-Selbstprüfung übersprungen ({str(e)[:120]})')
    spuren['hihat'] = hh_y
    gespielt['hihat'] = hh_info.pop('_schlaege', [])
    # 28. Nov 2026 (Money, Inhaber: „Timing stimmt an manchen Stellen nicht“): jede Spur wird nachgehört. Der Anschlag
    # (50 % der Hülle) jedes klaren Schlags wird mit dem Original verglichen; der typische Versatz (Median, kommt vom
    # Anschlag der Aufnahme, nicht vom Schlagzeuger) wird für die ganze Spur ausgeglichen.
    try:
        ref_ = dict(kick=(ost('kick'), 40, 150), snare=(ost('snare'), 1000, 5000), hihat=(q53, 6000, 14000))
        vs_ = {}
        for t_, (o_, lo_, hi_) in ref_.items():
            if t_ not in spuren:
                continue
            d_ = _anschlag_versatz(spuren[t_].mean(1), o_, lo_, hi_)
            if t_ == 'snare' and d_ is not None and ber.get('snare_auf_kick', 0) >= 0.3 * max(1, len(gespielt.get('snare', []))):
                # Kick + Rim/Snare auf demselben Schlag bleiben GENAU zusammen (Inhaber, Zion: sonst Doppelschlag) –
                # die Snare bekommt dann denselben Ausgleich wie die Kick
                d_ = dict(d_, median_ms=(vs_.get('kick') or {}).get('median_ms', 0.0), wie_kick=True)
            vs_[t_] = d_
            if d_ is not None and 2.0 <= abs(d_['median_ms']) <= 25.0:
                sh_ = -int(round(d_['median_ms'] / 1000 * SR))
                y_ = np.zeros_like(spuren[t_])
                if sh_ < 0:
                    y_[:sh_] = spuren[t_][-sh_:]
                else:
                    y_[sh_:] = spuren[t_][:len(y_) - sh_]
                spuren[t_] = y_
                gespielt[t_] = [(int(g_[0]) + sh_,) + tuple(g_[1:]) for g_ in gespielt.get(t_, [])]
                d_['ausgeglichen_ms'] = round(-sh_ / SR * 1000, 1)
        ber['timing_selbstpruefung'] = vs_
    except Exception as e_:  # noqa: BLE001
        ber['timing_selbstpruefung'] = dict(fehler=str(e_)[:200])
    ber['_gespielt'] = gespielt          # 23. Nov: genau die gespielten Schläge -> MIDI (wird vor dem Bericht entfernt)
    ber['hihat'] = dict(muster=hh_info, korrektur_db=korr, tempo=round(HP['tempo'], 1), schlaege=len(HP['plan']), stellen_53er_db=HP.get('stellen_53er_db'), raster=HP.get('raster'), tamburin=HP.get('tamburin'), katalog=HP.get('katalog'), zaehlung=HP.get('zaehlung'),
                       hh_53er_luecken_s=HP.get('hh_53er_luecken_s'))
    return spuren, ber
