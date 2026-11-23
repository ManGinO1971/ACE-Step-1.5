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
    """Grundton einer Tom (pyin über den Ausklang 20–300 ms; sonst stärkste Spitze im Spektrum)."""
    w = m[s + int(.02 * SR):s + int(.30 * SR)]
    if len(w) > 2048:
        try:
            f0, v, _ = librosa.pyin(w, fmin=40, fmax=450, sr=SR, frame_length=2048, hop_length=256)
            f0 = f0[v & np.isfinite(f0)]
            if len(f0) >= 3:
                return float(np.median(f0))
        except Exception:
            pass
    return max(_merkmale(m, s)[1], 40.0)


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
        if 'st' not in s_:
            s_['st'] = 12 * np.log2(_tom_ton(s_['x'].mean(1), 0) / 100.0)
    vergeben = set(); wahl = [None] * len(hits); info = {}; fehlt = []
    for nr, g_ in enumerate(gruppen):
        idx = np.where(grp == g_)[0]; ziel = float(np.median(st[idx])); hz = 100 * 2 ** (ziel / 12)
        frei = [s_ for s_ in P if s_['name'] not in vergeben] or P
        # Ton zuerst: alle Aufnahmen, die sich höchstens ±3 Halbtöne auf den Ton stimmen lassen; darunter entscheidet
        # die Klangfarbe am Anschlag (wie bisher bei Rise Up). Sonst die mit dem nächsten Ton.
        nah = [s_ for s_ in frei if abs(s_['st'] - ziel) <= max_stimmen]
        mo = [_merkmale(orig, hits_o[i]) for i in idx]
        ref_ton = float(np.median([m_[1] for m_ in mo]))
        if nah:
            haupt = min(nah, key=lambda s_: (np.mean([_abst(m_, s_['m'], tonw=3.0) for m_ in mo]), s_['name']))
        else:
            haupt = min(frei, key=lambda s_: (abs(s_['st'] - ziel), s_['name']))
        lagen = [s_ for s_ in P if s_['kat'] == haupt['kat'] and abs(s_['st'] - haupt['st']) <= 1.0 and s_['name'] not in vergeben][:4] or [haupt]
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
        info[f'tom_{nr + 1}'] = dict(schlaege=len(idx), ton_original_hz=round(hz), art=haupt['kat'], dateien=[t['name'] for t in tr])
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

def hihat_hoermodell(song, hh53, hhds, per, andere, samples, melde=print):
    """Muster der Hi-Hat aus dem ganzen Song. Rückgabe dict(muster, plan, tempo) wie im Test (plan2.json)."""
    n = min(map(len, (song, hh53, hhds, per)))
    song, S53, DS, PE = song[:n], hh53[:n], hhds[:n], per[:n]
    HH = 0.5 * (S53 / (np.abs(S53).max() + 1e-9) + DS / (np.abs(DS).max() + 1e-9)); PEn = PE / (np.abs(S53).max() + 1e-9) * 0.5
    AND = np.array(andere)
    tempo, beats = librosa.beat.beat_track(y=song, sr=SR, units='samples', start_bpm=85); beats = np.array(beats)
    env = np.abs(HH)
    raster_info = None
    if os.environ.get('HH_RASTER', '1') == '1':
        beats, raster_info = _raster_pruefen(HH, beats)
        tempo = 60 * SR / float(np.median(np.diff(beats)))

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
    s53 = np.median(p53[aktiv], 0); s53 = s53 - s53.max()
    stellen = {}
    for k in range(4):
        idx = [i for i in range(len(R)) if aktiv[i] and lv[i, k] > gmax - 30]
        if len(idx) < 0.5 * aktiv.sum():
            continue
        if s53[k] <= -18:
            continue                                        # die 53er-Hi-Hat hört hier (fast) nichts – keine Hi-Hat
        if np.mean([stoert(R[i][k][0]) for i in idx]) > 0.6 and s53[k] <= -12:
            continue                                        # Übersprechen von Kick/Snare – keine Hi-Hat
        stellen[k] = idx
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
        kand = sorted(((ab(p), p['name'], p['kat'], p['quelle']) for p in kette))
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
    plan = []
    for i in range(len(R)):
        if not aktiv[i]:
            continue
        for k in ks:
            m = muster[k]; ziel = beats[i] + (beats[i + 1] - beats[i]) * k / 4 + m['timing_ms'] / 1000 * SR
            eig = float(np.clip(R[i][k][0] - ziel, -.010 * SR, .010 * SR)) if lv[i, k] > gmax - 30 and not stoert(R[i][k][0]) else 0.0
            pz = m['pegel_je_zaehlzeit'].get(i % 4, m['pegel'])
            dl = float(np.clip(lv[i, k] - gmax - pz, -1.5, 1.5)) if lv[i, k] > gmax - 30 and not stoert(R[i][k][0]) else 0.0
            plan.append(dict(t=int(ziel + 0.8 * eig), stelle=int(k), pegel=pz + dl))
    return dict(muster={str(k): v for k, v in muster.items()}, plan=plan, tempo=float(np.atleast_1d(tempo)[0]),
                raster=raster_info, beats=[int(b) for b in beats], stellen_53er_db=[round(float(v), 1) for v in s53])


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
        for i, (s, lv) in enumerate(hits):
            smp = wahl[i]
            if smp is None:
                continue
            if teil:
                gespielt.setdefault(teil, []).append((int(s), float(lv), smp))   # 23. Nov: fürs MIDI (genau was klingt)
            x_ = smp['x'] * 10 ** (lv / 20)
            if stopp and stopp[i]:
                L = min(len(x_), stopp[i]); f_ = int(.06 * SR); x_ = x_[:L].copy()
                x_[max(0, L - f_):] *= np.linspace(1, 0, min(f_, L))[:, None]
            y[s:s + len(x_)] += x_
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
    spuren['kick'] = setze(kh, _gruppen_wahl(LIB, ber, 'kick', 'kick', kh, ost('kick'), max_gr=1, tonw=1.0, k_trommel=1), orig=ost('kick'), teil='kick')
    # Snare: EINE Haupt-Snare, hohe Snare nur an G-Stellen
    sh = _schlaege(gm['snare'])
    hz = np.array(fb.get('snare_hoch_zeiten') or []) * SR
    ist_hoch = [bool(len(hz)) and np.min(np.abs(hz - s)) < 0.04 * SR for s, _ in sh]
    wahl_s = [None] * len(sh)
    for name_, sel, nur_ in (('snare_haupt', [i for i in range(len(sh)) if not ist_hoch[i]], ['Snare/1_Normal']),
                             ('snare_hoch', [i for i in range(len(sh)) if ist_hoch[i]], ['Snare/4_Hohe', 'Snare/2_Hell'])):
        if not sel:
            continue
        w_ = _gruppen_wahl(LIB, ber, name_, 'snare', [sh[i] for i in sel], ost('snare'), max_gr=1, k_trommel=1, nur=nur_)
        for i, x_ in zip(sel, w_):
            wahl_s[i] = x_
    spuren['snare'] = setze(sh, wahl_s, orig=ost('snare'), teil='snare')
    # Toms
    th = _schlaege(gm['toms'])
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
    HP = hihat_hoermodell(song_hm, q53, ost('hihat'), per, andere, samples, melde)
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
    ber['_gespielt'] = gespielt          # 23. Nov: genau die gespielten Schläge -> MIDI (wird vor dem Bericht entfernt)
    ber['hihat'] = dict(muster=hh_info, korrektur_db=korr, tempo=round(HP['tempo'], 1), schlaege=len(HP['plan']), stellen_53er_db=HP.get('stellen_53er_db'), raster=HP.get('raster'))
    return spuren, ber
