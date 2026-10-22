"""HIELTECH-Anpassungen für Seed-VC inference.py (mehrfach ausführbar).
1. Speichern über soundfile (torchcodec fehlt auf dem Mac)
2. Stimmprobe 15 s (Entscheidung Inhaber 22. Okt 2026)
3. Oktav-Sperre wie beim Lead (aus: HIELTECH_SVC_LOCK=0)
4. Backing-Lage (HIELTECH_SVC_LAGE): original | oktave | terz | quarte | quinte | sexte, jeweils auch _unten (z.B. terz_unten)
   terz/quinte = Harmonie in der Tonart (Standard A-Moll, HIELTECH_SVC_TONART=a-moll),
   Vibrato und Gleiten bleiben unverändert, nur der Ton wird tonartgerecht verschoben.
7. Ziel-Datei: HIELTECH_SVC_F0_OUT=<pfad.npy> speichert die endgültige Zielmelodie (für den Vocoder)
6. Oktave nach der Lage: HIELTECH_SVC_OKTAVE (z.B. -1 für den Mann)
5. Stabil (vor der Lage): Ausreißer-Reparatur wie Lead, Ton-Halt gegen Leiern (HIELTECH_SVC_HALT, Standard 0.3),
   ruhige Übergänge tief<->hoch (HIELTECH_SVC_GLEIT=1). Vibrato 4-8,5 Hz bleibt voll erhalten.
"""
import re

OKTAVE = r'''        # HIELTECH-Oktave
        # HIELTECH_SVC_OKTAVE: ganze Oktaven NACH der Lage (Mann z.B. -1 = Terz eine Oktave tiefer), Standard 0
        _okt = int(os.environ.get("HIELTECH_SVC_OKTAVE", "0"))
        if _okt != 0:
            _f = F0_alt[0].detach().cpu().numpy().astype("float64")
            _f = np.where(_f > 1, _f * 2.0 ** _okt, _f)
            print(f"HIELTECH Oktave: {_okt:+d}")
            F0_alt = torch.from_numpy(_f).to(F0_alt.device, F0_alt.dtype)[None]
'''
ZIELDUMP = r'''        # HIELTECH-Ziel-Datei
        # HIELTECH_SVC_F0_OUT: die endgültige Zielmelodie (10-ms-Raster) speichern – für den Vocoder danach
        _zout = os.environ.get("HIELTECH_SVC_F0_OUT", "")
        if _zout:
            np.save(_zout, F0_alt[0].detach().cpu().numpy().astype("float64"))
'''
STABIL = r'''        # HIELTECH-Stabil
        # wie beim Lead (Melodie-Reparatur) + Ton-Halt gegen "Leiern" + saubere Übergänge.
        # HIELTECH_SVC_HALT: Anteil des langsamen Wackelns im gehaltenen Ton, der bleibt (1 = aus, Standard 0.3)
        # HIELTECH_SVC_GLEIT: 1 = Übergänge tief<->hoch als ruhige Gleitkurve (Standard 1)
        _halt = float(os.environ.get("HIELTECH_SVC_HALT", "0.3"))
        _gleit = os.environ.get("HIELTECH_SVC_GLEIT", "1") != "0"
        if _halt < 1.0 or _gleit:
            from scipy.ndimage import median_filter as _medf
            from scipy.signal import butter as _butter, sosfiltfilt as _sff
            _f = F0_alt[0].detach().cpu().numpy().astype("float64")
            _v = _f > 1
            _m = np.where(_v, 69 + 12 * np.log2(np.maximum(_f, 1e-6) / 440.0), np.nan)
            _n = len(_m)
            # 1. Aussetzer wie beim Lead: kurze Ausreißer (<= 80 ms, > 2 HT neben 120-ms-Umgebung) glätten
            _rep = 0
            for _i in np.flatnonzero(_v):
                _w = _m[max(0, _i - 6):_i + 7]
                _w = _w[~np.isnan(_w)]
                if len(_w) >= 5 and abs(_m[_i] - np.median(_w)) > 2.0:
                    _m[_i] = np.median(_w)
                    _rep += 1
            # 2. Töne finden (gehalten >= 120 ms)
            _toene = []
            _i = 0
            while _i < _n:
                if _v[_i]:
                    _j = _i
                    while _j < _n and _v[_j]:
                        _j += 1
                    _q = np.round(_medf(_m[_i:_j], 7, mode="nearest"))
                    _k = 0
                    while _k < len(_q):
                        _l = _k
                        while _l < len(_q) and _q[_l] == _q[_k]:
                            _l += 1
                        if _l - _k >= 12:
                            _toene.append((_i + _k, _i + _l))
                        _k = _l
                    _i = _j
                else:
                    _i += 1
            _sos_v = _butter(2, [4.0, 8.5], btype="band", fs=100, output="sos")
            _neu = _m.copy()
            _mitte = {}
            for _a, _b in _toene:
                _x = _m[_a:_b]
                _c = float(np.median(_x))
                _vib = _sff(_sos_v, _x) if _b - _a > 15 else np.zeros_like(_x)
                _slow = _x - _c - _vib
                _neu[_a:_b] = _c + _vib + _halt * _slow
                _mitte[_a] = _c
                _mitte[_b - 1] = _c
            # 3. Übergänge zwischen zwei gehaltenen Tönen (<= 250 ms, durchgehend gesungen): ruhige Gleitkurve
            _gl = 0
            if _gleit:
                for (_a1, _b1), (_a2, _b2) in zip(_toene[:-1], _toene[1:]):
                    _len = _a2 - _b1
                    if 0 < _len <= 25 and _v[_b1 - 1:_a2 + 1].all():
                        _c1 = _neu[_b1 - 1]
                        _c2 = _neu[_a2]
                        _t = (np.arange(1, _len + 1)) / (_len + 1)
                        _neu[_b1:_a2] = _c1 + (_c2 - _c1) * (_t * _t * (3 - 2 * _t))
                        _gl += 1
            _f = np.where(_v, 440.0 * 2 ** ((_neu - 69) / 12.0), _f)
            print(f"HIELTECH Stabil: {len(_toene)} Töne gehalten (Wackeln x{_halt:.2f}), {_gl} Übergänge geglättet, {_rep} Ausreißer repariert")
            F0_alt = torch.from_numpy(_f).to(F0_alt.device, F0_alt.dtype)[None]
'''

p = 'inference.py'
# immer vom unveränderten Seed-VC-Original ausgehen (git), damit neue Fassungen der Blöcke greifen
import subprocess
try:
    s = subprocess.check_output(['git', 'show', 'HEAD:inference.py'], text=True)
except Exception:
    s = open(p).read()
    assert '# HIELTECH-' not in s, 'Seed-VC-Original nicht gefunden (git) – bitte melden'
s = s.replace('vc_wave.cpu(), sr)', 'vc_wave.cpu().numpy().T, sr)')
s = s.replace('    torchaudio.save(os.path.join', '    __import__("soundfile").write(os.path.join')
s = re.sub(r'ref_audio\[:sr \* \d+\]', 'ref_audio[:sr * 15]', s)
MARK = '# HIELTECH-Oktav-Sperre'
if MARK not in s:
    anchor = '        F0_alt = torch.from_numpy(F0_alt).to(device)[None]\n'
    assert anchor in s, 'Ankerzeile nicht gefunden'
    s = s.replace(anchor, anchor + '''        ''' + MARK + '''
        if os.environ.get("HIELTECH_SVC_LOCK", "1") != "0":
            _f = F0_alt[0].detach().cpu().numpy().astype("float64")
            _v = _f > 1
            _st = np.where(_v, 12 * np.log2(np.maximum(_f, 1e-6) / 440.0), np.nan)
            _fix = 0
            for _i in np.flatnonzero(_v):
                _w = _st[max(0, _i - 25):_i + 26]
                _w = _w[~np.isnan(_w)]
                if len(_w) < 10:
                    continue
                _d = _st[_i] - np.median(_w)
                if abs(abs(_d) - 12) < 1.5:
                    _f[_i] = _f[_i] * (0.5 if _d > 0 else 2.0)
                    _fix += 1
            print(f"HIELTECH Oktav-Sperre: {_fix} Frames zurückgeklappt")
            F0_alt = torch.from_numpy(_f).to(F0_alt.device, F0_alt.dtype)[None]
''')
MARK2 = '# HIELTECH-Lage'
if MARK2 not in s:
    anchor2 = '        voiced_F0_ori = F0_ori[F0_ori > 1]\n'
    assert anchor2 in s, 'Ankerzeile 2 nicht gefunden'
    s = s.replace(anchor2, '''        ''' + MARK2 + '''
        _lage = os.environ.get("HIELTECH_SVC_LAGE", "original").lower()
        if _lage != "original":
            _f = F0_alt[0].detach().cpu().numpy().astype("float64")
            _v = _f > 1
            _m = np.where(_v, 69 + 12 * np.log2(np.maximum(_f, 1e-6) / 440.0), np.nan)
            if _lage == "oktave":
                _sh = np.where(_v, 12.0, 0.0)
            else:
                _ton = os.environ.get("HIELTECH_SVC_TONART", "a-moll").lower()
                _namen = {"c": 0, "cis": 1, "des": 1, "d": 2, "dis": 3, "es": 3, "e": 4, "f": 5, "fis": 6, "ges": 6,
                          "g": 7, "gis": 8, "as": 8, "a": 9, "ais": 10, "b": 10, "h": 11}
                _g, _art = _ton.split("-")
                _root = _namen[_g]
                _iv = [0, 2, 3, 5, 7, 8, 10] if _art.startswith("m") else [0, 2, 4, 5, 7, 9, 11]
                _skala = sorted({(_root + i) % 12 for i in _iv})
                _unten = _lage.endswith("_unten")
                _schritte = {"terz": 2, "quarte": 3, "quinte": 4, "sexte": 5}[_lage.replace("_unten", "")]
                # Ton je Frame aus geglättetem Verlauf (Vibrato ändert den Ton nicht)
                _glatt = _m.copy()
                for _i in np.flatnonzero(_v):
                    _w = _m[max(0, _i - 12):_i + 13]
                    _w = _w[~np.isnan(_w)]
                    _glatt[_i] = np.median(_w)
                _sh = np.zeros_like(_m)
                for _i in np.flatnonzero(_v):
                    _n = int(round(_glatt[_i]))
                    _pc = _n % 12
                    _k = min(range(7), key=lambda j: min((_pc - _skala[j]) % 12, (_skala[j] - _pc) % 12))
                    _basis = _n + ((_skala[_k] - _pc + 6) % 12 - 6)
                    if _unten:
                        _ziel_pc = _skala[(_k - _schritte) % 7]
                        _ziel = _basis - ((_skala[_k] - _ziel_pc) % 12)
                    else:
                        _ziel_pc = _skala[(_k + _schritte) % 7]
                        _ziel = _basis + ((_ziel_pc - _skala[_k]) % 12)
                    _sh[_i] = _ziel - _basis
                # Wechsel des Abstands nur an echten Tonwechseln: kurze Ausreißer (< 60 ms) glätten
                _sh_v = _sh.copy()
                _idx = np.flatnonzero(_v)
                for _a in range(len(_idx)):
                    _i = _idx[_a]
                    _w = _sh[max(0, _i - 3):_i + 4][_v[max(0, _i - 3):_i + 4]]
                    _vals, _cnt = np.unique(_w, return_counts=True)
                    _sh_v[_i] = _vals[np.argmax(_cnt)]
                _sh = _sh_v
            _f = np.where(_v, _f * 2 ** (_sh / 12.0), _f)
            print(f"HIELTECH Lage: {_lage} (Verschiebung im Mittel {np.mean(_sh[_v]):+.1f} Halbtöne)")
            F0_alt = torch.from_numpy(_f).to(F0_alt.device, F0_alt.dtype)[None]
''' + anchor2)
MARK3 = '# HIELTECH-Stabil'
if MARK3 not in s:
    anchor3 = '        # HIELTECH-Lage\n'
    assert anchor3 in s, 'Ankerzeile 3 nicht gefunden'
    s = s.replace(anchor3, STABIL + anchor3)
MARK4 = '# HIELTECH-Oktave'
if MARK4 not in s:
    anchor4 = '        voiced_F0_ori = F0_ori[F0_ori > 1]\n'
    s = s.replace(anchor4, OKTAVE + anchor4)
MARK5 = '# HIELTECH-Ziel-Datei'
if MARK5 not in s:
    anchor5 = '        voiced_F0_ori = F0_ori[F0_ori > 1]\n'
    s = s.replace(anchor5, ZIELDUMP + anchor5)
open(p, 'w').write(s)
print('Patch OK:', 'soundfile' in s, 'sr * 15' in s, MARK in s, MARK2 in s, MARK3 in s, MARK4 in s, MARK5 in s)
