"""
HELTECH Stems – eigener SoundFont-Abspieler (SF2), reines numpy/scipy, ohne Zusatzprogramm.

Spielt Noten (Start, Ende, Tonhöhe, Anschlag) mit einem Instrument aus einer SF2-Datei (z. B. GeneralUser GS):
Zonen nach Ton-/Anschlagbereich, Stereo-Samples, Schleifen (Sustain), Lautstärke-Hüllkurve
(Delay/Attack/Hold/Decay/Sustain/Release inkl. Taste→Decay), Grundstimmung, Dämpfung, Panorama,
Tiefpassfilter (initialFilterFc) und die Standard-Anschlagskurve.
"""
import numpy as np

SR = 44100
G = dict(start=0, end=1, loopstart=2, loopend=3, startcoarse=4, fc=8, q=9, endcoarse=12, pan=17,
         delay=33, attack=34, hold=35, decay=36, sustain=37, release=38, key2hold=39, key2decay=40,
         instrument=41, keyrange=43, velrange=44, loopstartcoarse=45, keynum=46, velocity=47, atten=48,
         loopendcoarse=50, coarse=51, fine=52, sampleid=53, modes=54, scale=56, excl=57, root=58)
STANDARD = {G["delay"]: -12000, G["attack"]: -12000, G["hold"]: -12000, G["decay"]: -12000, G["release"]: -12000,
            G["sustain"]: 0, G["atten"]: 0, G["pan"]: 0, G["coarse"]: 0, G["fine"]: 0, G["scale"]: 100,
            G["fc"]: 13500, G["q"]: 0, G["key2hold"]: 0, G["key2decay"]: 0, G["modes"]: 0}
ADDITIV = {G[k] for k in ("delay", "attack", "hold", "decay", "release", "sustain", "atten", "pan", "coarse",
                          "fine", "fc", "q", "key2hold", "key2decay")}


def tc2s(tc):
    return 0.0 if tc <= -12000 else 2.0 ** (tc / 1200.0)


class SoundFont:
    def __init__(self, pfad):
        from sf2_lesen import SF2
        self.sf = SF2(pfad)

    def presets(self):
        return self.sf.presets()

    def instrument(self, bank, programm):
        return Instrument(self, bank, programm)


class Instrument:
    def __init__(self, sf, bank, programm):
        self.sf = sf
        self.name, pz = sf.sf.preset_zonen(bank, programm)
        self.zonen = []                       # (keylo,keyhi,vello,velhi, gens, sample)
        pglobal = {}
        if pz and G["instrument"] not in pz[0]:
            pglobal = pz[0]
            pz = pz[1:]
        for pzone in pz:
            if G["instrument"] not in pzone:
                continue
            pg = dict(pglobal)
            pg.update(pzone)
            pk = pg.get(G["keyrange"], (0, 127))
            pv = pg.get(G["velrange"], (0, 127))
            iz = sf.sf.instrument_zonen(pzone[G["instrument"]])
            iglobal = {}
            if iz and G["sampleid"] not in iz[0]:
                iglobal = iz[0]
                iz = iz[1:]
            for izone in iz:
                if G["sampleid"] not in izone:
                    continue
                ig = dict(STANDARD)
                ig.update(iglobal)
                ig.update(izone)
                ik = ig.get(G["keyrange"], (0, 127))
                iv = ig.get(G["velrange"], (0, 127))
                for k, v in pg.items():          # Preset-Werte wirken additiv
                    if k in ADDITIV:
                        ig[k] = ig.get(k, STANDARD.get(k, 0)) + v
                kl, kh = max(pk[0], ik[0]), min(pk[1], ik[1])
                vl, vh = max(pv[0], iv[0]), min(pv[1], iv[1])
                if kl <= kh and vl <= vh:
                    self.zonen.append((kl, kh, vl, vh, ig, sf.sf.samples[izone[G["sampleid"]]]))

    def note(self, taste, anschlag, dauer, nachklang_max=6.0):
        """Rendert eine Note (Stereo, SR). dauer = gehaltene Zeit in s (bis Note aus)."""
        from scipy.signal import lfilter
        teile = []
        for kl, kh, vl, vh, g, s in self.zonen:
            if not (kl <= taste <= kh and vl <= anschlag <= vh):
                continue
            daten = self.sf.sf.smpl
            a0 = g.get(G["start"], 0) + 32768 * g.get(G["startcoarse"], 0)
            e0 = g.get(G["end"], 0) + 32768 * g.get(G["endcoarse"], 0)
            ls0 = g.get(G["loopstart"], 0) + 32768 * g.get(G["loopstartcoarse"], 0)
            le0 = g.get(G["loopend"], 0) + 32768 * g.get(G["loopendcoarse"], 0)
            anf = s.start + max(0, a0)                 # absolute Positionen im smpl-Block
            end = min(len(daten), s.end + e0)
            ls = s.start_loop + ls0
            le = s.end_loop + le0
            modus = g.get(G["modes"], 0) & 3
            schleife = modus in (1, 3) and le - ls > 8
            root = g.get(G["root"], -1)
            root = s.original_pitch if root is None or root < 0 else root
            cent = (taste - root) * g.get(G["scale"], 100) + 100 * g.get(G["coarse"], 0) + g.get(G["fine"], 0) \
                + s.pitch_correction
            schritt = 2.0 ** (cent / 1200.0) * s.sample_rate / SR
            # Hüllkurve
            t_del = tc2s(g[G["delay"]])
            t_att = tc2s(g[G["attack"]])
            t_hold = tc2s(g[G["hold"]] + (60 - taste) * g.get(G["key2hold"], 0))
            t_dec = tc2s(g[G["decay"]] + (60 - taste) * g.get(G["key2decay"], 0))
            sus_db = -min(max(g[G["sustain"]], 0), 1440) / 10.0
            t_rel = min(tc2s(g[G["release"]]), nachklang_max)
            n_an = int(dauer * SR)
            n = n_an + int((t_rel + 0.02) * SR)
            t = np.arange(n) / SR
            env_db = np.full(n, -100.0)
            # Attack (linear in Amplitude)
            a = (t >= t_del) & (t < t_del + t_att)
            env = np.zeros(n)
            env[a] = (t[a] - t_del) / max(t_att, 1e-6)
            h = (t >= t_del + t_att)
            env_db[h] = 0.0
            d0 = t_del + t_att + t_hold
            d = t >= d0
            if t_dec > 0:
                env_db[d] = np.maximum(sus_db, -100.0 * (t[d] - d0) / t_dec)   # Decay: 100 dB in t_dec
            else:
                env_db[d] = sus_db
            env = np.where(a, env, 10 ** (env_db / 20))
            # Release ab Note aus
            if n_an < n:
                lvl = env[n_an - 1] if n_an > 0 else 0.0
                lvl_db = 20 * np.log10(max(lvl, 1e-5))
                r = np.arange(n - n_an) / SR
                env[n_an:] = 10 ** ((lvl_db - 100.0 * r / max(t_rel, 1e-3)) / 20) if t_rel > 0 else 0
            # Lesepositionen (mit Schleife)
            pos = anf + np.arange(n) * schritt
            if schleife:
                ll = le - ls
                drin = pos >= le
                pos[drin] = ls + np.mod(pos[drin] - ls, ll)
            ok = pos < end - 1
            x = np.zeros(n, np.float32)
            i = pos[ok].astype(np.int64)
            fr = (pos[ok] - i).astype(np.float32)
            x[ok] = daten[i] * (1 - fr) + daten[np.minimum(i + 1, len(daten) - 1)] * fr
            x *= env.astype(np.float32)
            # Dämpfung + Anschlag (Standard-Kurve: konkav, 96 dB)
            att_db = 0.4 * g[G["atten"]] / 10.0       # Zentibel × 0,4 (wie EMU/FluidSynth) -> dB
            vel_db = 40.0 * np.log10(127.0 / max(anschlag, 1))
            x *= 10 ** (-(att_db + vel_db) / 20)
            # Tiefpass (initialFilterFc), Anschlag macht heller
            fc_cent = g[G["fc"]] - (127 - anschlag) * 18
            if fc_cent < 13400:
                fc = min(8.176 * 2 ** (fc_cent / 1200.0), SR * 0.45)
                k = np.exp(-2 * np.pi * fc / SR)
                x = lfilter([1 - k], [1, -k], x).astype(np.float32)
            pan = np.clip(g[G["pan"]] / 500.0, -1, 1)
            w = (pan + 1) * np.pi / 4
            teile.append(np.stack([x * np.cos(w), x * np.sin(w)], 1))
        if not teile:
            return np.zeros((1, 2), np.float32)
        L = max(len(p) for p in teile)
        out = np.zeros((L, 2), np.float32)
        for p in teile:
            out[:len(p)] += p
        return out


def spielen(instr, noten, laenge_s, nachklang_max=6.0):
    """noten: Liste (start_s, ende_s, taste, anschlag). Rückgabe Stereo-Array."""
    y = np.zeros((int((laenge_s + nachklang_max) * SR), 2), np.float32)
    cache = {}
    for st, en, k, v in noten:
        d = max(0.03, en - st)
        key = (int(k), int(v), round(d, 2))
        z = cache.get(key)
        if z is None:
            z = instr.note(int(k), int(v), d, nachklang_max)
            if len(cache) < 4000:
                cache[key] = z
        a = int(st * SR)
        L = min(len(z), len(y) - a)
        if L > 0:
            y[a:a + L] += z[:L]
    return y[:int(laenge_s * SR)]
