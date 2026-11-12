"""Eigener, schlanker SF2-Leser (RIFF/sfbk) – ohne Fremdbibliothek. Liefert Presets, Zonen und Samples."""
import struct
import numpy as np


class Sample:
    __slots__ = ("name", "start", "end", "start_loop", "end_loop", "sample_rate", "original_pitch",
                 "pitch_correction", "link", "typ")


class SF2:
    def __init__(self, pfad):
        with open(pfad, "rb") as f:
            d = f.read()
        if d[:4] != b"RIFF" or d[8:12] != b"sfbk":
            raise ValueError("keine SoundFont-2-Datei")
        self.smpl = None
        teile = {}
        i = 12
        while i < len(d) - 8:
            cid, ln = d[i:i + 4], struct.unpack("<I", d[i + 4:i + 8])[0]
            if cid == b"LIST":
                art = d[i + 8:i + 12]
                j, ende = i + 12, i + 8 + ln
                while j < ende - 8:
                    sid, sl = d[j:j + 4], struct.unpack("<I", d[j + 4:j + 8])[0]
                    teile[(art, sid)] = (j + 8, sl)
                    j += 8 + sl + (sl & 1)
            i += 8 + ln + (ln & 1)
        o, l = teile[(b"sdta", b"smpl")]
        self.smpl = np.frombuffer(d, dtype="<i2", count=l // 2, offset=o).astype(np.float32) / 32768.0

        def recs(name, groesse, fmt):
            o, l = teile[(b"pdta", name)]
            return [struct.unpack(fmt, d[o + k * groesse:o + (k + 1) * groesse]) for k in range(l // groesse)]
        self.phdr = recs(b"phdr", 38, "<20sHHHIII")
        self.pbag = recs(b"pbag", 4, "<HH")
        self.pgen = recs(b"pgen", 4, "<Hh")
        self.inst = recs(b"inst", 22, "<20sH")
        self.ibag = recs(b"ibag", 4, "<HH")
        self.igen = recs(b"igen", 4, "<Hh")
        self.samples = []
        for r in recs(b"shdr", 46, "<20sIIIIIBbHH")[:-1]:
            s = Sample()
            s.name = r[0].split(b"\0")[0].decode("latin-1")
            s.start, s.end, s.start_loop, s.end_loop, s.sample_rate = r[1:6]
            s.original_pitch, s.pitch_correction, s.link, s.typ = r[6], r[7], r[8], r[9]
            self.samples.append(s)

    @staticmethod
    def _zonen(hdr, bags, gens, endgen):
        """je Header-Eintrag: Liste von Zonen (dict oper->wert); Bereiche (43/44) als (lo,hi)."""
        out = []
        for h in range(len(hdr) - 1):
            b0, b1 = hdr[h][-1] if len(hdr[h]) == 2 else hdr[h][3], hdr[h + 1][-1] if len(hdr[h + 1]) == 2 else hdr[h + 1][3]
            zonen = []
            for b in range(b0, b1):
                g0, g1 = bags[b][0], bags[b + 1][0]
                z = {}
                for op, amt in gens[g0:g1]:
                    if op in (43, 44):
                        u = amt & 0xFFFF
                        z[op] = (u & 0xFF, u >> 8)
                    else:
                        z[op] = amt
                zonen.append(z)
            out.append(zonen)
        return out

    def presets(self):
        return sorted((p[2], p[1], p[0].split(b"\0")[0].decode("latin-1")) for p in self.phdr[:-1])

    def preset_zonen(self, bank, programm):
        alle = self._zonen(self.phdr, self.pbag, self.pgen, 41)
        for k, p in enumerate(self.phdr[:-1]):
            if p[1] == programm and p[2] == bank:
                return p[0].split(b"\0")[0].decode("latin-1"), alle[k]
        raise ValueError(f"Klang {bank}:{programm} nicht in der SoundFont")

    def instrument_zonen(self, idx):
        if not hasattr(self, "_iz"):
            self._iz = self._zonen(self.inst, self.ibag, self.igen, 53)
        return self._iz[idx]
