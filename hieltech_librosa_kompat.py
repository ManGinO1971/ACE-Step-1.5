"""HIELTECH (21. Nov 2026): audio-separator 0.44.5 (dieselbe Fassung wie auf dem Mac) ruft librosa.get_duration(filename=…)
auf. Neuere librosa-Fassungen kennen nur noch path=. Dieser kleine Zusatz übersetzt filename= in path= – sonst nichts.
Er wird beim Start jeder Python-Umgebung geladen (zz_hieltech_kompat.pth), greift aber erst, wenn librosa importiert wird."""
import sys
import importlib.abc
import importlib.util


def _patch(m):
    try:
        import inspect
        g = m.get_duration
        if "filename" in inspect.signature(g).parameters:
            return
        def get_duration(*a, filename=None, **k):
            if filename is not None and "path" not in k:
                k["path"] = filename
            return g(*a, **k)
        get_duration.__doc__ = g.__doc__
        m.get_duration = get_duration
    except Exception:  # noqa: BLE001 – nie den Import von librosa stören
        pass


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "librosa":
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        spec = importlib.util.find_spec("librosa")
        if spec is None or spec.loader is None:
            return spec
        orig = spec.loader.exec_module

        def exec_module(mod):
            orig(mod)
            _patch(mod)
        spec.loader.exec_module = exec_module
        return spec


if "librosa" in sys.modules:
    _patch(sys.modules["librosa"])
else:
    sys.meta_path.insert(0, _Finder())
