# hieltech_reparieren – Reparatur-Module der HELTECH-Stems-App auf der GPU

Derselbe Code wie in der Mac-App (`~/HELTECH_Stems/<modul>_neu/`), gleiche Ordner-Namen. Aufgerufen von
`runpod_entrypoint.py` über die Route `/v1/hieltech_stems_reparieren_start` (Tabelle `_HIELTECH_REPARIEREN_MODULE`).
Die Klänge (Samples, SoundFont) liegen NICHT im Repo, sondern als Klang-Pakete auf dem Volume
(`/runpod-volume/hieltech_stems_klaenge/<paket>/`); die Mac-App lädt sie bei Bedarf hoch, der Server legt sie als
Verknüpfung in den Code-Ordner (z. B. `bass_neu/samples`). Noten-Erkennung (Basic Pitch) läuft in der eigenen
Umgebung `/opt/hieltech_noten_venv` (siehe Dockerfile.runpod).

| Modul | Ordner | Klang-Pakete | Stand |
|---|---|---|---|
| drums | drum_neu/ | drums | 12. Nov 2026 |
| bass | bass_neu/ | bass_neu | 12. Nov 2026 |
| gitarre (Skank) | gitarre_neu/ | gitarre_neu | 12. Nov 2026 |
| piano | piano_neu/ | piano_neu (SoundFont) | 12. Nov 2026 |
| orgel | orgel_neu/ | piano_neu | 12. Nov 2026 |
| picking | picking_neu/ | picking_neu, piano_neu | 12. Nov 2026 |
| perc | perc_neu/ | perc_neu | 12. Nov 2026 |
| lead (Gitarre) | lead_neu/ | lead_neu, piano_neu | 12. Nov 2026 |
| blaeser | blaeser_neu/ | piano_neu, lead_neu | 12. Nov 2026 |
