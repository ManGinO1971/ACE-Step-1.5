# HIELTECH – Stand „Background-Chor“ (Knopf mit zwei Menschen), 23. Okt 2026

Vom Inhaber per Hörtest freigegeben (Root Nah Bow.1, erste Minute, 22. Okt 2026):
Frau 1 = Vocoder C, Frau 2 = Vocoder A, Mann = Vocoder B, Humanizer zuschaltbar.
Git-Markierung dieses Stands: `stand-background-chor-2026-10-23`

## Was der Knopf macht
Ein Mann und zwei Frauen singen den GANZEN Gesang des Songs (Lead + Background) neu –
Melodie, Vibrato und Timing vom Original, Klangfarbe aus den eigenen Stimmproben.
Drei Spuren im Verlauf („Backing 1/2/3“); der Kunde schneidet mit der Schere weg,
was er nicht braucht. Optional je Stimme ein eigenes .pth-Modell, optional Humanizer.

## Ablauf (Route `/v1/hieltech_backing_chor_start`, ein Auftrag)
1. `backing_chor_prepare_step.py` – „Mischung 8“: drei Trennmodelle, Min-Spec-Ensemble
   (pro Frequenz/Zeitpunkt das Leiseste):
   `vocals_mel_band_roformer.ckpt`, `model_bs_roformer_ep_317_sdr_12.9755.ckpt`,
   `mel_band_roformer_kim_ft2_bleedless_unwa.ckpt`
   → Hall raus `dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt`
   → Rauschen raus `denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt`
   → Tonart-Schätzung (nur Rückfall, wenn die App keine Tonart schickt)
2. `hieltech_chor/seedvc_multi.py` (Umgebung `/opt/seedvc_venv`, Seed-VC `/opt/seed-vc`,
   Commit 51383ef, angepasst mit `seedvc_hieltech_patch.py`):
   Gesangsmodell 44,1 kHz mit Tonhöhe, 50 Schritte, cfg 0.7, Stimmprobe 15 s,
   keine Auto-Tonlage. Vor dem Singen wird die Melodie vorbereitet:
   Oktav-Sperre → Stabil (Ausreißer-Reparatur, Ton-Halt 0.3, ruhige Übergänge) →
   Lage → Oktave. Die endgültige Zielmelodie wird gespeichert (`<rolle>_ziel.npy`).
3. Je Stimme die Lead-V2-Vocoder-Serie: `gate_vocal_stem.py` → RMVPE (0.03) →
   `f0_halt_vib.py` (nur Frau 1 und Mann) → `nsf_vocoder_step.py … <ziel> --follow`
   (Vocoder C mit Melodie-Lock + Melodie folgen)
4. Optional `rvc_convert_step.py` (.pth der Stimme, Transposition 0, rmvpe, Index wie Lead)
5. Optional `humanizer.py` (je Phrase ±25 ms spontan vor/hinter, Pegel ±1 dB, Pause ≥ 80 ms)
6. VPS (runpod-handler.js `attemptBackingChor`): Effekte/Hall wie alle Background-Knöpfe,
   Lautheit Original −6 dB.

## Stimmen (Rollen) – `_HIELTECH_CHOR_ROLES` in runpod_entrypoint.py
| Rolle | Stimmprobe | Lage | Ton-Halt | Vocoder |
|---|---|---|---|---|
| mann  | stimme_Mann.wav (ACE-Step Cover 0.3 „Tenor V1“)  | original (wie im Song) | 0.3 | B: f0_halt_vib 0.3 / 1.0 |
| frau2 | stimme_Frau2.wav (Cover30_V2, Favorit) | terz (A-Moll o. Songtonart) | 0.3 | A: ohne Zusatz |
| frau1 | stimme_Frau1.wav (Cover30_V1) | terz | 0.3 | C: f0_halt_vib 0.3 / 1.4 |

Zuordnung: Stimme männlich → mann; weiblich → erste frau2, zweite frau1, dritte frau2.
Originale der Stimmen: Mac `~/ACE-Step-1.5/lead-neu-test/STIMMEN-ORIGINAL-Frau1-Frau2/` (mit Rezept).

## Schalter (RunPod-Umgebungsvariablen)
- `HIELTECH_CHOR_STEPS` (50), `HIELTECH_SEEDVC_PYTHON` (/opt/seedvc_venv/bin/python)
- `HIELTECH_CHOR_DEREVERB`, `HIELTECH_CHOR_DENOISE` (Modellnamen)
- VPS: `BACKING_CHOR_MAX_WAIT_MS` (1 560 000 = 26 Min)

## Rückweg
- Ganzer Stand: `git checkout stand-lead-v2-2026-10-21 -- <datei>` (vorher) bzw. Revert des Chor-Commits.
- Alte Background-Wege (original/pth) sind im Code noch vorhanden, in der App aber entfernt.
