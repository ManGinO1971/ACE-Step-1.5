# HIELTECH – Stand „Vocals V2“ (Lead neu), 21. Okt 2026

Vom Inhaber per Hörtest freigegeben (Rock pon rock, Fassung F: „Höhen perfekt“, Tiefen gut).
Git-Markierung dieses Stands: `stand-lead-v2-2026-10-21`

## Ablauf (Knopf „Vocals V2“ in der App)
1. `/v1/hieltech_lead_v2_prepare` – `lead_v2_prepare_step.py`
   Mel-Band-RoFormer (Gesang) -> Karaoke-RoFormer (Lead/Background)
   -> Song ohne Background, Lead-Referenz, Background-Referenz
2. `/release_task` Extract, Base-Modell `acestep-v15-base`, track vocals,
   50 Schritte, Guidance 7, 2 Varianten, nur Lead-Text ((…) raus, {…} bleibt),
   Beschreibung ohne Background/Instrumente + a-cappella-Zusatz (VPS: runpod-handler.js)
3. `/v1/hieltech_lead_v2_post`
   - `lead_v2_pick_step.py`: beste Variante = wenigste Lücken (0,5 s gerundet), dann mehr Wörter
   - `lead_mask_step.py`: Zeitmaske (−30 dB, Halten 300 ms, Background −10 dB)
     + **Text-Maske** (weiter als 1 s von jeder Lead-Textzeile -> stumm)
   - `gate_vocal_stem.py` (Live-Werte)
   - `f0_rmvpe_lead_step.py` (RMVPE, Schwelle 0.03)
   - `nsf_vocoder_step.py` = Vocoder C (PC-NSF-HiFiGAN 44.1k)
     + **Melodie-Lock** (Oktave wie Original-Lead, Toleranz 1,5 HT)
     + **Melodie folgen** (kurze schiefe Stellen ≤ 200 ms, > 0,8 HT -> Original)
   - `rvc_convert_step.py` (.pth mit gleichnamiger .index, index_rate 0.5, rmvpe)
     bei vollem GPU-Speicher: nochmal, dann Prozessor (fällt nie weg)
4. VPS: Effekte nur wenn angehakt, Hall „Reverb Raum“ (mit .pth: Kammer lang), Lautheit wie Original

## Schalter (RunPod-Umgebungsvariablen, ohne Neubau änderbar)
- `HIELTECH_LEAD_V2_FEATURES` – Standard `melody_lock,lyrics_mask,melody_follow`; `none` = alles aus (Stand 20. Okt)
- `NSF_LOCK_TOL_ST` (1.5), `NSF_FOLLOW_TOL_ST` (0.8), `NSF_FOLLOW_MAX_MS` (200)
- `MASK_KEEP_PAD_S` (1.0), `MASK_KEEP_MIN_CONF` (0.5)
- `HIELTECH_RVC_INDEX_RATE` (0.5), `HIELTECH_RVC_USE_INDEX` (1), `HIELTECH_RVC_DEVICE` (leer = automatisch)

## Dateien (md5)
- runpod_entrypoint.py      103d2adef4a3c42f188b498867bacf1e
- nsf_vocoder_step.py       070997b0626e7a1d95e139ddf31f127c
- lead_mask_step.py         41b7872cdc5eaa3093ff5296c2f2d05f
- lead_v2_pick_step.py      40541688a6962e458e5577809d1b8b9b
- rvc_convert_step.py       acf82c79b7082224e2527a3a8941bcb1
- lead_v2_prepare_step.py   dd3c246f4e243370cc58da42f6dd5b83
- f0_rmvpe_lead_step.py     f58acf86e232889c64801de09898633f
- Dockerfile.runpod         ee527b489e0aba85ffe123adae205221
- VPS: runpod-handler.js 3592b1a99131773c8a1cbc2afe83abc5, server.js 48cd8b9c0638a0571bc964edde7ab71e

## Rückwege
- Verbesserungen aus: bei RunPod `HIELTECH_LEAD_V2_FEATURES=none` setzen (kein Neubau nötig)
- ganzer Stand: `git checkout stand-lead-v2-2026-10-21 -- <datei>`
- alter Rückweg „T5 Extract“: ~/ACE-Step-1.5/lead-neu-test/STAND-T5/

## Offen
- Lizenz Vocoder C (Gewichte CC BY-NC-SA 4.0) vor zahlenden Kunden klären
- GPU-Tempo: 2–3× langsamer als im ersten Test – GPU-Typ prüfen (RunPod-Log „GPU-Speicher“)
