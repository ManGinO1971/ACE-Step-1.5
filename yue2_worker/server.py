"""HIELTECH – YuE2-3B + MLTNT-Reggae-Stile als eigener RunPod-Load-Balancer-Worker (28. Okt 2026).

Läuft getrennt vom ACE-Step-Worker (eigener RunPod-Endpoint, eigenes Image aus Dockerfile.yue2).
Innen arbeitet ComfyUI (v0.43, natives YuE2) mit dem FS_Audio-Suite-Knotenpaket, gesteuert über
ComfyUIs eigene HTTP-Schnittstelle auf 127.0.0.1:8188. Nach außen gibt es dieselben Routen wie beim
ACE-Step-Worker, damit runpod-handler.js runRunPodJob() + fetchHieltechResultInChunks() wiederverwenden kann:

  GET  /ping                         200 = bereit, 204 = startet noch (Modelle laden / ComfyUI startet)
  POST /v1/hieltech_mltnt_start      JSON {variant, style, lyrics, seed, length} -> {ok, job_id}
  GET  /v1/hieltech_job_status       ?job_id= -> {status: running|done, elapsed, result}
  GET  /v1/hieltech_fetch_chunk      ?token=&offset=&length=&done=
  GET  /v1/hieltech_mltnt_info       Zustand (bereit?, Dateien, letzter Fehler)

Lizenz: YuE2-3B und die MLTNT-LoRAs stehen unter CC BY-NC 4.0 – nur nicht-kommerziell.
Namensnennung: "MLTNT LoRAs by becausereasons". In der App deshalb nur für den Admin freigeschaltet.
"""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

from fastapi import FastAPI, Request, Response

VOLUME = os.environ.get("YUE2_HOME", "/runpod-volume/yue2")
MODELS = os.path.join(VOLUME, "models")
COMFY_DIR = os.environ.get("COMFY_DIR", "/opt/ComfyUI")
COMFY_URL = "http://127.0.0.1:8188"
OUT_DIR = "/tmp/yue2_out"
HANDOFF_DIR = "/tmp/hieltech_handoff"
CKPT_REPO, CKPT_FILE = "Comfy-Org/YuE2", "checkpoints/yue2_3b_bf16.safetensors"
LORA_REPO = os.environ.get("MLTNT_REPO", "becausereasons/yue2-mltnt-militant-reggae")

# Empfohlene Einstellungen laut Modellkarte (strength_clip = Planer, strength_model = Decoder)
VARIANTS = {
    "steppers":   {"file": "mltnt_steppers.safetensors",   "clip": 1.0, "model": 1.0, "cfg": 1.0, "st": 0.7, "mt": 1.0},
    "roots":      {"file": "mltnt_roots.safetensors",      "clip": 1.0, "model": 1.0, "cfg": 1.0, "st": 0.7, "mt": 1.0},
    "frontline":  {"file": "mltnt_frontline.safetensors",  "clip": 1.0, "model": 1.0, "cfg": 1.0, "st": 0.7, "mt": 1.0},
    "fusion":     {"file": "mltnt_fusion.safetensors",     "clip": 1.0, "model": 1.5, "cfg": 1.4, "st": 0.9, "mt": 1.2},
    "soundclash": {"file": "mltnt_soundclash.safetensors", "clip": 0.5, "model": 1.0, "cfg": 1.0, "st": 0.7, "mt": 1.0},
    "chanter":    {"file": "mltnt_chanter.safetensors",    "clip": 1.0, "model": 1.0, "cfg": 1.0, "st": 0.7, "mt": 1.0},
}

STATE = {"ready": False, "phase": "start", "error": None, "comfy": None}
JOBS = {}
JOBS_LOCK = threading.Lock()
GEN_LOCK = threading.Lock()  # ein Song nach dem anderen pro Worker
app = FastAPI()


def log(msg):
    print(f"[HIELTECH-YuE2] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- Modelle aufs Volume
def _download_models():
    from huggingface_hub import hf_hub_download, list_repo_files
    token = os.environ.get("HF_TOKEN") or None
    for sub in ("checkpoints", "loras", "audio_encoders"):
        os.makedirs(os.path.join(MODELS, sub), exist_ok=True)
    lock_path = os.path.join(VOLUME, ".download.lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)  # mehrere Worker teilen sich das Volume
        try:
            ckpt = os.path.join(MODELS, CKPT_FILE)
            if not (os.path.isfile(ckpt) and os.path.getsize(ckpt) > 1_000_000_000):
                STATE["phase"] = "lade YuE2-3B (einmalig, mehrere GB)"
                log(STATE["phase"])
                hf_hub_download(CKPT_REPO, CKPT_FILE, local_dir=MODELS, token=token)
            wanted = {v["file"] for v in VARIANTS.values()}
            missing = [f for f in wanted if not os.path.isfile(os.path.join(MODELS, "loras", f))]
            if missing:
                STATE["phase"] = f"lade {len(missing)} MLTNT-Stile (einmalig)"
                log(STATE["phase"])
                files = [f for f in list_repo_files(LORA_REPO, token=token) if os.path.basename(f) in missing]
                for f in files:
                    p = hf_hub_download(LORA_REPO, f, local_dir=os.path.join(VOLUME, "hf_mltnt"), token=token)
                    dst = os.path.join(MODELS, "loras", os.path.basename(f))
                    if not os.path.exists(dst):
                        try:
                            os.symlink(p, dst)  # spart Platz (keine zweite Kopie)
                        except OSError:
                            shutil.copyfile(p, dst + ".part")
                            os.replace(dst + ".part", dst)
                still = [f for f in wanted if not os.path.isfile(os.path.join(MODELS, "loras", f))]
                if still:
                    log(f"Nicht im Repo gefunden: {still}")
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _http(method, path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(COMFY_URL + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def _start_comfy():
    yaml_path = "/tmp/hieltech_extra_model_paths.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"hieltech:\n  base_path: {MODELS}\n  checkpoints: checkpoints\n  loras: loras\n  audio_encoders: audio_encoders\n")
    os.makedirs(OUT_DIR, exist_ok=True)
    STATE["phase"] = "starte ComfyUI"
    cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188",
           "--extra-model-paths-config", yaml_path, "--output-directory", OUT_DIR,
           "--disable-auto-launch", "--dont-print-server"] + os.environ.get("COMFY_EXTRA_ARGS", "").split()
    STATE["comfy"] = subprocess.Popen(cmd, cwd=COMFY_DIR, stdout=sys.stderr, stderr=sys.stderr)
    t0 = time.time()
    while time.time() - t0 < 900:
        if STATE["comfy"].poll() is not None:
            raise RuntimeError(f"ComfyUI beendet (Code {STATE['comfy'].returncode})")
        try:
            info = _http("GET", "/object_info/FSAudioSampler", timeout=10)
            if "FSAudioSampler" in info:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    raise RuntimeError("ComfyUI nicht rechtzeitig bereit")


def _boot():
    try:
        _download_models()
        _start_comfy()
        STATE.update(ready=True, phase="bereit")
        log("bereit")
    except Exception as e:  # noqa: BLE001
        STATE.update(error=f"{type(e).__name__}: {e}", phase="Fehler beim Start")
        log(STATE["error"])


# ---------------------------------------------------------------- Erzeugung
def _workflow(v, style, lyrics, seed, length, job_id):
    return {
        "1": {"class_type": "FSAudioLoraLoader",
              "inputs": {"lora_name": v["file"], "strength_model": v["model"], "strength_clip": v["clip"]}},
        "2": {"class_type": "FSAudioModelLoader",
              "inputs": {"yue2_checkpoint": os.path.basename(CKPT_FILE), "melody_transcriber": "none", "loras": ["1", 0]}},
        "3": {"class_type": "FSAudioSampler",
              "inputs": {"pipe": ["2", 0], "style": style, "lyrics": lyrics, "seed": seed,
                         "song_length_cap": length, "score_mode": "full", "steps": 32,
                         "sampler": "dpm_2", "scheduler": "sgm_uniform", "Weirdness (cfg)": v["cfg"],
                         "score_temperature": v["st"], "music_temperature": v["mt"], "repetition_penalty": 1.2}},
        "4": {"class_type": "FSAudioOutput",
              "inputs": {"song": ["3", 0], "filename_prefix": f"mltnt/{job_id}", "score": ["3", 1], "info": ["3", 2]}},
    }


def _store_handoff(path, ext):
    os.makedirs(HANDOFF_DIR, exist_ok=True)
    now = time.time()
    for n in os.listdir(HANDOFF_DIR):
        p = os.path.join(HANDOFF_DIR, n)
        try:
            if now - os.path.getmtime(p) > 3600:
                os.remove(p)
        except OSError:
            pass
    token = uuid.uuid4().hex
    dst = os.path.join(HANDOFF_DIR, f"{token}.{ext}")
    shutil.copyfile(path, dst)
    return dst, token


def _generate(job_id, body):
    variant = str(body.get("variant") or "").lower()
    v = VARIANTS.get(variant)
    if not v:
        return {"ok": False, "error": "unknown_variant", "message": f"Unbekannter Stil {variant!r}"}
    style = " ".join(str(body.get("style") or "").split())[:1500]
    if not style.lower().startswith("mltnt"):
        style = f"mltnt {style}".strip()
    lyrics = str(body.get("lyrics") or "").replace("\r", "").strip()[:6000]
    if not lyrics:
        return {"ok": False, "error": "no_lyrics", "message": "Keine Lyrics"}
    seed = int(body.get("seed") or 0) & 0xFFFFFFFF
    length = float(max(30, min(360, int(body.get("length") or 240))))
    with GEN_LOCK:
        t0 = time.time()
        wf = _workflow(v, style, lyrics, seed, length, job_id)
        try:
            resp = _http("POST", "/prompt", {"prompt": wf, "client_id": job_id}, timeout=60)
        except urllib.error.HTTPError as e:
            return {"ok": False, "error": "comfy_rejected", "message": e.read().decode(errors="replace")[:800]}
        pid = resp.get("prompt_id")
        if not pid:
            return {"ok": False, "error": "comfy_rejected", "message": json.dumps(resp)[:800]}
        while time.time() - t0 < 1500:
            time.sleep(2)
            hist = _http("GET", f"/history/{pid}", timeout=30).get(pid)
            if not hist:
                continue
            st = hist.get("status") or {}
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                detail = msgs[-1][1].get("exception_message", "") if msgs else json.dumps(st)[:500]
                return {"ok": False, "error": "generation_failed", "message": str(detail)[:800]}
            if not st.get("completed"):
                continue
            audio = ((hist.get("outputs") or {}).get("4") or {}).get("audio") or []
            if not audio:
                return {"ok": False, "error": "no_audio", "message": "ComfyUI lieferte keine Audiodatei"}
            a = audio[0]
            flac = os.path.join(OUT_DIR, a.get("subfolder", ""), a["filename"])
            mp3 = flac[:-5] + ".mp3"
            ext = "mp3"
            if shutil.which("ffmpeg"):
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", flac, "-b:a", "320k", mp3], check=True, timeout=300)
            else:  # z. B. Mac ohne ffmpeg: FLAC direkt liefern
                shutil.copyfile(flac, mp3 + ".flac"); mp3 = mp3 + ".flac"; ext = "flac"
            info = {}
            try:
                with open(flac[:-5] + ".json") as f:
                    info = json.load(f)
            except (OSError, ValueError):
                pass
            hp, token = _store_handoff(mp3, ext)
            for p in (flac, mp3, flac[:-5] + ".json", flac[:-5] + ".abc"):
                try:
                    os.remove(p)
                except OSError:
                    pass
            secs = round(time.time() - t0)
            log(f"{variant}: fertig in {secs}s, Song {info.get('seconds', '?')}s")
            return {"ok": True, "result_token": token, "size_bytes": os.path.getsize(hp), "audio_ext": ext,
                    "variant": variant, "seed": seed, "song_seconds": info.get("seconds"), "gpu_seconds": secs}
        return {"ok": False, "error": "timeout", "message": "Erzeugung dauerte über 25 Minuten"}


@app.get("/ping")
async def ping():
    return Response(status_code=200 if STATE["ready"] else 204)


@app.get("/v1/hieltech_mltnt_info")
async def info():
    loras = sorted(os.listdir(os.path.join(MODELS, "loras"))) if os.path.isdir(os.path.join(MODELS, "loras")) else []
    return {"ready": STATE["ready"], "phase": STATE["phase"], "error": STATE["error"], "loras": loras,
            "variants": list(VARIANTS)}


@app.post("/v1/hieltech_mltnt_start")
async def mltnt_start(request: Request):
    if not STATE["ready"]:
        if STATE["error"]:
            return {"ok": False, "error": "worker_start_failed", "message": STATE["error"]}
        return Response(status_code=503, content=f"startet noch: {STATE['phase']}".encode())
    try:
        body = await request.json()
    except ValueError:
        body = {}
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        for k in [k for k, j in JOBS.items() if time.time() - j["started"] > 7200]:
            JOBS.pop(k, None)
        JOBS[job_id] = {"status": "running", "started": time.time()}

    def run():
        try:
            res = _generate(job_id, body)
        except Exception as e:  # noqa: BLE001
            res = {"ok": False, "error": "job_exception", "message": f"{type(e).__name__}: {e}"}
        if os.environ.get("HIELTECH_FREE_AFTER") == "1":  # Mac: Arbeitsspeicher nach jedem Song wieder freigeben
            try:
                _http("POST", "/free", {"unload_models": True, "free_memory": True}, timeout=30)
                log("Arbeitsspeicher freigegeben")
            except Exception as e:  # noqa: BLE001
                log(f"Freigeben fehlgeschlagen: {e}")
        with JOBS_LOCK:
            JOBS[job_id].update(status="done", result=res, finished=time.time())

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "job_id": job_id}


@app.get("/v1/hieltech_job_status")
async def job_status(job_id: str):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if not j:
            return Response(status_code=404)
        out = {"status": j["status"], "elapsed": round(time.time() - j["started"])}
        if j["status"] == "done":
            out["result"] = j.get("result")
    return out


@app.get("/v1/hieltech_fetch_chunk")
async def fetch_chunk(token: str, offset: int = 0, length: int = 18_000_000, done: str = "false"):
    if not token.isalnum():
        return Response(status_code=400)
    hits = [n for n in os.listdir(HANDOFF_DIR) if n.startswith(token + ".")] if os.path.isdir(HANDOFF_DIR) else []
    if not hits:
        return Response(status_code=404)
    path = os.path.join(HANDOFF_DIR, hits[0])
    with open(path, "rb") as f:
        f.seek(max(0, offset))
        data = f.read(max(0, min(length, 25_000_000)))
    if done == "true":
        try:
            os.remove(path)
        except OSError:
            pass
    return Response(content=data, media_type="application/octet-stream")


threading.Thread(target=_boot, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
