"""
RunPod-Load-Balancing-Einstiegspunkt fuer den bestehenden ACE-Step-1.5
REST-API-Server.

Diese Datei aendert das originale acestep-Paket NICHT - sie macht nur:
  1) importiert das bestehende FastAPI-"app"-Objekt (unveraendert),
  2) fuegt eine zusaetzliche Route "/ping" hinzu, die RunPods
     Load-Balancing-Endpoints fuer Gesundheitschecks brauchen,
  3) startet uvicorn auf dem Port, den RunPod erwartet.

Nichts hier hat mit Preisen/Lizenzen/Geschaeftslogik zu tun - es startet
nur den bestehenden, quelloffenen REST-Server so, dass RunPod damit reden
kann.
"""
import os
from fastapi import Response
from acestep.api_server import app

@app.get("/ping")
async def _runpod_ping():
    return Response(status_code=200)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
