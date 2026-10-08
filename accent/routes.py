"""Local-only API for the experimental accent page (/accent.html).

GET  /api/accent/reference     map payload written by build_index.py
POST /api/accent               raw 16 kHz float32 samples, as /api/similar takes them;
                               returns the take's accent coordinates and region profile
GET  /api/accent/audio/{id}    a reference speaker's Speech Accent Archive recording

The embedder (PyTorch) loads on the first take, so the server starts as fast as before.
"""

import asyncio
import os
import re

import numpy as np
from aiohttp import web

from .space import BY_NAME, SR, AccentSpace

MIN_SECONDS = 5
SPEAKER = re.compile(r"^english\d+$")


def add_accent_routes(app, data, read_audio, respond, gate):
    name = os.environ.get("KOENAMI_ACCENT", "gemma2-a10ms")
    index, reference = data / f"accent-index-{name}.npz", data / f"accent-reference-{name}.json"
    if not index.exists() or not reference.exists():
        print(f"Accent page is off: no {index.name}; run accent/build_index.py", flush=True)
        return
    space, d = AccentSpace.load(index)
    ids, z_ref = [str(i) for i in d["ids"]], d["z"]
    embedder = {}

    def embed(x):
        if "model" not in embedder:
            embedder["model"] = BY_NAME[space.embedder]()
        return embedder["model"](x)

    async def accent_reference(request):
        return web.FileResponse(reference)

    async def accent(request):
        x = await read_audio(request)
        if len(x) < SR * MIN_SECONDS:
            raise web.HTTPBadRequest(text=f"Record at least {MIN_SECONDS} seconds; the whole paragraph works best.")
        async with gate:
            v = await asyncio.to_thread(embed, x)
        z = space.transform(v)
        profile = space.region_profile(z)
        near = np.argsort(np.linalg.norm(z_ref - z, axis=1))[:5]
        return respond({
            "embedder": space.embedder,
            "seconds": round(len(x) / SR, 1),
            "z": z.round(4).tolist(),
            "profile": {r: round(float(p), 4) for r, p in zip(space.regions, profile)},
            "nearest": [ids[i] for i in near],
        })

    async def accent_audio(request):
        sid = request.match_info["id"]
        path = data / "saa_mp3" / f"{sid}.mp3"
        if not SPEAKER.match(sid) or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Content-Type": "audio/mpeg"})

    app.router.add_get("/api/accent/reference", accent_reference)
    app.router.add_post("/api/accent", accent)
    app.router.add_get("/api/accent/audio/{id}", accent_audio)
    print(f"Accent page: {len(ids)} reference speakers, {space.embedder}", flush=True)
