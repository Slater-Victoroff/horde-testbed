from typing import List, Dict
import uuid
import random
from pathlib import Path
from json import JSONEncoder

def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)

_default.default = JSONEncoder().default
JSONEncoder.default = _default

from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from pressure_math import generate_fields
from enemies import ENEMY_REGISTRY
from world_config import WORLD_SETUP
from models import FrameUpdate

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["http://<your-ec2-ip>"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CORSMiddlewareForStaticFiles(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        print(f"Intercepted request for: {request.url.path}")
        response = await call_next(request)
        # Add CORS headers to static file responses
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

# Directory where static files are stored
STATIC_DIR = Path("/app/static")

@app.get("/assets/{file_path:path}")
async def serve_static(file_path: str):
    """
    Custom route to serve static files with CORS headers.
    """
    file_location = STATIC_DIR / file_path
    if not file_location.exists():
        return {"error": "File not found"}, 404

    response = FileResponse(file_location)
    # Add CORS headers
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

app.add_middleware(CORSMiddlewareForStaticFiles)

# Simple spawn buffer
ENEMY_SPAWN_QUEUE = []

@app.post("/frame-update")
async def frame_update(request: Request):
    global ENEMY_SPAWN_QUEUE

    data = await request.json()
    game_state = FrameUpdate.parse_obj(data)

    # Use world_config parameters
    field_resolution = WORLD_SETUP["minimap_resolution"]
    grid_size = (field_resolution, field_resolution)
    ground_size = WORLD_SETUP["ground_size"]

    # Default fields
    pressure_field = np.zeros(grid_size, dtype=np.float32)
    flow_field = np.zeros((*grid_size, 2), dtype=np.float32)
    difficulty = 0.0

    if game_state.enemiesState:
        pressure_field, flow_field = generate_fields(game_state.playerState, game_state.enemiesState, grid_size, ground_size)
    else:
        pressure_field = np.zeros(grid_size, dtype=np.float32)
        flow_field = np.zeros((*grid_size, 2), dtype=np.float32)

    # Spawn logic
    if random.random() < 0.3:
        spawn_id = str(uuid.uuid4())
        radius_span = (5, 15)
        distance = random.uniform(*radius_span)
        angle = random.uniform(0, 2 * np.pi)
        x = game_state.playerState.position[0] + distance * np.cos(angle)
        z = game_state.playerState.position[2] + distance * np.sin(angle)
        ENEMY_SPAWN_QUEUE.append({
            "id": spawn_id,
            "position": [x, 0, z],
            "kind": random.choice(list(ENEMY_REGISTRY.keys())),
        })

    # Send response to frontend
    spawns = ENEMY_SPAWN_QUEUE
    ENEMY_SPAWN_QUEUE = []  # Clear after sending

    return {
        "spawn": spawns,
        "pressure_field": pressure_field.tolist(),
        "flow_field": flow_field.tolist(),
        "difficulty": difficulty,
    }

@app.get("/enemy-kernels")
async def get_enemy_kernels():
    """
    Returns the available enemy kernels.
    """
    return ENEMY_REGISTRY

@app.get("/world-setup")
async def get_world_setup():
    """
    Returns the game setup configuration.
    """
    return WORLD_SETUP
