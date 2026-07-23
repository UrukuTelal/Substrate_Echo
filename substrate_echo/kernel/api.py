"""Substrate API — Two planes.

Control Plane (REST):
  POST /kernel/checkpoint    — save kernel state
  POST /kernel/load          — load kernel state
  GET  /kernel/state         — full snapshot
  GET  /kernel/topology      — basin topology history
  GET  /kernel/abstraction   — abstraction events
  GET  /kernel/embodiments   — connected embodiments
  GET  /health               — liveness

Cognitive Plane (WebSocket):
  Client publishes: Observation, Goal, Reward, EmbodimentState
  Kernel responds:  CognitiveState (action, prediction, metrics)
"""
from __future__ import annotations
import json
from typing import Optional
from pathlib import Path

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from . import (
    SubstrateKernel, KernelConfig,
    Observation, Goal, Reward, Action, Prediction,
    EmbodimentState, CognitiveState,
)


if HAS_FASTAPI:
    class ObservationMsg(BaseModel):
        vector: list[float]
        modality: str = "generic"
        embodiment_id: str = "default"
        metadata: dict = {}

    class GoalMsg(BaseModel):
        target: list[float]
        priority: float = 0.5
        description: str = ""
        embodiment_id: str = "default"

    class RewardMsg(BaseModel):
        value: float
        target_attractor: Optional[int] = None
        embodiment_id: str = "default"

    class EmbodimentMsg(BaseModel):
        embodiment_id: str
        embodiment_type: str = "generic"
        available_modalities: list[str] = ["generic"]
        is_active: bool = True

    class CheckpointMsg(BaseModel):
        path: str = "checkpoint.json"


def create_app(kernel: Optional[SubstrateKernel] = None) -> "FastAPI":
    if not HAS_FASTAPI:
        raise ImportError("pip install fastapi uvicorn")

    app = FastAPI(title="Substrate Kernel", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _k = kernel or SubstrateKernel()

    # ── Control Plane (REST) ─────────────────────────────────

    @app.get("/health")
    def health():
        return {"status": "ok", "tick": _k._tick}

    @app.get("/kernel/state")
    def kernel_state():
        return _k.get_snapshot()

    @app.get("/kernel/topology")
    def kernel_topology():
        return _k.get_topology_history()

    @app.get("/kernel/abstraction")
    def kernel_abstraction():
        return _k.get_abstraction_events()

    @app.get("/kernel/embodiments")
    def kernel_embodiments():
        return _k.get_embodiments()

    @app.post("/kernel/checkpoint")
    def kernel_checkpoint(msg: CheckpointMsg):
        import json as _json
        snapshot = _k.get_snapshot()
        path = Path(msg.path)
        path.write_text(_json.dumps(snapshot, indent=2))
        return {"status": "saved", "path": str(path)}

    # ── Cognitive Plane (WebSocket) ──────────────────────────

    @app.websocket("/cognitive")
    async def cognitive_stream(ws: WebSocket):
        """Streaming cognitive interface.

        Client sends JSON messages:
          {"type": "observation", "vector": [...], ...}
          {"type": "goal", "target": [...], ...}
          {"type": "reward", "value": 0.5, ...}
          {"type": "embodiment", "embodiment_id": "...", ...}

        Kernel responds with CognitiveState JSON.
        """
        await ws.accept()
        try:
            while True:
                data = await ws.receive_text()
                msg = json.loads(data)
                msg_type = msg.get("type", "observation")

                if msg_type == "observation":
                    obs = Observation(
                        vector=msg["vector"],
                        modality=msg.get("modality", "generic"),
                        embodiment_id=msg.get("embodiment_id", "default"),
                        metadata=msg.get("metadata", {}),
                    )
                    state = _k.publish_observation(obs)

                elif msg_type == "goal":
                    goal = Goal(
                        target=msg["target"],
                        priority=msg.get("priority", 0.5),
                        description=msg.get("description", ""),
                        embodiment_id=msg.get("embodiment_id", "default"),
                    )
                    _k.publish_goal(goal)
                    # Still return current state
                    state = CognitiveState(
                        tick=_k._tick,
                        n_attractors=len(_k._base_attractors),
                        active_goals=len(_k._goals),
                    )

                elif msg_type == "reward":
                    reward = Reward(
                        value=msg["value"],
                        target_attractor=msg.get("target_attractor"),
                        embodiment_id=msg.get("embodiment_id", "default"),
                    )
                    _k.publish_reward(reward)
                    state = CognitiveState(tick=_k._tick)

                elif msg_type == "embodiment":
                    emb = EmbodimentState(
                        embodiment_id=msg["embodiment_id"],
                        embodiment_type=msg.get("embodiment_type", "generic"),
                        available_modalities=msg.get("available_modalities", ["generic"]),
                        is_active=msg.get("is_active", True),
                    )
                    _k.publish_embodiment_state(emb)
                    state = CognitiveState(tick=_k._tick)

                else:
                    state = CognitiveState(tick=_k._tick)

                # Serialize response
                action_dict = None
                if state.action:
                    action_dict = {
                        "vector": state.action.vector,
                        "confidence": state.action.confidence,
                        "source": state.action.source,
                    }
                pred_dict = None
                if state.prediction:
                    pred_dict = {
                        "expected_next": state.prediction.expected_next,
                        "confidence": state.prediction.confidence,
                    }

                response = {
                    "tick": state.tick,
                    "action": action_dict,
                    "prediction": pred_dict,
                    "n_attractors": state.n_attractors,
                    "n_meta_attractors": state.n_meta_attractors,
                    "coherence": state.coherence,
                    "basin_balance": state.basin_balance,
                    "mean_depth": state.mean_depth,
                    "volume_entropy": state.volume_entropy,
                    "active_goals": state.active_goals,
                    "active_embodiments": state.active_embodiments,
                }
                await ws.send_text(json.dumps(response))

        except WebSocketDisconnect:
            pass

    return app
