from typing import List, Dict, Optional
import numpy as np
from pydantic import BaseModel

from enemies import ENEMY_REGISTRY

class PlayerState(BaseModel):
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_raw(cls, raw: List[float]):
        """
        Create a PlayerState instance from a raw list of floats.
        """
        if len(raw) != 9:
            raise ValueError("PlayerState must have exactly 9 elements.")
        data = np.array(raw, dtype=np.float32)
        return cls(
            position=data[:3],
            velocity=data[3:6],
            acceleration=data[6:9],
        )


class EnemiesState(BaseModel):
    states: np.ndarray
    kinds: List[str]
    kind_masks: Dict[str, np.ndarray] = {}

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_raw(cls, states: List[List[float]], kinds: List[str]):
        """
        Create an EnemiesState instance from raw states and kinds.
        """
        states_array = np.array(states, dtype=np.float32).reshape(-1, 9) if states else np.empty((0, 9), dtype=np.float32)
        kinds_array = np.array(kinds, dtype=str) if kinds else np.empty((0,), dtype=str)

        # Validate kinds against ENEMY_REGISTRY
        unrecognized_kinds = set(kinds_array) - set(ENEMY_REGISTRY.keys())
        if unrecognized_kinds:
            raise ValueError(f"Unrecognized enemy kinds: {unrecognized_kinds}")

        # Compute kind_masks
        kind_masks = {
            kind: kinds_array == kind for kind in ENEMY_REGISTRY.keys()
        }

        return cls(states=states_array, kinds=kinds_array, kind_masks=kind_masks)

    def position_array(self, kind: str) -> np.ndarray:
        mask = self.kind_masks.get(kind, np.zeros(self.states.shape[0], dtype=bool))
        return self.states[mask, :3]

    def velocity_array(self, kind: str) -> np.ndarray:
        mask = self.kind_masks.get(kind, np.zeros(self.states.shape[0], dtype=bool))
        return self.states[mask, 3:6]

    def acceleration_array(self, kind: str) -> np.ndarray:
        mask = self.kind_masks.get(kind, np.zeros(self.states.shape[0], dtype=bool))
        return self.states[mask, 6:9]


class FrameUpdate(BaseModel):
    playerState: PlayerState
    enemiesState: Optional[EnemiesState] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def parse_obj(cls, obj):
        # Convert raw playerState to PlayerState instance
        obj["playerState"] = cls._parse_player_state(obj.get("playerState"))

        # Convert raw enemies to EnemiesState instance
        obj["enemiesState"] = cls._parse_enemies_state(obj.get("enemies"))

        return super().parse_obj(obj)

    @staticmethod
    def _parse_player_state(player_state):
        if isinstance(player_state, list):
            return PlayerState.from_raw(raw=player_state)
        raise ValueError("Invalid playerState format")

    @staticmethod
    def _parse_enemies_state(enemies):
        if isinstance(enemies, dict) and enemies.get("states") and enemies.get("kinds"):
            return EnemiesState.from_raw(states=enemies.get("states"), kinds=enemies.get("kinds"))
        return None
