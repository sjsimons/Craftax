"""Language wrapper for Craftax Classic following BALROG format."""

from typing import Tuple, Optional, Dict, Any
import jax
import jax.numpy as jnp
from gymnax.environments import spaces

from craftax.craftax_classic.envs.craftax_symbolic_env import (
    CraftaxClassicSymbolicEnvNoAutoReset,
)
from craftax.craftax_classic.envs.craftax_state import EnvState, EnvParams, StaticEnvParams
from craftax.craftax_classic.renderer import render_craftax_text_balrog


class CraftaxClassicTextEnvNoAutoReset(CraftaxClassicSymbolicEnvNoAutoReset):
    """
    Craftax Classic wrapper with text observations in BALROG format.

    Args:
        static_env_params: Static environment parameters
        unique_items: Only show closest instance of each item/mob type
        precise_location: Use precise location format (e.g., "1 step north and 2 steps east")
                         vs simple format (e.g., "3 steps to your north-east")
    """

    def __init__(
        self,
        static_env_params: StaticEnvParams = None,
        unique_items: bool = True,
        precise_location: bool = False,
    ):
        super().__init__(static_env_params)
        self.unique_items = unique_items
        self.precise_location = precise_location

    def step_env(
        self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[Dict[str, Any], EnvState, float, bool, dict]:
        """Step environment and return combined symbolic and text observations."""
        symbolic_obs, state, reward, done, info = super().step_env(rng, state, action, params)

        text_obs = render_craftax_text_balrog(
            state,
            unique_items=self.unique_items,
            precise_location=self.precise_location
        )

        obs = {
            "symbolic": symbolic_obs,
            "text": text_obs,
        }

        return obs, state, reward, done, info

    def reset_env(
        self, rng: jax.Array, params: EnvParams
    ) -> Tuple[Dict[str, Any], EnvState]:
        """Reset environment and return combined symbolic and text observations."""
        symbolic_obs, state = super().reset_env(rng, params)

        text_obs = render_craftax_text_balrog(
            state,
            unique_items=self.unique_items,
            precise_location=self.precise_location
        )

        obs = {
            "symbolic": symbolic_obs,
            "text": text_obs,
        }

        return obs, state

    @property
    def name(self) -> str:
        return "Craftax-Classic-Text-NoAutoReset-v1"

    def observation_space(self, params: EnvParams) -> spaces.Dict:
        """Return observation space dictionary (symbolic only; text is a string)."""
        symbolic_space = super().observation_space(params)
        return spaces.Dict({
            "symbolic": symbolic_space,
        })


def make_craftax_classic_text_env(
    static_env_params: Optional[StaticEnvParams] = None,
    unique_items: bool = True,
    precise_location: bool = False,
) -> CraftaxClassicTextEnvNoAutoReset:
    """Create a Craftax Classic environment with text observations."""
    return CraftaxClassicTextEnvNoAutoReset(
        static_env_params=static_env_params,
        unique_items=unique_items,
        precise_location=precise_location,
    )
