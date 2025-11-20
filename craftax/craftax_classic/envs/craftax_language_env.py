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
    Craftax Classic environment wrapper that provides text observations in BALROG format.

    This wrapper augments the standard Craftax Classic environment with natural language
    descriptions of the game state, following the BALROG crafter format.

    Args:
        static_env_params: Static environment parameters
        unique_items: If True, only show closest instance of each item/mob type in text
        precise_location: If True, use precise location format (e.g., "1 step north and 2 steps east")
                         If False, use simple format (e.g., "3 steps to your north-east")
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
        """
        Step the environment and return text observation.

        Returns:
            obs: Dictionary with 'symbolic' (array) and 'text' (string) observations
            state: Updated environment state
            reward: Reward value
            done: Whether episode is terminal
            info: Additional information dictionary
        """
        # Call parent step to get symbolic observation and state
        symbolic_obs, state, reward, done, info = super().step_env(rng, state, action, params)

        # Generate text observation
        text_obs = render_craftax_text_balrog(
            state,
            unique_items=self.unique_items,
            precise_location=self.precise_location
        )

        # Return combined observation
        obs = {
            "symbolic": symbolic_obs,
            "text": text_obs,
        }

        return obs, state, reward, done, info

    def reset_env(
        self, rng: jax.Array, params: EnvParams
    ) -> Tuple[Dict[str, Any], EnvState]:
        """
        Reset the environment and return initial text observation.

        Returns:
            obs: Dictionary with 'symbolic' (array) and 'text' (string) observations
            state: Initial environment state
        """
        # Call parent reset to get symbolic observation and state
        symbolic_obs, state = super().reset_env(rng, params)

        # Generate text observation
        text_obs = render_craftax_text_balrog(
            state,
            unique_items=self.unique_items,
            precise_location=self.precise_location
        )

        # Return combined observation
        obs = {
            "symbolic": symbolic_obs,
            "text": text_obs,
        }

        return obs, state

    @property
    def name(self) -> str:
        return "Craftax-Classic-Text-NoAutoReset-v1"

    def observation_space(self, params: EnvParams) -> spaces.Dict:
        """
        Return observation space as a dictionary containing both symbolic and text observations.

        Note: The text observation is not included in the space definition as it's a string.
              Only the symbolic observation space is formally defined.
        """
        # Get parent's symbolic observation space
        symbolic_space = super().observation_space(params)

        # Return as dictionary space
        # Note: gymnax doesn't have a good string space type, so we just document
        # that 'text' key will contain a string
        return spaces.Dict({
            "symbolic": symbolic_space,
            # "text" will be a string, but gymnax doesn't have a String space type
        })


# Convenience function to create the environment
def make_craftax_classic_text_env(
    static_env_params: Optional[StaticEnvParams] = None,
    unique_items: bool = True,
    precise_location: bool = False,
) -> CraftaxClassicTextEnvNoAutoReset:
    """
    Create a Craftax Classic environment with text observations.

    Args:
        static_env_params: Static environment parameters (optional)
        unique_items: If True, only show closest instance of each item/mob type in text
        precise_location: If True, use precise location format, otherwise use simple format

    Returns:
        CraftaxClassicTextEnvNoAutoReset environment instance
    """
    return CraftaxClassicTextEnvNoAutoReset(
        static_env_params=static_env_params,
        unique_items=unique_items,
        precise_location=precise_location,
    )
