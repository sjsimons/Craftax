"""
Standalone Craftax Classic Language Wrapper

This is a self-contained wrapper that can be copied to any project.
It uses craftax_classic as an external library dependency.

Usage:
    Simply copy this file to your project and import it:

    from standalone_craftax_wrapper import CraftaxClassicLanguageWrapper

    env = CraftaxClassicLanguageWrapper()
    obs, state = env.reset(rng, params)
    # obs contains both 'symbolic' and 'text' observations

Design:
    - No internal craftax imports (only public API)
    - All rendering logic self-contained in this file
    - Drop-in replacement for base Craftax Classic environment
    - Compatible with any project that has craftax installed

Similar to BALROG's CrafterLanguageWrapper design.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple

# Import craftax_classic as an external library (only public API)
from craftax.craftax_classic.envs.craftax_symbolic_env import (
    CraftaxClassicSymbolicEnvNoAutoReset,
)
from craftax.craftax_classic.envs.craftax_state import EnvState, EnvParams
from craftax.craftax_classic.constants import OBS_DIM, MAX_OBS_DIM, BlockType, Action


class CraftaxClassicLanguageWrapper(CraftaxClassicSymbolicEnvNoAutoReset):
    """
    Standalone language wrapper for Craftax Classic.

    This wrapper extends the base environment to provide natural language
    text observations in addition to symbolic observations, following the
    BALROG crafter format.

    All rendering logic is self-contained in this file - no dependencies
    on craftax internals beyond the public API.

    Args:
        static_env_params: Static environment parameters (optional)
        unique_items: If True, only show closest instance of each item/mob type
        precise_location: If True, use precise location descriptions
    """

    def __init__(
        self,
        static_env_params=None,
        unique_items: bool = True,
        precise_location: bool = False,
    ):
        super().__init__(static_env_params)
        self.unique_items = unique_items
        self.precise_location = precise_location

    def step_env(
        self, rng: jax.Array, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[Dict[str, Any], EnvState, float, bool, dict]:
        """Step environment and return observations with text."""
        symbolic_obs, state, reward, done, info = super().step_env(rng, state, action, params)

        # Generate text observation (note: this happens outside of traced computation)
        text_obs = self._render_text(state)

        obs = {
            "symbolic": symbolic_obs,
            "text": text_obs,
        }

        return obs, state, reward, done, info

    def reset_env(
        self, rng: jax.Array, params: EnvParams
    ) -> Tuple[Dict[str, Any], EnvState]:
        """Reset environment and return observations with text."""
        symbolic_obs, state = super().reset_env(rng, params)

        # Generate text observation (note: this happens outside of traced computation)
        text_obs = self._render_text(state)

        obs = {
            "symbolic": symbolic_obs,
            "text": text_obs,
        }

        return obs, state

    @property
    def name(self) -> str:
        return "Craftax-Classic-Language-Wrapper-v1"

    # =========================================================================
    # Self-contained rendering logic (no internal dependencies)
    # =========================================================================

    def _render_text(self, state: EnvState) -> str:
        """
        Render state as text in BALROG format.

        This is completely self-contained and doesn't depend on any
        internal craftax rendering modules.

        Note: Converts JAX arrays to numpy/Python values to avoid issues during JIT compilation.
        """
        import numpy as np

        result = ""

        # Status section (sleeping/dead) - convert to Python values
        is_sleeping = bool(np.array(state.is_sleeping))
        player_health = int(np.array(state.player_health))

        if is_sleeping:
            result += "You are sleeping, and will not be able take actions until energy is full.\n\n"
        elif player_health <= 0:
            result += "You died.\n\n"

        # Get map view
        obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)
        padded_grid = jnp.pad(
            state.map,
            (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
            constant_values=BlockType.OUT_OF_BOUNDS.value,
        )
        tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2
        map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

        # Get mobs view
        mob_map = jnp.zeros((*OBS_DIM, 4), dtype=jnp.int32)

        def _add_mob_to_map(carry, mob_index):
            mob_map, mobs, mob_type_index = carry
            local_position = (
                mobs.position[mob_index]
                - state.player_position
                + jnp.array([OBS_DIM[0], OBS_DIM[1]]) // 2
            )
            on_screen = jnp.logical_and(
                local_position >= 0, local_position < jnp.array([OBS_DIM[0], OBS_DIM[1]])
            ).all()
            on_screen *= mobs.mask[mob_index]
            mob_map = mob_map.at[local_position[0], local_position[1], mob_type_index].set(
                on_screen.astype(jnp.int32)
            )
            return (mob_map, mobs, mob_type_index), None

        (mob_map, _, _), _ = jax.lax.scan(
            _add_mob_to_map, (mob_map, state.zombies, 0), jnp.arange(state.zombies.mask.shape[0])
        )
        (mob_map, _, _), _ = jax.lax.scan(
            _add_mob_to_map, (mob_map, state.cows, 1), jnp.arange(state.cows.mask.shape[0])
        )
        (mob_map, _, _), _ = jax.lax.scan(
            _add_mob_to_map, (mob_map, state.skeletons, 2), jnp.arange(state.skeletons.mask.shape[0])
        )
        (mob_map, _, _), _ = jax.lax.scan(
            _add_mob_to_map, (mob_map, state.arrows, 3), jnp.arange(state.arrows.mask.shape[0])
        )

        # Determine what player is facing
        direction_offsets = {
            1: (0, -1),  # left
            2: (0, 1),   # right
            3: (-1, 0),  # up
            4: (1, 0),   # down
        }

        center = (OBS_DIM[0] // 2, OBS_DIM[1] // 2)
        player_direction = int(np.array(state.player_direction))
        facing_offset = direction_offsets.get(player_direction, (0, 0))
        facing_pos = (center[0] + facing_offset[0], center[1] + facing_offset[1])

        if 0 <= facing_pos[0] < OBS_DIM[0] and 0 <= facing_pos[1] < OBS_DIM[1]:
            # Convert to numpy for conditional checks
            facing_mob_vals = np.array(mob_map[facing_pos[0], facing_pos[1]])
            if facing_mob_vals.max() > 0.5:
                mob_id = int(facing_mob_vals.argmax())
                mob_names = ["zombie", "cow", "skeleton", "arrow"]
                facing_item = mob_names[mob_id]
            else:
                facing_block = int(np.array(map_view[facing_pos[0], facing_pos[1]]))
                if facing_block in [BlockType.GRASS.value, BlockType.PATH.value]:
                    facing_item = "nothing"
                else:
                    facing_item = BlockType(facing_block).name.lower()
        else:
            facing_item = "nothing"

        # Convert to numpy for iteration
        mob_map_np = np.array(mob_map)
        map_view_np = np.array(map_view)

        # Collect all visible objects
        obj_info_list = []
        for x in range(OBS_DIM[0]):
            for y in range(OBS_DIM[1]):
                if x == center[0] and y == center[1]:
                    continue  # Skip player position

                dx = y - center[1]
                dy = x - center[0]

                # Check for mobs
                if mob_map_np[x, y].max() > 0.5:
                    mob_id = int(mob_map_np[x, y].argmax())
                    mob_names = ["zombie", "cow", "skeleton", "arrow"]
                    obj_info_list.append((mob_names[mob_id], dx, dy))

                # Check for blocks (skip grass and path)
                block_type = int(map_view_np[x, y])
                if block_type not in [BlockType.GRASS.value, BlockType.PATH.value,
                                     BlockType.OUT_OF_BOUNDS.value, BlockType.INVALID.value]:
                    obj_info_list.append((BlockType(block_type).name.lower(), dx, dy))

        # Filter to unique items if requested
        if self.unique_items:
            closest_items = {}
            for item_name, dx, dy in obj_info_list:
                distance = abs(dx) + abs(dy)
                if item_name not in closest_items or distance < closest_items[item_name][0]:
                    closest_items[item_name] = (distance, dx, dy)
            obj_info_list = [(name, dx, dy) for name, (_, dx, dy) in closest_items.items()]

        # Format environment description
        if obj_info_list:
            env_desc = "You see:\n" + "\n".join(
                [f"- {name} {self._describe_location(dx, dy)}" for name, dx, dy in obj_info_list]
            )
        else:
            env_desc = "You see nothing away from you."

        result += env_desc + "\n\n"
        result += f"You face {facing_item} at your front.\n\n"

        # Status section - convert to Python ints
        status_lines = [
            f"- health: {int(np.array(state.player_health))}/9",
            f"- food: {int(np.array(state.player_food))}/9",
            f"- drink: {int(np.array(state.player_drink))}/9",
            f"- energy: {int(np.array(state.player_energy))}/9",
        ]
        result += "Your status:\n" + "\n".join(status_lines) + "\n\n"

        # Inventory items - convert to Python ints
        inventory_items = []
        wood = int(np.array(state.inventory.wood))
        if wood > 0:
            inventory_items.append(f"- wood: {wood}")
        stone = int(np.array(state.inventory.stone))
        if stone > 0:
            inventory_items.append(f"- stone: {stone}")
        coal = int(np.array(state.inventory.coal))
        if coal > 0:
            inventory_items.append(f"- coal: {coal}")
        iron = int(np.array(state.inventory.iron))
        if iron > 0:
            inventory_items.append(f"- iron: {iron}")
        diamond = int(np.array(state.inventory.diamond))
        if diamond > 0:
            inventory_items.append(f"- diamond: {diamond}")
        sapling = int(np.array(state.inventory.sapling))
        if sapling > 0:
            inventory_items.append(f"- sapling: {sapling}")
        wood_pickaxe = int(np.array(state.inventory.wood_pickaxe))
        if wood_pickaxe > 0:
            inventory_items.append(f"- wood_pickaxe: {wood_pickaxe}")
        stone_pickaxe = int(np.array(state.inventory.stone_pickaxe))
        if stone_pickaxe > 0:
            inventory_items.append(f"- stone_pickaxe: {stone_pickaxe}")
        iron_pickaxe = int(np.array(state.inventory.iron_pickaxe))
        if iron_pickaxe > 0:
            inventory_items.append(f"- iron_pickaxe: {iron_pickaxe}")
        wood_sword = int(np.array(state.inventory.wood_sword))
        if wood_sword > 0:
            inventory_items.append(f"- wood_sword: {wood_sword}")
        stone_sword = int(np.array(state.inventory.stone_sword))
        if stone_sword > 0:
            inventory_items.append(f"- stone_sword: {stone_sword}")
        iron_sword = int(np.array(state.inventory.iron_sword))
        if iron_sword > 0:
            inventory_items.append(f"- iron_sword: {iron_sword}")

        if inventory_items:
            result += "Your inventory:\n" + "\n".join(inventory_items)
        else:
            result += "You have nothing in your inventory."

        return result.strip()

    def _describe_location(self, dx: int, dy: int) -> str:
        """Describe relative location based on configuration."""
        if self.precise_location:
            return self._describe_location_precise(dx, dy)
        else:
            return self._describe_location_simple(dx, dy)

    def _describe_location_precise(self, dx: int, dy: int) -> str:
        """Precise format: '1 step north and 2 steps east'"""
        desc = []
        if dy < 0:
            desc.append(f"{abs(dy)} step{'s' if abs(dy) > 1 else ''} north")
        elif dy > 0:
            desc.append(f"{abs(dy)} step{'s' if abs(dy) > 1 else ''} south")
        if dx < 0:
            desc.append(f"{abs(dx)} step{'s' if abs(dx) > 1 else ''} west")
        elif dx > 0:
            desc.append(f"{abs(dx)} step{'s' if abs(dx) > 1 else ''} east")
        return " and ".join(desc) if desc else "at your location"

    def _describe_location_simple(self, dx: int, dy: int) -> str:
        """Simple format: '3 steps to your north-east'"""
        directions = []
        if dy < 0:
            directions.append("north")
        elif dy > 0:
            directions.append("south")
        if dx < 0:
            directions.append("west")
        elif dx > 0:
            directions.append("east")

        distance = abs(dx) + abs(dy)
        if distance == 0:
            return "at your location"
        return f"{distance} step{'s' if distance > 1 else ''} to your {'-'.join(directions)}"


# ============================================================================
# Convenience factory function (optional)
# ============================================================================

def make_language_env(unique_items=True, precise_location=False, static_env_params=None):
    """
    Factory function to create a Craftax Classic language wrapper.

    Args:
        unique_items: Only show closest instance of each item type
        precise_location: Use precise location format
        static_env_params: Static environment parameters

    Returns:
        CraftaxClassicLanguageWrapper instance

    Example:
        env = make_language_env(unique_items=True)
        env_params = env.default_params
        obs, state = env.reset(rng, env_params)
        print(obs['text'])
    """
    return CraftaxClassicLanguageWrapper(
        static_env_params=static_env_params,
        unique_items=unique_items,
        precise_location=precise_location,
    )
