"""
Craftax specific renderers and converters.

This module contains Craftax implementations of the generic protocols.
"""

from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax.constants import MAX_OBS_DIM, OBS_DIM, BlockType, ItemType, Action
from craftax.craftax.craftax_state import EnvState
from craftax.environment_base.util import load_compressed_pickle


class BalrogRenderer:
    """
    BALROG format renderer for Craftax.

    Converts: EnvState → symbolic observation → text (BALROG format)
    """

    def __init__(
        self,
        unique_items: bool = True,
        precise_location: bool = False,
    ):
        """
        Initialize BalrogRenderer.

        Args:
            unique_items: If True, only show closest instance of each item/mob type
            precise_location: If True, use precise location descriptions
        """
        self.unique_items = unique_items
        self.precise_location = precise_location

    def render(self, state: EnvState) -> str:
        """Render state as text in BALROG format."""
        result = ""

        # Status alerts
        is_sleeping = bool(np.array(state.is_sleeping))
        is_resting = bool(np.array(state.is_resting))
        player_health = float(np.array(state.player_health))

        if is_sleeping:
            result += "You are sleeping, and will not be able take actions until energy is full.\n\n"
        elif is_resting:
            result += "You are resting to regain mana.\n\n"
        elif player_health <= 0:
            result += "You died.\n\n"

        # Get map view for current level
        current_map = state.map[state.player_level]
        obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)
        padded_grid = jnp.pad(
            current_map,
            (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
            constant_values=BlockType.OUT_OF_BOUNDS.value,
        )
        tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2
        map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

        # Get item map view
        padded_items_map = jnp.pad(
            state.item_map[state.player_level],
            (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
            constant_values=ItemType.NONE.value,
        )
        item_map_view = jax.lax.dynamic_slice(padded_items_map, tl_corner, OBS_DIM)

        # Get light map view
        padded_light_map = jnp.pad(
            state.light_map[state.player_level],
            (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
            constant_values=0.0,
        )
        light_map_view = jax.lax.dynamic_slice(padded_light_map, tl_corner, OBS_DIM)

        # Build mob map
        mob_types_per_class = 8
        mob_map = jnp.zeros((*OBS_DIM, 5 * mob_types_per_class), dtype=jnp.int32)

        def _add_mob_to_map(carry, mob_index):
            mob_map, mobs, mob_class_index = carry
            local_position = (
                mobs.position[mob_index]
                - state.player_position
                + jnp.array([OBS_DIM[0], OBS_DIM[1]]) // 2
            )
            on_screen = jnp.logical_and(
                local_position >= 0,
                local_position < jnp.array([OBS_DIM[0], OBS_DIM[1]]),
            ).all()
            on_screen *= mobs.mask[mob_index]

            mob_identifier = mob_class_index * mob_types_per_class + mobs.type_id[mob_index]
            mob_map = mob_map.at[local_position[0], local_position[1], mob_identifier].set(
                on_screen.astype(jnp.int32)
            )
            return (mob_map, mobs, mob_class_index), None

        # Add mobs from current level
        current_level = state.player_level

        (mob_map, _, _), _ = jax.lax.scan(
            _add_mob_to_map,
            (mob_map, jax.tree_util.tree_map(lambda x: x[current_level], state.melee_mobs), 0),
            jnp.arange(state.melee_mobs.mask.shape[1]),
        )
        (mob_map, _, _), _ = jax.lax.scan(
            _add_mob_to_map,
            (mob_map, jax.tree_util.tree_map(lambda x: x[current_level], state.passive_mobs), 1),
            jnp.arange(state.passive_mobs.mask.shape[1]),
        )
        (mob_map, _, _), _ = jax.lax.scan(
            _add_mob_to_map,
            (mob_map, jax.tree_util.tree_map(lambda x: x[current_level], state.ranged_mobs), 2),
            jnp.arange(state.ranged_mobs.mask.shape[1]),
        )
        (mob_map, _, _), _ = jax.lax.scan(
            _add_mob_to_map,
            (mob_map, jax.tree_util.tree_map(lambda x: x[current_level], state.mob_projectiles), 3),
            jnp.arange(state.mob_projectiles.mask.shape[1]),
        )
        (mob_map, _, _), _ = jax.lax.scan(
            _add_mob_to_map,
            (mob_map, jax.tree_util.tree_map(lambda x: x[current_level], state.player_projectiles), 4),
            jnp.arange(state.player_projectiles.mask.shape[1]),
        )

        # Determine what player is facing
        direction_offsets = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        center = (OBS_DIM[0] // 2, OBS_DIM[1] // 2)
        player_direction = int(np.array(state.player_direction))
        facing_offset = direction_offsets.get(player_direction, (0, 0))
        facing_pos = (center[0] + facing_offset[0], center[1] + facing_offset[1])

        if 0 <= facing_pos[0] < OBS_DIM[0] and 0 <= facing_pos[1] < OBS_DIM[1]:
            facing_item = self._get_tile_description(
                facing_pos, mob_map, item_map_view, map_view, light_map_view
            )
        else:
            facing_item = "nothing"

        # Collect visible objects
        mob_map_np = np.array(mob_map)
        map_view_np = np.array(map_view)
        item_map_view_np = np.array(item_map_view)
        light_map_view_np = np.array(light_map_view)
        obj_info_list = []

        for x in range(OBS_DIM[0]):
            for y in range(OBS_DIM[1]):
                if x == center[0] and y == center[1]:
                    continue

                # Check if tile is visible
                if light_map_view_np[x, y] < 0.05:
                    continue

                dx = y - center[1]
                dy = x - center[0]

                # Add mobs
                if mob_map_np[x, y].max() > 0.5:
                    mob_id = int(mob_map_np[x, y].argmax())
                    mob_name = self._mob_id_to_name(mob_id)
                    obj_info_list.append((mob_name, dx, dy))

                # Add items
                if item_map_view_np[x, y] != ItemType.NONE.value:
                    item_name = ItemType(item_map_view_np[x, y]).name.lower()
                    obj_info_list.append((item_name, dx, dy))

                # Add blocks (excluding common ground types)
                block_type = int(map_view_np[x, y])
                if block_type not in [
                    BlockType.GRASS.value,
                    BlockType.PATH.value,
                    BlockType.OUT_OF_BOUNDS.value,
                    BlockType.INVALID.value,
                    BlockType.DARKNESS.value,
                ]:
                    obj_info_list.append((BlockType(block_type).name.lower(), dx, dy))

        # Filter to closest of each item type if requested
        if self.unique_items:
            closest_items = {}
            for item_name, dx, dy in obj_info_list:
                distance = abs(dx) + abs(dy)
                if (
                    item_name not in closest_items
                    or distance < closest_items[item_name][0]
                ):
                    closest_items[item_name] = (distance, dx, dy)
            obj_info_list = [
                (name, dx, dy) for name, (_, dx, dy) in closest_items.items()
            ]

        # Format visible objects
        if obj_info_list:
            env_desc = "You see:\n" + "\n".join(
                [
                    f"- {name} {self._describe_location(dx, dy)}"
                    for name, dx, dy in obj_info_list
                ]
            )
        else:
            env_desc = "You see nothing away from you."

        result += env_desc + "\n\n"
        result += f"You face {facing_item} at your front.\n\n"

        # Player status
        status_lines = [
            f"- health: {float(np.array(state.player_health)):.1f}/9",
            f"- food: {int(np.array(state.player_food))}/9",
            f"- drink: {int(np.array(state.player_drink))}/9",
            f"- energy: {int(np.array(state.player_energy))}/9",
            f"- mana: {int(np.array(state.player_mana))}/9",
        ]
        result += "Your status:\n" + "\n".join(status_lines) + "\n\n"

        # Attributes
        xp = int(np.array(state.player_xp))
        dexterity = int(np.array(state.player_dexterity))
        strength = int(np.array(state.player_strength))
        intelligence = int(np.array(state.player_intelligence))

        if xp > 0 or dexterity > 0 or strength > 0 or intelligence > 0:
            attr_lines = []
            if xp > 0:
                attr_lines.append(f"- xp: {xp}")
            if dexterity > 0:
                attr_lines.append(f"- dexterity: {dexterity}")
            if strength > 0:
                attr_lines.append(f"- strength: {strength}")
            if intelligence > 0:
                attr_lines.append(f"- intelligence: {intelligence}")
            result += "Your attributes:\n" + "\n".join(attr_lines) + "\n\n"

        # Inventory
        inventory_items = []

        # Resources
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
        sapphire = int(np.array(state.inventory.sapphire))
        if sapphire > 0:
            inventory_items.append(f"- sapphire: {sapphire}")
        ruby = int(np.array(state.inventory.ruby))
        if ruby > 0:
            inventory_items.append(f"- ruby: {ruby}")
        sapling = int(np.array(state.inventory.sapling))
        if sapling > 0:
            inventory_items.append(f"- sapling: {sapling}")

        # Tools
        pickaxe_level = int(np.array(state.inventory.pickaxe))
        if pickaxe_level > 0:
            material = self._level_to_material(pickaxe_level)
            inventory_items.append(f"- {material.lower()}_pickaxe: 1")

        sword_level = int(np.array(state.inventory.sword))
        if sword_level > 0:
            material = self._level_to_material(sword_level)
            enchantment = int(np.array(state.sword_enchantment))
            if enchantment > 0:
                ench_name = self._level_to_enchantment(enchantment)
                inventory_items.append(f"- {material.lower()}_sword ({ench_name.lower()}_enchanted): 1")
            else:
                inventory_items.append(f"- {material.lower()}_sword: 1")

        bow_level = int(np.array(state.inventory.bow))
        if bow_level > 0:
            bow_enchantment = int(np.array(state.bow_enchantment))
            if bow_enchantment > 0:
                ench_name = self._level_to_enchantment(bow_enchantment)
                inventory_items.append(f"- bow ({ench_name.lower()}_enchanted): 1")
            else:
                inventory_items.append(f"- bow: 1")

        # Consumables
        arrows = int(np.array(state.inventory.arrows))
        if arrows > 0:
            inventory_items.append(f"- arrows: {arrows}")

        torches = int(np.array(state.inventory.torches))
        if torches > 0:
            inventory_items.append(f"- torches: {torches}")

        books = int(np.array(state.inventory.books))
        if books > 0:
            inventory_items.append(f"- books: {books}")

        # Armour
        armour_pieces = ["helmet", "chestplate", "leggings", "boots"]
        armour_array = np.array(state.inventory.armour)
        armour_enchantments = np.array(state.armour_enchantments)
        for i, piece_name in enumerate(armour_pieces):
            armour_level = int(armour_array[i])
            if armour_level > 0:
                material = self._get_armour_material(armour_level)
                enchantment = int(armour_enchantments[i])
                if enchantment > 0:
                    ench_name = self._level_to_enchantment(enchantment)
                    inventory_items.append(f"- {material.lower()}_{piece_name} ({ench_name.lower()}_enchanted): 1")
                else:
                    inventory_items.append(f"- {material.lower()}_{piece_name}: 1")

        # Potions
        potion_colors = ["red", "green", "blue", "pink", "cyan", "yellow"]
        potions_array = np.array(state.inventory.potions)
        for i, color in enumerate(potion_colors):
            potion_count = int(potions_array[i])
            if potion_count > 0:
                inventory_items.append(f"- {color}_potion: {potion_count}")

        # Dungeon level
        player_level = int(np.array(state.player_level))
        result += f"Current dungeon level: {player_level}\n\n"

        if inventory_items:
            result += "Your inventory:\n" + "\n".join(inventory_items)
        else:
            result += "You have nothing in your inventory."

        return result.strip()

    def _get_tile_description(self, pos, mob_map, item_map_view, map_view, light_map_view):
        """Get description of what's at a specific tile."""
        x, y = pos

        # Check if visible
        if np.array(light_map_view[x, y]) < 0.05:
            return "darkness"

        # Check for mobs
        mob_vals = np.array(mob_map[x, y])
        if mob_vals.max() > 0.5:
            mob_id = int(mob_vals.argmax())
            return self._mob_id_to_name(mob_id)

        # Check for items
        item_type = int(np.array(item_map_view[x, y]))
        if item_type != ItemType.NONE.value:
            return ItemType(item_type).name.lower()

        # Check for blocks
        block_type = int(np.array(map_view[x, y]))
        if block_type in [BlockType.GRASS.value, BlockType.PATH.value]:
            return "nothing"
        else:
            return BlockType(block_type).name.lower()

    def _mob_id_to_name(self, mob_id: int) -> str:
        """Convert mob identifier to name."""
        # Melee mobs (0-7)
        melee_names = ["zombie", "gnome_warrior", "orc_soldier", "lizard", "knight", "troll", "pigman", "frost_troll"]
        # Passive mobs (8-15)
        passive_names = ["cow", "bat", "snail", "", "", "", "", ""]
        # Ranged mobs (16-23)
        ranged_names = ["skeleton", "gnome_archer", "orc_mage", "kobold", "knight_archer", "deep_thing", "fire_elemental", "ice_elemental"]
        # Mob projectiles (24-31)
        mob_proj_names = ["arrow", "dagger", "fireball", "iceball", "arrow", "slimeball", "fireball", "iceball"]
        # Player projectiles (32-39)
        player_proj_names = ["arrow", "dagger", "fireball", "iceball", "arrow", "slimeball", "fireball", "iceball"]

        if 0 <= mob_id < 8:
            return melee_names[mob_id]
        elif 8 <= mob_id < 16:
            return passive_names[mob_id - 8] if passive_names[mob_id - 8] else "unknown"
        elif 16 <= mob_id < 24:
            return ranged_names[mob_id - 16]
        elif 24 <= mob_id < 32:
            return mob_proj_names[mob_id - 24]
        elif 32 <= mob_id < 40:
            return f"{player_proj_names[mob_id - 32]}_player"
        else:
            return "unknown"

    def _level_to_material(self, level: int) -> str:
        """Convert tool level to material name."""
        if level == 1:
            return "Wood"
        elif level == 2:
            return "Stone"
        elif level == 3:
            return "Iron"
        elif level == 4:
            return "Diamond"
        return "Unknown"

    def _get_armour_material(self, level: int) -> str:
        """Convert armour level to material name."""
        if level == 1:
            return "Iron"
        elif level == 2:
            return "Diamond"
        return "Unknown"

    def _level_to_enchantment(self, level: int) -> str:
        """Convert enchantment level to name."""
        if level == 0:
            return "None"
        elif level == 1:
            return "Fire"
        elif level == 2:
            return "Ice"
        return "Unknown"

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


class CraftaxPixelRenderer:
    """
    Pixel renderer for Craftax (for future use).

    Renders: EnvState → pixel observation (RGB array/image)
    """

    def render(self, state: EnvState) -> Any:
        """
        Render state as pixel observation.

        Args:
            state: Craftax environment state

        Returns:
            Pixel observation (RGB array/image)
        """
        # TODO: Implement pixel rendering
        raise NotImplementedError("Pixel rendering not yet implemented")


class CraftaxConverter:
    """
    Craftax trajectory converter.

    Implements TrajectoryConverterProtocol for Craftax.
    """

    def __init__(self, renderer: Any):
        """
        Initialize converter with a renderer.

        Args:
            renderer: TextRenderer instance (e.g., BalrogRenderer)
        """
        self.renderer = renderer

    def render(self, state: Any) -> str:
        """Render state to text using the configured renderer."""
        return self.renderer.render(state)

    def load_trajectory(self, file_path: str) -> Dict[str, List]:
        """Load Craftax trajectory file."""
        return load_compressed_pickle(file_path)

    def action_to_text(self, action: int) -> str:
        """Convert Craftax action number to text description."""
        action_descriptions = {
            0: "do nothing",
            1: "move left",
            2: "move right",
            3: "move up",
            4: "move down",
            5: "interact with object in front",
            6: "sleep",
            7: "place stone",
            8: "place crafting table",
            9: "place furnace",
            10: "place plant",
            11: "craft wooden pickaxe",
            12: "craft stone pickaxe",
            13: "craft iron pickaxe",
            14: "craft wooden sword",
            15: "craft stone sword",
            16: "craft iron sword",
            17: "rest",
            18: "descend stairs",
            19: "ascend stairs",
            20: "craft diamond pickaxe",
            21: "craft diamond sword",
            22: "craft iron armour",
            23: "craft diamond armour",
            24: "shoot arrow",
            25: "craft arrow",
            26: "cast fireball",
            27: "cast iceball",
            28: "place torch",
            29: "drink red potion",
            30: "drink green potion",
            31: "drink blue potion",
            32: "drink pink potion",
            33: "drink cyan potion",
            34: "drink yellow potion",
            35: "read book",
            36: "enchant sword",
            37: "enchant armour",
            38: "craft torch",
            39: "level up dexterity",
            40: "level up strength",
            41: "level up intelligence",
            42: "enchant bow",
        }
        return action_descriptions.get(action, f"unknown action {action}")


# Factory functions
def create_balrog_renderer(unique_items: bool = True, precise_location: bool = False):
    """Create a BalrogRenderer instance."""
    return BalrogRenderer(unique_items=unique_items, precise_location=precise_location)


def create_pixel_renderer(**kwargs):
    """Create a CraftaxPixelRenderer instance (for future use)."""
    return CraftaxPixelRenderer()


def make_craftax_converter(
    dir_path: str, renderer: Any, num_workers: int = 1, **renderer_kwargs
):
    """
    Create a TrajectoryTextConverter configured for Craftax.

    Args:
        dir_path: Path to trajectory directory
        renderer: TextRenderer instance
        num_workers: Number of worker processes for parallel rendering
        **renderer_kwargs: Additional renderer arguments

    Returns:
        TrajectoryTextConverter instance
    """
    from craftax.environment_base.trajectory import TrajectoryTextConverter

    converter = CraftaxConverter(renderer)
    return TrajectoryTextConverter(dir_path, converter, num_workers=num_workers)


def make_craftax_converter_from_functions(
    dir_path: str,
    unique_items: bool = True,
    precise_location: bool = False,
    num_workers: int = 1,
):
    """
    Create a converter configured for Craftax (backward compatibility).

    Args:
        dir_path: Path to trajectory directory
        unique_items: If True, only show closest instance of each item/mob type
        precise_location: If True, use precise location descriptions
        num_workers: Number of worker processes for parallel rendering

    Returns:
        TrajectoryTextConverter instance
    """
    from craftax.environment_base.trajectory import TrajectoryTextConverter

    renderer = create_balrog_renderer(
        unique_items=unique_items, precise_location=precise_location
    )
    converter = CraftaxConverter(renderer)

    return TrajectoryTextConverter(dir_path, converter, num_workers=num_workers)


def craftax_render_function(state, unique_items=True, precise_location=False):
    """
    Render Craftax state to text using standalone wrapper (backward compatibility).

    Args:
        state: Craftax environment state
        unique_items: If True, only show closest instance of each item/mob type
        precise_location: If True, use precise location descriptions

    Returns:
        Text representation of the state
    """
    renderer = create_balrog_renderer(
        unique_items=unique_items, precise_location=precise_location
    )
    return renderer.render(state)
