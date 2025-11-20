from functools import partial

from craftax.craftax_classic.constants import *


def render_craftax_symbolic(state):
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # Map
    padded_grid = jnp.pad(
        state.map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2

    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)
    map_view_one_hot = jax.nn.one_hot(map_view, num_classes=len(BlockType))

    # Mobs
    mob_map = jnp.zeros((*OBS_DIM, 4), dtype=jnp.uint8)  # 4 types of mobs

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
            on_screen.astype(jnp.uint8)
        )

        return (mob_map, mobs, mob_type_index), None

    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.zombies, 0),
        jnp.arange(state.zombies.mask.shape[0]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map, (mob_map, state.cows, 1), jnp.arange(state.cows.mask.shape[0])
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.skeletons, 2),
        jnp.arange(state.skeletons.mask.shape[0]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.arrows, 3),
        jnp.arange(state.arrows.mask.shape[0]),
    )

    all_map = jnp.concatenate([map_view_one_hot, mob_map], axis=-1)

    # Inventory
    inventory = (
        jnp.array(
            [
                state.inventory.wood,
                state.inventory.stone,
                state.inventory.coal,
                state.inventory.iron,
                state.inventory.diamond,
                state.inventory.sapling,
                state.inventory.wood_pickaxe,
                state.inventory.stone_pickaxe,
                state.inventory.iron_pickaxe,
                state.inventory.wood_sword,
                state.inventory.stone_sword,
                state.inventory.iron_sword,
            ]
        ).astype(jnp.float16)
        / 10.0
    )

    intrinsics = (
        jnp.array(
            [
                state.player_health,
                state.player_food,
                state.player_drink,
                state.player_energy,
            ]
        ).astype(jnp.float16)
        / 10.0
    )

    direction = jax.nn.one_hot(state.player_direction - 1, num_classes=4)

    all_flattened = jnp.concatenate(
        [
            all_map.flatten(),
            inventory,
            intrinsics,
            direction,
            jnp.array([state.light_level, state.is_sleeping]),
        ]
    )

    return all_flattened


@partial(jax.jit, static_argnums=(1,))
def render_craftax_pixels(state, block_pixel_size):
    textures = TEXTURES[block_pixel_size]
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # RENDER MAP
    # Get view of map
    padded_grid = jnp.pad(
        state.map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2

    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

    # Render map tiles
    map_pixels_indexes = jnp.repeat(
        jnp.repeat(map_view, repeats=block_pixel_size, axis=0),
        repeats=block_pixel_size,
        axis=1,
    )
    map_pixels_indexes = jnp.expand_dims(map_pixels_indexes, axis=-1)
    map_pixels_indexes = jnp.repeat(map_pixels_indexes, repeats=3, axis=2)

    map_pixels = jnp.zeros(
        (OBS_DIM[0] * block_pixel_size, OBS_DIM[1] * block_pixel_size, 3),
        dtype=jnp.float32,
    )

    def _add_block_type_to_pixels(pixels, block_index):
        return (
            pixels
            + textures["full_map_block_textures"][block_index]
            * (map_pixels_indexes == block_index),
            None,
        )

    map_pixels, _ = jax.lax.scan(
        _add_block_type_to_pixels, map_pixels, jnp.arange(len(BlockType))
    )

    # Render player
    player_texture_index = jax.lax.select(
        state.is_sleeping, 4, state.player_direction - 1
    )
    map_pixels = (
        map_pixels
        * (1 - textures["full_map_player_textures_alpha"][player_texture_index])
        + textures["full_map_player_textures"][player_texture_index]
        * textures["full_map_player_textures_alpha"][player_texture_index]
    )

    # Render mobs
    # Zombies

    def _add_zombie_to_pixels(pixels, zombie_index):
        local_position = (
            state.zombies.position[zombie_index]
            - state.player_position
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all()
        on_screen *= state.zombies.mask[zombie_index]

        zombie_texture = textures["zombie_texture"] * on_screen

        zombie_texture_with_background = (
            1 - textures["zombie_texture_alpha"] * on_screen
        )

        zombie_texture_with_background = (
            zombie_texture_with_background
            * jax.lax.dynamic_slice(
                pixels,
                (
                    local_position[0] * block_pixel_size,
                    local_position[1] * block_pixel_size,
                    0,
                ),
                (block_pixel_size, block_pixel_size, 3),
            )
        )

        zombie_texture_with_background = (
            zombie_texture_with_background
            + zombie_texture * textures["zombie_texture_alpha"]
        )

        pixels = jax.lax.dynamic_update_slice(
            pixels,
            zombie_texture_with_background,
            (
                local_position[0] * block_pixel_size,
                local_position[1] * block_pixel_size,
                0,
            ),
        )

        return pixels, None

    map_pixels, _ = jax.lax.scan(
        _add_zombie_to_pixels, map_pixels, jnp.arange(state.zombies.mask.shape[0])
    )

    def _add_cow_to_pixels(pixels, cow_index):
        local_position = (
            state.cows.position[cow_index]
            - state.player_position
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all()
        on_screen *= state.cows.mask[cow_index]

        cow_texture = textures["cow_texture"] * on_screen

        cow_texture_with_background = 1 - textures["cow_texture_alpha"] * on_screen

        cow_texture_with_background = (
            cow_texture_with_background
            * jax.lax.dynamic_slice(
                pixels,
                (
                    local_position[0] * block_pixel_size,
                    local_position[1] * block_pixel_size,
                    0,
                ),
                (block_pixel_size, block_pixel_size, 3),
            )
        )

        cow_texture_with_background = (
            cow_texture_with_background + cow_texture * textures["cow_texture_alpha"]
        )

        pixels = jax.lax.dynamic_update_slice(
            pixels,
            cow_texture_with_background,
            (
                local_position[0] * block_pixel_size,
                local_position[1] * block_pixel_size,
                0,
            ),
        )

        return pixels, None

    map_pixels, _ = jax.lax.scan(
        _add_cow_to_pixels, map_pixels, jnp.arange(state.cows.mask.shape[0])
    )

    def _add_skeleton_to_pixels(pixels, skeleton_index):
        local_position = (
            state.skeletons.position[skeleton_index]
            - state.player_position
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all()
        on_screen *= state.skeletons.mask[skeleton_index]

        skeleton_texture = textures["skeleton_texture"] * on_screen

        skeleton_texture_with_background = (
            1 - textures["skeleton_texture_alpha"] * on_screen
        )

        skeleton_texture_with_background = (
            skeleton_texture_with_background
            * jax.lax.dynamic_slice(
                pixels,
                (
                    local_position[0] * block_pixel_size,
                    local_position[1] * block_pixel_size,
                    0,
                ),
                (block_pixel_size, block_pixel_size, 3),
            )
        )

        skeleton_texture_with_background = (
            skeleton_texture_with_background
            + skeleton_texture * textures["skeleton_texture_alpha"]
        )

        pixels = jax.lax.dynamic_update_slice(
            pixels,
            skeleton_texture_with_background,
            (
                local_position[0] * block_pixel_size,
                local_position[1] * block_pixel_size,
                0,
            ),
        )

        return pixels, None

    map_pixels, _ = jax.lax.scan(
        _add_skeleton_to_pixels, map_pixels, jnp.arange(state.skeletons.mask.shape[0])
    )

    def _add_arrow_to_pixels(pixels, arrow_index):
        local_position = (
            state.arrows.position[arrow_index]
            - state.player_position
            + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < obs_dim_array
        ).all()
        on_screen *= state.arrows.mask[arrow_index]

        arrow_texture = textures["arrow_texture"]
        arrow_texture_alpha = textures["arrow_texture_alpha"]

        flipped_arrow_texture = jnp.flip(arrow_texture, axis=0)
        flipped_arrow_texture_alpha = jnp.flip(arrow_texture_alpha, axis=0)
        flip_arrow = jnp.logical_or(
            state.arrow_directions[arrow_index, 0] > 0,
            state.arrow_directions[arrow_index, 1] > 0,
        )

        arrow_texture = jax.lax.select(
            flip_arrow,
            flipped_arrow_texture,
            arrow_texture,
        )
        arrow_texture_alpha = jax.lax.select(
            flip_arrow,
            flipped_arrow_texture_alpha,
            arrow_texture_alpha,
        )

        transposed_arrow_texture = jnp.transpose(arrow_texture, (1, 0, 2))
        transposed_arrow_texture_alpha = jnp.transpose(arrow_texture_alpha, (1, 0, 2))

        arrow_texture = jax.lax.select(
            state.arrow_directions[arrow_index, 1] != 0,
            transposed_arrow_texture,
            arrow_texture,
        )
        arrow_texture_alpha = jax.lax.select(
            state.arrow_directions[arrow_index, 1] != 0,
            transposed_arrow_texture_alpha,
            arrow_texture_alpha,
        )

        arrow_texture = arrow_texture * on_screen
        arrow_texture_with_background = 1 - arrow_texture_alpha * on_screen

        arrow_texture_with_background = (
            arrow_texture_with_background
            * jax.lax.dynamic_slice(
                pixels,
                (
                    local_position[0] * block_pixel_size,
                    local_position[1] * block_pixel_size,
                    0,
                ),
                (block_pixel_size, block_pixel_size, 3),
            )
        )

        arrow_texture_with_background = (
            arrow_texture_with_background + arrow_texture * arrow_texture_alpha
        )

        pixels = jax.lax.dynamic_update_slice(
            pixels,
            arrow_texture_with_background,
            (
                local_position[0] * block_pixel_size,
                local_position[1] * block_pixel_size,
                0,
            ),
        )

        return pixels, None

    map_pixels, _ = jax.lax.scan(
        _add_arrow_to_pixels, map_pixels, jnp.arange(state.arrows.mask.shape[0])
    )

    # Apply night
    daylight = state.light_level

    night_static_intensity = 2 * (0.5 - daylight)
    night_static_intensity = jnp.maximum(night_static_intensity, 0.0)
    night_with_static = (
        jax.random.uniform(state.state_rng, map_pixels.shape[:2]) * 95 + 32
    )
    night_static_mask = (
        night_static_intensity * textures["night_noise_intensity_texture"]
    )
    night_with_static = (
        1 - night_static_mask
    ) * map_pixels + night_static_mask * night_with_static[:, :, None]

    night_pixels = jax.lax.select(daylight < 0.5, night_with_static, map_pixels)

    # Enhance
    enhance_factor = 0.4
    lum = (
        0.299 * night_pixels[:, :, 0]
        + 0.587 * night_pixels[:, :, 1]
        + 0.114 * night_pixels[:, :, 2]
    )
    lum = jnp.expand_dims(lum, axis=-1).repeat(3, axis=-1)
    night_pixels = night_pixels * enhance_factor + (1 - enhance_factor) * lum

    # Tint
    night_pixels = 0.5 * night_pixels + 0.5 * textures["night_texture"]

    # Blend with map pixels
    map_pixels = daylight * map_pixels + (1 - daylight) * night_pixels

    # Apply sleep
    sleep_pixels = (
        0.299 * map_pixels[:, :, 0]
        + 0.587 * map_pixels[:, :, 1]
        + 0.114 * map_pixels[:, :, 2]
    )
    sleep_pixels = (0.5 * sleep_pixels)[:, :, None] + (0.5 * jnp.array([0, 0, 16]))[
        None, None, :
    ]
    map_pixels = (1 - state.is_sleeping) * map_pixels + state.is_sleeping * sleep_pixels

    # Render mob map
    # mob_map_pixels = (
    #     jnp.array([[[128, 0, 0]]]).repeat(OBS_DIM, axis=0).repeat(OBS_DIM, axis=1)
    # )
    # padded_mob_map = jnp.pad(state.mob_map, OBS_DIM // 2 + 2, constant_values=False)
    # mob_map_view = jax.lax.dynamic_slice(padded_mob_map, tl_corner, (OBS_DIM, OBS_DIM))
    # mob_map_pixels = mob_map_pixels * jnp.expand_dims(mob_map_view, axis=-1)
    # mob_map_pixels = mob_map_pixels.repeat(block_pixel_size, axis=0).repeat(
    #     block_pixel_size, axis=1
    # )
    # map_pixels = map_pixels + mob_map_pixels

    # RENDER INVENTORY
    inv_pixel_left_space = (block_pixel_size - int(0.8 * block_pixel_size)) // 2 - 1
    inv_pixel_right_space = (
        block_pixel_size - int(0.8 * block_pixel_size) - inv_pixel_left_space
    )

    inv_pixels = jnp.zeros(
        (INVENTORY_OBS_HEIGHT * block_pixel_size, OBS_DIM[1] * block_pixel_size, 3),
        dtype=jnp.float32,
    )

    number_size = int(block_pixel_size * 0.6)
    number_offset = block_pixel_size - number_size

    def _render_number(pixels, number, x, y):
        pixels = pixels.at[
            y * block_pixel_size + number_offset - 1 : (y + 1) * block_pixel_size - 1,
            x * block_pixel_size + number_offset - 1 : (x + 1) * block_pixel_size - 1,
        ].mul(1 - textures["number_textures_alpha"][number])

        pixels = pixels.at[
            y * block_pixel_size + number_offset - 1 : (y + 1) * block_pixel_size - 1,
            x * block_pixel_size + number_offset - 1 : (x + 1) * block_pixel_size - 1,
        ].add(textures["number_textures"][number])

        return pixels

    # Render player stats
    health_texture = jax.lax.select(
        state.player_health > 0,
        textures["health_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
    ].set(health_texture)
    inv_pixels = _render_number(inv_pixels, state.player_health, 0, 0)

    hunger_texture = jax.lax.select(
        state.player_food > 0,
        textures["hunger_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        block_pixel_size
        + inv_pixel_left_space : block_pixel_size * 2
        - inv_pixel_right_space,
    ].set(hunger_texture)
    inv_pixels = _render_number(inv_pixels, state.player_food, 1, 0)

    thirst_texture = jax.lax.select(
        state.player_drink > 0,
        textures["thirst_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        block_pixel_size * 2
        + inv_pixel_left_space : block_pixel_size * 3
        - inv_pixel_right_space,
    ].set(thirst_texture)
    inv_pixels = _render_number(inv_pixels, state.player_drink, 2, 0)

    energy_texture = jax.lax.select(
        state.player_energy > 0,
        textures["energy_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        block_pixel_size * 3
        + inv_pixel_left_space : block_pixel_size * 4
        - inv_pixel_right_space,
    ].set(energy_texture)
    inv_pixels = _render_number(inv_pixels, state.player_energy, 3, 0)

    # Render inventory

    inv_wood_texture = jax.lax.select(
        state.inventory.wood > 0,
        textures["smaller_block_textures"][BlockType.WOOD.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        block_pixel_size * 5
        + inv_pixel_left_space : block_pixel_size * 6
        - inv_pixel_right_space,
    ].set(inv_wood_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.wood, 5, 0)

    inv_stone_texture = jax.lax.select(
        state.inventory.stone > 0,
        textures["smaller_block_textures"][BlockType.STONE.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        block_pixel_size * 6
        + inv_pixel_left_space : block_pixel_size * 7
        - inv_pixel_right_space,
    ].set(inv_stone_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.stone, 6, 0)

    inv_coal_texture = jax.lax.select(
        state.inventory.coal > 0,
        textures["smaller_block_textures"][BlockType.COAL.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        block_pixel_size * 7
        + inv_pixel_left_space : block_pixel_size * 8
        - inv_pixel_right_space,
    ].set(inv_coal_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.coal, 7, 0)

    inv_iron_texture = jax.lax.select(
        state.inventory.iron > 0,
        textures["smaller_block_textures"][BlockType.IRON.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        block_pixel_size * 8
        + inv_pixel_left_space : block_pixel_size * 9
        - inv_pixel_right_space,
    ].set(inv_iron_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.iron, 8, 0)

    inv_diamond_texture = jax.lax.select(
        state.inventory.diamond > 0,
        textures["smaller_block_textures"][BlockType.DIAMOND.value],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        block_pixel_size
        + inv_pixel_left_space : 2 * block_pixel_size
        - inv_pixel_right_space,
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
    ].set(inv_diamond_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.diamond, 0, 1)

    inv_sapling_texture = jax.lax.select(
        state.inventory.sapling > 0,
        textures["sapling_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        inv_pixel_left_space : block_pixel_size - inv_pixel_right_space,
        block_pixel_size * 4
        + inv_pixel_left_space : block_pixel_size * 5
        - inv_pixel_right_space,
    ].set(inv_sapling_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.sapling, 4, 0)

    # Render tools
    # Wooden pickaxe
    wooden_pickaxe_maybe_texture = jax.lax.select(
        state.inventory.wood_pickaxe > 0,
        textures["wood_pickaxe_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        block_pixel_size
        + inv_pixel_left_space : block_pixel_size * 2
        - inv_pixel_right_space,
        block_pixel_size
        + inv_pixel_left_space : block_pixel_size * 2
        - inv_pixel_right_space,
    ].set(wooden_pickaxe_maybe_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.wood_pickaxe, 1, 1)

    # Stone pickaxe
    stone_pickaxe_maybe_texture = jax.lax.select(
        state.inventory.stone_pickaxe > 0,
        textures["stone_pickaxe_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        block_pixel_size
        + inv_pixel_left_space : block_pixel_size * 2
        - inv_pixel_right_space,
        block_pixel_size * 2
        + inv_pixel_left_space : block_pixel_size * 3
        - inv_pixel_right_space,
    ].set(stone_pickaxe_maybe_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.stone_pickaxe, 2, 1)

    # Iron pickaxe
    iron_pickaxe_maybe_texture = jax.lax.select(
        state.inventory.iron_pickaxe > 0,
        textures["iron_pickaxe_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        block_pixel_size
        + inv_pixel_left_space : block_pixel_size * 2
        - inv_pixel_right_space,
        block_pixel_size * 3
        + inv_pixel_left_space : block_pixel_size * 4
        - inv_pixel_right_space,
    ].set(iron_pickaxe_maybe_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.iron_pickaxe, 3, 1)

    # Wooden sword
    wooden_sword_maybe_texture = jax.lax.select(
        state.inventory.wood_sword > 0,
        textures["wood_sword_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        block_pixel_size
        + inv_pixel_left_space : block_pixel_size * 2
        - inv_pixel_right_space,
        block_pixel_size * 4
        + inv_pixel_left_space : block_pixel_size * 5
        - inv_pixel_right_space,
    ].set(wooden_sword_maybe_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.wood_sword, 4, 1)

    # Stone sword
    stone_sword_maybe_texture = jax.lax.select(
        state.inventory.stone_sword > 0,
        textures["stone_sword_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        block_pixel_size
        + inv_pixel_left_space : block_pixel_size * 2
        - inv_pixel_right_space,
        block_pixel_size * 5
        + inv_pixel_left_space : block_pixel_size * 6
        - inv_pixel_right_space,
    ].set(stone_sword_maybe_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.stone_sword, 5, 1)

    # Iron sword
    iron_sword_maybe_texture = jax.lax.select(
        state.inventory.iron_sword > 0,
        textures["iron_sword_texture"],
        textures["smaller_empty_texture"],
    )
    inv_pixels = inv_pixels.at[
        block_pixel_size
        + inv_pixel_left_space : block_pixel_size * 2
        - inv_pixel_right_space,
        block_pixel_size * 6
        + inv_pixel_left_space : block_pixel_size * 7
        - inv_pixel_right_space,
    ].set(iron_sword_maybe_texture)
    inv_pixels = _render_number(inv_pixels, state.inventory.iron_sword, 6, 1)

    # Combine map and inventory
    pixels = jnp.concatenate([map_pixels, inv_pixels], axis=0)

    # # Downscale by 2
    # pixels = pixels[::downscale, ::downscale]

    return pixels


def render_pixels_empty(block_pixel_size):
    pixels = jnp.zeros(
        (
            OBS_DIM * block_pixel_size,
            (OBS_DIM + INVENTORY_OBS_HEIGHT) * block_pixel_size,
            3,
        ),
        dtype=jnp.float32,
    )
    return pixels


def render_craftax_text(state):
    text_obs = "Map:\n"

    # Map
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # Map
    padded_grid = jnp.pad(
        state.map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2

    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

    # Mobs
    mob_map = jnp.zeros((*OBS_DIM, 4), dtype=jnp.int32)  # 4 types: zombie, cow, skeleton, arrow

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
        _add_mob_to_map,
        (mob_map, state.zombies, 0),
        jnp.arange(state.zombies.mask.shape[0]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.cows, 1),
        jnp.arange(state.cows.mask.shape[0]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.skeletons, 2),
        jnp.arange(state.skeletons.mask.shape[0]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.arrows, 3),
        jnp.arange(state.arrows.mask.shape[0]),
    )

    def mob_id_to_name(id):
        if id == 0:
            return "Zombie"
        elif id == 1:
            return "Cow"
        elif id == 2:
            return "Skeleton"
        elif id == 3:
            return "Arrow"

    for x in range(OBS_DIM[0]):
        for y in range(OBS_DIM[1]):
            text_obs += f"{y - OBS_DIM[1] // 2}, {x - OBS_DIM[0] // 2}: "
            if mob_map[x, y].max() > 0.5:
                text_obs += mob_id_to_name(mob_map[x, y].argmax()) + " on "
            text_obs += BlockType(map_view[x, y]).name.lower() + "\n"

    # Inventory
    text_obs += "\nInventory:\n"
    text_obs += f"Wood: {state.inventory.wood}\n"
    text_obs += f"Stone: {state.inventory.stone}\n"
    text_obs += f"Coal: {state.inventory.coal}\n"
    text_obs += f"Iron: {state.inventory.iron}\n"
    text_obs += f"Diamond: {state.inventory.diamond}\n"
    text_obs += f"Sapling: {state.inventory.sapling}\n"

    if state.inventory.wood_pickaxe > 0:
        text_obs += "Wood Pickaxe\n"
    if state.inventory.stone_pickaxe > 0:
        text_obs += "Stone Pickaxe\n"
    if state.inventory.iron_pickaxe > 0:
        text_obs += "Iron Pickaxe\n"
    if state.inventory.wood_sword > 0:
        text_obs += "Wood Sword\n"
    if state.inventory.stone_sword > 0:
        text_obs += "Stone Sword\n"
    if state.inventory.iron_sword > 0:
        text_obs += "Iron Sword\n"

    text_obs += f"Health: {state.player_health}\n"
    text_obs += f"Food: {state.player_food}\n"
    text_obs += f"Drink: {state.player_drink}\n"
    text_obs += f"Energy: {state.player_energy}\n"

    text_obs += f"Direction: {Action(state.player_direction).name.lower()}\n"

    text_obs += f"Light: {state.light_level}\n"
    text_obs += f"Is Sleeping: {state.is_sleeping}\n"

    return text_obs


def render_craftax_text_balrog(state, unique_items=True, precise_location=False):
    """
    Render craftax_classic state as text following BALROG crafter format.

    Args:
        state: The craftax_classic environment state
        unique_items: If True, only show closest instance of each item/mob type
        precise_location: If True, use precise location format (e.g., "1 step north and 2 steps east")
                         If False, use simple format (e.g., "3 steps to your north-east")

    Returns:
        String containing the text observation in BALROG format
    """
    result = ""

    # Status section (sleeping/dead)
    if state.is_sleeping:
        result += "You are sleeping, and will not be able take actions until energy is full.\n\n"
    elif state.player_health <= 0:
        result += "You died.\n\n"

    # Environment section
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # Get map view
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

    # Helper functions for location descriptions
    def describe_loc_precise(dx, dy):
        """Describe location in precise format: '1 step north and 2 steps east'"""
        desc = []
        if dy < 0:  # North
            desc.append(f"{abs(dy)} step{'s' if abs(dy) > 1 else ''} north")
        elif dy > 0:  # South
            desc.append(f"{abs(dy)} step{'s' if abs(dy) > 1 else ''} south")
        if dx < 0:  # West
            desc.append(f"{abs(dx)} step{'s' if abs(dx) > 1 else ''} west")
        elif dx > 0:  # East
            desc.append(f"{abs(dx)} step{'s' if abs(dx) > 1 else ''} east")
        return " and ".join(desc) if desc else "at your location"

    def describe_loc_simple(dx, dy):
        """Describe location in simple format: '3 steps to your north-east'"""
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

    describe_loc = describe_loc_precise if precise_location else describe_loc_simple

    # Determine what player is facing (player direction: 1=left, 2=right, 3=up, 4=down)
    direction_offsets = {
        1: (0, -1),  # left
        2: (0, 1),   # right
        3: (-1, 0),  # up
        4: (1, 0),   # down
    }

    center = (OBS_DIM[0] // 2, OBS_DIM[1] // 2)
    facing_offset = direction_offsets.get(int(state.player_direction), (0, 0))
    facing_pos = (center[0] + facing_offset[0], center[1] + facing_offset[1])

    if 0 <= facing_pos[0] < OBS_DIM[0] and 0 <= facing_pos[1] < OBS_DIM[1]:
        # Check for mob first
        if mob_map[facing_pos[0], facing_pos[1]].max() > 0.5:
            mob_id = int(mob_map[facing_pos[0], facing_pos[1]].argmax())
            mob_names = ["zombie", "cow", "skeleton", "arrow"]
            facing_item = mob_names[mob_id]
        else:
            facing_block = int(map_view[facing_pos[0], facing_pos[1]])
            # Skip grass and path (too common/boring)
            if facing_block in [BlockType.GRASS.value, BlockType.PATH.value]:
                facing_item = "nothing"
            else:
                facing_item = BlockType(facing_block).name.lower()
    else:
        facing_item = "nothing"

    # Collect all visible objects
    obj_info_list = []
    for x in range(OBS_DIM[0]):
        for y in range(OBS_DIM[1]):
            if x == center[0] and y == center[1]:
                continue  # Skip player position

            dx = y - center[1]
            dy = x - center[0]

            # Check for mobs
            if mob_map[x, y].max() > 0.5:
                mob_id = int(mob_map[x, y].argmax())
                mob_names = ["zombie", "cow", "skeleton", "arrow"]
                obj_info_list.append((mob_names[mob_id], dx, dy))

            # Check for blocks (skip grass and path)
            block_type = int(map_view[x, y])
            if block_type not in [BlockType.GRASS.value, BlockType.PATH.value,
                                 BlockType.OUT_OF_BOUNDS.value, BlockType.INVALID.value]:
                obj_info_list.append((BlockType(block_type).name.lower(), dx, dy))

    # Filter to unique items (closest of each type) if requested
    if unique_items:
        closest_items = {}
        for item_name, dx, dy in obj_info_list:
            distance = abs(dx) + abs(dy)
            if item_name not in closest_items or distance < closest_items[item_name][0]:
                closest_items[item_name] = (distance, dx, dy)
        obj_info_list = [(name, dx, dy) for name, (_, dx, dy) in closest_items.items()]

    # Format environment description
    if obj_info_list:
        env_desc = "You see:\n" + "\n".join(
            [f"- {name} {describe_loc(dx, dy)}" for name, dx, dy in obj_info_list]
        )
    else:
        env_desc = "You see nothing away from you."

    result += env_desc + "\n\n"
    result += f"You face {facing_item} at your front.\n\n"

    # Inventory and status section
    status_lines = [
        f"- health: {state.player_health}/9",
        f"- food: {state.player_food}/9",
        f"- drink: {state.player_drink}/9",
        f"- energy: {state.player_energy}/9",
    ]
    result += "Your status:\n" + "\n".join(status_lines) + "\n\n"

    # Inventory items
    inventory_items = []
    if state.inventory.wood > 0:
        inventory_items.append(f"- wood: {state.inventory.wood}")
    if state.inventory.stone > 0:
        inventory_items.append(f"- stone: {state.inventory.stone}")
    if state.inventory.coal > 0:
        inventory_items.append(f"- coal: {state.inventory.coal}")
    if state.inventory.iron > 0:
        inventory_items.append(f"- iron: {state.inventory.iron}")
    if state.inventory.diamond > 0:
        inventory_items.append(f"- diamond: {state.inventory.diamond}")
    if state.inventory.sapling > 0:
        inventory_items.append(f"- sapling: {state.inventory.sapling}")
    if state.inventory.wood_pickaxe > 0:
        inventory_items.append(f"- wood_pickaxe: {state.inventory.wood_pickaxe}")
    if state.inventory.stone_pickaxe > 0:
        inventory_items.append(f"- stone_pickaxe: {state.inventory.stone_pickaxe}")
    if state.inventory.iron_pickaxe > 0:
        inventory_items.append(f"- iron_pickaxe: {state.inventory.iron_pickaxe}")
    if state.inventory.wood_sword > 0:
        inventory_items.append(f"- wood_sword: {state.inventory.wood_sword}")
    if state.inventory.stone_sword > 0:
        inventory_items.append(f"- stone_sword: {state.inventory.stone_sword}")
    if state.inventory.iron_sword > 0:
        inventory_items.append(f"- iron_sword: {state.inventory.iron_sword}")

    if inventory_items:
        result += "Your inventory:\n" + "\n".join(inventory_items)
    else:
        result += "You have nothing in your inventory."

    return result.strip()
