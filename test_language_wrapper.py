#!/usr/bin/env python3
"""
Test script for the Craftax Classic language wrapper.

This demonstrates how to use the BALROG-style text environment wrapper
as a drop-in replacement for the standard Craftax Classic environment.
"""

import jax
import jax.numpy as jnp

from craftax.craftax_env import make_craftax_env_from_name


def main():
    print("=" * 80)
    print("CRAFTAX CLASSIC LANGUAGE WRAPPER TEST")
    print("=" * 80)
    print("\nThis demonstrates using the text environment wrapper following BALROG format.")
    print("The wrapper can be used just like the standard Craftax Classic environment,")
    print("but returns both symbolic and text observations.\n")

    # Create environment using the factory function
    print("Creating environment: Craftax-Classic-Text-v1")
    env = make_craftax_env_from_name("Craftax-Classic-Text-v1", auto_reset=False)
    env_params = env.default_params

    # Initialize environment
    rng = jax.random.PRNGKey(42)
    rng, reset_rng = jax.random.split(rng)

    print("Resetting environment...")
    obs, state = env.reset(reset_rng, env_params)

    print("\n" + "=" * 80)
    print("INITIAL OBSERVATION")
    print("=" * 80)
    print(f"\nObservation keys: {obs.keys()}")
    print(f"Symbolic observation shape: {obs['symbolic'].shape}")
    print(f"\nText observation:\n")
    print(obs['text'])

    # Take some actions
    print("\n\n" + "=" * 80)
    print("TAKING ACTIONS")
    print("=" * 80)

    actions_to_take = [
        (3, "Move North"),
        (2, "Move East"),
        (5, "Do (interact)"),
        (1, "Move West"),
    ]

    for action_id, action_name in actions_to_take:
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_rng, state, action_id, env_params)

        print(f"\n--- Action: {action_name} ---")
        if reward != 0:
            print(f"Reward: {reward}")

        # Show achievements if any were unlocked
        if "achievements" in info and info["achievements"].sum() > 0:
            print(f"Total achievements: {info['achievements'].sum()}")

    print("\n" + "=" * 80)
    print("OBSERVATION AFTER ACTIONS")
    print("=" * 80)
    print(f"\nText observation:\n")
    print(obs['text'])

    # Demonstrate creating wrapper with different parameters
    print("\n\n" + "=" * 80)
    print("CUSTOM WRAPPER PARAMETERS")
    print("=" * 80)

    # Import the wrapper directly for custom parameters
    from craftax.craftax_classic.envs.craftax_language_env import (
        CraftaxClassicTextEnvNoAutoReset
    )

    # Create with precise location format
    print("\nCreating environment with precise_location=True...")
    env_precise = CraftaxClassicTextEnvNoAutoReset(
        unique_items=True,
        precise_location=True
    )

    rng, reset_rng = jax.random.split(rng)
    obs_precise, state_precise = env_precise.reset(reset_rng, env_params)

    print("\nText observation with precise locations:\n")
    print(obs_precise['text'])

    # Create with all items (not just unique)
    print("\n\n" + "=" * 80)
    print("Creating environment with unique_items=False...")
    env_all = CraftaxClassicTextEnvNoAutoReset(
        unique_items=False,
        precise_location=False
    )

    rng, reset_rng = jax.random.split(rng)
    obs_all, state_all = env_all.reset(reset_rng, env_params)

    print("\nText observation with all items (may be very long):\n")
    # Only show first 1000 chars to avoid excessive output
    text = obs_all['text']
    if len(text) > 1000:
        print(text[:1000] + "\n... [truncated]")
    else:
        print(text)

    print("\n\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nThe language wrapper successfully provides:")
    print("  - Standard symbolic observations (compatible with RL training)")
    print("  - Natural language text observations (BALROG format)")
    print("  - Configurable text detail (unique_items, precise_location)")
    print("  - Drop-in replacement for standard Craftax Classic environment")


if __name__ == "__main__":
    main()
