#!/usr/bin/env python3
"""Test script for the craftax_classic BALROG-style text renderer."""

import jax
import jax.numpy as jnp

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.renderer import render_craftax_text_balrog


def main():
    # Create environment
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    env_params = env.default_params

    # Initialize environment
    rng = jax.random.PRNGKey(42)
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, env_params)

    print("=" * 80)
    print("BALROG-STYLE TEXT RENDERER - INITIAL STATE")
    print("=" * 80)
    print("\n## Simple location format (default):")
    print("-" * 80)
    text_output = render_craftax_text_balrog(state, unique_items=True, precise_location=False)
    print(text_output)

    print("\n\n" + "=" * 80)
    print("## Precise location format:")
    print("-" * 80)
    text_output_precise = render_craftax_text_balrog(state, unique_items=True, precise_location=True)
    print(text_output_precise)

    # Take some actions to show dynamic changes
    print("\n\n" + "=" * 80)
    print("AFTER 20 RANDOM ACTIONS")
    print("=" * 80)

    for i in range(20):
        rng, action_rng, step_rng = jax.random.split(rng, 3)
        action = jax.random.randint(action_rng, (), 0, env.action_space(env_params).n)
        obs, state, reward, done, info = env.step(step_rng, state, action, env_params)
        if reward != 0:
            print(f"Step {i+1}: Reward: {reward}")

    print("\n## Simple location format:")
    print("-" * 80)
    text_output = render_craftax_text_balrog(state, unique_items=True, precise_location=False)
    print(text_output)

    print("\n\n" + "=" * 80)
    print("## With all items (not just unique/closest):")
    print("-" * 80)
    text_output_all = render_craftax_text_balrog(state, unique_items=False, precise_location=False)
    print(text_output_all)

    print("\n\n" + "=" * 80)
    print("TEST COMPLETE - BALROG-style renderer is working!")
    print("=" * 80)


if __name__ == "__main__":
    main()
