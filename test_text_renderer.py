#!/usr/bin/env python3
"""Simple test script for the craftax_classic text renderer."""

import jax
import jax.numpy as jnp

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.renderer import render_craftax_text


def main():
    # Create environment
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    env_params = env.default_params

    # Initialize environment
    rng = jax.random.PRNGKey(0)
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, env_params)

    print("=" * 80)
    print("INITIAL STATE")
    print("=" * 80)
    text_output = render_craftax_text(state)
    print(text_output)

    # Take a few random actions
    print("\n" + "=" * 80)
    print("AFTER 10 RANDOM ACTIONS")
    print("=" * 80)

    step_fn = jax.jit(env.step)

    for i in range(10):
        rng, action_rng = jax.random.split(rng)
        action = jax.random.randint(action_rng, (), 0, env.action_space(env_params).n)
        obs, state, reward, done, info = step_fn(rng, state, action, env_params)
        if reward != 0:
            print(f"Step {i+1}: Action {action}, Reward: {reward}")

    text_output = render_craftax_text(state)
    print(text_output)

    print("\n" + "=" * 80)
    print("TEST COMPLETE - Text renderer is working!")
    print("=" * 80)


if __name__ == "__main__":
    main()
