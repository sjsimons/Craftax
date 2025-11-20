#!/usr/bin/env python3
"""
Test script demonstrating the standalone wrapper.

This shows how you would use the wrapper in a completely separate project
where you just have craftax installed as a library.

Usage in your own project:
    1. Copy standalone_craftax_wrapper.py to your project
    2. Install craftax: pip install craftax (or from github)
    3. Import and use:

    from standalone_craftax_wrapper import CraftaxClassicLanguageWrapper
    env = CraftaxClassicLanguageWrapper()
"""

import jax

# This is the ONLY import you need - just copy the wrapper file to your project
from standalone_craftax_wrapper import CraftaxClassicLanguageWrapper, make_language_env


def demo_basic_usage():
    """Basic usage - exactly like BALROG."""
    print("=" * 80)
    print("DEMO 1: Basic Usage (BALROG-style)")
    print("=" * 80)

    # Create environment (works just like base environment)
    env = CraftaxClassicLanguageWrapper(unique_items=True, precise_location=False)
    env_params = env.default_params

    # Reset
    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng, env_params)

    print("\nObservation type:", type(obs))
    print("Observation keys:", obs.keys())
    print("\nSymbolic observation shape:", obs['symbolic'].shape)
    print("\nText observation:")
    print("-" * 80)
    print(obs['text'])
    print("-" * 80)


def demo_factory_function():
    """Using the factory function."""
    print("\n\n" + "=" * 80)
    print("DEMO 2: Factory Function")
    print("=" * 80)

    # Alternative: use factory function
    env = make_language_env(unique_items=True, precise_location=False)
    env_params = env.default_params

    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng, env_params)

    print("\nUsing factory function to create environment")
    print("Text observation preview:")
    print(obs['text'][:200] + "...")


def demo_step_actions():
    """Step through environment with actions."""
    print("\n\n" + "=" * 80)
    print("DEMO 3: Stepping Through Environment")
    print("=" * 80)

    env = CraftaxClassicLanguageWrapper()
    env_params = env.default_params

    rng = jax.random.PRNGKey(123)
    obs, state = env.reset(rng, env_params)

    print("\nTaking 5 random actions:")

    for i in range(5):
        rng, action_rng, step_rng = jax.random.split(rng, 3)
        action = jax.random.randint(action_rng, (), 0, env.action_space(env_params).n)

        obs, state, reward, done, info = env.step(step_rng, state, action, env_params)

        print(f"\nStep {i+1}: Action={action}, Reward={reward:.2f}")
        if reward != 0:
            print("  Text observation changed:")
            print("  " + obs['text'][:150].replace('\n', '\n  ') + "...")


def demo_different_configs():
    """Compare different wrapper configurations."""
    print("\n\n" + "=" * 80)
    print("DEMO 4: Different Configurations")
    print("=" * 80)

    rng = jax.random.PRNGKey(999)

    configs = [
        {"unique_items": True, "precise_location": False},
        {"unique_items": True, "precise_location": True},
        {"unique_items": False, "precise_location": False},
    ]

    for i, config in enumerate(configs, 1):
        print(f"\nConfig {i}: {config}")
        print("-" * 40)

        env = CraftaxClassicLanguageWrapper(**config)
        env_params = env.default_params

        rng, reset_rng = jax.random.split(rng)
        obs, state = env.reset(reset_rng, env_params)

        # Show first 200 chars of text
        print(obs['text'][:200] + "...")


def demo_llm_integration():
    """Demonstrate integration with LLM workflow."""
    print("\n\n" + "=" * 80)
    print("DEMO 5: LLM Integration Example")
    print("=" * 80)

    env = CraftaxClassicLanguageWrapper()
    env_params = env.default_params

    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng, env_params)

    print("\nExample LLM prompt construction:")
    print("-" * 80)

    # Construct LLM prompt from text observation
    prompt = f"""You are an agent playing a survival crafting game. Based on the current observation, decide what action to take.

Current situation:
{obs['text']}

Available actions:
0: Noop
1: Move Left
2: Move Right
3: Move Up
4: Move Down
5: Do (interact)
6: Sleep
7-10: Place items
11-16: Craft tools

What action should you take? (respond with just the number)
"""

    print(prompt)
    print("-" * 80)

    print("\n(In real usage, you would send this to your LLM)")
    print("Then use the LLM's response as the action:")

    # Simulate LLM responding with action 3 (Move Up)
    llm_action = 3

    rng, step_rng = jax.random.split(rng)
    obs, state, reward, done, info = env.step(step_rng, state, llm_action, env_params)

    print(f"\nExecuted action {llm_action}")
    print(f"Reward: {reward}")
    print(f"\nNew observation:")
    print(obs['text'][:250] + "...")


def demo_pure_import():
    """Show that this works with ONLY the wrapper file + craftax library."""
    print("\n\n" + "=" * 80)
    print("DEMO 6: Standalone Import (Copy to Any Project)")
    print("=" * 80)

    print("""
To use in a completely separate project:

1. Install craftax:
   pip install craftax
   # or: pip install git+https://github.com/MichaelTMatthews/Craftax.git

2. Copy standalone_craftax_wrapper.py to your project:
   cp standalone_craftax_wrapper.py /path/to/your/project/

3. Import and use:
   from standalone_craftax_wrapper import CraftaxClassicLanguageWrapper

   env = CraftaxClassicLanguageWrapper()
   env_params = env.default_params
   obs, state = env.reset(rng, env_params)

   # obs['text'] contains natural language
   # obs['symbolic'] contains standard observation array

4. That's it! No need for any craftax source code.

The wrapper file is completely self-contained and only uses
the public craftax API (imports from craftax.craftax_classic).
""")

    # Demonstrate it works
    env = CraftaxClassicLanguageWrapper()
    env_params = env.default_params
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng, env_params)

    print("\n✓ Wrapper working with just library import!")
    print(f"✓ Environment name: {env.name}")
    print(f"✓ Observation has text: {'text' in obs}")
    print(f"✓ Observation has symbolic: {'symbolic' in obs}")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("STANDALONE WRAPPER DEMONSTRATION")
    print("Showing how to use Craftax Classic with text observations")
    print("in a completely separate project (BALROG-style)")
    print("=" * 80)

    demo_basic_usage()
    demo_factory_function()
    demo_step_actions()
    demo_different_configs()
    demo_llm_integration()
    demo_pure_import()

    print("\n\n" + "=" * 80)
    print("ALL DEMOS COMPLETE")
    print("=" * 80)
    print("\n✓ The standalone wrapper is ready to use in any project!")
    print("✓ Just copy standalone_craftax_wrapper.py and import it")
    print("✓ No need to access craftax source code internals")
    print()


if __name__ == "__main__":
    main()
