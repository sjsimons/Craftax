#!/usr/bin/env python3
"""
Test script for trajectory converter.

Demonstrates:
1. Converting trajectory files to text transitions
2. Batch processing multiple files
3. Retrieving states on-demand by ID
4. Usage patterns for LLM prompting workflows
"""

import jax
import tempfile
import os
from pathlib import Path

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.trajectory_converter import (
    make_craftax_converter,
    CraftaxTransition,
)
from craftax.environment_base.util import save_compressed_pickle


def generate_test_trajectories(output_dir: str, num_files: int = 3, steps_per_file: int = 20):
    """Generate some test trajectory files for demonstration."""
    print(f"Generating {num_files} test trajectory files with {steps_per_file} steps each...")

    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    env_params = env.default_params

    os.makedirs(output_dir, exist_ok=True)

    for file_idx in range(num_files):
        # Generate trajectory
        rng = jax.random.PRNGKey(file_idx)
        rng, reset_rng = jax.random.split(rng)
        obs, state = env.reset(reset_rng, env_params)

        states = [state]
        actions = []
        rewards = []
        dones = []

        for step in range(steps_per_file):
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            action = jax.random.randint(action_rng, (), 0, env.action_space(env_params).n)
            obs, state, reward, done, info = env.step(step_rng, state, action, env_params)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        # Save trajectory
        trajectory = {
            'state': states,
            'action': actions,
            'reward': rewards,
            'done': dones,
        }

        file_path = os.path.join(output_dir, f"run{file_idx}.pbz2")
        save_compressed_pickle(file_path, trajectory)
        print(f"  Created {file_path}")

    print(f"✓ Generated {num_files} trajectory files\n")


def demo_basic_conversion():
    """Demonstrate basic single file conversion."""
    print("=" * 80)
    print("DEMO 1: Basic Single File Conversion")
    print("=" * 80)

    # Create temporary directory with test data
    with tempfile.TemporaryDirectory() as temp_dir:
        generate_test_trajectories(temp_dir, num_files=1, steps_per_file=10)

        # Create converter
        converter = make_craftax_converter(unique_items=True, precise_location=False)

        # Convert single file
        file_path = os.path.join(temp_dir, "run0.pbz2")
        transitions = converter.convert_file(file_path)

        print(f"\nConverted {len(transitions)} transitions")
        print("\nFirst transition:")
        print(f"  ID: {transitions[0].id}")
        print(f"  Action: {transitions[0].action}")
        print(f"  Text state preview:\n{transitions[0].text_state[:200]}...")
        print(f"\n  Next state preview:\n{transitions[0].text_next_state[:200]}...")


def demo_batch_conversion():
    """Demonstrate batch directory conversion."""
    print("\n\n" + "=" * 80)
    print("DEMO 2: Batch Directory Conversion")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as temp_dir:
        generate_test_trajectories(temp_dir, num_files=3, steps_per_file=15)

        # Create converter
        converter = make_craftax_converter(unique_items=True)

        # Convert entire directory
        transitions = converter.convert_directory(temp_dir)

        print(f"\nTransition IDs from all files:")
        for i in [0, 1, -2, -1]:  # Show first two and last two
            print(f"  {transitions[i].id}")
        print(f"  ... ({len(transitions)} total transitions)")


def demo_on_demand_retrieval():
    """Demonstrate on-demand state retrieval by ID."""
    print("\n\n" + "=" * 80)
    print("DEMO 3: On-Demand State Retrieval")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as temp_dir:
        generate_test_trajectories(temp_dir, num_files=2, steps_per_file=10)

        converter = make_craftax_converter(unique_items=True)
        transitions = converter.convert_directory(temp_dir)

        # Retrieve specific transition by ID
        test_id = transitions[5].id
        print(f"\nRetrieving transition: {test_id}")

        # Get the original state
        original_state = converter.get_state_by_id(test_id)
        print(f"✓ Retrieved original state (type: {type(original_state).__name__})")

        # Get the full transition (re-rendered)
        transition = converter.get_transition_by_id(test_id)
        print(f"✓ Re-rendered transition")
        print(f"  Action: {transition.action}")
        print(f"  File: {Path(transition.file_path).name}")
        print(f"  Step: {transition.step_idx}")


def demo_llm_workflow():
    """Demonstrate typical LLM training workflow with batching."""
    print("\n\n" + "=" * 80)
    print("DEMO 4: LLM Training Workflow (Batch Processing)")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as temp_dir:
        generate_test_trajectories(temp_dir, num_files=2, steps_per_file=25)

        converter = make_craftax_converter(unique_items=True, precise_location=False)
        transitions = converter.convert_directory(temp_dir)

        print(f"\nSimulating LLM batch prompting workflow...")
        print(f"Total transitions: {len(transitions)}")

        batch_size = 8
        num_batches = (len(transitions) + batch_size - 1) // batch_size

        print(f"Batch size: {batch_size}")
        print(f"Number of batches: {num_batches}\n")

        # Simulate batch processing
        for batch_idx in range(min(2, num_batches)):  # Just show first 2 batches
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(transitions))
            batch = transitions[start_idx:end_idx]

            print(f"Batch {batch_idx + 1}:")
            print(f"  Transitions: {[t.id for t in batch[:3]]} ... ({len(batch)} total)")

            # Extract text states and IDs
            text_states = [t.text_state for t in batch]
            actions = [t.action for t in batch]
            ids = [t.id for t in batch]

            # Simulate LLM prompting
            print(f"  → Would prompt LLM with {len(text_states)} text observations")

            # Simulate getting responses
            # In real usage: responses = llm.batch_predict(text_states)
            mock_responses = [f"Response for {id}" for id in ids]

            # Associate responses with IDs
            results = {id: response for id, response in zip(ids, mock_responses)}
            print(f"  ✓ Associated {len(results)} responses with transition IDs")

            # Later, can retrieve original states if needed
            if batch_idx == 0:
                sample_id = ids[0]
                original_state = converter.get_state_by_id(sample_id)
                print(f"  ✓ Can retrieve original state for {sample_id} on-demand")

            print()


def demo_custom_render_params():
    """Demonstrate using different render parameters."""
    print("\n\n" + "=" * 80)
    print("DEMO 5: Custom Render Parameters")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as temp_dir:
        generate_test_trajectories(temp_dir, num_files=1, steps_per_file=5)

        print("\nComparing different render configurations:\n")

        # Simple location format
        print("1. Simple location format (unique_items=True):")
        converter1 = make_craftax_converter(unique_items=True, precise_location=False)
        trans1 = converter1.convert_file(os.path.join(temp_dir, "run0.pbz2"))
        print(trans1[0].text_state[:250] + "...\n")

        # Precise location format
        print("2. Precise location format (unique_items=True, precise_location=True):")
        converter2 = make_craftax_converter(unique_items=True, precise_location=True)
        trans2 = converter2.convert_file(os.path.join(temp_dir, "run0.pbz2"))
        print(trans2[0].text_state[:250] + "...\n")


def demo_portability():
    """Show how to port to another codebase."""
    print("\n\n" + "=" * 80)
    print("DEMO 6: Portability Guide")
    print("=" * 80)

    print("""
To port this converter to a different codebase:

1. Copy `trajectory_converter.py` to your project

2. Implement your own render function:
   ```python
   def my_render_function(state, **kwargs):
       # Your environment-specific rendering logic
       return text_observation_string
   ```

3. Implement your own trajectory loader:
   ```python
   def my_load_function(file_path):
       # Your file format loading logic
       return {
           'state': [...],
           'action': [...],
           # ... other fields
       }
   ```

4. Create your converter:
   ```python
   from trajectory_converter import TrajectoryTextConverter

   converter = TrajectoryTextConverter(
       render_function=my_render_function,
       load_trajectory_function=my_load_function,
       render_kwargs={'my_param': True},
   )

   transitions = converter.convert_directory("my_data/")
   ```

5. The rest of the API stays the same!
   - convert_file() / convert_directory()
   - get_state_by_id()
   - get_transition_by_id()
   - CraftaxTransition dataclass

The converter is completely decoupled from Craftax internals.
""")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("TRAJECTORY CONVERTER TEST SUITE")
    print("=" * 80)

    demo_basic_conversion()
    demo_batch_conversion()
    demo_on_demand_retrieval()
    demo_llm_workflow()
    demo_custom_render_params()
    demo_portability()

    print("\n" + "=" * 80)
    print("ALL DEMOS COMPLETE")
    print("=" * 80)
    print("\nThe trajectory converter is ready to use!")
    print("\nQuick start:")
    print("  from craftax.craftax_classic.trajectory_converter import make_craftax_converter")
    print("  converter = make_craftax_converter()")
    print("  transitions = converter.convert_directory('path/to/trajectories/')")
    print()


if __name__ == "__main__":
    main()
