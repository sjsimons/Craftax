# Craftax Classic Language Wrapper

A BALROG-style text observation wrapper for Craftax Classic that provides natural language descriptions of the game state.

## Quick Start

### Using the Factory Function

```python
import jax
from craftax.craftax_env import make_craftax_env_from_name

# Create the text environment
env = make_craftax_env_from_name("Craftax-Classic-Text-v1", auto_reset=False)
env_params = env.default_params

# Use it like any Craftax environment
rng = jax.random.PRNGKey(0)
obs, state = env.reset(rng, env_params)

# Observations are now dictionaries with 'symbolic' and 'text' keys
print(obs['text'])  # Natural language description
# obs['symbolic'] contains the standard symbolic observation array
```

### Using the Wrapper Directly

```python
from craftax.craftax_classic.envs import CraftaxClassicTextEnvNoAutoReset

# Create with custom parameters
env = CraftaxClassicTextEnvNoAutoReset(
    unique_items=True,        # Only show closest instance of each item type
    precise_location=False    # Use simple location format (e.g., "3 steps to your north-east")
)

env_params = env.default_params
rng = jax.random.PRNGKey(0)
obs, state = env.reset(rng, env_params)
```

## Configuration Options

### `unique_items` (default: `True`)
- **True**: Show only the closest instance of each item/mob type
  - Example: "You see:\n- tree 2 steps to your north\n- zombie 3 steps to your east"
- **False**: Show all visible items (can produce very long descriptions)
  - Example: "You see:\n- tree 2 steps to your north\n- tree 5 steps to your south\n- tree 1 step to your west\n..."

### `precise_location` (default: `False`)
- **False**: Simple location format (Manhattan distance + direction)
  - Example: "3 steps to your north-east"
- **True**: Precise location format (separate north/south and east/west)
  - Example: "1 step north and 2 steps east"

## Text Observation Format

The text observations follow the BALROG crafter format:

```
You see:
- tree 2 steps to your north
- stone 3 steps to your south-west
- zombie 1 step to your east

You face grass at your front.

Your status:
- health: 9/9
- food: 9/9
- drink: 9/9
- energy: 9/9

Your inventory:
- wood: 5
- stone: 2
```

## Example: Training an RL Agent

```python
import jax
from craftax.craftax_env import make_craftax_env_from_name

env = make_craftax_env_from_name("Craftax-Classic-Text-v1", auto_reset=False)
env_params = env.default_params

rng = jax.random.PRNGKey(0)
obs, state = env.reset(rng, env_params)

# Use symbolic observation for RL training
symbolic_obs = obs['symbolic']  # Standard observation array for neural networks

# Use text observation for language models or debugging
text_obs = obs['text']  # Natural language description
print(text_obs)

# Step through environment
action = 3  # Move North
rng, step_rng = jax.random.split(rng)
obs, state, reward, done, info = env.step(step_rng, state, action, env_params)
```

## Example: Language Model Integration

```python
from craftax.craftax_classic.envs import CraftaxClassicTextEnvNoAutoReset

env = CraftaxClassicTextEnvNoAutoReset(unique_items=True, precise_location=False)
env_params = env.default_params

rng = jax.random.PRNGKey(0)
obs, state = env.reset(rng, env_params)

# Send text observation to language model
prompt = f"""
You are playing a survival crafting game. Here is your current situation:

{obs['text']}

What action should you take? Available actions are:
0: Noop, 1: Move Left, 2: Move Right, 3: Move Up, 4: Move Down,
5: Do (interact), 6: Sleep, 7-10: Place items, 11-16: Craft items
"""

# Get action from language model, then step environment
action = get_action_from_llm(prompt)  # Your LLM integration
obs, state, reward, done, info = env.step(rng, state, action, env_params)
```

## Testing

Run the test script to see the wrapper in action:

```bash
python test_language_wrapper.py
```

This demonstrates:
- Creating the environment with default parameters
- Taking actions and viewing text observations
- Using custom parameters (precise_location, unique_items)
- Compatibility with standard Craftax Classic interface

## Comparison with BALROG

This wrapper follows the same pattern as BALROG's `CrafterLanguageWrapper`:
- Environment wrapper that extends the base environment
- Returns structured observations with text descriptions
- Configurable text detail level
- Maintains full compatibility with base environment

Key differences:
- Uses JAX/Gymnax instead of OpenAI Gym
- Integrated directly into Craftax codebase
- Optimized for Craftax Classic game mechanics
