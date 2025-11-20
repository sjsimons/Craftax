# Standalone Craftax Classic Language Wrapper

A **single-file, portable wrapper** that adds natural language text observations to Craftax Classic. Designed to be copied to any project and used with Craftax as an external library - just like BALROG does with Crafter.

## 🎯 Key Features

- ✅ **Single file** - Just copy `standalone_craftax_wrapper.py` to your project
- ✅ **Zero internal dependencies** - Only uses public Craftax API
- ✅ **Self-contained** - All rendering logic included
- ✅ **BALROG-compatible** - Same design pattern as BALROG's CrafterLanguageWrapper
- ✅ **Drop-in replacement** - Works exactly like base Craftax Classic environment
- ✅ **Dual observations** - Returns both symbolic (for RL) and text (for LLMs)

## 🚀 Quick Start

### Step 1: Install Craftax

```bash
pip install craftax
# or from source:
# pip install git+https://github.com/MichaelTMatthews/Craftax.git
```

### Step 2: Copy the Wrapper

Copy `standalone_craftax_wrapper.py` to your project:

```bash
cp standalone_craftax_wrapper.py /path/to/your/project/
```

### Step 3: Use It

```python
import jax
from standalone_craftax_wrapper import CraftaxClassicLanguageWrapper

# Create environment
env = CraftaxClassicLanguageWrapper(unique_items=True, precise_location=False)
env_params = env.default_params

# Reset and get initial observation
rng = jax.random.PRNGKey(0)
obs, state = env.reset(rng, env_params)

# Access observations
print(obs['text'])       # Natural language description
print(obs['symbolic'])   # Standard observation array

# Step through environment
action = 3  # Move North
obs, state, reward, done, info = env.step(rng, state, action, env_params)
```

That's it! No need to clone the Craftax repo or access any internal code.

## 📖 Usage Examples

### Basic Usage

```python
from standalone_craftax_wrapper import CraftaxClassicLanguageWrapper

env = CraftaxClassicLanguageWrapper()
env_params = env.default_params

rng = jax.random.PRNGKey(0)
obs, state = env.reset(rng, env_params)

# Text observation in BALROG format
print(obs['text'])
# Output:
# You see:
# - tree 2 steps to your north
# - stone 3 steps to your south-west
#
# You face grass at your front.
#
# Your status:
# - health: 9/9
# - food: 9/9
# - drink: 9/9
# - energy: 9/9
#
# You have nothing in your inventory.
```

### Using the Factory Function

```python
from standalone_craftax_wrapper import make_language_env

# Convenience factory
env = make_language_env(unique_items=True, precise_location=False)
obs, state = env.reset(rng, env_params)
```

### LLM Integration Example

```python
import jax
from standalone_craftax_wrapper import CraftaxClassicLanguageWrapper

env = CraftaxClassicLanguageWrapper()
env_params = env.default_params
rng = jax.random.PRNGKey(0)

obs, state = env.reset(rng, env_params)

# Construct LLM prompt from text observation
prompt = f"""You are playing a survival game. Current situation:

{obs['text']}

Available actions: 0-Noop, 1-Left, 2-Right, 3-Up, 4-Down, 5-Interact, 6-Sleep, 7-16: Craft/Place

What action? (respond with number)"""

# Get LLM response
action = your_llm.predict(prompt)  # e.g., returns 3

# Execute action
rng, step_rng = jax.random.split(rng)
obs, state, reward, done, info = env.step(step_rng, state, action, env_params)
```

### Batch Processing for Training

```python
import jax
from standalone_craftax_wrapper import CraftaxClassicLanguageWrapper

env = CraftaxClassicLanguageWrapper()
env_params = env.default_params

# Generate batch of observations
batch_size = 32
rngs = jax.random.split(jax.random.PRNGKey(0), batch_size)

observations = []
states = []

for rng in rngs:
    obs, state = env.reset(rng, env_params)
    observations.append(obs['text'])
    states.append(state)

# Batch prompt LLM
actions = your_llm.batch_predict(observations)

# Step all environments
new_observations = []
for i, action in enumerate(actions):
    rng = jax.random.split(rngs[i])[0]
    obs, state, reward, done, info = env.step(rng, states[i], action, env_params)
    new_observations.append(obs['text'])
```

## ⚙️ Configuration

### Constructor Parameters

```python
CraftaxClassicLanguageWrapper(
    static_env_params=None,    # Static environment parameters (optional)
    unique_items=True,          # Only show closest instance of each item type
    precise_location=False,     # Use precise location format
)
```

### Location Format Options

**Simple format** (`precise_location=False`, default):
```
"3 steps to your north-east"
"1 step to your west"
```

**Precise format** (`precise_location=True`):
```
"1 step north and 2 steps east"
"3 steps west"
```

### Item Filtering Options

**Unique items** (`unique_items=True`, default):
- Shows only the closest instance of each item/mob type
- Cleaner, less verbose output
- Example: "- tree 2 steps to your north"

**All items** (`unique_items=False`):
- Shows every visible item/mob
- More detailed but can be very long
- Example: "- tree 2 steps north\n- tree 5 steps south\n- tree 1 step west\n..."

## 🔍 Observation Format

The wrapper returns a dictionary with two keys:

```python
{
    'symbolic': jax.Array,  # Standard symbolic observation (shape depends on env)
    'text': str,            # Natural language description (BALROG format)
}
```

### Text Observation Structure

```
[Status alert if sleeping/dead]

You see:
- <item> <location>
- <item> <location>
...

You face <item> at your front.

Your status:
- health: X/9
- food: X/9
- drink: X/9
- energy: X/9

Your inventory:
- <item>: <count>
- <item>: <count>
...
```

## 🏗️ Architecture

### Design Principles

1. **No internal dependencies**: Only imports from `craftax.craftax_classic` public API
2. **Self-contained rendering**: All text rendering logic is in the wrapper file
3. **Inheritance-based**: Extends `CraftaxClassicSymbolicEnvNoAutoReset`
4. **BALROG-compatible**: Same pattern as BALROG's environment wrappers

### File Structure

```
standalone_craftax_wrapper.py
├── CraftaxClassicLanguageWrapper (main class)
│   ├── step_env() - Returns dict with 'symbolic' and 'text'
│   ├── reset_env() - Returns dict with 'symbolic' and 'text'
│   ├── _render_text() - Self-contained BALROG-format renderer
│   ├── _describe_location() - Location description logic
│   └── _describe_location_precise/simple() - Format variants
└── make_language_env() - Factory function
```

## 🔄 Comparison with Other Components

This repository has three different ways to use text observations:

| Component | Use Case | Dependencies |
|-----------|----------|--------------|
| **standalone_craftax_wrapper.py** | **Use in external projects** | Only public Craftax API |
| `craftax_language_env.py` | Internal to Craftax codebase | Internal renderer module |
| `trajectory_converter.py` | Offline trajectory conversion | Render function injection |

**Use the standalone wrapper when:**
- Working in a separate project
- Don't want to modify Craftax codebase
- Want BALROG-style single-file import
- Need portable, self-contained solution

## 📝 API Reference

### `CraftaxClassicLanguageWrapper`

Main wrapper class.

**Methods:**
- `__init__(static_env_params=None, unique_items=True, precise_location=False)`
- `step_env(rng, state, action, params)` → Returns `(obs_dict, state, reward, done, info)`
- `reset_env(rng, params)` → Returns `(obs_dict, state)`
- `name` property → Returns `"Craftax-Classic-Language-Wrapper-v1"`

Inherits all other methods from `CraftaxClassicSymbolicEnvNoAutoReset`.

### `make_language_env()`

Factory function for convenience.

**Parameters:**
- `unique_items: bool = True`
- `precise_location: bool = False`
- `static_env_params = None`

**Returns:** `CraftaxClassicLanguageWrapper` instance

## 🧪 Testing

Run the test suite to see all features:

```bash
python test_standalone_wrapper.py
```

This demonstrates:
- Basic usage
- Factory function
- Stepping through environment
- Different configurations
- LLM integration example
- Pure import (showing it works standalone)

## 💡 Tips

1. **Performance**: The text rendering adds minimal overhead (~1ms per step)

2. **Memory**: Text observations are small strings (<1KB typically)

3. **Batching**: For LLM training, batch size 32-128 works well

4. **JIT Compilation**: The environment is JAX-compatible and can be JIT compiled (though text rendering itself is not JIT-compiled)

5. **Customization**: To modify text format, edit `_render_text()` method in the wrapper

## 🤝 Compatibility

- **Craftax version**: Tested with Craftax 1.5.0+
- **JAX version**: Compatible with JAX 0.4.0+
- **Python version**: Requires Python 3.9+

## 📦 Installation in Your Project

### Option 1: Direct Copy

```bash
# Copy the wrapper file
cp standalone_craftax_wrapper.py your_project/

# Install craftax
pip install craftax
```

### Option 2: As a Module

```python
# your_project/craftax_wrapper.py
from standalone_craftax_wrapper import CraftaxClassicLanguageWrapper
```

### Option 3: With Modifications

Copy and modify `standalone_craftax_wrapper.py` to customize:
- Text format
- Location descriptions
- Status/inventory display
- Additional observations

## 🐛 Troubleshooting

**"No module named 'craftax'"**
- Install craftax: `pip install craftax`

**"AttributeError" on environment methods**
- Make sure you're using Craftax 1.5.0 or later
- Update: `pip install --upgrade craftax`

**Text observations seem incomplete**
- Check `unique_items` setting (True shows only closest items)
- Verify state is properly initialized

**Memory issues with large batches**
- Text observations are small; likely an issue with symbolic observations
- Reduce batch size or use streaming processing

## 📚 Further Reading

- [BALROG paper](https://github.com/balrog-ai/BALROG) - Original language wrapper design
- [Craftax documentation](https://github.com/MichaelTMatthews/Craftax)
- BALROG-style wrappers for other environments

## 🙏 Credits

- Wrapper design inspired by BALROG's CrafterLanguageWrapper
- Built on top of Craftax by Michael Matthews
- Text rendering follows BALROG crafter format
