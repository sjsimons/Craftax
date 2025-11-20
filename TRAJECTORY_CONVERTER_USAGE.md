# Trajectory Converter Usage Guide

A portable, generic module for converting Craftax Classic trajectory files to text observations for LLM training.

## 🎯 Design Goals

- **Portable**: Easy to copy to other codebases with minimal changes
- **Memory efficient**: On-demand state loading, doesn't keep all states in memory
- **Flexible**: Configurable render parameters
- **LLM-ready**: Designed for batch prompting workflows

## 🚀 Quick Start

### Basic Usage

```python
from craftax.craftax_classic.trajectory_converter import make_craftax_converter

# Create converter
converter = make_craftax_converter(unique_items=True, precise_location=False)

# Convert all trajectory files in a directory
transitions = converter.convert_directory("data/trajectories/")

# Each transition contains: id, text_state, action, text_next_state
for trans in transitions[:5]:
    print(f"{trans.id}: {trans.action}")
```

### LLM Training Workflow

```python
from craftax.craftax_classic.trajectory_converter import make_craftax_converter

# Setup
converter = make_craftax_converter(unique_items=True)
transitions = converter.convert_directory("data/runs/")

# Batch processing for LLM
batch_size = 64
for i in range(0, len(transitions), batch_size):
    batch = transitions[i:i+batch_size]

    # Extract text and IDs
    text_states = [t.text_state for t in batch]
    actions = [t.action for t in batch]
    ids = [t.id for t in batch]

    # Prompt LLM
    responses = your_llm.batch_predict(text_states)

    # Associate responses with IDs
    results = {id: response for id, response in zip(ids, responses)}

    # Later: retrieve original state if needed
    if need_original_state:
        state = converter.get_state_by_id(ids[0])
```

## 📊 Transition Format

Each `CraftaxTransition` contains:

```python
@dataclass
class CraftaxTransition:
    id: str                  # e.g., "run1_000005"
    text_state: str          # Current state as natural language
    action: int              # Action taken (0-16)
    text_next_state: str     # Next state as natural language
    file_path: str           # Path to original .pbz2 file
    step_idx: int            # Step index in trajectory
```

### Example Transition

```python
transition = transitions[5]

print(f"ID: {transition.id}")
# → "run1_000005"

print(f"Action: {transition.action}")
# → 3  (Move North)

print(transition.text_state)
# → You see:
#   - tree 2 steps to your north
#   - stone 3 steps to your south-west
#
#   You face grass at your front.
#
#   Your status:
#   - health: 9/9
#   - food: 8/9
#   ...

print(transition.text_next_state)
# → (text observation after taking action 3)
```

## 🔧 Configuration Options

### Render Parameters

```python
# Simple location format (default)
converter = make_craftax_converter(
    unique_items=True,        # Only show closest of each type
    precise_location=False    # "3 steps to your north-east"
)

# Precise location format
converter = make_craftax_converter(
    unique_items=True,
    precise_location=True     # "1 step north and 2 steps east"
)

# Show all items (can be verbose)
converter = make_craftax_converter(
    unique_items=False,       # Show every visible item
    precise_location=False
)
```

## 📁 File Handling

### Convert Single File

```python
transitions = converter.convert_file(
    "data/trajectories/run1.pbz2",
    file_id="run1"  # Optional, defaults to filename
)
```

### Convert Directory

```python
# Convert all .pbz2 files
transitions = converter.convert_directory(
    "data/trajectories/",
    pattern="*.pbz2"  # Optional, customize pattern
)

# Convert specific pattern
transitions = converter.convert_directory(
    "data/runs/",
    pattern="run_*.pbz2"
)
```

## 🔍 On-Demand State Retrieval

The converter stores a file index, allowing you to retrieve original states without keeping everything in memory.

```python
# Convert trajectories (only stores text, not states)
transitions = converter.convert_directory("data/runs/")

# Later: get original state by ID
transition_id = "run1_000042"
original_state = converter.get_state_by_id(transition_id)

# Or get full transition (re-rendered from state)
transition = converter.get_transition_by_id(transition_id)
```

## 📦 Expected Trajectory Format

Your trajectory files should contain:

```python
{
    'state': [state0, state1, state2, ...],      # List of EnvState objects
    'action': [action0, action1, ...],           # List of actions
    'reward': [reward0, reward1, ...],           # Optional
    'done': [done0, done1, ...],                 # Optional
}
```

The converter uses `state[i]`, `action[i]`, and `state[i+1]` to create transitions.

## 🔄 Porting to Another Codebase

The converter is designed to be portable. To use with a different environment:

### Step 1: Copy the module

```bash
cp craftax/craftax_classic/trajectory_converter.py your_project/
```

### Step 2: Implement your render function

```python
def your_render_function(state, **kwargs):
    """Convert your environment state to text."""
    # Your custom rendering logic
    return text_observation
```

### Step 3: Implement your loader function

```python
def your_load_function(file_path):
    """Load your trajectory format."""
    # Your custom loading logic
    return {
        'state': [...],
        'action': [...],
    }
```

### Step 4: Create your converter

```python
from trajectory_converter import TrajectoryTextConverter

converter = TrajectoryTextConverter(
    render_function=your_render_function,
    load_trajectory_function=your_load_function,
    render_kwargs={'your_param': True},
)

# Use exactly the same API!
transitions = converter.convert_directory("your_data/")
```

**No other code changes needed!** The entire API (`convert_file`, `convert_directory`, `get_state_by_id`, etc.) works identically.

## 🎓 Advanced Usage Examples

### Filter by Step Index

```python
transitions = converter.convert_directory("data/runs/")

# Only get transitions from first 10 steps of each trajectory
early_transitions = [t for t in transitions if t.step_idx < 10]
```

### Group by File

```python
from collections import defaultdict

transitions = converter.convert_directory("data/runs/")

# Group by source file
by_file = defaultdict(list)
for trans in transitions:
    file_id = trans.id.rsplit('_', 1)[0]  # "run1_000005" -> "run1"
    by_file[file_id].append(trans)

print(f"Transitions per file:")
for file_id, trans_list in by_file.items():
    print(f"  {file_id}: {len(trans_list)}")
```

### Custom Batching Strategy

```python
import random

transitions = converter.convert_directory("data/runs/")

# Shuffle for random batching
random.shuffle(transitions)

# Create batches
batches = [transitions[i:i+64] for i in range(0, len(transitions), 64)]

for batch_idx, batch in enumerate(batches):
    print(f"Processing batch {batch_idx}: {len(batch)} transitions")
    # Your LLM training code here
```

### Save Text Observations

```python
import json

transitions = converter.convert_directory("data/runs/")

# Save as JSONL for easy loading
with open("text_observations.jsonl", "w") as f:
    for trans in transitions:
        data = {
            'id': trans.id,
            'text_state': trans.text_state,
            'action': trans.action,
            'text_next_state': trans.text_next_state,
        }
        f.write(json.dumps(data) + '\n')
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_trajectory_converter.py
```

This demonstrates:
- Single file conversion
- Batch directory conversion
- On-demand state retrieval
- LLM workflow patterns
- Custom render parameters
- Portability guide

## 💡 Tips

1. **Memory Management**: The converter only stores text, not states. Use `get_state_by_id()` to retrieve states on-demand when needed.

2. **Batch Size**: For LLM training, batch size of 32-128 works well depending on your GPU memory.

3. **ID Format**: IDs are formatted as `{file_id}_{step:06d}` (e.g., "run1_000005"). The zero-padding ensures proper sorting.

4. **Text Length**: BALROG format typically produces 200-500 character observations depending on visible items.

5. **Performance**: Converting 100k transitions takes ~1-2 minutes (depends on trajectory size and render complexity).

## ❓ Common Issues

**Q: "Unknown file_id" error when using `get_state_by_id()`**

A: You can only retrieve states from files that were converted in the current converter instance. Either convert the file again or keep your converter instance.

**Q: Transitions seem to skip the last step**

A: This is expected! We need both `state[i]` and `state[i+1]` to create a transition, so a trajectory with N states produces N-1 transitions.

**Q: Want to include reward/done in transitions?**

A: Extend the `CraftaxTransition` dataclass:

```python
@dataclass
class ExtendedTransition(CraftaxTransition):
    reward: float = 0.0
    done: bool = False
```

Then modify the converter to populate these fields from your trajectory data.
