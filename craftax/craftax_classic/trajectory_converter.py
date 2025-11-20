"""
Trajectory to text converter - converts trajectory data to text observations.

To use with different environments:
1. Provide a render_function that converts state to text
2. Provide a load_trajectory_function for your file format
3. Use TrajectoryTextConverter with your custom functions

For Craftax Classic: Uses standalone_craftax_wrapper.py renderer with internal fallback.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable, Any, Optional, Dict, Tuple


@dataclass
class CraftaxTransition:
    """A single transition in text format."""
    id: str
    text_state: str
    action: int
    text_next_state: str
    file_path: str
    step_idx: int

    def __repr__(self):
        text_preview = self.text_state[:50] + "..." if len(self.text_state) > 50 else self.text_state
        return f"CraftaxTransition(id='{self.id}', action={self.action}, text='{text_preview}')"


class TrajectoryTextConverter:
    """
    Converter for trajectory files to text transitions.

    Args:
        render_function: Function that takes (state, **kwargs) and returns text string
        load_trajectory_function: Function that takes file_path and returns trajectory dict
                                 Expected dict format: {'state': [...], 'action': [...], ...}
        render_kwargs: Keyword arguments to pass to render_function
    """

    def __init__(
        self,
        render_function: Callable[[Any], str],
        load_trajectory_function: Callable[[str], Dict[str, List]],
        render_kwargs: Optional[Dict] = None,
    ):
        self.render_function = render_function
        self.load_trajectory_function = load_trajectory_function
        self.render_kwargs = render_kwargs or {}
        self._file_index: Dict[str, str] = {}

    def convert_file(
        self,
        file_path: str,
        file_id: Optional[str] = None,
    ) -> List[CraftaxTransition]:
        """
        Convert a single trajectory file to text transitions.

        Args:
            file_path: Path to trajectory file
            file_id: Identifier for this file (defaults to filename without extension)

        Returns:
            List of CraftaxTransition objects
        """
        if file_id is None:
            file_id = Path(file_path).stem

        self._file_index[file_id] = file_path

        trajectory = self.load_trajectory_function(file_path)
        states = trajectory['state']
        actions = trajectory['action']

        transitions = []
        for i in range(len(states) - 1):
            state = states[i]
            next_state = states[i + 1]
            action = actions[i]

            text_state = self.render_function(state, **self.render_kwargs)
            text_next_state = self.render_function(next_state, **self.render_kwargs)

            transition = CraftaxTransition(
                id=f"{file_id}_{i:06d}",
                text_state=text_state,
                action=int(action),
                text_next_state=text_next_state,
                file_path=file_path,
                step_idx=i,
            )
            transitions.append(transition)

        return transitions

    def convert_directory(
        self,
        directory_path: str,
        pattern: str = "*.pbz2",
    ) -> List[CraftaxTransition]:
        """
        Batch convert all trajectory files in a directory.

        Args:
            directory_path: Path to directory containing trajectory files
            pattern: Glob pattern for matching files (default: "*.pbz2")

        Returns:
            List of all CraftaxTransition objects from all files
        """
        directory = Path(directory_path)
        files = sorted(directory.glob(pattern))

        if not files:
            raise ValueError(f"No files matching '{pattern}' found in {directory_path}")

        all_transitions = []
        for file_path in files:
            print(f"Converting {file_path.name}...")
            transitions = self.convert_file(str(file_path))
            all_transitions.extend(transitions)
            print(f"  → {len(transitions)} transitions")

        print(f"\nTotal: {len(all_transitions)} transitions from {len(files)} files")
        return all_transitions

    def get_state_by_id(self, transition_id: str) -> Any:
        """Retrieve the original state object for a transition ID."""
        parts = transition_id.rsplit('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid transition ID format: {transition_id}")

        file_id, step_str = parts
        step_idx = int(step_str)

        if file_id not in self._file_index:
            raise ValueError(f"Unknown file_id: {file_id}")

        file_path = self._file_index[file_id]
        trajectory = self.load_trajectory_function(file_path)
        states = trajectory['state']

        if step_idx >= len(states):
            raise ValueError(f"Step index {step_idx} out of range for {file_id}")

        return states[step_idx]

    def get_transition_by_id(
        self,
        transition_id: str,
        include_states: bool = False,
    ) -> CraftaxTransition:
        """Retrieve a full transition by ID."""
        parts = transition_id.rsplit('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid transition ID format: {transition_id}")

        file_id, step_str = parts
        step_idx = int(step_str)

        if file_id not in self._file_index:
            raise ValueError(f"Unknown file_id: {file_id}")

        file_path = self._file_index[file_id]
        trajectory = self.load_trajectory_function(file_path)
        states = trajectory['state']
        actions = trajectory['action']

        if step_idx >= len(states) - 1:
            raise ValueError(f"Step index {step_idx} out of range")

        state = states[step_idx]
        next_state = states[step_idx + 1]
        action = actions[step_idx]

        text_state = self.render_function(state, **self.render_kwargs)
        text_next_state = self.render_function(next_state, **self.render_kwargs)

        return CraftaxTransition(
            id=transition_id,
            text_state=text_state,
            action=int(action),
            text_next_state=text_next_state,
            file_path=file_path,
            step_idx=step_idx,
        )


def craftax_render_function(state, unique_items=True, precise_location=False):
    """Render Craftax state to text using standalone wrapper (with internal fallback)."""
    try:
        from standalone_craftax_wrapper import CraftaxClassicLanguageWrapper
    except ImportError:
        from craftax.craftax_classic.renderer import render_craftax_text_balrog
        return render_craftax_text_balrog(state, unique_items, precise_location)

    wrapper = CraftaxClassicLanguageWrapper(
        unique_items=unique_items,
        precise_location=precise_location
    )
    return wrapper._render_text(state)


def craftax_load_trajectory(file_path: str) -> Dict[str, List]:
    """Load Craftax trajectory file."""
    from craftax.environment_base.util import load_compressed_pickle
    return load_compressed_pickle(file_path)


def make_craftax_converter(unique_items=True, precise_location=False):
    """Create a converter configured for Craftax Classic."""
    return TrajectoryTextConverter(
        render_function=craftax_render_function,
        load_trajectory_function=craftax_load_trajectory,
        render_kwargs={'unique_items': unique_items, 'precise_location': precise_location},
    )
