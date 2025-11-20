"""
Trajectory to text converter - Portable module for converting trajectory data to text observations.

This module is designed to be portable across codebases. To use in a different project:
1. Copy this file to your project
2. Copy standalone_craftax_wrapper.py to your project (for Craftax Classic)
   OR implement your own render_function that takes a state and returns text
3. Implement your own load_trajectory_file function for your file format
4. Use the TrajectoryTextConverter with your custom functions

Note: The default Craftax converter uses standalone_craftax_wrapper.py for text rendering.
      If standalone_craftax_wrapper.py is not available, it falls back to the internal renderer.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable, Any, Optional, Dict, Tuple


@dataclass
class CraftaxTransition:
    """
    A single transition in text format.

    This dataclass is generic and can be used with any environment.
    """
    id: str  # Unique identifier (e.g., "run1_005")
    text_state: str  # Current state as text
    action: int  # Action taken
    text_next_state: str  # Next state as text

    # Metadata for retrieval
    file_path: str  # Path to original file
    step_idx: int  # Step index in the trajectory

    def __repr__(self):
        text_preview = self.text_state[:50] + "..." if len(self.text_state) > 50 else self.text_state
        return f"CraftaxTransition(id='{self.id}', action={self.action}, text='{text_preview}')"


class TrajectoryTextConverter:
    """
    Generic converter for trajectory files to text transitions.

    Design principles for portability:
    - Takes a render_function as input (swap for different environments)
    - Takes a load_function as input (swap for different file formats)
    - Stores minimal state (just file index for on-demand loading)
    - No hard dependencies on specific environment implementations

    Args:
        render_function: Function that takes (state, **kwargs) and returns text string
        load_trajectory_function: Function that takes file_path and returns trajectory dict
                                 Expected dict format: {'state': [...], 'action': [...], ...}
        render_kwargs: Keyword arguments to pass to render_function (e.g., unique_items=True)
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

        # File index for on-demand state retrieval
        self._file_index: Dict[str, str] = {}  # Maps file_id -> file_path

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
            file_id = Path(file_path).stem  # e.g., "run1" from "run1.pbz2"

        # Store file path for on-demand loading
        self._file_index[file_id] = file_path

        # Load trajectory
        trajectory = self.load_trajectory_function(file_path)
        states = trajectory['state']
        actions = trajectory['action']

        # Convert to transitions
        transitions = []
        for i in range(len(states) - 1):  # -1 because we need next_state
            state = states[i]
            next_state = states[i + 1]
            action = actions[i]

            # Render states to text
            text_state = self.render_function(state, **self.render_kwargs)
            text_next_state = self.render_function(next_state, **self.render_kwargs)

            # Create transition
            transition = CraftaxTransition(
                id=f"{file_id}_{i:06d}",  # e.g., "run1_000005"
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
        """
        Retrieve the original state object for a transition ID.

        This enables on-demand loading without keeping all states in memory.

        Args:
            transition_id: Transition ID (e.g., "run1_000005")

        Returns:
            The original state object
        """
        # Parse ID
        parts = transition_id.rsplit('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid transition ID format: {transition_id}")

        file_id, step_str = parts
        step_idx = int(step_str)

        # Get file path
        if file_id not in self._file_index:
            raise ValueError(f"Unknown file_id: {file_id}. Was this trajectory converted?")

        file_path = self._file_index[file_id]

        # Load and return state
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
        """
        Retrieve a full transition by ID, optionally including original state objects.

        Args:
            transition_id: Transition ID (e.g., "run1_000005")
            include_states: If True, loads and includes original states (memory intensive)

        Returns:
            CraftaxTransition object (re-rendered from original states)
        """
        # Parse ID
        parts = transition_id.rsplit('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid transition ID format: {transition_id}")

        file_id, step_str = parts
        step_idx = int(step_str)

        # Get file path
        if file_id not in self._file_index:
            raise ValueError(f"Unknown file_id: {file_id}")

        file_path = self._file_index[file_id]

        # Load trajectory
        trajectory = self.load_trajectory_function(file_path)
        states = trajectory['state']
        actions = trajectory['action']

        if step_idx >= len(states) - 1:
            raise ValueError(f"Step index {step_idx} out of range")

        # Get states
        state = states[step_idx]
        next_state = states[step_idx + 1]
        action = actions[step_idx]

        # Render to text
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


# ============================================================================
# Craftax-specific utilities (swap these out for different projects)
# ============================================================================

def craftax_render_function(state, unique_items=True, precise_location=False):
    """
    Craftax-specific render function.

    Uses the standalone wrapper's text renderer for consistency.
    To use with a different environment, replace this with your own render function.

    Note: This requires standalone_craftax_wrapper.py to be in your Python path.
          If using in a different project, copy standalone_craftax_wrapper.py
          to your project directory.
    """
    try:
        from standalone_craftax_wrapper import CraftaxClassicLanguageWrapper
    except ImportError:
        # Fallback to internal renderer if standalone wrapper not available
        from craftax.craftax_classic.renderer import render_craftax_text_balrog
        return render_craftax_text_balrog(state, unique_items, precise_location)

    # Create a temporary wrapper instance with the desired config
    wrapper = CraftaxClassicLanguageWrapper(
        unique_items=unique_items,
        precise_location=precise_location
    )

    # Use the wrapper's text renderer
    return wrapper._render_text(state)


def craftax_load_trajectory(file_path: str) -> Dict[str, List]:
    """
    Craftax-specific trajectory loader.

    To use with a different file format, replace this with your own loader.

    Returns:
        Dict with keys: 'state', 'action', 'reward', 'done'
    """
    from craftax.environment_base.util import load_compressed_pickle
    return load_compressed_pickle(file_path)


def make_craftax_converter(unique_items=True, precise_location=False):
    """
    Factory function to create a converter configured for Craftax Classic.

    Example:
        converter = make_craftax_converter(unique_items=True)
        transitions = converter.convert_directory("data/trajectories/")
    """
    return TrajectoryTextConverter(
        render_function=craftax_render_function,
        load_trajectory_function=craftax_load_trajectory,
        render_kwargs={'unique_items': unique_items, 'precise_location': precise_location},
    )
