"""
Trajectory to text converter - converts trajectory data to text observations.

To use with different environments:
1. Provide a render_function that converts state to text
2. Provide a load_trajectory_function for your file format
3. Use TrajectoryTextConverter with your custom functions

For Craftax Classic: Uses standalone_craftax_wrapper.py renderer.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable, Any, Optional, Dict, Tuple


def action_to_text(action: int) -> str:
    """Convert action number to text description."""
    action_descriptions = {
        0: "do nothing",
        1: "move left",
        2: "move right",
        3: "move up",
        4: "move down",
        5: "interact with object in front",
        6: "sleep",
        7: "place stone",
        8: "place crafting table",
        9: "place furnace",
        10: "place plant",
        11: "craft wooden pickaxe",
        12: "craft stone pickaxe",
        13: "craft iron pickaxe",
        14: "craft wooden sword",
        15: "craft stone sword",
        16: "craft iron sword",
    }
    return action_descriptions.get(action, f"unknown action {action}")


@dataclass
class CraftaxTransition:
    """A single transition in text format."""
    id: str
    text_state: str
    action: int
    text_action: str
    text_next_state: str
    file_path: str
    step_idx: int

    def __repr__(self):
        text_preview = self.text_state[:50] + "..." if len(self.text_state) > 50 else self.text_state
        return f"CraftaxTransition(id='{self.id}', action={self.text_action}, text='{text_preview}')"


class TrajectoryTextConverter:
    """
    Converter for trajectory files to text transitions.

    Loads a trajectory file to memory and provides on-demand text rendering
    with random sampling and batch retrieval capabilities.

    Args:
        file_path: Path to trajectory file
        render_function: Function that takes (state, **kwargs) and returns text string
        load_trajectory_function: Function that takes file_path and returns trajectory dict
                                 Expected dict format: {'state': [...], 'action': [...], ...}
        render_kwargs: Keyword arguments to pass to render_function
    """

    def __init__(
        self,
        file_path: str,
        render_function: Callable[[Any], str],
        load_trajectory_function: Callable[[str], Dict[str, List]],
        render_kwargs: Optional[Dict] = None,
    ):
        self.file_path = file_path
        self.render_function = render_function
        self.render_kwargs = render_kwargs or {}

        # Load trajectory to memory
        trajectory = load_trajectory_function(file_path)
        self.states = trajectory['state']
        self.actions = trajectory['action']

        # Calculate available transition IDs
        self.file_id = Path(file_path).stem
        self.transition_ids = [f"{self.file_id}_{i:06d}" for i in range(len(self.states) - 1)]

    def sample_transitions(self, n: int) -> List[CraftaxTransition]:
        """
        Randomly sample n transitions from the trajectory.

        Args:
            n: Number of transitions to sample

        Returns:
            List of CraftaxTransition objects
        """
        import random
        sampled_ids = random.sample(self.transition_ids, min(n, len(self.transition_ids)))
        return self.get_transitions(sampled_ids)

    def get_transitions(self, transition_ids: List[str]) -> List[CraftaxTransition]:
        """
        Retrieve a batch of transitions by their IDs.

        Args:
            transition_ids: List of transition IDs to retrieve

        Returns:
            List of CraftaxTransition objects
        """
        transitions = []
        for tid in transition_ids:
            step_idx = int(tid.rsplit('_', 1)[1])

            if step_idx >= len(self.states) - 1:
                raise ValueError(f"Step index {step_idx} out of range")

            state = self.states[step_idx]
            next_state = self.states[step_idx + 1]
            action = self.actions[step_idx]

            text_state = self.render_function(state, **self.render_kwargs)
            text_next_state = self.render_function(next_state, **self.render_kwargs)
            text_action = action_to_text(int(action))

            transition = CraftaxTransition(
                id=tid,
                text_state=text_state,
                action=int(action),
                text_action=text_action,
                text_next_state=text_next_state,
                file_path=self.file_path,
                step_idx=step_idx,
            )
            transitions.append(transition)

        return transitions


def craftax_render_function(state, unique_items=True, precise_location=False):
    """Render Craftax state to text using standalone wrapper."""
    from standalone_craftax_wrapper import CraftaxClassicLanguageWrapper

    wrapper = CraftaxClassicLanguageWrapper(
        unique_items=unique_items,
        precise_location=precise_location
    )
    return wrapper._render_text(state)


def craftax_load_trajectory(file_path: str) -> Dict[str, List]:
    """Load Craftax trajectory file."""
    from craftax.environment_base.util import load_compressed_pickle
    return load_compressed_pickle(file_path)


def make_craftax_converter(file_path: str, unique_items=True, precise_location=False):
    """Create a converter configured for Craftax Classic."""
    return TrajectoryTextConverter(
        file_path=file_path,
        render_function=craftax_render_function,
        load_trajectory_function=craftax_load_trajectory,
        render_kwargs={'unique_items': unique_items, 'precise_location': precise_location},
    )
