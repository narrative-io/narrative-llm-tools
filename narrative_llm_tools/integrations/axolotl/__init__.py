"""
A simple Axolotl plugin that demonstrates basic plugin functionality.
This plugin serves as a minimal example of the plugin system.
"""

import logging

from axolotl.integrations.base import BasePlugin

LOG = logging.getLogger("narrative_llm_tools.integrations.axolotl")


class NarrativePlugin(BasePlugin):
    """
    A minimal Axolotl plugin implementation that just imports a library.

    This plugin demonstrates the basic structure of an Axolotl plugin
    while keeping implementation details to a minimum.
    """

    def __init__(self) -> None:
        """Initialize the Narrative Plugin."""
        super().__init__()  # type: ignore
        try:
            import narrative_llm_tools

            self.narrative_llm_tools = narrative_llm_tools
        except ImportError as e:
            raise ImportError("Required package 'narrative_llm_tools' is not installed") from e

    def get_input_args(self) -> str | None:
        """
        Get the input arguments for the plugin.
        """
        return "narrative_llm_tools.integrations.axolotl.NarrativePluginArgs"
