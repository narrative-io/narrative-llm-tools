"""
Module for handling Narrative I/O Pluging input arguments.
"""

import logging

from pydantic import BaseModel

LOG = logging.getLogger("narrative_llm_tools.integrations.axolotl.args")


class NarrativePluginArgs(BaseModel):
    """
    Input args for Narrative Plugin.
    """

    pass
