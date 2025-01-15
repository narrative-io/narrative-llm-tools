import logging
from collections.abc import Hashable
from typing import Literal

from cachetools import LRUCache, cached
from cachetools.keys import hashkey
from lmformatenforcer import (  # type: ignore
    CharacterLevelParser,
    CharacterLevelParserConfig,
    JsonSchemaParser,
    SequenceParser,
)
from lmformatenforcer.integrations.transformers import (  # type: ignore
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import PreTrainedTokenizerBase  # type: ignore

from narrative_llm_tools.tools.json_schema_tools import JsonSchemaTools
from narrative_llm_tools.utils.types import TransformersPrefixAllowedTokensFn

logger = logging.getLogger("narrative-llm-tools")


class ThoughtsParser(CharacterLevelParser):  # type: ignore
    """A CharacterLevelParser that only allows generation to
    end if the string ends with <|end_thought|>"""

    END_SEQUENCE = "<|end_thought|>"
    WINDOW_SIZE = len(END_SEQUENCE)

    def __init__(self, config: CharacterLevelParserConfig | None = None, is_optional: bool = False):
        super().__init__(config)
        self._text_window = ""
        self._consecutive_whitespace_count = 0
        self._end_sequence_match_length = 0
        self._is_optional = is_optional

    def add_character(self, new_character: str) -> "ThoughtsParser":
        """Returns a new parser with the character added to the current text."""
        # Track consecutive whitespaces (includes spaces, tabs, newlines, returns, etc.)
        new_consecutive_whitespace_count = (
            self._consecutive_whitespace_count + 1 if new_character.isspace() else 0
        )

        new_parser = ThoughtsParser(self.config)
        new_parser._text_window = (self._text_window + new_character)[-self.WINDOW_SIZE :]
        new_parser._consecutive_whitespace_count = new_consecutive_whitespace_count

        if new_character == self.END_SEQUENCE[new_parser._end_sequence_match_length]:
            new_parser._end_sequence_match_length = self._end_sequence_match_length + 1
        else:
            # If we failed to match, check if we're starting a new match
            new_parser._end_sequence_match_length = (
                1 if new_character == self.END_SEQUENCE[0] else 0
            )

        return new_parser

    def get_allowed_characters(self) -> str:
        """Returns allowed characters based on config and current state."""
        if self._consecutive_whitespace_count >= self.config.max_consecutive_whitespaces:
            filtered_chars: str = "".join([c for c in self.config.alphabet if not c.isspace()])
            return filtered_chars
        return str(self.config.alphabet)

    def can_end(self) -> bool:
        """Returns True only if the current text ends with <|end_thought|>"""
        return (self._text_window == self.END_SEQUENCE) or (
            self._is_optional and self._text_window == ""
        )

    def cache_key(self) -> Hashable | None:
        """Returns a tuple of just the essential state:
        - How many consecutive whitespaces we have
        - How many characters we've matched of the end sequence
        - Whether the parser is optional
        - Whether the text window is empty

        ## TODO: Is this the best way to do this? At any state we can choose any character
        ## to add.  The only thing we are changing is if we can add a whitespace character
        ## and/or if we've matched the end sequence.
        """
        return (
            self._consecutive_whitespace_count,
            self._end_sequence_match_length,
            self._is_optional,
            self._text_window == "",
        )

    def shortcut_key(self) -> Hashable | None:
        """Same as cache_key since we need both pieces of information"""
        return self.cache_key()


def _cache_key(
    tokenizer: PreTrainedTokenizerBase, tool_set: JsonSchemaTools
) -> tuple[Hashable, ...]:
    return hashkey(tokenizer.name_or_path, tool_set)


@cached(cache=LRUCache(maxsize=100), key=_cache_key)
def get_format_enforcer(
    tokenizer: PreTrainedTokenizerBase,
    tool_set: JsonSchemaTools,
    chain_of_thought: Literal["required", "allowed"] | None = None,
) -> TransformersPrefixAllowedTokensFn:
    """
    Retrieves or creates a format enforcer function for the given tool set.

    Note:
        Building the prefix_fn in an expensive operation, so we cache up to 100 of them.  Repeated
        calls using the same tool_set will return the cached prefix_fn and will be noiticably
        faster to the users.

    Args:
        tool_set (dict): Dictionary describing the tool configuration.

    Returns:
        TransformersPrefixAllowedTokensFn: The callable that enforces the format of the tool set.
    """
    logger.info(f"Creating format enforcer with tool_set: {tool_set}")
    tools_json_parser = JsonSchemaParser(tool_set.model_dump())

    if chain_of_thought == "required":
        parser = SequenceParser([ThoughtsParser(), tools_json_parser])
    elif chain_of_thought == "allowed":
        parser = SequenceParser([ThoughtsParser(is_optional=True), tools_json_parser])
    else:
        parser = tools_json_parser

    try:
        enforcer: TransformersPrefixAllowedTokensFn = build_transformers_prefix_allowed_tokens_fn(
            tokenizer, parser
        )
        return enforcer
    except Exception:
        logger.error(f"Failed to create format enforcer. Tool set: {tool_set}", exc_info=True)
        raise
