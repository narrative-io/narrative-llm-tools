import logging
from collections.abc import Hashable

from cachetools import LRUCache, cached
from cachetools.keys import hashkey
from lmformatenforcer import JsonSchemaParser  # type: ignore
from lmformatenforcer.integrations.transformers import (  # type: ignore
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import PreTrainedTokenizerBase  # type: ignore

from narrative_llm_tools.tools.json_schema_tools import JsonSchemaTools
from narrative_llm_tools.utils.types import TransformersPrefixAllowedTokensFn

logger = logging.getLogger("narrative-llm-tools")


def _cache_key(
    tokenizer: PreTrainedTokenizerBase, tool_set: JsonSchemaTools
) -> tuple[Hashable, ...]:
    return hashkey(tokenizer.name_or_path, tool_set)


@cached(cache=LRUCache(maxsize=100), key=_cache_key)
def get_format_enforcer(
    tokenizer: PreTrainedTokenizerBase, tool_set: JsonSchemaTools
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
    try:
        enforcer: TransformersPrefixAllowedTokensFn = build_transformers_prefix_allowed_tokens_fn(
            tokenizer, JsonSchemaParser(tool_set.model_dump())
        )
        return enforcer
    except Exception:
        logger.error(f"Failed to create format enforcer. Tool set: {tool_set}", exc_info=True)
        raise
