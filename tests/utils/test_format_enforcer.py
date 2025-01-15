from collections.abc import Hashable
from dataclasses import dataclass
from unittest.mock import Mock, patch
import pytest
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer

from narrative_llm_tools.tools.json_schema_tools import JsonSchemaTools
from narrative_llm_tools.utils.format_enforcer import ThoughtsParser, TransformersPrefixAllowedTokensFn, get_format_enforcer


class SimpleSchema(BaseModel):
    name: str
    age: int

@pytest.fixture
def test_schema():
    return {
                    "type": "array",
                    "items": {
                    "anyOf": [
                        {
                        "type": "object",
                        "required": ["name", "parameters"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": ["my_generic_tool"],
                                "description": "The unique name of this tool."
                            },
                            "parameters": {
                            "type": "object",
                            "properties": {
                                "param1": {
                                    "type": "string",
                                    "description": "A required string parameter for this tool."
                                },
                                "param2": {
                                    "type": "integer",
                                    "description": "An optional integer parameter for this tool."
                                }
                            },
                            "required": ["param1"],
                            "additionalProperties": False
                            }
                        },
                        "additionalProperties": False
                        },
                        {
                        "type": "object",
                        "required": ["name", "parameters"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": ["text_summarizer"],
                                "description": "This tool is a text summarizer."
                            },
                            "parameters": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "The text to summarize."
                                },
                                "summary_length": {
                                    "type": "integer",
                                    "description": "Desired length of the summary."
                                }
                            },
                            "required": ["text"],
                            "additionalProperties": False
                            }
                        },
                        "additionalProperties": False
                        }
                    ]
                    }
                }

def test_get_format_enforcer_basic(test_schema):
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    schema = JsonSchemaTools.model_validate(test_schema)

    # Execute
    enforcer = get_format_enforcer(tokenizer, schema)

    # Assert
    assert callable(enforcer)

def test_format_enforcer_caching(test_schema):
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    schema = JsonSchemaTools.model_validate(test_schema)

    # Execute
    enforcer1 = get_format_enforcer(tokenizer, schema)
    enforcer2 = get_format_enforcer(tokenizer, schema)

    # Assert
    assert enforcer1 is enforcer2  # Should return cached instance

def test_format_enforcer_different_schemas(test_schema):
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    schema1 = JsonSchemaTools.model_validate(test_schema)
    schema2 = JsonSchemaTools.model_validate({
                    "type": "array",
                    "items": {
                    "anyOf": [
                        {
                        "type": "object",
                        "required": ["name", "parameters"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": ["my_generic_tool"],
                                "description": "The unique name of this tool."
                            },
                            "parameters": {
                            "type": "object",
                            "properties": {
                                "param1": {
                                    "type": "string",
                                    "description": "A required string parameter for this tool."
                                },
                                "param2": {
                                    "type": "integer",
                                    "description": "An optional integer parameter for this tool."
                                }
                            },
                            "required": ["param1"],
                            "additionalProperties": False
                            }
                        },
                        "additionalProperties": False
                        }]}})

    # Execute
    enforcer1 = get_format_enforcer(tokenizer, schema1)
    enforcer2 = get_format_enforcer(tokenizer, schema2)

    # Assert
    assert enforcer1 is not enforcer2  # Should be different instances

def test_format_enforcer_invalid_schema():
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    with pytest.raises(Exception):
        invalid_schema = JsonSchemaTools.model_validate({
            "type": "invalid_type",  # Invalid schema type
            "properties": {}
        })

        get_format_enforcer(tokenizer, invalid_schema)

def test_format_enforcer_functionality(test_schema):
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    schema = JsonSchemaTools.model_validate(test_schema)

    enforcer = get_format_enforcer(tokenizer, schema)

    # Create a simple input tensor
    input_ids = tokenizer.encode('{"', return_tensors='pt')[0]

    # Execute
    allowed_tokens = enforcer(0, input_ids)

    # Assert
    assert isinstance(allowed_tokens, list)
    assert len(allowed_tokens) > 0

def test_get_format_enforcer_exception():
    # Mock dependencies
    mock_tokenizer = Mock()
    mock_tokenizer.name_or_path = "test-tokenizer"
    
    mock_tool_set = Mock()
    mock_tool_set.model_dump.return_value = {"some": "schema"}

    # Patch the build_transformers_prefix_allowed_tokens_fn to raise an exception
    with patch(
        "narrative_llm_tools.utils.format_enforcer.build_transformers_prefix_allowed_tokens_fn",
        side_effect=ValueError("Test error")
    ):
        # Verify that the exception is propagated
        with pytest.raises(Exception):
            get_format_enforcer(mock_tokenizer, mock_tool_set)

@dataclass
class CharacterLevelParserConfig:
    alphabet: str = "abcdefghijklmnopqrstuvwxyz <>\n\t|_"
    max_consecutive_whitespaces: int = 2

class CharacterLevelParser:
    def __init__(self, config: CharacterLevelParserConfig | None = None):
        self.config = config or CharacterLevelParserConfig()

class TestThoughtsParser:
    @pytest.fixture
    def default_config(self):
        return CharacterLevelParserConfig()

    @pytest.fixture
    def parser(self, default_config):
        return ThoughtsParser(default_config)

    @pytest.fixture
    def optional_parser(self, default_config):
        return ThoughtsParser(default_config, is_optional=True)

    def test_initialization(self, parser):
        assert parser._text_window == ""
        assert parser._consecutive_whitespace_count == 0
        assert parser._end_sequence_match_length == 0
        assert not parser._is_optional

    def test_optional_initialization(self, optional_parser):
        assert optional_parser._is_optional
        assert optional_parser.can_end()

    def test_add_character_basic(self, parser):
        new_parser = parser.add_character("a")
        assert new_parser._text_window == "a"
        assert new_parser._consecutive_whitespace_count == 0

    def test_add_character_whitespace(self, parser):
        parser1 = parser.add_character(" ")
        assert parser1._consecutive_whitespace_count == 1
        
        parser2 = parser1.add_character(" ")
        assert parser2._consecutive_whitespace_count == 2
        
        parser3 = parser2.add_character("a")
        assert parser3._consecutive_whitespace_count == 0

    def test_window_size_limit(self, parser):
        # Add more characters than window size
        current_parser = parser
        test_string = "hello" + ThoughtsParser.END_SEQUENCE
        
        for char in test_string:
            current_parser = current_parser.add_character(char)
        
        assert len(current_parser._text_window) == ThoughtsParser.WINDOW_SIZE
        assert current_parser._text_window == ThoughtsParser.END_SEQUENCE

    def test_end_sequence_matching(self, parser):
        current_parser = parser
        
        # Add the end sequence character by character
        for char in ThoughtsParser.END_SEQUENCE:
            current_parser = current_parser.add_character(char)
            
        assert current_parser.can_end()

    def test_end_sequence_partial_match(self, parser):
        current_parser = parser
        
        # Add partial end sequence
        partial_sequence = ThoughtsParser.END_SEQUENCE[:-1]
        for char in partial_sequence:
            current_parser = current_parser.add_character(char)
            
        assert not current_parser.can_end()

    def test_end_sequence_reset(self, parser):
        current_parser = parser
        
        # Start matching end sequence
        for char in "<|end_":
            current_parser = current_parser.add_character(char)
        
        # Break the sequence
        current_parser = current_parser.add_character("x")
        assert current_parser._end_sequence_match_length == 0

    def test_get_allowed_characters_normal(self, parser):
        allowed = parser.get_allowed_characters()
        assert allowed == parser.config.alphabet

    def test_get_allowed_characters_max_whitespace(self, parser):
        current_parser = parser
        
        # Add max consecutive whitespaces
        for _ in range(parser.config.max_consecutive_whitespaces):
            current_parser = current_parser.add_character(" ")
        
        allowed = current_parser.get_allowed_characters()
        assert " " not in allowed
        assert "\n" not in allowed
        assert "\t" not in allowed

    def test_cache_key(self, parser):
        key = parser.cache_key()
        assert isinstance(key, Hashable)
        assert len(key) == 4
        assert key == (0, 0, False, True)

    def test_cache_key_changes(self, parser):
        initial_key = parser.cache_key()
        
        # Add a character and check if key changes
        new_parser = parser.add_character("a")
        new_key = new_parser.cache_key()
        
        assert initial_key != new_key

    def test_shortcut_key(self, parser):
        assert parser.shortcut_key() == parser.cache_key()

    @pytest.mark.parametrize("text,expected_result", [
        ("", False),
        ("some random text", False),
        ("<|end_thought|>", True),
        ("text<|end_thought|>", True),
    ])
    def test_various_end_conditions(self, parser, text, expected_result):
        current_parser = parser
        for char in text:
            current_parser = current_parser.add_character(char)
        assert current_parser.can_end() == expected_result

    def test_optional_parser_empty_string(self, optional_parser):
        assert optional_parser.can_end()

    def test_optional_parser_with_content(self, optional_parser):
        parser = optional_parser.add_character("a")
        assert not parser.can_end()
        
        # Add end sequence
        current_parser = parser
        for char in ThoughtsParser.END_SEQUENCE:
            current_parser = current_parser.add_character(char)
        assert current_parser.can_end()

    @pytest.mark.parametrize("whitespace_char", [" ", "\n", "\t"])
    def test_different_whitespace_characters(self, parser, whitespace_char):
        current_parser = parser
        
        # Add multiple whitespace characters
        for _ in range(3):
            current_parser = current_parser.add_character(whitespace_char)
            
        # Verify that non-whitespace characters are still allowed
        allowed = current_parser.get_allowed_characters()
        assert "a" in allowed
        assert whitespace_char not in allowed