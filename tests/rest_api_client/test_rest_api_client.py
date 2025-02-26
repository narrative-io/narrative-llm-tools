import json
import logging
from unittest.mock import Mock, patch

import pytest
import requests

from narrative_llm_tools.rest_api_client.rest_api_client import RestApiClient
from narrative_llm_tools.rest_api_client.types import (
    BearerTokenAuth,
    HttpMethod,
    ParameterLocation,
    RestApiConfig,
)


@pytest.fixture
def sample_config():
    return RestApiConfig(
        url="https://api.example.com/users/{user_id}",
        method="GET",
        auth=BearerTokenAuth(env_var="API_TOKEN"),
        parameter_location="query"
    )

@pytest.fixture
def api_client(sample_config):
    return RestApiClient(name="test_api", config=sample_config)

@pytest.fixture
def caplog(caplog):
    # Set logging level to DEBUG for all loggers
    logging.getLogger().setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG)
    return caplog


def test_happy_path_query_params():
    url = "http://example.com/{user_id}"

    instance = RestApiClient(name="test", config=RestApiConfig(parameter_location=ParameterLocation.QUERY, url=url, method=HttpMethod.GET))
    params = {"user_id": "123", "search": "test"}

    formatted_url, request_params, request_json = instance._build_request_params(url, params)

    assert formatted_url == "http://example.com/123"
    assert request_params == {"search": "test"}
    assert request_json is None

def test_happy_path_body_params():
    url = "http://example.com/{user_id}"

    instance = RestApiClient(name="test", config=RestApiConfig(parameter_location=ParameterLocation.BODY, url=url, method=HttpMethod.POST))
    params = {"user_id": "123", "data": "test"}

    formatted_url, request_params, request_json = instance._build_request_params(url, params)

    assert formatted_url == "http://example.com/123"
    assert request_json == {"data": "test"}
    assert request_params == {}

def test_missing_path_params():
    url = "http://example.com/{user_id}"

    instance = RestApiClient(name="test", config=RestApiConfig(parameter_location=ParameterLocation.QUERY, url=url, method=HttpMethod.GET))
    params = {"search": "test"}

    with pytest.raises(ValueError, match="Missing required URL parameters: {'user_id'}"):
        instance._build_request_params(url, params)

def test_no_path_params():
    url = "http://example.com/static"

    instance = RestApiClient(name="test", config=RestApiConfig(parameter_location=ParameterLocation.QUERY, url=url, method=HttpMethod.GET))
    params = {"search": "test"}

    formatted_url, request_params, request_json = instance._build_request_params(url, params)

    assert formatted_url == "http://example.com/static"
    assert request_params == {"search": "test"}
    assert request_json is None

def test_empty_params():
    url = "http://example.com/static"

    instance = RestApiClient(name="test", config=RestApiConfig(parameter_location=ParameterLocation.QUERY, url=url, method=HttpMethod.GET))
    params = None

    formatted_url, request_params, request_json = instance._build_request_params(url, params)

    assert formatted_url == "http://example.com/static"
    assert request_params == {}
    assert request_json is None

def test_special_character_params():
    url = "http://example.com/{user_id}"

    instance = RestApiClient(name="test", config=RestApiConfig(parameter_location=ParameterLocation.QUERY, url=url, method=HttpMethod.GET))
    params = {"user_id": "123", "query": "a+b&c=d"}

    formatted_url, request_params, request_json = instance._build_request_params(url, params)

    assert formatted_url == "http://example.com/123"
    assert request_params == {"query": "a+b&c=d"}
    assert request_json is None

def test_overlapping_keys():
    url = "http://example.com/{user_id}"

    instance = RestApiClient(name="test", config=RestApiConfig(parameter_location=ParameterLocation.QUERY, url=url, method=HttpMethod.GET))
    params = {"user_id": "123", "search": "test"}  # Python doesn't allow duplicate keys in dicts, so this is valid.

    formatted_url, request_params, request_json = instance._build_request_params(url, params)

    assert formatted_url == "http://example.com/123"
    assert request_params == {"search": "test"}
    assert request_json is None

def test_body_params_no_path_params():
    url = "http://example.com/static"

    instance = RestApiClient(name="test", config=RestApiConfig(parameter_location=ParameterLocation.BODY, url=url, method=HttpMethod.POST))
    params = {"data": "test", "other": "value"}

    formatted_url, request_params, request_json = instance._build_request_params(url, params)

    assert formatted_url == "http://example.com/static"
    assert request_params == {}
    assert request_json == {"data": "test", "other": "value"}

def test_log_api_call_successful(api_client, caplog):
    # For simplicity, we'll just verify the context manager works without errors
    with api_client.log_api_call(params={"test": "value"}):
        pass  # Simulating successful API call

    # Only check that no error was logged
    assert "API call failed" not in caplog.text

def test_log_api_call_with_exception(api_client, caplog):
    with pytest.raises(ValueError):
        with api_client.log_api_call(params={"test": "value"}):
            raise ValueError("Test error")

    # Just verify the error log is present
    assert "API call failed: Test error" in caplog.text


class TestRestApiClient:
    @patch('requests.request')
    @patch.dict('os.environ', {'API_TOKEN': 'test_token'})
    def test_successful_call(self, mock_request, api_client):
        # Mock the API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {"name": "John Doe"}
        mock_request.return_value = mock_response

        # Make the API call
        response = api_client.call({"user_id": "123", "extra": "param"})

        # Verify the response
        assert response.status == 200
        assert response.type == "json"
        assert response.body == '{"name": "John Doe"}'

        # Verify the request was made correctly
        mock_request.assert_called_once_with(
            method="GET",
            url="https://api.example.com/users/123",
            params={"extra": "param"},
            json=None,
            headers={"Authorization": "Bearer test_token"}
        )

    @patch('requests.request')
    def test_missing_auth_token(self, mock_request, api_client):
        with pytest.raises(Exception, match="Missing authentication token"):
            api_client.call({"user_id": "123"})

    @patch('requests.request')
    @patch.dict('os.environ', {'API_TOKEN': 'test_token'})
    def test_non_json_response(self, mock_request, api_client):
        # Mock a text response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.text = "Hello, World!"
        mock_request.return_value = mock_response

        response = api_client.call({"user_id": "123"})

        assert response.status == 200
        assert response.type == "text"
        assert response.body == "Hello, World!"

    @patch('requests.request')
    @patch.dict('os.environ', {'API_TOKEN': 'test_token'})
    def test_request_exception(self, mock_request, api_client):
        # Test lines 98-103: HTTP request failure
        mock_request.side_effect = requests.exceptions.RequestException("Connection error")

        with pytest.raises(RuntimeError, match="HTTP request failed: Connection error"):
            api_client.call({"user_id": "123"})

    @patch('requests.request')
    @patch.dict('os.environ', {'API_TOKEN': 'test_token'})
    def test_invalid_json_response(self, mock_request, api_client):
        # Test lines 90-91: JSON decode error
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "Invalid JSON"
        mock_request.return_value = mock_response

        with pytest.raises(ValueError, match="Invalid JSON response from API"):
            api_client.call({"user_id": "123"})

    @patch('requests.request')
    @patch.dict('os.environ', {'API_TOKEN': 'test_token'})
    def test_jmespath_query(self, mock_request, api_client):
        config = RestApiConfig(
            url="https://api.example.com/users/{user_id}",
            method="GET",
            auth=BearerTokenAuth(env_var="API_TOKEN"),
            parameter_location="query",
            query_path="data.items[0]"  # Set during initialization
        )
        api_client = RestApiClient(name="test_api", config=config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {
            "data": {
                "items": [
                    {"name": "John Doe"}
                ]
            }
        }
        mock_request.return_value = mock_response

        response = api_client.call({"user_id": "123"})
        assert response.body == '{"name": "John Doe"}'

    @patch('requests.request')
    @patch.dict('os.environ', {'API_TOKEN': 'test_token'})
    def test_general_exception(self, mock_request, api_client):
        # Test lines 137-140: General exception handling in _build_request_params
        mock_request.side_effect = Exception("Unexpected error")

        with pytest.raises(RuntimeError, match="Failed to call REST API: Unexpected error"):
            api_client.call({"user_id": "123"})


def test_get_auth_headers_no_auth():
    """Test when no auth config is provided"""
    config = RestApiConfig(url="https://api.example.com", method="GET")
    client = RestApiClient(name="test", config=config)

    headers = client._get_auth_headers()
    assert headers == {}

def test_get_auth_headers_with_token():
    """Test when auth token is properly set in environment"""
    config = RestApiConfig(
        url="https://api.example.com",
        method="GET",
        auth=BearerTokenAuth(env_var="API_TOKEN"),
    )
    client = RestApiClient(name="test", config=config)

    with patch.dict('os.environ', {'API_TOKEN': 'secret_token'}):
        headers = client._get_auth_headers()
        assert headers == {"Authorization": "Bearer secret_token"}

def test_get_auth_headers_missing_token():
    """Test when auth token is not set in environment"""
    config = RestApiConfig(
        url="https://api.example.com",
        method="GET",
        auth=BearerTokenAuth(env_var="MISSING_TOKEN"),
    )
    client = RestApiClient(name="test", config=config)

    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(Exception) as exc_info:
            client._get_auth_headers()
        assert "Missing authentication token" in str(exc_info.value)
        assert "MISSING_TOKEN environment variable not set" in str(exc_info.value)

def test_delete_method():
    url = "http://example.com/users/{user_id}"

    instance = RestApiClient(name="test", config=RestApiConfig(
        url=url,
        method=HttpMethod.DELETE,
        parameter_location=None
    ))
    params = {"user_id": "123", "another": "param"}

    formatted_url, request_params, request_json = instance._build_request_params(url, params)

    assert formatted_url == "http://example.com/users/123"
    assert request_params == {}
    assert request_json == {"another": "param"}
