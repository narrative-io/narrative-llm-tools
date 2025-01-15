import pytest
from unittest.mock import Mock
import requests

from narrative_llm_tools.narrative_api.narrative_api_client import NarrativeAPIClient, NarrativeAPIError
from narrative_llm_tools.narrative_api.narrative_api_config import NarrativeAPIConfig

@pytest.fixture
def mock_config():
    return NarrativeAPIConfig(
        base_url="https://api.example.com",
        api_key="test-key",
        verify_ssl=True,
        timeout=30,
    )

@pytest.fixture
def api_client(mock_config):
    return NarrativeAPIClient(mock_config)

class TestNarrativeAPIClient:
    def test_init_creates_session_with_auth_headers(self, api_client):
        assert api_client.session.headers["Authorization"] == "Bearer test-key"
        assert api_client.session.headers["Content-Type"] == "application/json"
        assert api_client.session.verify is True

    @pytest.mark.parametrize(
        "method,endpoint,expected_url",
        [
            ("GET", "/test", "https://api.example.com/test"),
            ("POST", "test", "https://api.example.com/test"),
            ("PUT", "/v1/test", "https://api.example.com/v1/test"),
        ],
    )
    def test_make_request_builds_correct_url(self, api_client, method, endpoint, expected_url, mocker):
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"data": "test"}
        mock_request = mocker.patch.object(api_client.session, "request", return_value=mock_response)

        response = api_client._make_request(method, endpoint)

        mock_request.assert_called_once()
        assert mock_request.call_args[1]["url"] == expected_url
        assert response.status_code == 200
        assert response.data == {"data": "test"}

    def test_make_request_handles_request_exception(self, api_client, mocker):
        mocker.patch.object(
            api_client.session,
            "request",
            side_effect=requests.RequestException("Network error"),
        )

        with pytest.raises(NarrativeAPIError) as exc_info:
            api_client._make_request("GET", "/test")
        
        assert "Network error" in str(exc_info.value)

    def test_upload_file(self, api_client, mocker):
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"status": "success"}
        mock_request = mocker.patch.object(api_client.session, "request", return_value=mock_response)

        response = api_client.upload_file("/upload", "test content")

        mock_request.assert_called_once()
        assert mock_request.call_args[1]["method"] == "PUT"
        assert "files" in mock_request.call_args[1]
        assert response.status_code == 200
        assert response.data == {"status": "success"}

    @pytest.mark.parametrize(
        "method_name,http_method",
        [
            ("get", "GET"),
            ("post", "POST"),
            ("put", "PUT"),
            ("delete", "DELETE"),
        ],
    )
    def test_http_methods(self, api_client, method_name, http_method, mocker):
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"data": "test"}
        mock_request = mocker.patch.object(api_client.session, "request", return_value=mock_response)

        method = getattr(api_client, method_name)
        response = method("/test", data={"key": "value"} if method_name != "delete" else None)

        mock_request.assert_called_once()
        assert mock_request.call_args[1]["method"] == http_method
        assert response.status_code == 200
        assert response.data == {"data": "test"}