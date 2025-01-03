import pytest
from narrative_llm_tools.rest_api_client.types import (
    RestApiConfig,
    HttpMethod,
    ParameterLocation,
    BearerTokenAuth,
    ReturnToLlmBehavior,
)


class TestRestApiConfig:
    def test_default_parameter_location_for_post(self):
        config = RestApiConfig(
            url="https://api.example.com",
            method=HttpMethod.POST,
        )
        assert config.parameter_location == ParameterLocation.BODY

    def test_default_parameter_location_for_put(self):
        config = RestApiConfig(
            url="https://api.example.com",
            method=HttpMethod.PUT,
        )
        assert config.parameter_location == ParameterLocation.BODY

    def test_default_parameter_location_for_get(self):
        config = RestApiConfig(
            url="https://api.example.com",
            method=HttpMethod.GET,
        )
        assert config.parameter_location == ParameterLocation.QUERY

    def test_default_parameter_location_for_delete(self):
        config = RestApiConfig(
            url="https://api.example.com",
            method=HttpMethod.DELETE,
        )
        assert config.parameter_location == ParameterLocation.BODY

    def test_explicit_parameter_location_overrides_default(self):
        config = RestApiConfig(
            url="https://api.example.com",
            method=HttpMethod.GET,
            parameter_location=ParameterLocation.BODY,
        )
        assert config.parameter_location == ParameterLocation.BODY

    def test_default_response_behavior(self):
        config = RestApiConfig(
            url="https://api.example.com",
            method=HttpMethod.GET,
        )
        assert "default" in config.response_behavior
        assert isinstance(config.response_behavior["default"], ReturnToLlmBehavior)
        assert config.response_behavior["default"].response is None

    def test_equality(self):
        config1 = RestApiConfig(
            url="https://api.example.com",
            method=HttpMethod.GET,
            auth=BearerTokenAuth(env_var="API_KEY"),
        )
        config2 = RestApiConfig(
            url="https://api.example.com",
            method=HttpMethod.GET,
            auth=BearerTokenAuth(env_var="API_KEY"),
        )
        config3 = RestApiConfig(
            url="https://api.example.com",
            method=HttpMethod.POST,
            auth=BearerTokenAuth(env_var="API_KEY"),
        )
        
        assert config1 == config2
        assert config1 != config3
        assert hash(config1) == hash(config2)
        assert hash(config1) != hash(config3)
