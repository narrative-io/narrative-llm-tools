import json
import logging
import os
import string
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import jmespath
import requests
from pydantic import BaseModel

from narrative_llm_tools.rest_api_client.types import (
    ParameterLocation,
    RestApiConfig,
    RestApiResponse,
)

logger = logging.getLogger(__name__)


class RestApiClient(BaseModel):
    """A tool for making REST API calls with configurable parameters and authentication.

    It supports various parameter locations (path, query, body),
    authentication via environment variables, and JSON response parsing with JMESPath.

    Attributes:
        name (str): The unique identifier for this API tool.
        config (RestApiConfig): Configuration object containing API endpoint details,
            authentication settings, and response handling options.
    """

    name: str
    config: RestApiConfig

    def __init__(self, name: str, config: RestApiConfig):
        """Initialize a new RestApiTool instance.

        Args:
            name (str): The name identifier for this API tool.
            config (RestApiConfig): Configuration object for the API endpoint.
        """
        super().__init__(name=name, config=config)
        self.name = name
        RestApiConfig.model_validate(config)
        self.config = config

    def call(self, params: dict[str, Any] | None = None) -> RestApiResponse:
        """Execute the API call with the provided parameters.

        Makes an HTTP request to the configured endpoint, handling parameter placement,
        authentication, and response processing according to the tool's configuration.

        Args:
            params (Optional[Dict[str, Any]], optional): Parameters to be sent with the request.
                The placement of these parameters (path, query, or body) is determined by
                the tool's configuration. Defaults to None.

        Returns:
            str: The processed API response. If a query_path is configured, returns the extracted
                data. Otherwise, returns the full response as a JSON string.

        Raises:
            RuntimeError: If the HTTP request fails or if there's an error processing the response.
            ValueError: If the API response contains invalid JSON.

        Example:
            >>> tool = RestApiTool("weather", config)
            >>> result = tool.call({"city": "London"})
        """
        headers = self._get_auth_headers()

        try:
            url, request_params, request_json = self._build_request_params(self.config.url, params)
            logger.debug(
                f"Calling tool {self.name} with parameters {params}, url {url}, "
                f"request_params {request_params}, request_json {request_json}"
            )

            with self.log_api_call(params or {}):
                response = requests.request(
                    method=self.config.method,
                    url=url,
                    params=request_params,
                    json=request_json,
                    headers=headers,
                )

            if response.headers.get("Content-Type") == "application/json":
                json_data = response.json()
                logger.debug(f"Response from tool {self.name}: {json_data}")

                if self.config.query_path:
                    json_data = jmespath.search(self.config.query_path, json_data)
                    logger.info(f"Transformed data from tool {self.name}: {json_data}")

                return RestApiResponse(
                    status=response.status_code,
                    type="json",
                    body=json.dumps(json_data),
                    request=json.dumps(request_json),
                )
            else:
                logger.debug(f"Response from tool {self.name}: {response.text}")
                return RestApiResponse(
                    status=response.status_code,
                    type="text",
                    body=response.text,
                    request=json.dumps(request_json),
                )

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HTTP request failed: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from API: {response.text[:200]}...") from e
        except Exception as e:
            raise RuntimeError(f"Failed to call REST API: {str(e)}") from e

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers based on configuration.

        Returns:
            Dict[str, str]: Headers containing authentication information.

        Raises:
            Exception: If the required authentication token is not set in environment variables.
        """
        headers = {}
        if self.config.auth:
            auth_token = os.getenv(self.config.auth.env_var)
            if not auth_token:
                raise Exception(
                    f"Missing authentication token: {self.config.auth.env_var} "
                    "environment variable not set"
                )
            headers["Authorization"] = f"Bearer {auth_token}"
        return headers

    def _build_request_params(
        self, url: str, params: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
        """Build the request parameters based on the configuration.

        Args:
            url: The base URL to use
            params: Parameters to be sent with the request

        Returns:
            tuple containing:
                - formatted URL
                - query parameters dict
                - request body dict (or None)

        Raises:
            ValueError: If required URL template parameters are missing
        """
        params = params or {}
        request_params = {}
        request_json = None

        path_params = {name for _, name, _, _ in string.Formatter().parse(url) if name is not None}

        if path_params:
            missing_params = path_params - params.keys()
            if missing_params:
                raise ValueError(f"Missing required URL parameters: {missing_params}")
            url = url.format(**{k: params[k] for k in path_params})

        remaining_params = {k: v for k, v in params.items() if k not in path_params}

        if remaining_params:
            if self.config.parameter_location == ParameterLocation.QUERY:
                request_params = remaining_params

            if self.config.parameter_location == ParameterLocation.BODY:
                request_json = remaining_params

        return url, request_params, request_json

    @contextmanager
    def log_api_call(self, params: dict[str, Any]) -> Generator[None, None, None]:
        start_time = time.time()
        logger.debug(f"Starting API call to {self.name}")

        try:
            yield
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
        finally:
            duration = time.time() - start_time
            logger.debug(f"API call completed in {duration:.2f}s")
