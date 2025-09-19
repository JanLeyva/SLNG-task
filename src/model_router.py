"""
AI Model Router for handling multiple AI providers.
"""

# 1st party
import asyncio
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict
import random
import hashlib

# 3rd party
import aiohttp
from pydantic import BaseModel, Field
from loguru import logger


class EndpointConfig(BaseModel):
    """Configuration for a single API endpoint."""

    name: str
    url: str
    type: str  # 'http' or 'websocket'
    weight: int
    timeout: float = 10


class RetryConfig(BaseModel):
    """Configuration for retry logic."""

    max_attempts: int = 3
    backoff_factor: float = 2.0


class CircuitBreakerConfig(BaseModel):
    """Configuration for the circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: int = 30  # in seconds


class AIModelRouterConfig(BaseModel):
    """Main configuration for the AIModelRouter."""

    endpoints: List[EndpointConfig]
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)


class AIModelRouter:
    """
    A robust AI model router that provides load balancing, retry logic,
    circuit breaking, and response caching for multiple AI providers.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the AIModelRouter.

        Args:
            config: A dictionary containing the configuration for the router.

        ---------------------------------------------------------------------
        exmple ->
                config = {
            'endpoints': [
                {
                    'name': 'stt-primary',
                    'url': 'https://api.provider1.com/v1/stt',
                    'type': 'http',
                    'weight': 70,
                    'timeout': 5.0
                },
                {
                    'name': 'stt-backup',
                    'url': 'wss://ws.provider2.com/stt',
                    'type': 'websocket', 
                    'weight': 30,
                    'timeout': 3.0
                }
            ],
            'retry_config': {
                'max_attempts': 3,
                'backoff_factor': 2.0
            },
            'circuit_breaker': {
                'failure_threshold': 5,
                'recovery_timeout': 30
            }
        }
        """
        self.config = AIModelRouterConfig(**config)
        self.endpoints = self.config.endpoints

        # Circuit breaker state
        self.circuit_states = {endpoint.name: "closed" for endpoint in self.endpoints}
        self.failure_counts = defaultdict(int)
        self.last_failure_time = defaultdict(float)

        # Caching
        self.cache = {}

        # For weighted load balancing
        self.total_weight = sum(endpoint.weight for endpoint in self.endpoints)

        # aiohttp session
        self.http_session = aiohttp.ClientSession()

        # Metrics
        self.metrics = {
            endpoint.name: {
                "request_count": 0,
                "success_count": 0,
                "error_count": 0,
                "total_latency": 0.0,
                "average_latency": 0.0,
            }
            for endpoint in self.endpoints
        }
        self.metrics_lock = asyncio.Lock()

    async def close(self):
        """Closes the aiohttp session."""
        await self.http_session.close()

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Returns the collected metrics."""
        return self.metrics

    async def _update_metrics(self, endpoint_name: str, latency: float, success: bool):
        """Updates metrics for a given endpoint."""
        async with self.metrics_lock:
            metrics = self.metrics[endpoint_name]
            metrics["request_count"] += 1
            metrics["total_latency"] += latency
            if success:
                metrics["success_count"] += 1
            else:
                metrics["error_count"] += 1

            if metrics["request_count"] > 0:
                metrics["average_latency"] = (
                    metrics["total_latency"] / metrics["request_count"]
                )

    def _get_cache_key(
        self, model_type: str, data: bytes, metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Generates a cache key for a request."""
        hasher = hashlib.sha256()
        hasher.update(model_type.encode())
        hasher.update(data)
        if metadata:
            hasher.update(str(sorted(metadata.items())).encode())
        return hasher.hexdigest()

    def _select_endpoint(self, exclude_names: Optional[List[str]] = None) -> Optional[EndpointConfig]:
        """
        Selects an endpoint based on weighted random selection,
        respecting the circuit breaker state.
        """
        if exclude_names is None:
            exclude_names = []

        active_endpoints = [
            endpoint
            for endpoint in self.endpoints
            if self.circuit_states[endpoint.name] != "open"
            and endpoint.name not in exclude_names
        ]
        if not active_endpoints:
            return None

        total_weight = sum(endpoint.weight for endpoint in active_endpoints)
        if total_weight == 0:
            return random.choice(active_endpoints) if active_endpoints else None

        selection = random.uniform(0, total_weight)
        current_weight = 0
        for endpoint in active_endpoints:
            current_weight += endpoint.weight
            if current_weight >= selection:
                return endpoint
        return None  # Should not be reached

    async def _make_http_request(
        self, endpoint: EndpointConfig, data: bytes, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Makes an HTTP request to the specified endpoint."""
        headers = {"Content-Type": "application/octet-stream"}
        if metadata:
            headers.update({f"X-Metadata-{k}": str(v) for k, v in metadata.items()})

        async with self.http_session.post(
            endpoint.url, data=data, headers=headers, timeout=endpoint.timeout
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def _make_websocket_request(
        self, endpoint: EndpointConfig, data: bytes, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Makes a WebSocket request to the specified endpoint."""
        try:
            async with self.http_session.ws_connect(
                endpoint.url, timeout=endpoint.timeout
            ) as ws:
                request_payload = {"data": data.decode("utf-8")}
                if metadata:
                    request_payload["metadata"] = metadata

                await ws.send_json(request_payload)

                response = await ws.receive_json()
                return response
        except aiohttp.WSServerHandshakeError as e:
            logger.error(f"WebSocket handshake error with {endpoint.name}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Error during WebSocket communication with {endpoint.name}: {e}"
            )
            raise

    async def inference(
        self, model_type: str, data: bytes, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Routes an inference request to an appropriate endpoint, with retries and circuit breaking.
        """
        cache_key = self._get_cache_key(model_type, data, metadata)

        # we should apply a FIFO to handle cache memory or move it into cache db such as Redis
        if cache_key in self.cache:
            logger.info(f"Returning cached response for {model_type}")
            return self.cache[cache_key]["response"]

        attempts = 0
        tried_endpoints: List[str] = []
        while attempts < self.config.retry_config.max_attempts:
            endpoint = None
            if attempts == 0 and metadata and "endpoint_pref" in metadata:
                endpoint = next(
                    (
                        ep
                        for ep in self.endpoints
                        if ep.name == metadata["endpoint_pref"]
                    ),
                    None,
                )

            if not endpoint:
                endpoint = self._select_endpoint(exclude_names=tried_endpoints)

            if not endpoint:
                logger.error("All endpoints are currently unavailable.")
                return {"error": "All endpoints are unavailable."}

            tried_endpoints.append(endpoint.name)
            start_time = time.time()
            try:
                if self.circuit_states[endpoint.name] == "half-open":
                    logger.info(
                        f"Endpoint {endpoint.name} is half-open, attempting a test request."
                    )

                if endpoint.type == "http":
                    response = await self._make_http_request(endpoint, data, metadata)
                elif endpoint.type == "websocket":
                    response = await self._make_websocket_request(
                        endpoint, data, metadata
                    )
                else:
                    raise ValueError(f"Unsupported endpoint type: {endpoint.type}")

                latency = time.time() - start_time
                await self._update_metrics(endpoint.name, latency, success=True)

                # Request was successful, reset circuit breaker
                self._reset_circuit(endpoint.name)

                # Cache the successful response
                self.cache[cache_key] = {"timestamp": time.time(), "response": response}

                return response

            except Exception as e:
                latency = time.time() - start_time
                await self._update_metrics(endpoint.name, latency, success=False)

                logger.warning(f"Request to {endpoint.name} failed: {e}")
                self._handle_failure(endpoint.name)
                attempts += 1
                if attempts < self.config.retry_config.max_attempts:
                    backoff_time = self.config.retry_config.backoff_factor**attempts
                    logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                    await asyncio.sleep(backoff_time)

        logger.error(
            f"Request failed after {self.config.retry_config.max_attempts} attempts."
        )
        return {"error": "Request failed after multiple retries."}

    def _handle_failure(self, endpoint_name: str):
        """Handles the logic for a failed request to an endpoint."""
        if self.circuit_states[endpoint_name] == "closed":
            self.failure_counts[endpoint_name] += 1
            if (
                self.failure_counts[endpoint_name]
                >= self.config.circuit_breaker.failure_threshold
            ):
                self._trip_circuit(endpoint_name)
        elif self.circuit_states[endpoint_name] == "half-open":
            self._trip_circuit(endpoint_name)

    def _trip_circuit(self, endpoint_name: str):
        """Trips the circuit breaker for an endpoint."""
        logger.warning(f"Circuit breaker for {endpoint_name} is now open.")
        self.circuit_states[endpoint_name] = "open"
        self.last_failure_time[endpoint_name] = time.time()

    def _reset_circuit(self, endpoint_name: str):
        """Resets the circuit breaker for an endpoint."""
        if self.circuit_states[endpoint_name] != "closed":
            logger.info(f"Circuit breaker for {endpoint_name} is now closed.")
            self.circuit_states[endpoint_name] = "closed"
            self.failure_counts[endpoint_name] = 0

    async def _check_endpoint_health(self, endpoint: EndpointConfig) -> bool:
        """Performs a lightweight health check on an endpoint."""
        try:
            if endpoint.type == "http":
                async with self.http_session.head(
                    endpoint.url, timeout=2.0, allow_redirects=True
                ) as response:
                    return True
            elif endpoint.type == "websocket":
                ws = await self.http_session.ws_connect(endpoint.url, timeout=2.0)
                await ws.close(code=1000)
                return True
        except Exception:
            return False
        return False

    async def health_check(self):
        """
        Checks the health of all endpoints and updates circuit breaker states.
        This can be run periodically in the background.
        """
        for endpoint in self.endpoints:
            # Fallback to time-based recovery
            if (
                self.circuit_states[endpoint.name] == "open"
                and (time.time() - self.last_failure_time[endpoint.name])
                > self.config.circuit_breaker.recovery_timeout
            ):
                logger.info(
                    f"Endpoint {endpoint.name} recovery timeout elapsed. Setting to half-open."
                )
                self.circuit_states[endpoint.name] = "half-open"

            # Active health check on half-open circuits to probe for recovery
            if self.circuit_states[endpoint.name] == "half-open":
                is_healthy = await self._check_endpoint_health(endpoint)
                if is_healthy:
                    self._reset_circuit(endpoint.name)
                else:
                    # It's still not healthy, trip it again
                    self._trip_circuit(endpoint.name)

        return {
            endpoint.name: {
                "state": self.circuit_states[endpoint.name],
                "failures": self.failure_counts[endpoint.name],
            }
            for endpoint in self.endpoints
        }
