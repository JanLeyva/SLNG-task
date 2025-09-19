import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.model_router import AIModelRouter


import pytest_asyncio


@pytest_asyncio.fixture
async def router():
    config = {
        "endpoints": [
            {
                "name": "stt-primary",
                "url": "http://localhost:8080/stt",
                "type": "http",
                "weight": 70,
                "timeout": 5.0,
            },
            {
                "name": "stt-backup",
                "url": "ws://localhost:8081/stt",
                "type": "websocket",
                "weight": 30,
                "timeout": 3.0,  # deprecated float in aiohttp, refactor with `ClientWSTimeout` in the future
            },
        ],
        "retry_config": {"max_attempts": 3, "backoff_factor": 1.0},
        "circuit_breaker": {"failure_threshold": 2, "recovery_timeout": 10},
    }
    router_instance = AIModelRouter(config)
    yield router_instance
    await router_instance.close()


@pytest.mark.asyncio
async def test_basic_routing(router):
    with patch.object(
        router, "_make_http_request", new_callable=AsyncMock
    ) as mock_http:
        mock_http.return_value = {"result": "success"}
        result = await router.inference("stt", "audio_data".encode())
        assert result == {"result": "success"}
        mock_http.assert_called_once()


@pytest.mark.asyncio
async def test_endpoint_failure_with_retry(router):
    with patch.object(
        router, "_make_http_request", new_callable=AsyncMock
    ) as mock_http:
        mock_http.side_effect = [Exception("Connection error"), {"result": "success"}]

        router._select_endpoint = MagicMock(return_value=router.endpoints[0])

        result = await router.inference("stt", "audio_data".encode())

        assert result == {"result": "success"}
        assert mock_http.call_count == 2


@pytest.mark.asyncio
async def test_endpoint_failure_all_retries(router):
    with patch.object(
        router, "_make_http_request", new_callable=AsyncMock
    ) as mock_http:
        mock_http.side_effect = Exception("Connection error")

        router._select_endpoint = MagicMock(return_value=router.endpoints[0])

        result = await router.inference("stt", "audio_data".encode())

        assert result == {"error": "Request failed after multiple retries."}
        assert mock_http.call_count == router.config.retry_config.max_attempts


@pytest.mark.asyncio
async def test_circuit_breaker_activation(router):
    with patch.object(
        router, "_make_http_request", new_callable=AsyncMock
    ) as mock_http:
        mock_http.side_effect = Exception("Connection error")

        for _ in range(router.config.circuit_breaker.failure_threshold):
            await router.inference("stt", "audio_data_cb".encode())

        assert router.circuit_states["stt-primary"] == "open"

        result = await router.inference("stt", "audio_data_cb".encode())
        assert result == {"error": "All endpoints are unavailable."}


@pytest.mark.asyncio
async def test_circuit_breaker_recovery(router):
    with patch.object(
        router, "_make_http_request", new_callable=AsyncMock
    ) as mock_http, patch(
        "src.model_router.AIModelRouter._check_endpoint_health", new_callable=AsyncMock
    ) as mock_health:
        endpoint = router.endpoints[0]
        router._select_endpoint = MagicMock(return_value=endpoint)

        # 1. Trip the circuit
        mock_http.side_effect = Exception("Connection error")
        for _ in range(router.config.circuit_breaker.failure_threshold):
            await router.inference("stt", "audio_data_rec".encode())
        assert router.circuit_states[endpoint.name] == "open"

        # 2. Test that a failed health probe keeps the circuit open
        await asyncio.sleep(router.config.circuit_breaker.recovery_timeout)
        mock_health.return_value = False
        await router.health_check()
        assert router.circuit_states[endpoint.name] == "open"

        # 3. Test that a successful health probe closes the circuit
        await asyncio.sleep(router.config.circuit_breaker.recovery_timeout)
        mock_health.return_value = True
        await router.health_check()
        assert router.circuit_states[endpoint.name] == "closed"

        # 4. Test that inference now works
        mock_http.side_effect = None
        mock_http.return_value = {"result": "success"}
        result = await router.inference("stt", "audio_data_rec_2".encode())
        assert result == {"result": "success"}


@pytest.mark.asyncio
async def test_concurrent_requests(router):
    with patch.object(
        router, "_make_http_request", new_callable=AsyncMock
    ) as mock_http:
        mock_http.return_value = {"result": "success"}

        tasks = [router.inference("stt", f"audio_data_{i}".encode()) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        for result in results:
            assert result == {"result": "success"}

        assert mock_http.call_count <= 10


@pytest.mark.asyncio
async def test_websocket_connection(router):
    with patch.object(
        router, "_make_websocket_request", new_callable=AsyncMock
    ) as mock_ws:
        mock_ws.return_value = {"result": "success_ws"}

        result = await router.inference(
            "stt", "audio_data_ws".encode(), metadata={"endpoint_pref": "stt-backup"}
        )

        assert result == {"result": "success_ws"}
        mock_ws.assert_called_once()
