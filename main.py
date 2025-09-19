# 1st party
import json
from typing import Any, Dict

# internal libs
from src.model_router import AIModelRouter, EndpointConfig

# 3rd party
import asyncio
from aiohttp import web
from loguru import logger

if __name__ == "__main__":
    # Mock AI Provider Servers
    async def mock_provider_http(request: web.Request) -> web.Response:
        """Simulates a successful HTTP AI provider."""
        # Simulate some processing time
        await asyncio.sleep(0.1)
        body = await request.read()
        if b"trigger failure" in body:
            logger.info(
                "--- Mock HTTP provider received 'trigger failure' in body, simulating failure ---"
            )
            return web.Response(status=500, text="Internal Server Error")
        return web.json_response({"provider": "http", "status": "success"})

    async def mock_provider_http_failing(request: web.Request) -> web.Response:
        """Simulates a failing HTTP AI provider."""
        return web.Response(status=500, text="Internal Server Error")

    async def mock_provider_websocket(request: web.Request) -> web.WebSocketResponse:
        """Simulates a WebSocket AI provider."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    msg_json = json.loads(msg.data)
                    if msg_json.get("data") and "trigger failure" in msg_json.get(
                        "data", ""
                    ):
                        logger.info(
                            "--- Mock WebSocket provider received 'trigger failure', simulating failure ---"
                        )
                        # Closing with a non-1000 code to indicate an error
                        await ws.close(code=4000, message=b"Simulated failure")
                        return ws

                    # Simulate processing
                    await asyncio.sleep(0.2)
                    await ws.send_json({"provider": "websocket", "status": "success"})
                except Exception as e:
                    await ws.send_json({"error": str(e)})
                finally:
                    # Close the connection after one message for this example
                    if not ws.closed:
                        await ws.close()
            elif msg.type == web.WSMsgType.ERROR:
                logger.info(
                    f"WebSocket connection closed with exception {ws.exception()}"
                )

        return ws

    async def run_mock_servers():
        """Runs all the mock servers."""
        app_http = web.Application()
        app_http.router.add_post("/v1/stt", mock_provider_http)

        app_http_failing = web.Application()
        app_http_failing.router.add_post("/v1/stt_failing", mock_provider_http_failing)

        app_websocket = web.Application()
        app_websocket.router.add_get("/stt", mock_provider_websocket)

        runner_http = web.AppRunner(app_http)
        await runner_http.setup()
        site_http = web.TCPSite(runner_http, "localhost", 8080)
        await site_http.start()

        runner_http_failing = web.AppRunner(app_http_failing)
        await runner_http_failing.setup()
        site_http_failing = web.TCPSite(runner_http_failing, "localhost", 8081)
        await site_http_failing.start()

        runner_websocket = web.AppRunner(app_websocket)
        await runner_websocket.setup()
        site_websocket = web.TCPSite(runner_websocket, "localhost", 8082)
        await site_websocket.start()

        logger.info("Mock servers running on ports 8080, 8081, 8082")

        # Keep servers running
        try:
            while True:
                await asyncio.sleep(3600)  # Keep alive
        finally:
            await runner_http.cleanup()
            await runner_http_failing.cleanup()
            await runner_websocket.cleanup()

    async def main():
        """Main function to demonstrate AIModelRouter."""
        server_task = asyncio.create_task(run_mock_servers())
        await asyncio.sleep(2)  # Give servers time to start

        config: Dict[str, Any] = {
            "endpoints": [
                {
                    "name": "http-primary",
                    "url": "http://localhost:8080/v1/stt",
                    "type": "http",
                    "weight": 60,
                    "timeout": 5.0,
                },
                {
                    "name": "http-failing",
                    "url": "http://localhost:8081/v1/stt_failing",
                    "type": "http",
                    "weight": 20,
                    "timeout": 2.0,
                },
                {
                    "name": "websocket-primary",
                    "url": "ws://localhost:8082/stt",
                    "type": "websocket",
                    "weight": 20,
                    "timeout": 5.0,
                },
            ],
            "retry_config": {"max_attempts": 3, "backoff_factor": 1.5},
            "circuit_breaker": {"failure_threshold": 3, "recovery_timeout": 10},
        }

        router = AIModelRouter(config)
        logger.info("-" * 40)
        logger.info("--- Initial Health Check ---")
        logger.info("-" * 40)
        health = await router.health_check()
        logger.info(json.dumps(health, indent=2))

        # 1. Basic routing
        logger.info("-" * 40)
        logger.info("--- Testing Basic Routing ---")
        logger.info("-" * 40)
        result = await router.inference("stt", "audio_data".encode())
        logger.info(f"Basic Routing Result: {result}")
        logger.info(f"metrics: {router.get_metrics()}")

        # 2. Endpoint failure with retry
        # Primary fails -> retry -> fallback to backup
        logger.info("-" * 40)
        logger.info("--- Testing Endpoint Failure with Retry ---")
        logger.info("-" * 40)
        result = await router.inference(
            "stt",
            "trigger failure".encode(),
            metadata={"endpoint_pref": "http-primary"},
        )
        logger.info(f"Failure with Retry Result: {result}")

        # 3. Circuit breaker activation
        # After 3 failures, endpoint should be temporarily disabled
        logger.info("-" * 40)
        logger.info("--- Testing Failing Endpoint and Circuit Breaker ---")
        logger.info("-" * 40)
        failure_threshold = config["circuit_breaker"]["failure_threshold"]
        for i in range(failure_threshold):
            logger.info(f"Attempt {i + 1} to trigger circuit breaker...")
            result = await router.inference("stt", f"trigger failure {i}".encode())
            logger.info(f"Result of attempt {i + 1}: {result}")

        logger.info("-" * 40)
        logger.info("--- Health Check After Failures ---")
        logger.info("-" * 40)
        health = await router.health_check()
        logger.info(json.dumps(health, indent=2))
        if any(h["state"] == "open" for h in health.values()):
            logger.info("SUCCESS: At least one circuit breaker is open.")
        else:
            logger.warning("WARNING: No circuit breaker opened.")

        logger.info("-" * 40)
        logger.info("--- Waiting for Recovery ---")
        logger.info("-" * 40)
        recovery_timeout = config["circuit_breaker"]["recovery_timeout"]
        logger.info(
            f"Waiting for {recovery_timeout + 1} seconds for recovery timeout..."
        )
        await asyncio.sleep(recovery_timeout + 1)

        logger.info("-" * 40)
        logger.info(
            "--- Health Check After Recovery Period (should move to half-open) ---"
        )
        logger.info("-" * 40)
        health = await router.health_check()
        logger.info(json.dumps(health, indent=2))
        if any(h["state"] == "half-open" for h in health.values()):
            logger.info("SUCCESS: At least one circuit moved to half-open.")
        else:
            logger.warning("WARNING: No circuit moved to half-open.")

        logger.info("-" * 40)
        logger.info("--- Another Health Check (should close circuits if healthy) ---")
        logger.info("-" * 40)
        health = await router.health_check()
        logger.info(json.dumps(health, indent=2))
        if all(h["state"] == "closed" for h in health.values()):
            logger.info("SUCCESS: All circuits are closed after recovery.")
        else:
            logger.warning("WARNING: Not all circuits closed after recovery.")

        # 4. Concurrent requests
        logger.info("-" * 40)
        logger.info("--- Testing Concurrent Requests ---")
        logger.info("-" * 40)
        tasks = [router.inference("stt", f"audio_data_{i}".encode()) for i in range(10)]
        results = await asyncio.gather(*tasks)
        logger.info(f"Concurrent Results: {results}")
        logger.info(f"metrics: {router.get_metrics()}")

        # 5. WebSocket connection management
        # Maintain persistent connections, handle reconnectio
        logger.info("-" * 40)
        logger.info("--- Testing Successful WebSocket Request ---")
        logger.info("-" * 40)
        endpoint_websocket = next(
            ep for ep in router.endpoints if ep.name == "websocket-primary"
        )
        response = await router._make_websocket_request(
            endpoint_websocket, "audio_data".encode(), metadata=None
        )
        logger.info(f"Response: {response}")

        await router.close()
        server_task.cancel()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down.")
