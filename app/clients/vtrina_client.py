from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger("comparador.vtrina")


class AppError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    def __str__(self) -> str:
        return f"AppError(status={self.status_code}, message={self.message})"


class VtrinaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_seconds: int = 10,
    ) -> None:
        env_url = base_url or os.getenv("VTRINA_BASE_URL")
        if not env_url:
            raise RuntimeError("VTRINA_BASE_URL environment variable is required")
        self.base_url = env_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def _request(
        self,
        path: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        message: str = "",
    ) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        session = await self._get_session()

        try:
            async with session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json_body,
                headers=headers,
            ) as resp:
                try:
                    response_data = await resp.json()
                except (ValueError, aiohttp.ContentTypeError):
                    response_data = await resp.text()

                if resp.status >= 400:
                    logger.warning(
                        "VtrinaClient: request failed with status=%s for %s",
                        resp.status,
                        message or path,
                    )
                    raise AppError(
                        f"Falha na requisição: {message or path}, Status: {resp.status}",
                        status_code=resp.status,
                    )

                return response_data

        except aiohttp.ClientError as exc:
            logger.exception(
                "VtrinaClient: request exception for %s",
                message or path,
            )
            raise AppError(
                f"Falha na requisição: {message or path}",
                status_code=None,
            ) from exc

    async def get_token(self, token: str) -> Any:
        if not token:
            return None
        return await self._request(
            "/api/token",
            method="GET",
            params={"token": token},
            message="Busca de token",
        )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
