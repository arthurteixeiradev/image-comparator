from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests

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
        session: Optional[requests.Session] = None,
    ) -> None:
        env_url = base_url or os.getenv("VTRINA_BASE_URL")
        if not env_url:
            raise RuntimeError("VTRINA_BASE_URL environment variable is required")
        self.base_url = env_url
        self.timeout = timeout_seconds
        self._session = session or requests.Session()

    def _request(
        self,
        path: str,
        method: str = "get",
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        message: str = "",
    ) -> Any:
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")

        try:
            resp = self._session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json_body,
                headers=headers,
                timeout=self.timeout,
            )

            try:
                response_data = resp.json()
            except ValueError:
                response_data = resp.text

            if resp.status_code >= 400:
                logger.warning(
                    "VtrinaClient: request failed with status=%s for %s",
                    resp.status_code,
                    message or path,
                )
                raise AppError(
                    f"Falha na requisição: {message or path}, Status: {resp.status_code}",
                    status_code=resp.status_code,
                )

            return response_data

        except requests.RequestException as exc:
            logger.exception(
                "VtrinaClient: request exception for %s",
                message or path,
            )
            raise AppError(
                f"Falha na requisição: {message or path}",
                status_code=None,
            ) from exc

    def get_token(self, token: str) -> Any:
        if not token:
            return None
        return self._request(
            "/api/token",
            method="get",
            params={"token": token},
            message="Busca de token",
        )
