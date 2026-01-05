from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import Header, HTTPException

from app.services.token_service import get_token_service

logger = logging.getLogger("comparador.auth")


def require_token(
    x_access_token: Optional[str] = Header(None, alias="x-access-token"),
) -> Dict[str, Any]:
    if not x_access_token:
        logger.debug("Autenticação falhou: header x-access-token ausente")
        raise HTTPException(status_code=401, detail="Unauthorized")

    token_service = get_token_service()
    try:
        token_obj = token_service.get(x_access_token)
    except Exception as exc:
        logger.exception("Erro ao buscar token no TokenService: %s", exc)
        raise HTTPException(status_code=500, detail="internal_server_error")

    if not token_obj:
        logger.debug("Autenticação falhou: token inválido ou não encontrado")
        raise HTTPException(status_code=401, detail="Unauthorized")

    return token_obj
