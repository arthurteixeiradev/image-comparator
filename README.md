# Comparador API

Serviço simples em FastAPI para comparar duas imagens. Projeto organizado seguindo separação de responsabilidades (Controller, Service, Schemas, Middleware, Config).

Como usar (desenvolvimento):

1. Criar e ativar um virtualenv

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Rodar em desenvolvimento

```bash
uvicorn app.main:app --reload --port 8000
```

3. Endpoint

- POST /api/compare
- Request body (application/json):

```json
{
    "imagem1": "https://example.com/img1.jpg",
    "imagem2": "https://example.com/img2.jpg"
}
```

- Campos opcionais: `algorithm` (one of `phash`, `dhash`), `threshold` (float entre 0 e 1)

Example (curl):

```bash
curl -X POST http://localhost:8000/api/compare \
	-H "Content-Type: application/json" \
	-d '{"imagem1":"https://example.com/img1.jpg","imagem2":"https://example.com/img2.jpg"}'
```

4. Resposta

Se as imagens são iguais:

```json
{
    "isEqual": true,
    "message": "Os arrays de imagens são iguais."
}
```

Se as imagens são diferentes:

```json
{
    "isEqual": false,
    "message": "Os arrays de imagens são diferentes."
}
```

Em caso de erro no processamento:

```json
{
    "isEqual": false,
    "message": "Erro ao processar as imagens."
}
```

O que o algoritmo faz:

- Baixa as duas imagens a partir das URLs (usa `aiohttp`, com timeout e checagem de `content-length`). Imagens muito grandes são rejeitadas; há redimensionamento para `max_dimension` quando configurado.

- Calcula hashes perceptuais com `imagehash`: `phash`, `dhash` (cada um com `hash_size` configurável).

- Converte a diferença entre hashes em distância de Hamming e em uma pontuação de similaridade para determinar se as imagens são iguais.

- O serviço usa cache em duas camadas: L1 em memória e L2 opcional em Redis (se habilitado). São armazenados hashes por URL+algoritmo e resultados de comparação por par de URLs ordenadas.

- Se o Redis estiver indisponível, o serviço faz degrade gracioso para o cache em memória e registra o problema nos logs.

Esses passos replicam a lógica presente no `app/services/comparator_service.py` para garantir comportamento idêntico ao código original.