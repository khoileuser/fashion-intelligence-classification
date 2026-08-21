# Application deployment

## Services

The root stack builds two targets from one `Dockerfile`:

- `server`: FastAPI on container port `8000`, with CPU PyTorch and read-only model/data mounts.
- `client`: the standalone Next.js production server on container port `3000`.

The browser only contacts the client. Requests under `/api/backend/*` are proxied from Next.js to `http://server:8000`, so `BACKEND_URL` must be a container-network address rather than a public browser URL.

## Required host files

Before starting the stack, make sure the seven outputs listed in the root README exist in the model directory. The gallery thumbnails also require the supplied dataset directory, including `train/images_train/`.

For a local checkout, the defaults are:

```dotenv
FASHION_MODELS_PATH=./models
FASHION_DATA_ROOT=./dataset
```

Both paths are bind-mounted read-only. A healthy API can start without trained files so that diagnostics remain available, but the interface reports **Models required** and inference returns HTTP 503 until all outputs are mounted.

## Portainer Git stack

1. Push the repository to the Git provider that Portainer can access.
2. In Portainer, create a stack from the Git repository and use `docker-compose.yml` as the Compose path.
3. Add `FASHION_MODELS_PATH` and `FASHION_DATA_ROOT` as stack environment variables. Use absolute Linux paths on the Docker host, for example `/srv/fashion/models` and `/srv/fashion/dataset`.
4. Optionally change `CLIENT_PORT`; it defaults to `3000`.
5. Deploy the stack and wait for both health checks to become healthy.

The host directories must be visible to the Docker daemon itself. A path on the computer running the browser is not sufficient when Portainer manages a remote Docker host.

## Operations

```bash
docker compose ps
docker compose logs -f server
docker compose logs -f client
docker compose build --pull
docker compose up -d
```

The stack uses `restart: unless-stopped`. Compose waits for the server health check before starting the client. To expose Swagger documentation remotely, set `API_BIND=0.0.0.0`; otherwise keep the safer loopback default and use the Next.js application as the public entry point.

The server image installs the CPU-only PyTorch wheel by default. Training is intentionally excluded from this runtime container. GPU inference would require a CUDA PyTorch index, NVIDIA container runtime configuration, and an explicit Compose device reservation.
