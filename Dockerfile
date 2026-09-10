# syntax=docker/dockerfile:1.7

FROM debian:bookworm-slim AS model-assets
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates git git-lfs \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /assets
COPY models/ ./models/
# Portainer checkouts may contain LFS pointers. Smudge downloads the exact
# referenced object anonymously from the public repository.
RUN git init -q \
    && git remote add origin https://github.com/khoileuser/fashion-intelligence-classification.git \
    && for file in models/*; do \
         if git lfs pointer --check --file="$file"; then \
           GIT_TERMINAL_PROMPT=0 git lfs smudge "$file" < "$file" > /tmp/model-asset \
           && mv /tmp/model-asset "$file" || exit 1; \
         fi; \
       done

FROM python:3.12-slim AS server

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/workspace/models \
    FASHION_DATA_ROOT=/workspace/dataset

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY app/server/requirements.txt /tmp/server-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/server-requirements.txt

RUN useradd --create-home --uid 10001 fashion \
    && mkdir -p /workspace/models /workspace/dataset \
    && chown -R fashion:fashion /workspace

COPY --chown=fashion:fashion app/server app/server
COPY --from=model-assets --chown=fashion:fashion /assets/models /workspace/models

USER fashion
EXPOSE 8000
CMD ["uvicorn", "app.server.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM oven/bun:1.4-alpine AS client-dependencies
WORKDIR /workspace/app/client
COPY app/client/package.json app/client/bun.lock ./
RUN bun install --frozen-lockfile

FROM node:26-alpine AS client-builder
WORKDIR /workspace/app/client
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=client-dependencies /workspace/app/client/node_modules ./node_modules
COPY app/client ./
RUN npm run build

FROM node:26-alpine AS client
WORKDIR /workspace
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000 \
    BACKEND_URL=http://server:8000

RUN addgroup --system --gid 10001 nodejs \
    && adduser --system --uid 10001 nextjs

COPY --from=client-builder --chown=nextjs:nodejs /workspace/app/client/.next/standalone ./
COPY --from=client-builder --chown=nextjs:nodejs /workspace/app/client/.next/static ./.next/static
COPY --from=client-builder --chown=nextjs:nodejs /workspace/app/client/public ./public

USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
