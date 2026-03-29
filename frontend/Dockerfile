FROM node:22-bookworm-slim AS base

WORKDIR /app

ENV NEXT_TELEMETRY_DISABLED=1 \
    CHROMIUM_PATH=/usr/bin/chromium

RUN apt-get update \
    && apt-get install -y --no-install-recommends chromium ca-certificates \
    && rm -rf /var/lib/apt/lists/*

FROM base AS deps

COPY package.json ./
COPY package-lock.json ./

RUN npm ci --no-audit --no-fund

FROM base AS quality

COPY --from=deps /app/node_modules ./node_modules
COPY . .

RUN npm run typecheck
RUN npm run lint
RUN npm run build

FROM base AS runner

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0 \
    CHROMIUM_PATH=/usr/bin/chromium \
    NEXT_INTERNAL_BASE_URL=http://127.0.0.1:3000

COPY --from=quality /app/public ./public
COPY --from=quality /app/.next/standalone ./
COPY --from=quality /app/.next/static ./.next/static

EXPOSE 3000

CMD ["node", "server.js"]
