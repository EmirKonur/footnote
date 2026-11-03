# Build context: the repository root.
FROM node:26-slim AS build
# Node 25+ no longer bundles corepack, so pnpm is installed explicitly at the version
# that wrote the lockfile.
RUN npm install --global pnpm@9.15.0
WORKDIR /app
COPY web/package.json web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY web/ ./
RUN pnpm build

FROM node:26-slim
WORKDIR /app
ENV NODE_ENV=production HOSTNAME=0.0.0.0 PORT=4411
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
USER node
EXPOSE 4411
CMD ["node", "server.js"]
