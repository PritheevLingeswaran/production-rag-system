FROM node:20-alpine AS deps

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-alpine AS builder

WORKDIR /app
COPY --from=deps /app/node_modules /app/node_modules
COPY . .
RUN npm run build

FROM node:20-alpine AS runtime

ENV NODE_ENV=production
WORKDIR /app

USER node

COPY --from=builder --chown=node:node /app/.next /app/.next
COPY --from=builder --chown=node:node /app/public /app/public
COPY --from=builder --chown=node:node /app/package.json /app/package.json
COPY --from=builder --chown=node:node /app/package-lock.json /app/package-lock.json
COPY --from=builder --chown=node:node /app/next.config.mjs /app/next.config.mjs
COPY --from=deps --chown=node:node /app/node_modules /app/node_modules

EXPOSE 3000

CMD ["npm", "run", "start", "--", "-H", "0.0.0.0", "-p", "3000"]
