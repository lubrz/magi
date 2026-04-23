# Build stage
FROM node:20-slim AS build
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Default nginx config is fine for this SPA
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
