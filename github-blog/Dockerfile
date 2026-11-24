# Stage 1: Build the Angular application
FROM oven/bun:1.4.5 AS angular
WORKDIR /app
COPY github-blog/package.json github-blog/bun.lock ./
RUN bun install --frozen-lockfile
COPY github-blog/ .
COPY docs/content/ /docs/content/
RUN bun run build

# Stage 2: Serve the application from Apache
FROM httpd:alpine3.15
WORKDIR /usr/local/apache2/htdocs/
COPY --from=angular /app/dist/github-blog .

# Enable .htaccess files and mod_rewrite
RUN sed -i 's#AllowOverride None#AllowOverride All#' /usr/local/apache2/conf/httpd.conf
RUN sed -i 's/#LoadModule rewrite_module modules\/mod_rewrite.so/LoadModule rewrite_module modules\/mod_rewrite.so/' /usr/local/apache2/conf/httpd.conf

# Add .htaccess file
COPY github-blog/.htaccess /usr/local/apache2/htdocs/