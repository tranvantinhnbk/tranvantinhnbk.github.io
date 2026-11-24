# Local build and GitHub Pages

The Angular build emits static HTML, JavaScript, CSS, PDFs, images, and model assets into `github-blog/dist/github-blog/`. GitHub Pages serves those files; no application server or Docker container is required. Angular routes and lazy components still run in the browser. This is a static-hosted client application, not pre-rendered HTML for every article.

## Local development

Install [Bun](https://bun.com/docs/installation) 1.4.5 (the version pinned in CI and `package.json`). Bun installs dependencies and runs the content pipeline and Angular CLI. `bun.lock` is the committed dependency lockfile; use frozen installation for reproducible builds.

From the repository root:

```bash
cd github-blog
bun install --frozen-lockfile
bun run start --host 127.0.0.1
```

Open `http://127.0.0.1:4200`. This runs Angular's development server, which supports deep-route refreshes.

## Production build on your machine

```bash
cd github-blog
bun install --frozen-lockfile
bun run build --configuration production --base-href /
cp dist/github-blog/index.html dist/github-blog/404.html
touch dist/github-blog/.nojekyll
python3 -m http.server 8080 --bind 127.0.0.1 --directory dist/github-blog
```

Open `http://127.0.0.1:8080`, then navigate from the homepage. Serve only the built output, never the repository root. Python's basic server does not use `404.html` as an SPA fallback; refreshing a nested route locally will show a server 404. Use the Angular development server for local route testing.

## Build automatically, deploy manually

The tracked workflow is `.github/workflows/pages.yml`. No Bitbucket pipeline is present or needed for this GitHub-hosted repository. Manual workflow runs use GitHub’s [Run workflow button](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).

Deployment is explicitly limited to `workflow_dispatch` on `main`:

1. A frontend pull request to `main` runs installation and the production build; it never deploys.
2. A push affecting `github-blog/` or the workflow on `main` installs and builds only. It never uploads a Pages artifact or deploys.
3. To deploy, use **Actions → Build / manual deploy GitHub Pages → Run workflow → Branch: main → Run workflow**. That manual run builds, uploads only the generated site, and deploys after success. Manual runs on other branches build only.
4. A failed build blocks deployment. The workflow uses the built-in token and Pages OIDC; no personal access token or Docker registry is needed.

One-time repository setting: **Settings → Pages → Build and deployment → Source → GitHub Actions**. Check the `github-pages` environment allows deployment from `main`. See [GitHub's publishing source instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) and [custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

The workflow file must first be committed and present on the default branch for GitHub to expose **Run workflow**. Pushing it does not deploy. Inspect automatic builds under **Actions → Build / manual deploy GitHub Pages**; the deploy job must be skipped on pushes and pull requests. When you manually run it from `main`, wait for both build and deploy to succeed, then use its deployment URL. The intended user-site URL is `https://tranvantinhnbk.github.io/`; live Pages settings and deployment have not been checked or changed in this task.

This workflow assumes this user-site repository is served at `/`, consistent with its name and existing absolute asset URLs. A project site under `/repository/` requires a different base href and an audit of absolute asset links.

## Direct links and drafts

The workflow copies `index.html` to `404.html` so Pages can bootstrap Angular for direct links such as `/about` and `/notes/two-sum`. Pages may still return HTTP 404 for these deep links even when Angular renders the page; this is a fallback, not true per-route HTML generation. If clean HTTP 200 responses and article SEO become requirements, add prerendering as a separate change. GitHub documents [custom 404 pages](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-custom-404-page-for-your-github-pages-site).

Markdown under `docs/content/drafts/` is not copied into the Angular output. Drafting and source review do not publish an article; the author's approval is required before blog generation.

## Validation status

The implementation now passes the production build, five content-rendering tests, five Angular route regression tests, and five real-reader UI tests. The UI tests use exact 390, 768, and 1366 px browser frames; they are not physical-device or screenshot checks. The production build still reports bundle/style and legacy dependency warnings. Full results are in `docs/design/reviews/math-reader-implementation.md`.

Useful checks from `github-blog/` with Bun on PATH:

```bash
bun run test:content
bun run test:routes
bun run test:ui
bun run build
```

The CI workflow audits the locked dependencies with `bun audit` and runs these focused checks before building/uploading the site. Pages uploads and deployment still require a manual run from `main`; normal pushes and pull requests never deploy. No GitHub workflow run or live Pages deployment has been executed or checked by the agent. On a nearly full disk, `CI=true` disables Angular's default local persistent build cache for that process.

## Profile migration smoke check

After starting the app, open `/profile`: it must show the original profile and PDF viewer. Use its Achievements and Project navigation, then return to Blog. Check `/profile/achievements`, `/profile/projects/mnist`, and the old `/about`, `/certificate`, `/projects/mnist` aliases. Check both `#CV` and `#AT` on `/profile` and `/about` to select the original documents. Finally open a blog note and return to Profile; verify the profile navigation does not duplicate the blog header. These are required runtime checks, not checks already completed.

## Bun migration — 2026-11-24

The existing npm lockfile was migrated using Bun's [lockfile migration](https://bun.com/docs/pm/lockfile), preserving resolved dependencies. Serve/build scripts explicitly invoke the Angular CLI with Bun. Browser tests retain Node 24 for Karma: a Bun runtime trial failed in Karma dependency injection and, with explicit annotations, its HTTP server stalled. Run those tests through `bun run test:routes` / `bun run test:ui` with Node 24 on PATH. CI installs Node only for that compatibility requirement. Use `bun run build`, rather than `bun build` (which invokes Bun's bundler). Docker builds now use the repository root as context: `docker build -f github-blog/Dockerfile -t angular .`; this includes the approved Markdown outside the frontend directory. Manual deployment conditions are unchanged.

Migration validation: frozen Bun installation, five content tests, five route regression tests, seven reader UI tests, the production build, and HTTP smoke checks for the home page, Part 2 route, and math asset pass. Existing build warnings remain. Docker build execution was unavailable because the Docker daemon is stopped; the Dockerfile changes have source inspection only. Bun was downloaded into temporary storage for these checks, not installed globally on the author's machine. Install Bun using the link above before running the new local commands.
