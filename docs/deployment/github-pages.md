# Local build and GitHub Pages

The Angular build emits static HTML, JavaScript, CSS, PDFs, images, and model assets into `github-blog/dist/github-blog/`. GitHub Pages serves those files; no application server or Docker container is required. Angular routes and lazy components still run in the browser. This is a static-hosted client application, not pre-rendered HTML for every article.

## Local development

Use Node 20 for this existing Angular 17 checkout, following its [documented compatibility](https://angular.dev/reference/versions). Angular 17 is now unsupported; upgrading the framework and Node together is separate work. Node/npm are currently missing on this machine, so these commands have not been run here.

From the repository root:

```bash
cd github-blog
npm ci
npm start -- --host 127.0.0.1
```

Open `http://127.0.0.1:4200`. This runs Angular's development server, which supports deep-route refreshes.

## Production build on your machine

```bash
cd github-blog
npm ci
npm run build -- --configuration production --base-href /
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

Workflow YAML and source paths can be inspected locally. A successful Angular build, dependency installation, GitHub Actions run, and live Pages deployment have not been verified: Node/npm and node_modules are absent locally. Treat the first CI run as required validation, not a deployment already completed.

## Profile migration smoke check

After starting the app, open `/profile`: it must show the original profile and PDF viewer. Use its Achievements and Project navigation, then return to Blog. Check `/profile/achievements`, `/profile/projects/mnist`, and the old `/about`, `/certificate`, `/projects/mnist` aliases. Check both `#CV` and `#AT` on `/profile` and `/about` to select the original documents. Finally open a blog note and return to Profile; verify the profile navigation does not duplicate the blog header. These are required runtime checks, not checks already completed.
