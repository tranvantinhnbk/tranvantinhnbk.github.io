# Dependency security remediation

Date: 2026-11-24. Scope: the Angular/Bun dependency tree, not a full application penetration test.

## Evidence and limits

The initial `bun audit --json` reports 251 advisory records across 64 affected packages: 6 critical, 120 high, 93 moderate, and 32 low. The production-reachable dependency subset reports 28 records across six packages: Angular common/compiler/core/router, form-data, and pdfjs-dist. A dependency being reachable from `dependencies` does not establish that its vulnerable code executes in the browser or is exploitable in this application.

Baseline evidence is in [dependency-audit-before.json](dependency-audit-before.json) and [dependency-audit-production-before.json](dependency-audit-production-before.json). These reflect the registry's advisory data at audit time; advisory counts can change independently of repository edits.

The repository's private Dependabot alert list was not retrieved. Public browsing could not fetch it, and the desktop browser attachment failed before navigation with a network-policy compatibility error. Thus local audit results must not be represented as GitHub alert closure. No alerts were dismissed or marked resolved manually.

## Changes and audit result

Angular 17 was upgraded to 21.2.26, with CLI/build tooling 21.2.31, Material/CDK 21.2.14, and TypeScript 5.9.3. Updated the PDF viewer to 10.4.0 (PDF.js 4.8.69), TensorFlow to 4.22.0, active syntax highlighting to 21.0.0, KaTeX to 0.19.1, and Markdown-it to 15.0.3. Removed unused `ngx-highlightjs` and `ngx-opencv` packages and their dead imports; the active highlighting directive and TensorFlow demo remain.

A complete dependency refresh removes the stale compatible transitive versions. Three remaining moderate advisories required explicit overrides: glob 14.0.6, minimatch 11.0.9, and UUID 11.1.1. Glob/minimatch rules target the modern declared dependency ranges, leaving the safe legacy glob 7/minimatch 3 API copies used by Karma intact. UUID 11 retains the CommonJS export used by SockJS. No advisory is ignored. Bun's version-scoped rules generate lockfile format 3; use the pinned Bun 1.4.5 rather than an older runtime.

Fresh independent full and production audits both return `{}` with exit 0: **251 advisory records across 64 packages to zero**, and production **28 records across six packages to zero**. See [after-audit evidence](dependency-audit-after.json), which includes the audited lockfile SHA-256. This means no known advisories were returned by that registry at that time, not a guarantee against all vulnerabilities.

## Compatibility and validation

The migration keeps the existing NgModule root explicit with `standalone: false`, uses TypeScript's bundler module resolution, and imports the maintained standalone highlighting directive. Angular 21 defaults to zoneless change detection; a real browser check exposed a lesson stuck on Loading after its fetch. The supported `bootstrapModule` `applicationProviders` option now explicitly supplies `provideZoneChangeDetection()` to preserve existing asynchronous updates. See [Angular's bootstrap guidance](https://v21.angular.dev/api/core/provideZoneChangeDetection). Route definitions are unchanged. Hashes of all 30 inspected original profile/achievement/demo templates and assets match their before-upgrade snapshot.

Frozen installation, five content tests, five route regressions, seven responsive-reader tests, and production build pass. The route tests stub legacy views, so they alone do not establish working PDF rendering or demo behavior. After the bootstrap fix, additional real production-browser checks rendered the full Part 2 article with 43 MathML elements, loaded the original profile and rendered its PDF page, loaded the coding solution in its shadow DOM, and loaded the MNIST canvas, with no observed uncaught JavaScript exceptions. Reader and original profile desktop screenshots were visually inspected. All four checks pass against the final built output; the seven reader tests also pass again after the bootstrap fix. These smoke checks do not validate model prediction accuracy or every legacy interaction.

The production build retains SCSS import deprecation, TensorFlow CommonJS, and bundle/style budget warnings. Initial bundle size increased from approximately 677 KB to 875 KB with the framework upgrade; budget limits were not raised to conceal this. Docker image/container CVEs are outside this lockfile audit, and Docker was not built or scanned.

## Maintenance

The existing Pages workflow now runs `bun audit` immediately after frozen installation, including development dependencies. A nonzero audit result stops the job. Tests use Node 24 for the legacy Karma browser server; Bun remains the install/serve/build tool. Pages upload and deployment still require a manual workflow dispatch on `main`.

Reproduce from `github-blog/`:

```sh
bun install --frozen-lockfile
bun audit
bun audit --prod
bun run test:content
bun run test:routes
bun run test:ui
bun run build --configuration production --base-href /
```

Browser tests require a compatible Node runtime and Chrome. Updating packages is separate from publishing the site. No commit, push, or deployment is part of this remediation.

## References

- [Bun audit behavior and flags](https://bun.com/docs/pm/cli/audit): reads the Bun lockfile and queries the registry advisory endpoint; the read-only audit does not apply fixes.
- [Angular version compatibility](https://angular.dev/reference/versions) and [support windows](https://angular.dev/reference/releases): framework upgrades must align compiler, TypeScript, and runtime requirements.
- Examples independently checked against GitHub's advisory pages: [piscina CVE-2026-102992](https://github.com/advisories/GHSA-67c8-pqhq-4rmx), [tar CVE-2026-59873](https://github.com/advisories/GHSA-23hp-3jrh-7fpw), and [websocket-driver CVE-2026-54466](https://github.com/advisories/GHSA-xv26-6w52-cph6). These are examples from the baseline, not the complete alert list.
