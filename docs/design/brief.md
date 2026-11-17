# General Engineering — Design brief

Status: approved simple blog with note sidebar; approved migration of the existing profile UI under Profile. Migration implemented locally on 2026-11-17. Build and browser validation remain pending.

## Purpose

An author's open notebook of ideas, abstractions, and knowledge across science and engineering. Help readers discover connections, understand reasoning, and build on the work. Supporting future SI/RSI research is a long-term ambition.

## Readers

Curious readers, researchers, engineers, and open-source contributors. Explanations should offer a clear entry point and room for technical depth.

## Proposed structure

| Screen | Reader purpose |
| --- | --- |
| Home | Understand the project and discover subjects and featured work |
| Library / subject | Browse and search articles across disciplines |
| Ideas | Explore abstractions, hypotheses, and cross-disciplinary connections |
| Experiments | Inspect models, demonstrations, and reproducible results |
| Open Source | Read source investigations and contribution notes |
| About | Understand the author, philosophy, and approach |
| Article | Read focused prose, equations, code, evidence, and references |

Subjects initially include Mathematics, Physics, Biology, Computer Science, Engineering, and Intelligence. Subjects classify content; Ideas, Experiments, and Open Source describe ways to explore it. Final navigation labels remain open.

## Design goals

- Calm, readable, distinctive, and comfortable for long articles.
- Make the full breadth of subjects visible.
- Support desktop and mobile, keyboard navigation, and readable contrast.
- Keep philosophy concise; let the content demonstrate value.
- Label sample content in previews.

## Confirmed requirements

- Reader navigation: a persistent left sidebar lists all notes, grouped by subject, with title/subject/type search and active-note highlighting. On phones it becomes a collapsible All notes panel. Keep this navigation available in the article reader. The user subsequently approved proceeding with a simple version of this layout.
- Static hosting must support lazy components. Keep article prose and the lightweight note index available independently of heavy interactive renderers. Load component code only when needed; mount on explicit interaction or near the viewport as appropriate. Use lightweight placeholders, loading/error states, and cleanup when navigating away. Static hosting does not require eagerly loading every renderer.
- For a large library, load a compact metadata index rather than every article body. Consider virtualizing the note list only if measured size warrants it; preserve search and keyboard navigation.

## Implementation decision — accepted 2026-11-17

- Proceed with a simple editorial notebook and the approved note sidebar. Keep the existing portfolio screens and URLs: `/about` (CV, with `#CV` supported), `/certificate`, `/projects/mnist`, and `/blog/leetcode`. Link to them from the notebook; keep a return link to the notebook in the portfolio navigation. Do not rewrite CV content or claim it has been updated.
- `/` is the new notebook homepage; `/notes/:slug` is the reader. Index only the four entries that have actual solution content; do not turn illustrative preview articles or unfinished index entries into published notes. Show other disciplines as future breadth, without implying existing articles.
- The sidebar imports metadata only. Reader code is loaded through a dynamic import. CV and certificates use lazy component routes; the existing project and blog modules retain lazy routes. The handwritten digit recognition model remains inside the existing demo, rather than being imported by the notebook.
- Keep the existing CV, certificates, demo, and coding pages intact. No dependencies added, no deployment performed.
- Validation: HTML parsing and source checks verified note slugs against the actual content, the preserved PDF assets and legacy route definitions, lazy import boundaries, and mobile/sidebar selectors. `git diff --check` passed. Full Angular build and tests could not run: Node/npm and node_modules are absent, and Docker's daemon is stopped. No new browser viewport or physical-device validation is claimed for the implementation. Earlier preview checks do not validate the Angular application.

## Existing Profile migration — accepted and implemented 2026-11-17

The author clarified that the existing profile page must be reused, with the rest of the site serving as the blog. The new Portfolio overview and colorful redesign are superseded; do not implement those mockups. The migration changes routes and navigation, not the original profile page design or its content.

| Purpose | Primary route | Existing alias |
| --- | --- | --- |
| Blog home | `/` | — |
| Profile / existing CV and transcript viewer | `/profile` | `/about` |
| Existing achievements/certificate gallery | `/profile/achievements` | `/certificate` |
| Existing digit recognition demo | `/profile/projects/mnist` | `/projects/mnist` |
| Coding collection | `/blog/leetcode` | unchanged |

The profile opens directly into the original AboutMe component; no overview or replacement profile component is introduced. Original page navigation now targets the Profile routes and includes a Blog return link. The blog header has one Profile entry; the coding collection remains blog content. Heavy pages retain lazy imports.

Compatibility paths are aliases using the same lazy loaders, rather than redirects, so original URL fragments and query parameters remain on the matched route. `/profile#CV`, `/profile#AT`, `/about#CV`, and `/about#AT` continue to use the existing fragment subscriber. Static Pages hosting still relies on the documented client-route fallback for direct links.

Validation: hashes of the original AboutMe, certificate, and project component trees plus their PDF/certificate assets match a snapshot taken before this migration. Source inspection checks canonical/legacy loader reuse and navigation targets. `git diff --check` passes. Angular build and browser/physical-device checks remain unverified because Node/npm and frontend dependencies are absent and connected browser navigation has failed. This records source evidence, not successful runtime navigation. No deployment performed.

## Superseded Portfolio styling exploration — 2026-11-17

The author wants the original portfolio UI or a more colorful treatment like it, rather than applying the notebook's muted style to the portfolio. Preserve the existing CV, certificate/achievement, and demo component designs. Keep the simple notebook as a separate reading surface. Portfolio route grouping remains proposed; styling approval does not imply route migration approval.

`previews/portfolio-colorful.html` is the revised draft, using the original blue/teal gradient, pink accents, pale-blue navigation, and existing avatar. The Portfolio overview has colorful CV/Achievements/Projects cards; notebook and reader views retain the simple treatment. The overview opens first; Notebook navigation returns to the reading layout. Representations of legacy subpages remain labeled handoffs; they are not replacements for the original components.

Source inspection covers responsive rules at 620/900 px, touch control sizing, visible focus, reduced-motion styling, labels, and navigation destinations. Browser widths 390/768/1366 px and physical devices remain unverified because connected browser navigation has failed. No Angular code or deployment configuration is changed by this styling revision.

## Superseded Portfolio placement exploration — 2026-11-17

Keep the notebook as the homepage with a single Portfolio navigation link. Group older material under a Portfolio overview with CV, Achievements, and Projects subnavigation. Proposed routes are `/portfolio`, `/portfolio/cv`, `/portfolio/achievements`, and `/portfolio/projects/mnist`. Preserve existing `/about`, `/certificate`, and `/projects/mnist` links via redirects/aliases when this placement is implemented; preserve CV/transcript fragments as well. Keep the original records, PDFs, and demo content. Load the PDF viewer, demo, and model on demand. This is a proposed grouping, not an approved Angular route migration.

`previews/portfolio-layout.html` carries the accepted simple notebook style through the proposed overview and subpages. Four real note titles remain in the sidebar; reader prose is illustrative. Achievement labels refer to existing repository records and have not been independently revalidated. CV/demo pages are handoff representations rather than embedded working legacy components.

Source review with ui-review criteria: navigation destinations, content labels, mobile toggle, and responsive CSS inspected. Browser targets are 390, 768, and 1366 px; browser validation remains pending, with no new screenshots or physical-device results claimed.

## Static deployment setup

`.github/workflows/pages.yml` automatically builds frontend pushes and pull requests to `main`, without deploying or uploading a Pages artifact. At the author’s explicit request, deployment is manual only: **Actions → Build / manual deploy GitHub Pages → Run workflow**, selecting `main`. Both artifact upload and deployment require `workflow_dispatch` on `main` and a successful build. `docs/deployment/github-pages.md` explains local commands, Pages settings, lazy components, the SPA 404 fallback, and validation limits. This configuration is local and unexecuted; no commit, push, repository-setting change, or deployment occurred. Angular source and dependency inputs are unchanged in this draft task.

## Preview and decisions

- `previews/portfolio-colorful.html`: superseded exploration; author chose the existing profile UI instead.
- `previews/portfolio-layout.html`: superseded grouping exploration; no Portfolio overview will be implemented.

- `previews/notebook-next.html`: next draft of the accepted simple notebook, with a searchable sidebar of the four real note titles, responsive phone toggle, example article reader, lazy interactive example, and CV/demo/certificate handoff screens. Reader prose and pseudocode are explicitly illustrative. Portfolio handoff screens describe preserved routes; they do not pretend to run the original Angular demo. Proposed refinement, awaiting review; this turn does not alter Angular.
- Validation for the next draft: HTML parsing, real note-slug matching, lazy import reference, sidebar/active-note markup, and `git diff --check` passed. ui-review source inspection found that sample prose and preserved portfolio handoffs are labeled. Browser navigation again failed in connected tooling before rendering; 390, 768, and 1366 px remain target widths, not checked widths. No new keyboard, interaction, contrast, physical-device, or screenshot validation is claimed.

- `previews/editorial-notebook.html`: initial editorial direction, awaiting review. Interactive inline preview source; not the Angular application.
- `previews/homepage-directions.html`: three proposed responsive homepage directions: Editorial notebook, Knowledge atlas, and Field notes. Includes subject browsing and a sample article reader. No direction accepted; all article content is illustrative.
- Browser checks for the three-direction preview: all variants inspected for horizontal overflow at browser widths of 390, 768, and 1366 px (embedded content widths 343, 721, and 1319 px); none detected. Editorial homepage visually inspected at laptop and phone widths; subject filtering and the sample article reader exercised. These are browser viewport checks, not physical-phone validation or a full accessibility audit.
- Sidebar revision: `previews/homepage-directions.html` now includes the all-notes sidebar in each direction, collapsible subject groups, search with empty results, an active-note state, and a phone toggle. Article views include an illustrative canvas renderer loaded with dynamic import from `previews/lazy-example.mjs` only after Open interactive example is selected. The preview includes loading, retry, and navigation-away guards; it does not implement the production Angular renderer architecture.
- Review of this revision: source inspection completed with ui-review criteria. Browser navigation failed in the connected browser tooling, so no new viewport, interaction, keyboard, contrast, or physical-phone checks are claimed. Earlier viewport results above apply to the previous revision only. Source breakpoints are 620 and 1050 px; intended recheck widths are 390, 768, and 1366 px.
- Open the preview over local HTTP for the lazy module (for example, serve `docs/design/previews/` and open `homepage-directions.html`). Angular source and dependencies are unchanged by this preview revision.
- Remaining review: final typography/theme refinements and browser verification. The simple sidebar layout and phone toggle are approved for this first implementation.
- Next: validate the Angular implementation in laptop and phone browsers, then refine the simple layout based on review.

## Workflow

Brief → visual directions → user choice → complete screens → UI review → Angular implementation → validation.

Use the `ui-draft` skill to create previews and `ui-review` to assess them. These are repository-local skills, not installed external plugins. Product Design is optional when connected.
