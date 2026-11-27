# General Engineering — Design brief

Status: approved blog/Profile separation and Math reader implemented locally. Production build and focused content, route, and browser-frame UI checks pass as of 2026-11-21; manual visual/accessibility checks and deployment remain separate.

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

## Math / Abstract Algebra section — requested 2026-11-18

Organize this learning conversation under subject **Math (Mathematics) → Abstract Algebra**, with **Part 1: Sets, operations, and closure** as the first learning session. Subject, section, and content type remain distinct; this is a learning-session draft, not a published article. Proposed URLs: `/math`, `/math/abstract-algebra`, and `/math/abstract-algebra/sets-operations-closure`. These routes are a UI/content proposal and are not added to Angular in this task.

`previews/math-abstract-algebra.html` reuses the accepted simple blog layout and adds nested sidebar navigation, title/subject search, a Math landing page, an Abstract Algebra section page, breadcrumbs, a lesson reader, a phone navigation toggle, and a practice hint. The four existing coding note titles remain in the preview index; their bodies are not duplicated. The broader science/engineering subjects remain visible on the preview homepage. The existing Profile UI is untouched.

Learning notes are saved in `docs/content/drafts/abstract-algebra/session-01.md` for author discussion and review. The preview lesson is based on this session; later topics are explicitly planned and not linked as completed articles. Blog generation still requires author approval of the Markdown.

Validation: HTML parsing, internal destination/source checks, draft labels, and `git diff --check` checked. Responsive rules use 620/900 px breakpoints. Connected browser navigation failed before rendering with a tooling error. Browser targets 390, 768, and 1366 px remain untested; no screenshot, viewport, interaction, or physical-device results are claimed. Preview and learning notes only; Angular and dependencies unchanged.

## Mathematical typography and explanation — requested 2026-11-19

Prioritize proper typesetting alongside clear explanations. Author Markdown uses `$...$` for inline math and `$$...$$` for display math, including `\mathbb`, fractions, set notation, quantifiers, and aligned proofs. Introduce notation in words and explain why each proof step follows. Use distinct definition, proposition, proof, counterexample, and exercise treatments without crowding the reading column.

`previews/math-reader.html` refines Part 1 using the accepted simple blog shell. It opens directly into the lesson, retains the Math/Abstract Algebra navigation, and offers an author-preview-only **Show LaTeX source** toggle. MathML is handwritten for this finite prototype, with corresponding LaTeX stored in semantic annotations; the preview is not an automatic Markdown/LaTeX compiler. It needs no remote fonts, math scripts, or dependency changes. Native MathML supports the rendered fractions and aligned proof; viewport and assistive-technology behavior still require browser testing. See [MathML semantics](https://developer.mozilla.org/en-US/docs/Web/MathML/Reference/Element/semantics).

The source lesson remains `docs/content/drafts/abstract-algebra/session-01.md`, now with LaTeX delimiters and the matching explanation/proof structure. Content remains draft for the author's review. Production renderer selection is still proposed: build-time KaTeX or another suitable renderer can later convert reviewed LaTeX to static HTML/MathML; this draft does not install or implement it. See [KaTeX delimiter documentation](https://katex.org/docs/autorender.html).

Source checks validate every MathML expression as XML, the annotation/source toggle wiring, lesson navigation, and draft labels. Connected browser navigation failed again before rendering due to a tooling error. Browser targets remain 390, 768, and 1366 px; viewport, physical-device, screen-reader, and source-toggle interaction checks are pending. No screenshots or browser validation are claimed. No Angular, package, or deployment changes in this task.

## Inline math reader review — 2026-11-20

`previews/math-reader-inline.html` presents the current accepted layout for direct review in conversation; `previews/math-reader-review.html` is its generated standalone inspection wrapper. It preserves the section/lesson reader, typeset expressions, author-only source toggle, and phone navigation. Optional host design controls adjust reading text size and column width. Navigation and saved review state stay local; no Angular, dependency, content-publication, or deployment changes.

Chrome inspection finally succeeded through native browser controls after the browser connector continued to be unavailable. The desktop default view and a 390 × 844 responsive viewport were visually inspected. The generated wrapper has horizontal inset spacing, so its embedded product is narrower than the outer browser viewport. The 390 px phone view wraps its header, breadcrumbs, and lesson title, and collapses the sidebar under All notes. Native browser accessibility observations confirmed the source checkbox switches between MathML and LaTeX, and the phone navigation expands the note tree. These are browser checks, not physical-device or screen-reader validation.

Inspection found and fixed (1) source-mode styles failing to match the inline scope root, and (2) the intended blackboard-bold integer symbol rendering as an italic Z; the preview now uses the literal ℤ token while retaining `\mathbb{Z}` in its LaTeX annotation. The standalone math-reader prototype receives the same symbol correction.

Remaining checks: 768 and 1366 px, full-article clipping/overflow measurement, search/empty results, exercise hint, complete keyboard navigation, appearance variants, and assistive technology. Attempts to continue wider inspection were interrupted when the user switched Chrome tabs; no further responsive results are claimed. The dedicated preview tab may retain its temporary 390 × 844 device-toolbar state because browser control was interrupted before reset. The wrapper console only showed blocked Lucide source-map requests from its helper runtime, with no observed article-script error; no security policy was weakened to load those maps.

Current review status: author review pending. Mathematical content remains in Markdown awaiting author approval before blog generation.

## Math reader implementation — accepted 2026-11-20

The author accepted the displayed math reader and explicitly requested `ui-implement` plus implementation, while requiring existing blog paths to remain unchanged. This covers conversion of the displayed Part 1 lesson; it does not approve future lessons or deployment. The new repository skill is `.agents/skills/ui-implement/SKILL.md`.

Implemented `/math`, `/math/abstract-algebra`, and `/math/abstract-algebra/sets-operations-closure` in the existing notebook shell. Root `/`, `/notes/:slug`, `/blog/leetcode`, `/blog/leetcode/solution/:name`, Profile routes, and original profile/demo aliases remain valid. Original blog paths retain their query strings and fragments. No replacement Portfolio page was added.

Approved Markdown is explicitly selected by `github-blog/content/articles.json`. A prebuild/prestart step renders it with Markdown HTML disabled and KaTeX trust disabled, generates static article HTML/MathML and a small metadata index, and leaves unapproved drafts out. Article bodies load on demand; KaTeX's JavaScript compiler runs at build time. Generated assets/catalog and dependency/build directories are ignored by Git. The public reader omits preview-only author controls; the hint remains interactive.

Validation and review: [implementation review](reviews/math-reader-implementation.md). Five content tests, five route regression tests, and five reader UI tests pass. The UI tests use exact 390, 768, and 1366 px browser-frame viewports and verify the real generated math plus navigation, search, retry, stale responses, and overflow. Production build passes with bundle/style/CommonJS warnings; physical-device and full manual accessibility/screenshot review remain unverified. The original 34 blog/profile/demo source files are unchanged. Skill validation and whitespace checks pass. No commit, push, or deployment performed by the agent.

## Part 1 implementation recheck — 2026-11-21

The author reaffirmed approval to implement Part 1. The approved implementation is present; fresh production build, content (5), route (5), and real-reader UI (5) checks pass. Existing root/blog/Profile URLs remain unchanged. The local Angular app and generated lesson asset return HTTP 200 for browser-style HTML requests. The actual lesson was exposed in Chrome's native accessibility tree; desktop visual capture and magnified device-toolbar capture were inconclusive, so no new screenshot/physical-device/accessibility audit is claimed.

Local review URL while the development server runs: `http://127.0.0.1:4200/math/abstract-algebra/sets-operations-closure`. No duplicate implementation, new article, commit, push, or deployment was needed or performed. See [implementation review](reviews/math-reader-implementation.md) for precise evidence and limits.

## Session 2 scope and Computing removal — 2026-11-22

Session 2 introduces **associativity, identity, and inverses** with examples, notation explanations, and exercises. Groups are a brief connection to a later lesson, not the main topic. The new Markdown draft is `docs/content/drafts/abstract-algebra/session-02.md`, awaiting author review before registration in the approved article manifest. Part 1 and its URL stay unchanged.

At the author's explicit request, Computing is removed from the visible blog catalog: subject list, search, sidebar, homepage note cards, and the original coding-index shortcut. The archive metadata remains available only for resolving existing `/notes/:slug` links. Original `/blog/leetcode` and `/blog/leetcode/solution/:name` routes and original coding content remain intact to honor the existing URL contract. Science/engineering subjects and the original Profile stay available. Historical preview files are retained as earlier drafts; they are not the current public catalog.

Validation for this change: Session 2 renders with the existing Markdown/KaTeX pipeline without being added to the approved manifest; six real-reader UI tests pass, including Computing absence, saved-note lookup, search, and exact-width frame checks. Production build passes with the existing budget/CommonJS warnings. Original blog routing/source files were not edited by this catalog change. No commit, push, or deployment performed.

## Part 2 implementation — accepted 2026-11-22

The author clarified the request by invoking `ui-implement` for the next session. This approves conversion of Part 2, **Associativity, identity, and inverses**, using the accepted reader. The earlier Session 2 draft status above is superseded. Part 3 remains an unregistered Markdown draft for later review.

Implemented `/math/abstract-algebra/associativity-identity-inverses` from `docs/content/drafts/abstract-algebra/session-02.md`. The approved manifest now contains Parts 1 and 2. Lesson numbers come from metadata in the sidebar, breadcrumbs, section index, and reader; previous/next links connect the two lessons. Both article bodies remain lazy-loaded static HTML/MathML generated from Markdown and LaTeX. Computing remains absent from the visible catalog, with saved coding URLs preserved. No route definitions, legacy profile/demo source, or deployment settings were changed for Part 2.

Validation: seven reader UI tests pass, including Part 2 navigation/math and exact 390/768/1366 px browser-frame overflow/menu checks; five content tests and five route regression tests pass; production build passes with the known budget/CommonJS warnings. The local lesson route and generated article return HTTP 200. These checks do not constitute a new manual screenshot, physical-device, or full accessibility audit. See [implementation review](reviews/math-reader-implementation.md). No commit, push, or deployment performed.

Local review: `http://127.0.0.1:4200/math/abstract-algebra/associativity-identity-inverses` while the development server runs.

## Preview and decisions

- `previews/math-reader-inline.html`: current in-conversation math reader; inspection wrapper is `previews/math-reader-review.html`.

- `previews/math-reader.html`: current math-reader refinement with typeset notation and explanations, awaiting review.

- `previews/math-abstract-algebra.html`: current Math → Abstract Algebra section/lesson draft, awaiting author review.

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

## Part 3 lesson preview — proposed 2026-11-24

The author requested the next learning section with `ui-draft`, following the even-integer exercise. Reused the accepted notebook/math reader rather than reopening the visual direction. Draft content remains `docs/content/drafts/abstract-algebra/session-03.md`: **Groups — putting the properties together**. It combines closure, associativity, identity, and inverses, distinguishes commutativity as optional, compares even and nonnegative integers, introduces square rotations, proves left cancellation, and leaves nonzero-rational multiplication as practice. Added a learning goal and inverse-notation explanation to the existing draft. Content is awaiting author review; the approved manifest still contains only Parts 1 and 2.

Preview: [Part 3 reader](previews/abstract-algebra-part-03.html), served locally at `http://127.0.0.1:4301/abstract-algebra-part-03.html` while the preview server runs. It includes home/Math/section navigation, searchable lesson links, the reader, native exercise disclosure, and a proposed lightweight square-rotation interaction. Existing Part 1/2 links explicitly open the local Angular app on port 4200. The demo applies counterclockwise quarter-turns, announces the resulting transformation, and resets to identity; a marked corner distinguishes orientation. No heavy renderer is needed. Standalone preview math is real generated MathML from the Markdown/LaTeX source, using the existing renderer and browser-native MathML display; production KaTeX HTML styling is not reproduced exactly.

Validation: isolated headless Chrome frames measured exact 390, 768, and 1366 px widths; no document horizontal overflow. All three pass search/empty-state/restore, library-to-reader navigation, four-turn identity/reset, hint disclosure, and 44 px button-height checks; phone menu and Escape-to-close checks pass at 390 px. Each frame renders 62 MathML elements. Phone and laptop screenshots were visually inspected for the visible upper reader layout; the full long article, contrast, and comprehensive keyboard/screen-reader behavior were not visually/audit checked. Source includes visible focus, labels, status announcements, and reduced-motion behavior. These are browser-frame checks, not physical-device results.

Review status: proposed content and rotation interaction, awaiting the author's feedback. No new layout choice is needed. No Angular source, dependencies, approved manifest, or deployment changes were made in this draft task. No commit, push, or deployment.

## Part 3 implementation — accepted 2026-11-24

The author reviewed the group introduction, completed its rational-number exercise, and explicitly requested `ui-implement` for this section. This accepts the existing Part 3 preview and authorizes conversion of that lesson, including its square-rotation example. The earlier proposed status above is superseded. Subgroups remain a later planned topic.

Implemented `/math/abstract-algebra/groups-and-first-examples`, registered as Part 3 in the approved manifest. The existing reader now lists all three lessons and connects Parts 2 and 3 with previous/next links. The Markdown remains the editable content source; an empty, recognized `interactive:square-rotations` fence inserts the fixed Angular rotation example between the square explanation and cancellation proof. The compiler emits a fixed marker between balanced article sections, disallows content/nested slots and duplicate slots, and keeps raw Markdown HTML disabled. Article HTML/MathML stays lazy-loaded. The small CSS/Angular interaction needs no heavy renderer or additional dependencies and is absent from other lessons. Rotation state and trailing article content reset on navigation. Preview-only author notices are omitted from the public reader.

Validation: six content tests, five route regression tests, and eight reader UI tests pass (19 total). The UI suite verifies Part 3 math and both article segments, hint behavior, lesson links, rotation/reset, navigation cleanup, and exact 390/768/1366 px browser-frame menu/overflow behavior. Production build passes with existing SCSS/CommonJS/budget warnings; the notebook stylesheet is now about 4.76 KB. An independent check of the real production bootstrap at exact 390, 768, and 1366 px browser viewports renders all 62 MathML elements, the trailing cancellation proof, and the rotation component; native Enter activation and the four-click identity cycle work at all widths, with no horizontal document overflow or observed uncaught JavaScript exceptions. Navigation back to Part 2 removes the example. Phone and laptop screenshots around the interactive example were visually inspected. These are browser checks, not physical-device or comprehensive accessibility results.

Self-review with `ui-review` found no blocking difference from the accepted preview. The production equations use the existing KaTeX HTML/MathML styling rather than the preview's native-MathML-only display. Route definitions, original profile/demo templates and assets, manual-only deployment, and the prior security fixes remain unchanged by this task. Whitespace checks pass. No commit, push, or deployment performed.

Local review while the production-output server runs: `http://127.0.0.1:4302/math/abstract-algebra/groups-and-first-examples`. This loopback server serves only the built output and uses a local SPA fallback; it does not change GitHub Pages behavior.

## Part 4 implementation — accepted 2026-11-24

The author reviewed the subgroup draft, discussed identity inheritance and inherited associativity, accepted those explanations, and explicitly requested `ui-implement`. This approves Part 4 in the already accepted math reader; no new visual direction or future-lesson approval is implied.

Implemented `/math/abstract-algebra/subgroups` from `docs/content/drafts/abstract-algebra/session-04.md`. Added the discussed proof that a subgroup cannot choose a different identity, making clear that the inverse calculation takes place in the ambient group. The lesson also covers the three checks, inherited associativity, examples/counterexamples, the nonempty one-step subgroup test, and practice. The approved manifest lists four lessons; existing numbering and previous/next navigation connect Parts 3 and 4. Generated subgroups and cyclic groups remain planned only. Reader layout, styles, component code, dependencies, and route definitions are unchanged. Article bodies still load on demand; Part 4 does not instantiate Part 3's rotation component.

Validation: six content tests, five route regressions, and nine reader UI tests pass (20 total), including Part 4 proof/table/math/hint, numbering/navigation, and exact-width 390/768/1366 px browser-frame layout checks. Production build passes with the existing SCSS/CommonJS/budget warnings. Independent real production-browser checks at exact 390, 768, and 1366 px viewports render all 123 MathML elements, the two-row operation table, and the identity proof, with no horizontal overflow or observed uncaught JavaScript exceptions. Native Enter opens the hint at each width. Navigating back through the reader loads Part 3, restores its rotation example, and exposes the next link to Part 4. Phone/laptop screenshots of the identity-proof section were visually inspected. These checks are not physical-device or comprehensive accessibility audits.

Self-review using `ui-review` found no blocking departure from the accepted reader. Original legacy template/asset hashes and the manual-only deployment gates remain unchanged. Whitespace check passes. No commit, push, or deployment performed.

Local review while the existing production-output server runs: `http://127.0.0.1:4302/math/abstract-algebra/subgroups`.

## Abstract Algebra navigation refinement — accepted 2026-11-24

The author requested a collapsible Abstract Algebra branch and a section contents page explaining each lesson with a TL;DR. This is a functional refinement within the already accepted notebook layout, not a new visual direction. Kept the existing theme, section URL, lesson URLs, and catalog metadata.

The sidebar now has separate sibling controls: a 44 px arrow button toggles the lesson list with an accessible expanded state and controlled-region label; the Abstract Algebra text remains a link to `/math/abstract-algebra`. The branch starts expanded. Entering a nonempty search expands it so matching lessons remain discoverable. The section page now explicitly presents an ordered Contents list: part number, lesson title/link, learning-session type, and a labeled TL;DR drawn from each approved lesson's existing summary. All four implemented lessons appear in order; planned future topics remain separate. Article bodies remain fetched only when a lesson is opened.

Validation: 11 reader UI tests and five route regressions pass under Bun. Added coverage for toggling without navigation, search revealing a collapsed matching branch, title-link navigation, all four TL;DR links, and exact 390/768/1366 px browser-frame layouts for both the contents page and lesson reader. Production build and whitespace checks pass, with existing warnings (notebook stylesheet about 5.13 KB). Independent production-browser checks at exact 390, 768, and 1366 px widths verify native Enter collapses the branch without leaving Math, then Enter on its separate title opens the section contents. Each view contains four TL;DR lesson links, has no horizontal document overflow, and records no uncaught JavaScript exception. Phone/laptop screenshots of the resulting contents page were visually inspected. Physical-device and comprehensive accessibility audits are not claimed.

Self-review with `ui-review` found no blocking issue. This task changes no article source, route definitions, dependencies, security behavior, or deployment gates. Existing unrelated Bun migration changes are preserved. No commit, push, or deployment performed. Local inspection: `http://127.0.0.1:4302/math/abstract-algebra` while the production-output server is running.

## Abstract Algebra Part 5 draft — proposed 2026-11-25

The proposed next lesson is **Generated subgroups and cyclic groups**. The author invited content preparation and a UI draft; article approval and Angular conversion are still pending. The Markdown is `docs/content/drafts/abstract-algebra/session-05.md`. It builds on Part 4's subgroup test, then covers integer powers and multiples, cyclic groups, finite and infinite element order, and a practice problem about multiples of six.

The standalone preview is `docs/design/previews/abstract-algebra-part-05.html`, reusing the accepted notebook and Math reader direction. It includes home/Math/lesson navigation, a searchable five-lesson list, the draft reader, a hint disclosure, and a proposed square rotation interaction that compares the subgroup generated by each rotation. The prose and order table provide a fallback for the interaction. Parts 1–4 link to the local Angular server on port 4200; those links require that server to be running. Part 5 stays in the preview and is not in the approved article manifest.

Headless Chrome checks at exact 390, 768, and 1366 px iframe viewports found no document overflow and passed search, navigation, hint, phone menu, 44 px buttons, and rotation cycles for all four choices. Each frame rendered 124 MathML nodes. A laptop screenshot was visually inspected; the 390 px headless top-level window screenshot was cropped by Chrome's window sizing, so it is not claimed as a phone visual audit. A dedicated visible Chrome window was opened at `http://127.0.0.1:4301/abstract-algebra-part-05.html` and its accessibility tree confirmed the article and interaction. The local loopback preview server runs only while its process remains active. No physical device or full accessibility audit was performed.

Review status: proposed for author review. The content, example, and interaction can be revised before the author explicitly approves blog generation. No Angular implementation or deployment is part of this draft.

## Abstract Algebra Part 5 implementation — accepted 2026-11-25

After reviewing the cyclic-group idea through examples and questions, the author explicitly invoked `ui-implement` for the next section. Part 5 is approved for local blog conversion. The prior proposed status above is superseded; this does not authorize deployment.

Implemented `/math/abstract-algebra/generated-subgroups-and-cyclic-groups` from `docs/content/drafts/abstract-algebra/session-05.md`. The approved manifest now lists five lessons, and the contents, sidebar, and previous/next links include Part 5. The lesson explains generation by integer multiples or powers, finite and infinite cyclic groups, and element order. A build-time `interactive:square-rotations` slot inserts a small generator comparison between the prose sections; the existing Part 3 quarter-turn example remains unchanged. The new interaction offers `e`, `r`, `r²`, and `r³`, shows the distinct rotations reached and first return, and resets when the choice or route changes. The text and order table remain usable without interacting. Cosets are now labeled as planned in the section page.

Validation: six content pipeline tests, five route regression tests, and twelve Chrome reader UI tests pass. The UI checks include Part 5 rendering and generator behavior, unchanged Part 3 behavior, navigation, and exact 390/768/1366 px browser-frame overflow checks for the Part 5 reader and Algebra contents. The production build passes with existing Sass, CommonJS, and bundle/style budget warnings. Source review against the accepted preview found the same reader hierarchy and controls; the production reader uses the existing KaTeX styling. Physical-device and comprehensive accessibility audits were not performed. Existing routes and manual-only deployment settings were not edited. No commit, push, or deployment was performed.

## Abstract Algebra Part 6 implementation — accepted 2026-11-25

After the author discussed the coset definition, requested the general setup before examples, and received the Section 6 Markdown draft, they explicitly invoked `ui-implement` for Cosets. This approves local conversion of `docs/content/drafts/abstract-algebra/session-06.md` in the established Math reader. No new visual direction was needed; the accepted notebook layout remains in use.

Implemented `/math/abstract-algebra/cosets-and-partitions`. The approved manifest now lists six lessons. The section contents, searchable sidebar, and previous/next links connect Parts 5 and 6; the planned-topics list now starts with Lagrange’s theorem. The Markdown keeps the abstract definition first: `H ≤ G`, `a ∈ G`, and `aH = {a ∗ h | h ∈ H}`. It then explains coset equality and partitioning before the integer-remainder and square-rotation examples. No interactive renderer was added; equations are pre-rendered by the existing trusted content pipeline and the article body loads on demand.

Validation: six content pipeline tests, five route regression tests, and thirteen Chrome reader UI tests pass. The UI checks include the definition-first order, rendered math, hint disclosure, Part 5 link, absence of a rotation widget on Part 6, and exact 390/768/1366 px browser-frame overflow checks for the Part 6 reader and Algebra contents. The production build passes with existing CommonJS and bundle/style budget warnings. Source review using `ui-review` criteria found no blocking departure from the accepted reader design: subject breadth, search, accessible native hint, focus styles, and mobile navigation remain in place. The route tests cover existing compatibility URLs; this turn did not include an independent direct-entry production-server or physical-device check. No deployment, commit, or push was performed.

## Abstract Algebra history page — proposed 2026-11-25

The author requested a nice UI for an Algebra history page. The proposed standalone preview is `docs/design/previews/algebra-history.html`, reusing the accepted editorial notebook theme and six-lesson Algebra structure. It shows a selective timeline from equation solving to group ideas and structural algebra, filterable by theme, with source links and paths back to existing lessons. It labels the timeline as selective and the page as a proposed preview. Historical summaries need author review before conversion into the Angular site. The existing `/math/abstract-algebra` page and lesson routes remain the current implemented experience.

Review status: proposed. Decide whether to add this as a separate `/math/abstract-algebra/history` context page and whether its four selected milestones are the right scope. Isolated Chrome iframe checks at exact 390, 768, and 1366 px passed: no document overflow, the Groups and All filters returned the expected cards, and the phone menu opened. A 1366 px screenshot was visually inspected. A dedicated visible Chrome window could not be reliably verified because focus changed during navigation; use the local file preview link. These are browser-frame checks, not physical-device checks. No Angular code, dependencies, manifest, or deployment changed.

## Abstract Algebra Part 7 draft — proposed 2026-11-26

The author requested `ui-implement` for Lagrange’s theorem and explicitly asked for its proof. The preceding exchange had only introduced the theorem, so the full article was not yet available for author review. Prepared `docs/content/drafts/abstract-algebra/session-07.md` with the coset-counting proof, the square-rotation example, the element-order consequence, the finite-group limitation, and exercises. The existing accepted Math reader direction would support it without a redesign. The article rendered through the current Markdown/KaTeX pipeline with 92 MathML expressions; no public article manifest or Angular files were changed. Review status: draft awaiting the author's review and explicit approval of this full article before blog conversion, as required by `AGENTS.md` and the `content-prepare` workflow. This is separate from the proposed history-page preview, whose bottom navigation was relabeled “Related lessons” for clarity.

## Abstract Algebra Part 7 implementation — accepted 2026-11-26

After receiving the full Part 7 draft, the author explicitly invoked `ui-implement` again and requested local serving to read it. This approves conversion of `docs/content/drafts/abstract-algebra/session-07.md` in the existing Math reader. The prior proposed status above is superseded.

Implemented `/math/abstract-algebra/lagranges-theorem` as Part 7 in the approved manifest. It appears in the section contents and searchable sidebar, with previous navigation to Part 6. The lesson includes a proof using coset coverage, disjointness, and a bijection between each coset and the subgroup; it then derives the element-order consequence and distinguishes necessary divisibility from existence. The section page now lists homomorphisms and quotients as future topics. No layout or route definitions changed.

Validation: six content pipeline checks, five route regression checks, and fourteen Chrome reader UI checks pass, including Part 7 proof/math/hint/navigation and exact 390/768/1366 px browser-frame overflow checks. The production build passes with existing CommonJS and bundle/style budget warnings. A local Angular server started successfully on `127.0.0.1:4305`; browser-style direct entry and the generated article asset both return HTTP 200. Isolated Chrome rendered the heading, proof, MathML, hint, and Part 6 link from the live route. Source review against the accepted Math reader found no blocking design change. This is local serving only; no commit, push, or deployment occurred.

## Abstract Algebra Part 8 visual draft — proposed 2026-11-26

The author clarified that one visual is enough: show two sets, source elements `a` and `b`, their product `a ∗ b`, and the corresponding mapped values. A further correction requested separate arrows for the rows and the target result explicitly labeled `f(a) ∗ f(b)`. The diagram now compares that target result with `f(a ∗ b)` in an equality statement. `docs/design/previews/abstract-algebra-part-08.html` is a standalone, responsive preview using the accepted notebook styling. It uses the parity example with source `(ℤ,+)` and target `({0,1},+ mod 2)`, explaining that the same `∗` notation refers to ordinary addition in the source and addition modulo 2 in the target. Part 8 is proposed, not registered as an article or implemented in Angular.

The simplified revision passed isolated headless Chrome browser-frame checks at exact 390, 768, and 1366 px: no horizontal document overflow, two sets rendered, the generic equation appeared, and changing `a` to `−3` updated the source result and all mapped values correctly. A 1366 px screenshot of the preview page was visually inspected. These are page-only browser checks, not screen recording, physical-device, or full accessibility checks. Review status: proposed for author feedback. No Angular code, dependencies, article manifest, or deployment changed.

## Abstract Algebra Part 8 implementation preparation — accepted visual, article pending 2026-11-27

The author accepted the pairwise two-set diagram and requested `ui-implement`. The accepted visual is `docs/design/previews/abstract-algebra-part-08.html`; the lesson title is **Homomorphisms**, meaning a map preserving the group operation. The Angular reader now supports a fixed `interactive:homomorphism` slot with three paired rows and an equality check for the parity map. The content compiler accepts only an empty top-level slot and still rejects multiple interactive slots. The draft Markdown is `docs/content/drafts/abstract-algebra/session-08.md`, including the general definition, the parity proof for all integers, a coset connection, and exercises. The article manifest still contains only Parts 1–7. The author has not yet reviewed and explicitly approved the full Part 8 Markdown for blog generation, so the new widget is prepared but not publicly reachable through a Part 8 lesson route.

Validation: seven content pipeline tests pass, including rendering the Part 8 draft and rejecting malformed/multiple slots. The production build passes with existing CommonJS and bundle/style budget warnings. Five route compatibility tests and fifteen reader UI tests pass in ChromeHeadless. The reader tests check the paired rows, negative integer mapping, invalid input, and exact 390, 768, and 1366 px browser-frame overflow and pairing alignment. Source review against the accepted preview found the same two-set order, three mappings, and equality explanation; the production reader uses its established narrower article column. There was no direct-entry Part 8 browser check because no approved public Part 8 route exists. No physical-device or full accessibility audit, commit, push, or deployment occurred.

## Abstract Algebra Part 8 implementation — accepted 2026-11-27

After reviewing the full Markdown handoff, the author invoked `ui-implement` again. This explicitly approves local conversion of `docs/content/drafts/abstract-algebra/session-08.md`. The previous article-pending status is superseded. Implemented `/math/abstract-algebra/homomorphisms` as Part 8 in the approved article manifest and existing Math reader. It appears in the section contents, searchable sidebar, and previous/next lesson navigation. The section's planned list now starts with quotient groups. The lesson contains the definition, pairwise two-group parity diagram, proof for all integers, a kernel/coset connection, and exercises. Its static HTML/MathML article body remains lazy-loaded; the fixed lightweight diagram is inserted only at its approved marker and resets when navigating away. Existing root, blog, Profile, and legacy routes were not changed, and deployment remains manual only.

Validation: seven content checks pass, including malformed slot rejection; five route compatibility checks and fifteen ChromeHeadless reader checks pass. The reader checks exercise Part 8's real generated article, proof after the diagram, negative and invalid inputs, cleanup after navigation, and exact 390/768/1366 px browser-frame overflow and row alignment. Production build passes with existing CommonJS and bundle/style budget warnings. A browser-style direct request to the fresh local Angular server returns HTTP 200 for `/math/abstract-algebra/homomorphisms`, and its generated article asset returns HTTP 200. Isolated headless Chrome rendered the heading, diagram, equality result, and proof from that live route. A 1366 px screenshot of the top of the live page was visually inspected; this does not replace a full manual accessibility or physical-device review. `ui-review` comparison found the accepted two-set hierarchy and three paired mappings in the implementation. No commit, push, or deployment occurred. Local reading URL while the server runs: `http://127.0.0.1:4306/math/abstract-algebra/homomorphisms`.

## Part 8 kernel and coset explanation — author-requested expansion 2026-11-27

The author asked for more explanation and examples in “A connection to cosets.” Expanded the approved Markdown source with explicit preimages of target `0` and `1`, even and odd integer examples, the coset difference test for `5` and `−3`, and a distinction between a target element (`f(5)=1`) and a coset (`5+2ℤ`). Added the two-coset addition example as a preview of quotient groups. The diagram and route are unchanged. The revised source rendered through the content pipeline; seven content checks and fifteen ChromeHeadless reader checks pass, including new assertions for the expanded passage. The running local server returned HTTP 200 for the refreshed article asset and served the new text. No new production build or route regression run was needed for this prose-only expansion; the preceding full build and route checks remain the validation for the implementation. No deployment occurred.

## Part 8 kernel and coset visualization — requested 2026-11-27

The author said the expanded prose still needed a visual. Added a second fixed interactive slot beside “A connection to cosets” in Part 8. A selectable row of integers from `−5` to `5` colors evens and odds, and two labeled buckets show the kernel `2ℤ` and odd coset `1+2ℤ`. Selecting a number highlights its bucket and states separately its single target value `f(n)` and the set `n+2ℤ`, including the even-difference test. The number buttons are keyboard and touch accessible; the numeric field accepts whole values in the displayed range. Existing text remains as the reading fallback. The content compiler accepts this second slot only when paired after the Part 8 homomorphism slot; malformed content, duplicate slots, and unsupported combinations fail at build time. The reader splits the lazily loaded static article into three HTML sections around the two fixed interactions and clears both on navigation.

Validation: seven content checks, fifteen ChromeHeadless reader checks, and the production build pass. Reader checks exercise an odd negative input, an even negative input, invalid input, button selection, navigation cleanup, and exact 390/768/1366 px browser-frame overflow. A long screenshot of the live local page was visually inspected with both diagrams and the new coset panel visible. Existing route definitions and deployment settings were not changed; no commit, push, or deployment occurred. The local reading URL remains `http://127.0.0.1:4306/math/abstract-algebra/homomorphisms` while that server runs.
