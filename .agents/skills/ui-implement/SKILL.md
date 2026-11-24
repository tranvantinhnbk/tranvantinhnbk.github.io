---
name: ui-implement
description: Implement an approved General Engineering UI draft in Angular, preserve existing routes and content, and validate the result. Use after design approval; use ui-draft for exploration and ui-review for review-only requests.
---

# UI implement

Read applicable `AGENTS.md`, `README.md`, `docs/design/brief.md`, the accepted preview, and the relevant application source. Reuse approval already given in the conversation. If no direction has been selected, prepare a concrete preview with `ui-draft`; ask only about unresolved choices that materially affect implementation. Do not reopen settled decisions or invent an additional approval gate.

## Preserve the project contract

Inventory existing route paths, fragments, query-dependent behavior, public assets, and links before editing. Keep the site's root and existing blog URLs stable. Add new subject/section paths alongside them unless a migration is explicitly requested. Preserve the original profile, CV, achievements, and demos. An approved visual change does not authorize rewriting their facts or silently moving their URLs.

Use the accepted layout and content hierarchy. Keep science and engineering breadth visible; distinguish subjects, sections, and content types. Use only reviewed content approved for conversion; do not publish illustrative preview prose or planned lessons as completed work. When current-session approval covers the displayed lesson, record that approval and proceed without asking again. Otherwise implement the reusable reader without inventing article approval.

## Implement

Work in the existing Angular application and respect its installed version. Keep changes focused, preserve unrelated work, and use existing components where appropriate. Reproduce the approved reading width, typography, navigation, equations, and phone behavior. Keep preview-only author controls out of public reader flows unless explicitly requested.

Prefer Markdown as the editable article source. Typeset LaTeX using a maintained renderer; pre-render at build time when appropriate for static hosting. Do not present a hand-maintained MathML mockup as a working Markdown compiler. Keep source trust boundaries explicit and fail clearly on malformed math. Lazy-load article bodies and heavy interactive renderers; provide meaningful loading/error states and ignore stale responses after navigation. Avoid loading CV viewers or demo models into the blog homepage.

Keep touch controls, keyboard navigation, focus handling, heading hierarchy, readable contrast, and equation/code overflow usable across widths. Do not change manual-only deployment into automatic publishing. Implementation does not authorize committing, pushing, or deploying.

## Validate and hand off

Run a production build and focused checks for the changed behavior when the toolchain permits. Test preservation of existing URLs, including direct entry and relevant fragments, not only new happy paths. Check the article pipeline with real content and meaningful invalid-input cases. Use browser inspection for representative phone, intermediate, and laptop widths; distinguish actual measurements, visual inspection, source checks, and physical-device checks. If a tool fails, record what remains unverified rather than claiming a pass.

Review the implementation against the accepted preview using `ui-review`. Fix observed blockers within scope, then record accepted decisions and actual validation in `docs/design/brief.md`. Report the implemented routes, material differences or limitations, validation results, and deployment status. Keep optional preferences separate from correctness blockers; do not require more approvals for already-authorized routine fixes.
