---
name: ui-review
description: Review General Engineering UI drafts or frontend changes for usability, accessibility, content clarity, and consistency with the chosen design.
---

# UI review

Read applicable `AGENTS.md` and `docs/design/brief.md`. Identify the draft or implementation being reviewed and whether a design direction has been accepted.

Inspect the primary reader journey: homepage → subject or library → article → related material or back navigation. Check the pages relevant to the task rather than demanding every page for a small change.

Review:
- Breadth: biology, physics, mathematics, computing, and other engineering fields remain discoverable.
- Structure: subjects, article types, and SI/RSI ambitions are understandable without competing navigation systems.
- Reading: heading hierarchy, prose width, equations, code, citations, and related links suit long articles.
- Interaction: visible focus, keyboard controls, meaningful labels, empty search results, and mobile navigation.
- Layout: no clipping or overlap at representative mobile and desktop widths; readable contrast in the supported themes.
- Honesty: sample content is identified and hypotheses are distinguishable from established results.

Use browser inspection when available. For implementation reviews, run the relevant build/checks and compare against the accepted preview. Report verified problems with a location, user impact, and a focused fix. Clearly identify untested behavior and proposed preferences. Do not rewrite the frontend during a review-only request.

Finish with the most important findings and readiness for the next phase. Do not mark a design accepted on the user's behalf.
