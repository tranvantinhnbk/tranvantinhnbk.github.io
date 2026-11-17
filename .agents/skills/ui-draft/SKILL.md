---
name: ui-draft
description: Draft or revise responsive General Engineering website previews for laptop and phone browsers before frontend implementation. Use for layout exploration, page flows, and interactive browser previews.
---

# UI draft

Read `README.md`, applicable `AGENTS.md`, and `docs/design/brief.md` from the repository root. Reuse accepted decisions; ask only about gaps that materially affect the design.

For a new visual direction, offer two or three meaningfully different homepage treatments. Once a direction is chosen, carry it through the remaining screens rather than continuing to generate alternatives.

Prioritize the website in laptop and phone browsers. Create a responsive, interactive browser preview with realistic navigation, subject browsing, and an article reader. Use one layout that adapts across viewport widths; choose breakpoints based on content and check widths between the laptop and phone views. Adapt navigation, reading width, images, code blocks, and equations for smaller screens, with touch-friendly controls on phones and keyboard access on laptops. Cover the screens requested. Label example articles and speculative features. Preserve the distinction between subjects (e.g. Physics) and content types (e.g. Experiment).

Use Product Design if connected and suitable. Otherwise create a local interactive preview with available tools; never claim a plugin was used without calling it. For inline previews, follow the visualize skill when available. Save drafts in `docs/design/previews/` without changing Angular code or dependencies during a preview-only task.

Use browser inspection when available to check representative laptop and phone viewports, plus intermediate widths. Check navigation and key interactions, clipping and unintended horizontal overflow, touch target usability, keyboard access, and text contrast. Record the viewport sizes checked. Distinguish source checks, browser viewport emulation, and physical-device checks; report any checks that could not run.

Record the preview location and current review status in `docs/design/brief.md`. Present the preview and the concrete design choices still needed. Wait for a direction choice before implementing unless the user already authorized that direction and implementation.
