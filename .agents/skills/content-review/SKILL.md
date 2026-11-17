---
name: content-review
description: Review General Engineering article drafts for reasoning, factual accuracy, source support, and code or equation correctness before blog generation. Use for content and citation review, rather than drafting a new article, reviewing UI, or publishing.
---

# Content review

Read applicable `AGENTS.md`, `README.md`, the target article, and its supplied sources. Locate the draft from the user's path or current task; if several drafts could be intended, ask which one rather than choosing silently. If no article exists, request the article or use `content-prepare` only when drafting is also requested.

Review the article as written. Do not rewrite its body during a review-only request. Suggested replacements belong in the review; apply fixes when the user also requests revision, then review the changed passages again. Preserve unrelated drafts, CV facts, Angular source, public indexes, and deployment settings.

## Check the article and its evidence

- Check that the title and summary match what the body establishes. Follow the central claim through its assumptions, reasoning, concrete examples, limitations, and conclusion. Identify missing steps, contradictory statements, and misleading terminology.
- Verify material factual claims against supplied artifacts or primary sources. Open cited sources and inspect the relevant section; a working link or search snippet alone is not verification. Browse for current or uncertain facts and for referenced sources whose contents were not supplied. When reviewing an older event, distinguish the historical claim from the current state.
- Check that each citation supports the nearby claim, including the stated version, date, scope, and strength of conclusion. Prefer original research, official documentation, or direct project artifacts. Record a corrected source when available. Do not add citations that were not read or pretend an inaccessible source was verified.
- Distinguish established results, author observations, interpretations, hypotheses, and illustrative examples. Keep SI/RSI as long-term ambitions unless specific claims have evidence. Distinguish a local patch or test from an upstream contribution accepted or released.
- Check relevant code, equations, units, edge cases, and reproducibility. Run small meaningful checks when feasible and authorized. State exactly what ran and what remains source-inspected; do not infer runtime correctness or benchmark results from reading code.
- Evaluate whether interactive examples have a useful explanation or fallback and whether their outputs actually support the article. UI layout and lazy-renderer implementation remain separate UI checks.

Concentrate on claims that affect the reader's understanding. Do not demand citations for every sentence, inflate stylistic preferences into correctness blockers, or remove a clearly labeled hypothesis merely because it is unproven. If a material source or check is unavailable, describe the unresolved claim and its impact; a central unsupported claim blocks readiness.

## Write the review

Save `review.md` beside the target draft, normally in `docs/content/drafts/<slug>/`. Preserve unrelated reviews. If a prior review exists, update its current verdict and retain useful unresolved findings. For published source outside the drafts folder, keep the review under `docs/content/reviews/` unless the user provides another location.

Include:

- Article path and title, review date, and review basis. Label same-agent review as self-review; never imply an independent reviewer or agent was used when it was not.
- Verdict: `needs revision` or `ready for blog conversion`. An outline alone is not a complete article. Readiness means the content has no unresolved material correctness or evidence blockers; it does not certify publication or UI readiness.
- Findings ordered by importance. Give each a heading or line reference, the specific problem, reader impact, supporting evidence, and a focused correction. Separate blockers from optional editorial suggestions. Say when there are no substantive findings.
- A compact evidence table for material claims checked: article location/claim, source URL or local artifact with relevant section, and result (`supported`, `partially supported`, `unsupported`, or `not verified`). Do not claim exhaustive verification unless it occurred.
- Code/example checks actually run, sources or tools unavailable, other review limits, and any remaining author questions.

If revision was authorized, record the fixes and update the verdict after rechecking. Otherwise leave the article unchanged and report the findings.

Return the verdict, review path, and highest-priority findings. Do not generate a blog page or publish merely because a review passed. A ready verdict is a technical assessment, not author approval. The author must review the Markdown and explicitly approve this article for blog generation before conversion proceeds; reuse that approval once given.
