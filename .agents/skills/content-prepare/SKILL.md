---
name: content-prepare
description: Collaborate on General Engineering article ideas and prepare Markdown drafts for the author's review before blog generation. Use for brainstorming, outlining, and drafting; use content-review for a dedicated source and correctness review.
---

# Content prepare

Read applicable `AGENTS.md`, `README.md`, and supplied material. Work with the author on the idea before treating it as a finished article. Preserve their meaning, voice, and the distinction between verified results, interpretations, hypotheses, and open questions. SI/RSI remain long-term ambitions unless a more specific claim has evidence.

## Discuss, then draft in Markdown

Match the author's current stage. During brainstorming, help clarify the question, intended reader, main point, examples, and uncertainties. Suggest a working title and outline when useful; do not force a full article or settle an open idea on the author's behalf. Ask focused questions where their intent matters and keep useful drafting moving where it does not.

Keep working material under `docs/content/drafts/<slug>/`, or reuse the location the author supplies. Save an idea/outline as `idea.md` when useful, and the article itself as `article.md` when drafting is requested. Mark work as a draft; keep unknown details open. Do not create empty artifacts merely to fill a workflow.

For a full article, include a working title, slug, one-sentence summary, subject, content type, and draft status. A subject such as Physics differs from a content type such as Experiment. Use the [article template](assets/article.md) when useful and adapt the body to the material. Lead with the actual question or supported finding, then explain through a concrete example and reasoning. Include relevant assumptions, limitations, and sources.

Verify material factual claims against supplied artifacts or appropriate primary sources. Browse for current or uncertain facts and source attribution. Put source links near the claims they support. Do not invent results, benchmarks, citations, dates, author experience, or contribution status. Distinguish a local observation, proposed fix, and confirmed upstream result. Qualify unsupported claims or leave them explicitly unresolved.

For interactive content, describe its purpose, inputs, expected output, fallback, and proposed loading trigger. Do not implement renderers, Angular pages, public indexes, or deployments during content preparation.

## Author review and revision

Present the Markdown path, working title, and the few questions or choices that need the author's input. Let the author read, discuss, and revise the content with you. Continue revisions when requested; do not infer approval from silence or from a general request to prepare content.

Perform ordinary drafting checks as you work. Use `content-review` when a dedicated article/source review is requested or needed; it records findings in `review.md`. A technical review verdict does not replace the author's review or approval. Do not require a separate formal review report for every brainstorming exchange.

Stop at the Markdown handoff until the author has reviewed and explicitly approved this article for blog generation. Prior broad conversion requests do not skip this author-review step. Once that approval exists, do not ask for it again unless a material content change needs further review. Blog generation and publication remain distinct actions with their own authorized scope.
