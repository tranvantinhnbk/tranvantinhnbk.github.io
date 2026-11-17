# Article preparation

1. Discuss the idea together: question, reader, working title, examples, and outline. Use `$content-prepare`; stay at the idea stage while the author is still exploring.
2. Draft in Markdown: `docs/content/drafts/<slug>/article.md`. An optional `idea.md` can hold early thinking.
3. The author reads the Markdown, discusses it with the assistant, and requests revisions. Continue until the content reflects their intent.
4. Use `$content-review` for a dedicated check of reasoning, code, and sources when requested or needed. Its `review.md` verdict is a technical assessment, not author approval.
5. Generate the blog page only after the author explicitly approves this article for generation. Existing approval remains valid; do not repeatedly request it. Publication is a separate action.

No public note, Angular page, or deployment changes during idea preparation.

Example: `Use $content-prepare to discuss this idea with me and draft it in Markdown for my review.`

Review only: `Use $content-review to review this article and verify its sources. Leave the article unchanged.`
