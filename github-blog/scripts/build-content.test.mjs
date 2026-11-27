import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, access, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { renderArticle, buildContent } from './build-content.mjs';

test('renders a safe rotation slot between balanced article sections and rejects malformed slots', () => {
  const slot = '```interactive:square-rotations\n```\n';
  const html = renderArticle('Before\n\n' + slot + '\nAfter');
  assert.match(html, /<\/div><!-- ge-interactive:square-rotations --><div class="ge-article">/);
  assert.match(html, /After/);
  assert.doesNotMatch(html, /<script|<button/);
  assert.throws(() => renderArticle(slot + '\n' + slot), /Only the homomorphism/);
  assert.throws(() => renderArticle('```interactive:square-rotations\nanything\n```'), /empty top-level/);
});

test('renders the reviewed homomorphism slot and rejects slot content', async () => {
  const source = await readFile(new URL('../../docs/content/drafts/abstract-algebra/session-08.md', import.meta.url), 'utf8');
  const html = renderArticle(source);
  assert.match(html, /<!-- ge-interactive:homomorphism -->/);
  assert.match(html, /<!-- ge-interactive:parity-cosets -->/);
  assert.ok(html.indexOf('ge-interactive:homomorphism') < html.indexOf('ge-interactive:parity-cosets'));
  assert.match(html, /Why parity always preserves addition/);
  assert.match(html, /<math/);
  assert.throws(() => renderArticle('```interactive:homomorphism\n<script>bad</script>\n```'), /empty top-level/);
  assert.throws(() => renderArticle('```interactive:homomorphism\n```\n\n```interactive:parity-cosets\nnot empty\n```'), /empty top-level/);
  assert.throws(() => renderArticle('```interactive:parity-cosets\n```'), /Only the homomorphism/);
  assert.throws(() => renderArticle('```interactive:homomorphism\n```\n\n```interactive:square-rotations\n```'), /Only the homomorphism/);
});

test('typesets inline fractions and aligned proofs, omitting the duplicate article title', () => {
  const html = renderArticle('# Article\n\nA fraction $\\frac{1}{2}$.\n\n$$\n\\begin{aligned}a+b&=2m+2n\\\\&=2(m+n)\\end{aligned}\n$$\n');
  assert.match(html, /<math/);
  assert.match(html, /katex-html/);
  assert.match(html, /application\/x-tex/);
  assert.doesNotMatch(html, /<h1/);
});

test('fails clearly on bad LaTeX or unclosed display math', () => {
  assert.throws(() => renderArticle('$\\unknownMathCommand{x}$'));
  assert.throws(() => renderArticle('$$\nx+1\n'), /Unclosed display math/);
});

test('preserves code fences and escapes raw HTML from Markdown', () => {
  const html = renderArticle('```text\n$notMath$\n```\n\n<img src=x onerror=alert(1)>\n\n[bad](javascript:alert(1))\n');
  assert.match(html, /\$notMath\$/);
  assert.doesNotMatch(html, /<math|<img|href="javascript:/);
  assert.match(html, /&lt;img/);
});

test('the practice hint is a native disclosure, not an exposed answer panel', () => {
  const html = renderArticle('**Hint.** Write each input as $2m+1$.');
  assert.match(html, /<details class="math-hint"><summary>A small hint<\/summary>/);
  assert.match(html, /<math/);
});

test('builds only explicitly approved content and leaves previous output intact on failure', async () => {
  const root = await mkdtemp(join(tmpdir(), 'ge-content-test-'));
  const app = join(root, 'app');
  const manifest = join(app, 'content/articles.json');
  await mkdir(join(app, 'content'), { recursive: true });
  await mkdir(join(app, 'src/app/pages/notebook'), { recursive: true });
  const entry = { approved: true, slug: 'example', title: 'Example', subject: 'Mathematics', section: 'Abstract Algebra', kind: 'Learning session', path: '/math/abstract-algebra/example', source: '../approved.md' };
  try {
    await writeFile(join(root, 'approved.md'), '# Example\n\n$1+1=2$');
    await writeFile(manifest, JSON.stringify([entry, { approved: false, source: '../missing-draft.md' }]));
    const catalog = await buildContent(app);
    assert.equal(catalog.length, 1);
    const output = join(app, 'src/assets/content/math/abstract-algebra/example.html');
    const original = await readFile(output, 'utf8');
    assert.ok(original.startsWith('<!-- ge-article:example -->'));
    assert.match(original, /<math/);
    await assert.rejects(access(join(app, 'src/assets/content/missing-draft.html')));
    await writeFile(join(root, 'approved.md'), '$$\n\\badCommand\n$$');
    await assert.rejects(buildContent(app));
    assert.equal(await readFile(output, 'utf8'), original);
    await writeFile(manifest, JSON.stringify([{ ...entry, source: '../../outside.md' }]));
    await assert.rejects(buildContent(app), /inside the repository/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
