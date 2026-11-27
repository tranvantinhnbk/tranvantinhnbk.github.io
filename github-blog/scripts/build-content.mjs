import { readFile, writeFile, mkdir, rm } from 'node:fs/promises';
import { resolve, dirname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import MarkdownIt from 'markdown-it';
import katex from 'katex';

export function createRenderer() {
  const md = new MarkdownIt({ html: false, linkify: false, typographer: true });
  const codeFence = md.renderer.rules.fence;
  md.renderer.rules.fence = (tokens, i, options, env, self) => {
    const kind = tokens[i].info.trim();
    if (!['interactive:square-rotations', 'interactive:homomorphism', 'interactive:parity-cosets'].includes(kind)) return codeFence(tokens, i, options, env, self);
    if (tokens[i].level !== 0 || tokens[i].content.trim()) throw new Error('Interactive example must be an empty top-level fence');
    return `</div><!-- ge-${kind} --><div class="ge-article">`;
  };
  const typeset = (tex, displayMode) => katex.renderToString(tex, {
    displayMode, throwOnError: true, trust: false, strict: 'error', output: 'htmlAndMathml',
  });
  md.inline.ruler.before('escape', 'inline_math', (state, silent) => {
    if (state.src[state.pos] !== '$' || state.src[state.pos + 1] === '$') return false;
    let end = state.pos + 1;
    while (end < state.posMax) {
      if (state.src[end] === '\n') return false;
      if (state.src[end] === '$') {
        let escapes = 0;
        for (let i = end - 1; i > state.pos && state.src[i] === '\\'; i--) escapes++;
        if (escapes % 2 === 0) break;
      }
      end++;
    }
    if (end >= state.posMax || end === state.pos + 1) return false;
    if (!silent) state.push('inline_math', 'math', 0).content = state.src.slice(state.pos + 1, end);
    state.pos = end + 1;
    return true;
  });
  md.block.ruler.before('fence', 'display_math', (state, start, end, silent) => {
    const line = state.src.slice(state.bMarks[start] + state.tShift[start], state.eMarks[start]).trim();
    if (line !== '$$' || state.sCount[start] - state.blkIndent >= 4) return false;
    if (silent) return true;
    let close = start + 1;
    while (close < end && state.src.slice(state.bMarks[close] + state.tShift[close], state.eMarks[close]).trim() !== '$$') close++;
    if (close === end) throw new Error(`Unclosed display math at line ${start + 1}`);
    const token = state.push('display_math', 'math', 0);
    token.content = state.getLines(start + 1, close, state.blkIndent, false).trim();
    state.line = close + 1;
    return true;
  });
  md.renderer.rules.table_open = () => '<div class="math-table" tabindex="0" role="region" aria-label="Operation examples"><table>\n';
  md.renderer.rules.table_close = () => '</table></div>\n';
  md.renderer.rules.hint_open = () => '<details class="math-hint"><summary>A small hint</summary><p>';
  md.renderer.rules.hint_close = () => '</p></details>\n';
  md.renderer.rules.inline_math = (tokens, i) => typeset(tokens[i].content, false);
  md.renderer.rules.display_math = (tokens, i) => `<div class="math-equation" tabindex="0" role="region" aria-label="Display equation">${typeset(tokens[i].content, true)}</div>\n`;
  return md;
}

export function renderArticle(source) {
  const body = source.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, '');
  const md = createRenderer();
  const tokens = md.parse(body, {});
  const slots = tokens.filter(token => token.type === 'fence' && ['interactive:square-rotations', 'interactive:homomorphism', 'interactive:parity-cosets'].includes(token.info.trim())).map(token => token.info.trim());
  const pairedParitySlots = slots.length === 2 && slots[0] === 'interactive:homomorphism' && slots[1] === 'interactive:parity-cosets';
  if ((slots.length > 1 && !pairedParitySlots) || (slots.includes('interactive:parity-cosets') && !pairedParitySlots)) throw new Error('Only the homomorphism and parity-cosets pair is supported together');
  // The application renders the approved metadata title, so omit the duplicate Markdown h1.
  if (tokens[0]?.type === 'heading_open' && tokens[0].tag === 'h1') tokens.splice(0, 3);
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i].type === 'paragraph_open' && /^\*\*(Definition|Proposition|Proof|Hint)/.test(tokens[i + 1]?.content || '')) {
      tokens[i].attrSet('class', 'math-callout');
    }
    if (tokens[i].type === 'paragraph_open' && /^\*\*By the end/.test(tokens[i + 1]?.content || '')) tokens[i].attrSet('class', 'math-goal');
    if (tokens[i].type === 'paragraph_open' && /^\*\*Hint\.\*\*/.test(tokens[i + 1]?.content || '')) {
      tokens[i].type = 'hint_open';
      tokens[i + 2].type = 'hint_close';
      tokens[i + 1].content = tokens[i + 1].content.replace(/^\*\*Hint\.\*\*\s*/, '');
      tokens[i + 1].children = md.parseInline(tokens[i + 1].content, {})[0].children;
    }
  }
  return `<div class="ge-article">${md.renderer.render(tokens, md.options, {})}</div>`;
}

export async function buildContent(appRoot = fileURLToPath(new URL('..', import.meta.url))) {
  const manifest = JSON.parse(await readFile(resolve(appRoot, 'content/articles.json'), 'utf8'));
  const catalog = [];
  const outputs = [];
  const seen = new Set();
  // Render everything first: a bad equation must fail before replacing the previous output.
  for (const article of manifest) {
    if (article.approved !== true) continue;
    if (!/^[a-z0-9-]+$/.test(article.slug) || typeof article.title !== 'string') throw new Error('Invalid article metadata');
    if (!/^\/math\/[a-z0-9/-]+$/.test(article.path) || article.path.includes('..') || seen.has(article.path)) throw new Error('Invalid or duplicate article route');
    seen.add(article.path);
    const sourcePath = resolve(appRoot, article.source);
    const repoRoot = resolve(appRoot, '..');
    if (relative(repoRoot, sourcePath).startsWith('..')) throw new Error('Article source must be inside the repository');
    const html = renderArticle(await readFile(sourcePath, 'utf8'));
    const asset = `/assets/content${article.path}.html`;
    outputs.push({ path: resolve(appRoot, `src${asset}`), html: `<!-- ge-article:${article.slug} -->\n${html}` });
    const { approved, source, ...metadata } = article;
    catalog.push({ ...metadata, source: 'article', asset });
  }
  await rm(resolve(appRoot, 'src/assets/content'), { recursive: true, force: true });
  for (const output of outputs) {
    await mkdir(dirname(output.path), { recursive: true });
    await writeFile(output.path, output.html);
  }
  const generated = resolve(appRoot, 'src/app/pages/notebook/generated-content.ts');
  await writeFile(generated, `// Generated from the explicitly approved article manifest. Do not edit.\nimport type { NoteMetadata } from './note-metadata';\nexport const mathNotes: NoteMetadata[] = ${JSON.stringify(catalog, null, 2)};\n`);
  return catalog;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const catalog = await buildContent();
  console.log(`Rendered ${catalog.length} approved article(s) with static LaTeX typesetting.`);
}
