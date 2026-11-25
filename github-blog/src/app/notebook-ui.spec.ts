import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { RouterTestingHarness } from '@angular/router/testing';
import { APP_ROUTES } from './app-routing.module';
import { NotebookComponent } from './pages/notebook/notebook.component';

const lessonUrl = '/math/abstract-algebra/sets-operations-closure';

describe('Implemented notebook reader and responsive navigation', () => {
  beforeEach(() => TestBed.configureTestingModule({ providers: [provideRouter(APP_ROUTES)] }));

  it('renders the real generated article with fractions, aligned proof, and a native hint', async () => {
    const harness = await RouterTestingHarness.create();
    const page = await harness.navigateByUrl(lessonUrl, NotebookComponent);
    await page.loadNote();
    harness.detectChanges();
    const root = harness.routeNativeElement!;
    expect(page.error).toBeFalse();
    expect(root.querySelector('.ge-article math')).not.toBeNull();
    expect(root.querySelector('.ge-article mfrac')).not.toBeNull();
    expect(root.querySelector('.ge-article mtable')).not.toBeNull();
    const hint = root.querySelector('details.math-hint') as HTMLDetailsElement;
    expect(hint.open).toBeFalse();
    (hint.querySelector('summary') as HTMLElement).click();
    expect(hint.open).toBeTrue();
  });

  it('collapses the Algebra lessons separately from opening the contents page and reveals search matches', async () => {
    const harness = await RouterTestingHarness.create();
    const page = await harness.navigateByUrl('/math', NotebookComponent);
    const root = harness.routeNativeElement!;
    const toggle = root.querySelector('.section-toggle') as HTMLButtonElement;
    const lessons = root.querySelector('#algebra-lessons') as HTMLElement;
    toggle.click(); harness.detectChanges();
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(lessons.hidden).toBeTrue();
    expect(page.page).toBe('math');
    const search = root.querySelector('input[type=search]') as HTMLInputElement;
    search.value = 'subgroups'; search.dispatchEvent(new Event('input')); harness.detectChanges();
    expect(lessons.hidden).toBeFalse();
    expect(lessons.querySelectorAll('a').length).toBe(2);
    search.value = ''; search.dispatchEvent(new Event('input')); harness.detectChanges();
    toggle.click(); harness.detectChanges();
    (root.querySelector('.section-heading a') as HTMLAnchorElement).click();
    await harness.fixture.whenStable(); harness.detectChanges();
    const overview = harness.routeNativeElement!;
    expect(overview.querySelector('#algebra-contents')?.textContent).toBe('Contents');
    const entries = Array.from(overview.querySelectorAll('.course-contents a'));
    expect(entries.length).toBe(6);
    expect(entries.every(entry => entry.textContent!.includes('TL;DR:'))).toBeTrue();
    expect(entries[0].textContent).toContain('closure');
    expect(entries[3].getAttribute('href')).toBe('/math/abstract-algebra/subgroups');
    expect(entries[4].getAttribute('href')).toBe('/math/abstract-algebra/generated-subgroups-and-cyclic-groups');
    expect(entries[5].getAttribute('href')).toBe('/math/abstract-algebra/cosets-and-partitions');
  });

  it('lists Parts 1 through 6 correctly and links between the real rendered lessons', async () => {
    const harness = await RouterTestingHarness.create();
    await harness.navigateByUrl('/math/abstract-algebra', NotebookComponent);
    const secondUrl = '/math/abstract-algebra/associativity-identity-inverses';
    expect(harness.routeNativeElement!.querySelectorAll('a.entry').length).toBe(6);
    expect(harness.routeNativeElement!.textContent).toContain('Part 2');
    const first = await harness.navigateByUrl(lessonUrl, NotebookComponent);
    expect(first.nextLesson?.path).toBe(secondUrl);
    expect(harness.routeNativeElement!.querySelector(`.reader-nav a[href="${secondUrl}"]`)).not.toBeNull();
    const second = await harness.navigateByUrl(secondUrl, NotebookComponent);
    await second.loadNote();
    harness.detectChanges();
    const root = harness.routeNativeElement!;
    expect(second.error).toBeFalse();
    expect(root.querySelector('article h1')?.textContent).toBe('Associativity, identity, and inverses');
    expect(root.querySelector('.breadcrumb')?.textContent).toContain('Part 2');
    expect(root.querySelector('.ge-article math')).not.toBeNull();
    expect(root.querySelector('.ge-article mfrac')).not.toBeNull();
    expect(second.previousLesson?.path).toBe(lessonUrl);
    expect(second.nextLesson?.path).toBe('/math/abstract-algebra/groups-and-first-examples');
    expect(second.mathNotes.some(note => note.slug === 'groups-and-first-examples')).toBeTrue();
  });

  it('renders Part 3 around a working rotation example and clears it when leaving', async () => {
    const harness = await RouterTestingHarness.create();
    const page = await harness.navigateByUrl('/math/abstract-algebra/groups-and-first-examples', NotebookComponent);
    await page.loadNote();
    harness.detectChanges();
    const root = harness.routeNativeElement!;
    expect(page.error).toBeFalse();
    expect(root.querySelector('article h1')?.textContent).toContain('Groups');
    expect(root.querySelector('.breadcrumb')?.textContent).toContain('Part 3');
    expect(root.querySelectorAll('.ge-article').length).toBe(2);
    expect(root.querySelector('.ge-article math')).not.toBeNull();
    const demo = root.querySelector('.rotation-example')!;
    expect(root.textContent).toContain('Left cancellation');
    expect(demo.nextElementSibling?.textContent).toContain('Your turn');
    const [turn, reset] = Array.from(demo.querySelectorAll('button'));
    turn.click(); harness.detectChanges();
    expect(demo.querySelector('[role=status]')?.textContent).toContain('90° counterclockwise');
    for (let i = 0; i < 3; i++) turn.click();
    harness.detectChanges();
    expect(demo.querySelector('[role=status]')?.textContent).toContain('identity');
    turn.click(); reset.click(); harness.detectChanges();
    expect(page.rotationTurns).toBe(0);
    expect(page.previousLesson?.part).toBe(2);
    expect(page.nextLesson?.path).toBe('/math/abstract-algebra/subgroups');
    const hint = root.querySelector('details.math-hint') as HTMLDetailsElement;
    (hint.querySelector('summary') as HTMLElement).click();
    expect(hint.open).toBeTrue();
    await harness.navigateByUrl(lessonUrl, NotebookComponent);
    await page.loadNote(); harness.detectChanges();
    expect(root.querySelector('.rotation-example')).toBeNull();
    expect(page.articleAfterHtml).toBeNull();
  });

  it('renders the approved subgroup lesson, identity proof, table, and hint without the rotation component', async () => {
    const harness = await RouterTestingHarness.create();
    const page = await harness.navigateByUrl('/math/abstract-algebra/subgroups', NotebookComponent);
    await page.loadNote(); harness.detectChanges();
    const root = harness.routeNativeElement!;
    expect(page.error).toBeFalse();
    expect(root.querySelector('article h1')?.textContent).toBe('Subgroups — groups inside groups');
    expect(root.querySelector('.breadcrumb')?.textContent).toContain('Part 4');
    expect(root.textContent).toContain('Could the subset have a different identity?');
    expect(root.textContent).toContain('We inherit associativity');
    expect(root.textContent).toContain('One-step subgroup test');
    expect(root.querySelector('.ge-article math')).not.toBeNull();
    expect(root.querySelectorAll('.math-table tbody tr').length).toBe(2);
    expect(root.querySelector('.rotation-example')).toBeNull();
    expect(page.previousLesson?.part).toBe(3);
    expect(root.querySelector('.reader-nav a[href="/math/abstract-algebra/groups-and-first-examples"]')).not.toBeNull();
    expect(page.nextLesson?.part).toBe(5);
    const hint = root.querySelector('details.math-hint') as HTMLDetailsElement;
    (hint.querySelector('summary') as HTMLElement).click();
    expect(hint.open).toBeTrue();
  });

  it('renders Part 5 and compares generators without changing the earlier rotation example', async () => {
    const harness = await RouterTestingHarness.create();
    const page = await harness.navigateByUrl('/math/abstract-algebra/generated-subgroups-and-cyclic-groups', NotebookComponent);
    await page.loadNote(); harness.detectChanges();
    const root = harness.routeNativeElement!;
    expect(page.error).toBeFalse();
    expect(root.querySelector('.breadcrumb')?.textContent).toContain('Part 5');
    expect(root.querySelectorAll('.ge-article').length).toBe(2);
    expect(root.textContent).toContain('Cyclic means one generator');
    expect(root.querySelector('.ge-article math')).not.toBeNull();
    expect(root.querySelectorAll('.math-table tbody tr').length).toBe(4);
    const demo = root.querySelector('.rotation-example')!;
    const choice = demo.querySelector('select') as HTMLSelectElement;
    choice.value = '2'; choice.dispatchEvent(new Event('change')); harness.detectChanges();
    const [apply, reset] = Array.from(demo.querySelectorAll('button'));
    apply.click(); apply.click(); harness.detectChanges();
    expect(demo.querySelector('[role=status]')?.textContent).toContain('order 2');
    expect(demo.textContent).toContain('e, r²');
    reset.click(); harness.detectChanges();
    expect(page.rotationTurns).toBe(0);
    expect(page.previousLesson?.part).toBe(4);
    expect(page.nextLesson?.path).toBe('/math/abstract-algebra/cosets-and-partitions');
    await harness.navigateByUrl('/math/abstract-algebra/subgroups', NotebookComponent);
    await page.loadNote(); harness.detectChanges();
    expect(root.querySelector('.rotation-example')).toBeNull();
  });

  it('renders Part 6 with the abstract definition before the integer examples and links back to Part 5', async () => {
    const harness = await RouterTestingHarness.create();
    const page = await harness.navigateByUrl('/math/abstract-algebra/cosets-and-partitions', NotebookComponent);
    await page.loadNote(); harness.detectChanges();
    const root = harness.routeNativeElement!;
    expect(page.error).toBeFalse();
    expect(root.querySelector('.breadcrumb')?.textContent).toContain('Part 6');
    expect(root.querySelector('article h1')?.textContent).toContain('Cosets');
    const headings = Array.from(root.querySelectorAll('.ge-article h2')).map(heading => heading.textContent);
    expect(headings.indexOf('Define the objects before the example')).toBeLessThan(headings.indexOf('Example: integer remainders are cosets'));
    expect(root.textContent).toContain('parent group');
    expect(root.textContent).toContain('Every element lands in one piece');
    expect(root.querySelectorAll('.ge-article math').length).toBeGreaterThan(0);
    expect(root.querySelector('.rotation-example')).toBeNull();
    const hint = root.querySelector('details.math-hint') as HTMLDetailsElement;
    (hint.querySelector('summary') as HTMLElement).click();
    expect(hint.open).toBeTrue();
    expect(page.previousLesson?.part).toBe(5);
    expect(page.nextLesson).toBeUndefined();
    expect(root.querySelector('.reader-nav a[href="/math/abstract-algebra/generated-subgroups-and-cyclic-groups"]')).not.toBeNull();
  });

  for (const [label, url] of [['Part 6', '/math/abstract-algebra/cosets-and-partitions'], ['the Algebra contents', '/math/abstract-algebra']]) {
  it(`adapts ${label} to exact phone, tablet, and laptop widths without horizontal overflow`, async () => {
    const harness = await RouterTestingHarness.create();
    const page = await harness.navigateByUrl(url, NotebookComponent);
    await page.loadNote();
    harness.detectChanges();
    const root = harness.routeNativeElement!;
    const originalParent = root.parentNode!;
    // Headless Chrome clamps small outer windows. Exact browser-frame viewports
    // exercise the real stylesheet's media queries at the intended widths.
    for (const width of [390, 768, 1366]) {
      const frame = document.createElement('iframe');
      frame.width = String(width);
      frame.height = '844';
      frame.srcdoc = '<!doctype html><html><head></head><body></body></html>';
      const loaded = new Promise<void>(resolve => frame.addEventListener('load', () => resolve(), { once: true }));
      document.body.appendChild(frame);
      await loaded;
      try {
        const frameDocument = frame.contentDocument!;
        for (const style of Array.from(document.querySelectorAll('style, link[rel=stylesheet]'))) frameDocument.head.appendChild(style.cloneNode(true));
        frameDocument.body.appendChild(root);
        await frameDocument.fonts.ready;
        await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
        expect(frame.contentWindow!.innerWidth).withContext('exact frame viewport').toBe(width);
        const button = root.querySelector('.notes-toggle') as HTMLElement;
        const sidebar = root.querySelector('aside') as HTMLElement;
        page.menuOpen = false;
        harness.detectChanges();
        const styleOf = (element: Element) => frame.contentWindow!.getComputedStyle(element);
        if (width <= 620) {
          expect(styleOf(button).display).not.toBe('none');
          expect(styleOf(sidebar).display).toBe('none');
          button.click();
          harness.detectChanges();
          expect(styleOf(sidebar).display).not.toBe('none');
          expect(button.getAttribute('aria-expanded')).toBe('true');
        } else {
          expect(styleOf(button).display).toBe('none');
          expect(styleOf(sidebar).display).not.toBe('none');
        }
        expect(frameDocument.documentElement.scrollWidth).withContext(`overflow at ${width}px`).toBeLessThanOrEqual(width + 1);
      } finally {
        originalParent.appendChild(root);
        frame.remove();
      }
    }
  });

  }

  it('shows an empty-search message and restores the notes when cleared', async () => {
    const harness = await RouterTestingHarness.create();
    await harness.navigateByUrl('/math', NotebookComponent);
    const root = harness.routeNativeElement!;
    const input = root.querySelector('input[type=search]') as HTMLInputElement;
    input.value = 'no-matching-note';
    input.dispatchEvent(new Event('input'));
    harness.detectChanges();
    expect(root.textContent).toContain('No matching notes. Try another search.');
    expect(root.querySelectorAll('aside details').length).toBe(0);
    input.value = '';
    input.dispatchEvent(new Event('input'));
    harness.detectChanges();
    expect(root.querySelectorAll('aside details').length).toBe(1);
    expect(root.textContent).not.toContain('Computing');
    expect(root.textContent).not.toContain('Original coding index');
  });

  it('keeps archived coding notes reachable without advertising Computing on the homepage', async () => {
    const harness = await RouterTestingHarness.create();
    const home = await harness.navigateByUrl('/', NotebookComponent);
    expect(home.notes.every(note => note.subject !== 'Computing')).toBeTrue();
    expect(harness.routeNativeElement!.textContent).not.toContain('Computing');
    expect(harness.routeNativeElement!.textContent).not.toContain('Two Sum');
    const archived = await harness.navigateByUrl('/notes/two-sum', NotebookComponent);
    expect(archived.currentNote?.slug).toBe('two-sum');
  });

  it('provides retry after a fetch failure', async () => {
    const fetchSpy = spyOn(window, 'fetch').and.rejectWith(new Error('Offline'));
    const harness = await RouterTestingHarness.create();
    const page = await harness.navigateByUrl(lessonUrl, NotebookComponent);
    await page.loadNote();
    harness.detectChanges();
    expect(harness.routeNativeElement!.querySelector('[role=alert]')?.textContent).toContain('could not load');
    fetchSpy.and.resolveTo(new Response('<!-- ge-article:sets-operations-closure -->\n<div class="ge-article">Recovered lesson</div>'));
    await page.loadNote();
    harness.detectChanges();
    expect(harness.routeNativeElement!.textContent).toContain('Recovered lesson');
    expect(page.error).toBeFalse();
  });

  it('ignores a late article response after navigation to an unknown lesson', async () => {
    let resolveResponse!: (response: Response) => void;
    spyOn(window, 'fetch').and.returnValue(new Promise<Response>(resolve => resolveResponse = resolve));
    const harness = await RouterTestingHarness.create();
    const page = await harness.navigateByUrl(lessonUrl, NotebookComponent);
    await harness.navigateByUrl('/math/abstract-algebra/missing', NotebookComponent);
    resolveResponse(new Response('<!-- ge-article:sets-operations-closure -->\n<div class="ge-article">Stale lesson</div>'));
    await new Promise(resolve => setTimeout(resolve, 0));
    harness.detectChanges();
    expect(page.articleHtml).toBeNull();
    expect(harness.routeNativeElement!.textContent).toContain('Note not found.');
    expect(harness.routeNativeElement!.textContent).not.toContain('Stale lesson');
  });
});
