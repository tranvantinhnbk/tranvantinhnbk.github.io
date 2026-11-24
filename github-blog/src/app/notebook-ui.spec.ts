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

  it('lists Parts 1, 2, and 3 correctly and links between the real rendered lessons', async () => {
    const harness = await RouterTestingHarness.create();
    await harness.navigateByUrl('/math/abstract-algebra', NotebookComponent);
    const secondUrl = '/math/abstract-algebra/associativity-identity-inverses';
    expect(harness.routeNativeElement!.querySelectorAll('a.entry').length).toBe(3);
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
    expect(page.nextLesson).toBeUndefined();
    const hint = root.querySelector('details.math-hint') as HTMLDetailsElement;
    (hint.querySelector('summary') as HTMLElement).click();
    expect(hint.open).toBeTrue();
    await harness.navigateByUrl(lessonUrl, NotebookComponent);
    await page.loadNote(); harness.detectChanges();
    expect(root.querySelector('.rotation-example')).toBeNull();
    expect(page.articleAfterHtml).toBeNull();
  });

  it('adapts Part 3 to exact phone, tablet, and laptop widths without horizontal overflow', async () => {
    const harness = await RouterTestingHarness.create();
    const page = await harness.navigateByUrl('/math/abstract-algebra/groups-and-first-examples', NotebookComponent);
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
