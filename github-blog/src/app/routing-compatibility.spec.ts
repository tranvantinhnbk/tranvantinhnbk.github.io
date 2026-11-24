import { TestBed } from '@angular/core/testing';
import { Router, provideRouter } from '@angular/router';
import { RouterTestingHarness } from '@angular/router/testing';
import { APP_ROUTES } from './app-routing.module';
import { NotebookComponent } from './pages/notebook/notebook.component';
import { AboutMeComponent } from './pages/about-me/about-me.component';
import { CertificateComponent } from './pages/certificate/certificate.component';
import { MnistComponent } from './pages/projects/mnist/mnist.component';
import { LeetcodeComponent } from './pages/blog/leetcode/leetcode.component';
import { SolutionComponent } from './pages/blog/leetcode/solution/solution.component';

describe('Blog URL compatibility and new Math routes', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideRouter(APP_ROUTES)] });
    // Test real route matching and fragment subscribers without booting PDF/model renderers.
    TestBed.overrideComponent(NotebookComponent, { set: { template: '<main>Notebook</main>', imports: [] } });
    TestBed.overrideComponent(AboutMeComponent, { set: { template: '<main>Profile</main>', imports: [] } });
    TestBed.overrideComponent(CertificateComponent, { set: { template: '<main>Achievements</main>', imports: [] } });
    TestBed.overrideComponent(MnistComponent, { set: { template: '<main>Demo</main>', imports: [] } });
    TestBed.overrideComponent(LeetcodeComponent, { set: { template: '<main>Coding index</main>', imports: [] } });
    TestBed.overrideComponent(SolutionComponent, { set: { template: '<main>Coding solution</main>', imports: [] } });
    spyOn(window, 'fetch').and.resolveTo(new Response('<!-- ge-article:sets-operations-closure -->\n<div class="ge-article">Lesson</div>'));
  });

  it('keeps the root blog and existing note URLs', async () => {
    const harness = await RouterTestingHarness.create();
    const home = await harness.navigateByUrl('/', NotebookComponent);
    expect(home.page).toBe('home');
    const note = await harness.navigateByUrl('/notes/two-sum', NotebookComponent);
    expect(note.currentNote?.slug).toBe('two-sum');
    expect(TestBed.inject(Router).url).toBe('/notes/two-sum');
  });

  it('adds Math and Abstract Algebra without redirecting the lesson URL', async () => {
    const harness = await RouterTestingHarness.create();
    expect((await harness.navigateByUrl('/math', NotebookComponent)).page).toBe('math');
    expect((await harness.navigateByUrl('/math/abstract-algebra', NotebookComponent)).page).toBe('algebra');
    const lesson = await harness.navigateByUrl('/math/abstract-algebra/sets-operations-closure', NotebookComponent);
    expect(lesson.currentNote?.source).toBe('article');
    expect(TestBed.inject(Router).url).toBe('/math/abstract-algebra/sets-operations-closure');
  });

  it('preserves the original coding index and solution paths, queries and fragments', async () => {
    const harness = await RouterTestingHarness.create();
    await harness.navigateByUrl('/blog/leetcode', LeetcodeComponent);
    const url = '/blog/leetcode/solution/two-sum?view=code#example';
    const solution = await harness.navigateByUrl(url, SolutionComponent);
    expect(Object.keys(solution.solution).length).toBeGreaterThan(0);
    expect(TestBed.inject(Router).url).toBe(url);
  });

  it('preserves CV/transcript fragments on both the original and Profile paths', async () => {
    const harness = await RouterTestingHarness.create();
    const old = await harness.navigateByUrl('/about#AT', AboutMeComponent);
    expect(old.pdfSrc).toBe('/assets/pdf/academic-transcript.pdf');
    expect(TestBed.inject(Router).url).toBe('/about#AT');
    const current = await harness.navigateByUrl('/profile#CV', AboutMeComponent);
    expect(current.pdfSrc).toBe('/assets/pdf/CV-Tran Van Tinh.pdf');
    expect(TestBed.inject(Router).url).toBe('/profile#CV');
  });

  it('retains both achievement and demo URL families', async () => {
    const harness = await RouterTestingHarness.create();
    for (const url of ['/certificate', '/profile/achievements']) {
      await harness.navigateByUrl(url, CertificateComponent);
      expect(TestBed.inject(Router).url).toBe(url);
    }
    for (const url of ['/projects/mnist', '/profile/projects/mnist']) {
      await harness.navigateByUrl(url, MnistComponent);
      expect(TestBed.inject(Router).url).toBe(url);
    }
  });
});
