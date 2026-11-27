import { CommonModule } from '@angular/common';
import { AfterViewInit, Component, DestroyRef, ElementRef, ViewChild, inject } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { ActivatedRoute, RouterModule } from '@angular/router';
import { notes, browseNotes } from './note-index';

@Component({
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './notebook.component.html',
  styleUrls: ['./notebook.component.scss'],
})
export class NotebookComponent implements AfterViewInit {
  readonly notes = browseNotes;
  readonly subjects = ['Mathematics', 'Physics', 'Biology', 'Engineering', 'Intelligence'];
  query = '';
  menuOpen = false;
  algebraExpanded = true;
  page = 'home';
  slug = '';
  loading = false;
  error = false;
  articleHtml: SafeHtml | null = null;
  articleAfterHtml: SafeHtml | null = null;
  articleTailHtml: SafeHtml | null = null;
  rotationExample = false;
  homomorphismExample = false;
  parityCosetsExample = false;
  cosetN: number | null = 5;
  readonly cosetSamples = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5];
  get cosetParity() { return this.parity(this.cosetN ?? 0); }
  get cosetDifference() { return (this.cosetN ?? 0) - this.cosetParity; }
  setCosetInput(value: string) {
    const parsed = /^-?(0|[1-9]\d*)$/.test(value) ? Number(value) : NaN;
    this.cosetN = Number.isInteger(parsed) && parsed >= -5 && parsed <= 5 ? parsed : null;
  }
  homA: number | null = 3;
  homB: number | null = 4;
  get homValid() { return this.homA !== null && this.homB !== null; }
  get homSum() { return (this.homA ?? 0) + (this.homB ?? 0); }
  parity(value: number) { return ((value % 2) + 2) % 2; }
  get homMappedA() { return this.parity(this.homA ?? 0); }
  get homMappedB() { return this.parity(this.homB ?? 0); }
  get homMappedProduct() { return this.parity(this.homSum); }
  get homTargetProduct() { return this.parity(this.homMappedA + this.homMappedB); }
  setHomInput(which: 'a' | 'b', value: string) {
    const parsed = /^-?(0|[1-9]\d*)$/.test(value) ? Number(value) : NaN;
    const next = Number.isSafeInteger(parsed) && parsed >= -20 && parsed <= 20 ? parsed : null;
    if (which === 'a') this.homA = next;
    else this.homB = next;
  }
  rotationTurns = 0;
  rotationStep = 1;
  get rotationLabel() {
    return ['e · 0° · identity', 'r · 90° counterclockwise', 'r² · 180°', 'r³ · 270° counterclockwise'][this.rotationTurns % 4];
  }
  get generatedOrder() { return this.rotationStep === 0 ? 1 : this.rotationStep === 2 ? 2 : 4; }
  get generatedPosition() { return this.rotationTurns * this.rotationStep % 4; }
  get generatedStatus() {
    const labels = ['e', 'r', 'r²', 'r³'];
    const returned = this.rotationTurns === this.generatedOrder ? ` · First return to identity; order ${this.generatedOrder}` : '';
    return `Applications: ${this.rotationTurns} · Current: ${labels[this.generatedPosition]}${returned}`;
  }
  get generatedElements() {
    const labels = ['e', 'r', 'r²', 'r³'];
    return [...new Set(Array.from({ length: Math.min(this.rotationTurns + 1, this.generatedOrder) }, (_, index) => labels[index * this.rotationStep % 4]))].join(', ');
  }
  get generatedCount() { return Math.min(this.rotationTurns + 1, this.generatedOrder); }
  setRotationStep(value: string) { this.rotationStep = Number(value); this.rotationTurns = 0; }
  code: { python: string; typescript: string } | null = null;
  private request = 0;
  private abort?: AbortController;
  private readonly route = inject(ActivatedRoute);
  private readonly sanitizer = inject(DomSanitizer);
  @ViewChild('content') content?: ElementRef<HTMLElement>;

  constructor() {
    this.route.paramMap.pipe(takeUntilDestroyed()).subscribe(params => {
      this.page = this.route.snapshot.data['page'] || 'home';
      this.slug = params.get('slug') || '';
      this.menuOpen = false;
      this.content?.nativeElement.focus();
      void this.loadNote();
    });
    inject(DestroyRef).onDestroy(() => { ++this.request; this.abort?.abort(); });
  }

  ngAfterViewInit() { this.content?.nativeElement.focus({ preventScroll: true }); }
  get currentNote() {
    const source = this.page === 'math-lesson' ? 'article' : 'coding';
    return notes.find(note => note.slug === this.slug && note.source === source);
  }
  get mathNotes() {
    return this.notes.filter(note => note.subject === 'Mathematics').sort((a, b) => (a.part || 0) - (b.part || 0));
  }
  get currentPartLabel() {
    const note = this.currentNote;
    return note ? (note.part ? `Part ${note.part}` : note.title) : 'Note not found';
  }
  get previousLesson() {
    const lessons = this.mathNotes;
    const index = lessons.findIndex(note => note.slug === this.currentNote?.slug);
    return index > 0 ? lessons[index - 1] : undefined;
  }
  get nextLesson() {
    const lessons = this.mathNotes;
    const index = lessons.findIndex(note => note.slug === this.currentNote?.slug);
    return index >= 0 ? lessons[index + 1] : undefined;
  }
  get filteredNotes() {
    const query = this.query.trim().toLowerCase();
    return this.notes.filter(note => `${note.title} ${note.subject} ${note.section || ''} ${note.kind}`.toLowerCase().includes(query));
  }
  bySubject(subject: string) { return this.filteredNotes.filter(note => note.subject === subject); }
  setQuery(value: string) {
    this.query = value;
    if (value.trim()) this.algebraExpanded = true;
  }
  openNote() { this.menuOpen = false; }
  skipToContent(event: Event) { event.preventDefault(); this.content?.nativeElement.focus(); }

  async loadNote() {
    const request = ++this.request;
    this.abort?.abort();
    this.articleHtml = null;
    this.articleAfterHtml = null;
    this.articleTailHtml = null;
    this.rotationExample = false;
    this.homomorphismExample = false;
    this.parityCosetsExample = false;
    this.cosetN = 5;
    this.homA = 3;
    this.homB = 4;
    this.rotationTurns = 0;
    this.rotationStep = 1;
    this.code = null;
    this.error = false;
    const note = this.currentNote;
    this.loading = !!note;
    if (!note) return;
    try {
      if (note.source === 'article' && note.asset) {
        this.abort = new AbortController();
        const response = await fetch(note.asset, { signal: this.abort.signal });
        if (!response.ok) throw new Error('Article unavailable');
        const html = await response.text();
        if (!html.startsWith(`<!-- ge-article:${note.slug} -->`)) throw new Error('Unexpected article response');
        if (request === this.request) {
          // Only our manifest-selected, build-generated asset reaches this boundary.
          // The builder disables raw Markdown HTML and KaTeX trust. Preserve KaTeX's
          // required layout styles and MathML, which Angular's generic sanitizer strips.
          const marker = html.includes('<!-- ge-interactive:homomorphism -->')
            ? '<!-- ge-interactive:homomorphism -->'
            : '<!-- ge-interactive:square-rotations -->';
          const [before, after] = html.split(marker);
          this.articleHtml = this.sanitizer.bypassSecurityTrustHtml(before);
          this.rotationExample = marker.includes('square-rotations') && after !== undefined;
          this.homomorphismExample = marker.includes('homomorphism') && after !== undefined;
          if (this.homomorphismExample && after !== undefined) {
            const [middle, tail] = after.split('<!-- ge-interactive:parity-cosets -->');
            this.articleAfterHtml = this.sanitizer.bypassSecurityTrustHtml(middle);
            this.parityCosetsExample = tail !== undefined;
            this.articleTailHtml = tail === undefined ? null : this.sanitizer.bypassSecurityTrustHtml(tail);
          } else {
            this.articleAfterHtml = after === undefined ? null : this.sanitizer.bypassSecurityTrustHtml(after);
          }
        }
      } else {
        const { solution } = await import('../blog/leetcode/solution/content/solution');
        if (request === this.request) this.code = solution[note.slug] || null;
      }
    } catch {
      if (request === this.request) this.error = true;
    } finally {
      if (request === this.request) this.loading = false;
    }
  }
}
