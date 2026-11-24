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
  page = 'home';
  slug = '';
  loading = false;
  error = false;
  articleHtml: SafeHtml | null = null;
  articleAfterHtml: SafeHtml | null = null;
  rotationExample = false;
  rotationTurns = 0;
  get rotationLabel() {
    return ['e · 0° · identity', 'r · 90° counterclockwise', 'r² · 180°', 'r³ · 270° counterclockwise'][this.rotationTurns % 4];
  }
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
  openNote() { this.menuOpen = false; }
  skipToContent(event: Event) { event.preventDefault(); this.content?.nativeElement.focus(); }

  async loadNote() {
    const request = ++this.request;
    this.abort?.abort();
    this.articleHtml = null;
    this.articleAfterHtml = null;
    this.rotationExample = false;
    this.rotationTurns = 0;
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
          const [before, after] = html.split('<!-- ge-interactive:square-rotations -->');
          this.articleHtml = this.sanitizer.bypassSecurityTrustHtml(before);
          this.rotationExample = after !== undefined;
          this.articleAfterHtml = after === undefined ? null : this.sanitizer.bypassSecurityTrustHtml(after);
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
