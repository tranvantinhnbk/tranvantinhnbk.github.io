import { CommonModule } from '@angular/common';
import { Component, DestroyRef, ElementRef, ViewChild, inject } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { ActivatedRoute, RouterModule } from '@angular/router';
import { notes } from './note-index';

@Component({
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './notebook.component.html',
  styleUrls: ['./notebook.component.scss'],
})
export class NotebookComponent {
  readonly notes = notes;
  readonly subjects = ['Mathematics', 'Physics', 'Biology', 'Computing', 'Engineering', 'Intelligence'];
  query = '';
  menuOpen = false;
  slug = '';
  loading = false;
  error = false;
  code: { python: string; typescript: string } | null = null;
  private request = 0;
  @ViewChild('content') content?: ElementRef<HTMLElement>;

  constructor() {
    inject(ActivatedRoute).paramMap.pipe(takeUntilDestroyed()).subscribe(params => {
      this.slug = params.get('slug') || '';
      this.menuOpen = false;
      void this.loadNote();
    });
    // Ignore a module response if the reader navigates away while it loads.
    inject(DestroyRef).onDestroy(() => ++this.request);
  }

  get currentNote() { return this.notes.find(note => note.slug === this.slug); }
  get filteredNotes() {
    const query = this.query.trim().toLowerCase();
    return this.notes.filter(note => `${note.title} ${note.subject} Coding note`.toLowerCase().includes(query));
  }
  bySubject(subject: string) { return this.filteredNotes.filter(note => note.subject === subject); }
  openNote() { this.menuOpen = false; this.content?.nativeElement.focus(); }

  async loadNote() {
    const request = ++this.request;
    this.code = null;
    this.error = false;
    this.loading = !!this.currentNote;
    if (!this.currentNote) return;
    try {
      const { solution } = await import('../blog/leetcode/solution/content/solution');
      if (request === this.request) this.code = solution[this.slug] || null;
    } catch {
      if (request === this.request) this.error = true;
    } finally {
      if (request === this.request) this.loading = false;
    }
  }
}
