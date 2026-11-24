// Only metadata enters the navigation bundle. Bodies and old demos stay lazy.
import { mathNotes } from './generated-content';
import type { NoteMetadata } from './note-metadata';

const codingNotes: NoteMetadata[] = [
  { slug: 'two-sum', title: 'Two Sum', subject: 'Computing', kind: 'Coding note', path: '/notes/two-sum', source: 'coding' },
  { slug: 'longest-substring-without-repeating-characters', title: 'Longest Substring Without Repeating Characters', subject: 'Computing', kind: 'Coding note', path: '/notes/longest-substring-without-repeating-characters', source: 'coding' },
  { slug: 'longest-palindromic-substring', title: 'Longest Palindromic Substring', subject: 'Computing', kind: 'Coding note', path: '/notes/longest-palindromic-substring', source: 'coding' },
  { slug: 'container-with-most-water', title: 'Container With Most Water', subject: 'Computing', kind: 'Coding note', path: '/notes/container-with-most-water', source: 'coding' },
];
export const notes: NoteMetadata[] = [...mathNotes, ...codingNotes];

// Archive lookup preserves saved coding URLs; only the current catalog is browsable.
export const browseNotes: NoteMetadata[] = notes.filter(note => note.subject !== 'Computing');
