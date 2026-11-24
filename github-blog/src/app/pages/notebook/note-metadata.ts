export interface NoteMetadata {
  slug: string;
  title: string;
  subject: string;
  section?: string;
  part?: number;
  kind: string;
  path: string;
  source: 'coding' | 'article';
  summary?: string;
  asset?: string;
}
