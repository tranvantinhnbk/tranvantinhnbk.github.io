import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { PageNotFoundComponent } from './pages/page-not-found/page-not-found.component';

const loadProfile = () => import('./pages/about-me/about-me.component').then(m => m.AboutMeComponent);
const loadAchievements = () => import('./pages/certificate/certificate.component').then(m => m.CertificateComponent);
const loadProjects = () => import('./pages/projects/projects.module').then(m => m.ProjectsModule);

export const APP_ROUTES: Routes = [
  { path: '', pathMatch: 'full', loadComponent: () => import('./pages/notebook/notebook.component').then(m => m.NotebookComponent) },
  { path: 'notes/:slug', data: { page: 'code' }, loadComponent: () => import('./pages/notebook/notebook.component').then(m => m.NotebookComponent) },
  { path: 'math', pathMatch: 'full', title: 'Math | General Engineering', data: { page: 'math' }, loadComponent: () => import('./pages/notebook/notebook.component').then(m => m.NotebookComponent) },
  { path: 'math/abstract-algebra', pathMatch: 'full', title: 'Abstract Algebra | General Engineering', data: { page: 'algebra' }, loadComponent: () => import('./pages/notebook/notebook.component').then(m => m.NotebookComponent) },
  { path: 'math/abstract-algebra/:slug', title: 'Abstract Algebra | General Engineering', data: { page: 'math-lesson' }, loadComponent: () => import('./pages/notebook/notebook.component').then(m => m.NotebookComponent) },
  {
    path: 'profile',
    children: [
      { path: '', pathMatch: 'full', loadComponent: loadProfile },
      { path: 'achievements', loadComponent: loadAchievements },
      { path: 'projects', loadChildren: loadProjects },
    ],
  },
  { path: 'blog', loadChildren: () => import('./pages/blog/blog.module').then(m => m.BlogModule) },
  // Compatibility aliases reuse the original pages, retaining query strings and CV/AT fragments.
  { path: 'about', loadComponent: loadProfile },
  { path: 'certificate', loadComponent: loadAchievements },
  { path: 'projects', loadChildren: loadProjects },
  { path: '**', component: PageNotFoundComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(APP_ROUTES)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
