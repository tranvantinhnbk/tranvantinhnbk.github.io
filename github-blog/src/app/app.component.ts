import { Router } from '@angular/router';
import { Component, inject } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'github-blog';
  private readonly router = inject(Router);
  get isNotebook() {
    const path = this.router.url.split(/[?#]/)[0];
    return path === '/' || path.startsWith('/notes/');
  }
}
