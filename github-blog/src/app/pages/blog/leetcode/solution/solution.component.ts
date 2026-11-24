import { Component, ViewEncapsulation } from '@angular/core';
import { HighlightJsDirective } from 'ngx-highlight-js';
import { solution } from './content/solution';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-solution',
  standalone: true,
  imports: [
    CommonModule,
    HighlightJsDirective,
  ],
  templateUrl: './solution.component.html',
  styleUrl: './solution.component.scss',
  encapsulation: ViewEncapsulation.ShadowDom,

})
export class SolutionComponent {
  constructor(private _route: ActivatedRoute) { }
  solution = {};
  ngOnInit(): void {
      const solutionName = this._route.snapshot.paramMap.get('name') || '';
      this.solution = solution[solutionName];
  }

}
