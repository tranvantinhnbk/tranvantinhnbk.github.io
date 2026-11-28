import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';

@Component({
  selector: 'app-quotient-example', standalone: true, imports: [CommonModule],
  template: `
  <section aria-labelledby="quotient-heading">
    <p class="eyebrow">Explore the quotient operation</p><h2 id="quotient-heading">Add two cosets.</h2>
    <p>H = 2ℤ. E = H contains all evens; O = 1 + H contains all odds. Each whole coset is one quotient element.</p>
    <div class="presets"><button *ngFor="let pair of pairs" type="button" (click)="choose(pair)" [attr.aria-pressed]="pair === name(a) + name(b)">{{ pair[0] }} + {{ pair[1] }}</button></div>
    <div class="inputs"><label>Representative a<input type="number" min="-9" max="9" step="1" [value]="a" (change)="set('a', $any($event.target).value)"></label><label>Representative b<input type="number" min="-9" max="9" step="1" [value]="b" (change)="set('b', $any($event.target).value)"></label></div>
    <p *ngIf="invalid" role="alert">Enter whole integers from −9 to 9.</p>
    <div class="flow"><div class="card" [class.odd]="name(a) === 'O'"><small>First quotient element</small><strong>{{ a }} + H = {{ name(a) }}</strong><span>a = {{ a }}</span></div><span class="operator">+</span><div class="card" [class.odd]="name(b) === 'O'"><small>Second quotient element</small><strong>{{ b }} + H = {{ name(b) }}</strong><span>b = {{ b }}</span></div><span class="operator">=</span><div class="card result" [class.odd]="name(a+b) === 'O'"><small>Resulting coset</small><strong>{{ a+b }} + H = {{ name(a+b) }}</strong><span>a + b = {{ a+b }}</span></div></div>
    <p class="equation" role="status">({{ a }} + H) + ({{ b }} + H) = {{ a+b }} + H. So {{ name(a) }} + {{ name(b) }} = {{ name(a+b) }}.</p>
    <div class="actions"><button type="button" (click)="changeRepresentatives()">Change both representatives, keep the cosets</button><button type="button" (click)="reset()">Reset example</button></div>
    <p class="status" role="status" aria-live="polite">{{ status }}</p>
    <table aria-label="Coset addition table"><thead><tr><th scope="col">+</th><th scope="col">E</th><th scope="col">O</th></tr></thead><tbody><tr *ngFor="let row of ['E','O']"><th scope="row">{{ row }}</th><td *ngFor="let col of ['E','O']" [class.active]="row === name(a) && col === name(b)">{{ row === col ? 'E' : 'O' }}</td></tr></tbody></table>
  </section>`,
  styles: [`:host{display:block;max-width:100%}section{border:1px solid #879782;border-radius:8px;padding:20px;margin:30px 0;font:15px/1.6 system-ui;color:#203129}.eyebrow{font-size:12px;color:#426b5e;text-transform:uppercase}h2{font:500 28px/1.3 Georgia,serif;margin:8px 0}button{font:inherit;min-height:44px;padding:8px 12px;border:1px solid #829e8b;border-radius:5px;background:#edf3e9;color:inherit;cursor:pointer}button[aria-pressed=true]{background:#277064;color:white}button:focus-visible,input:focus-visible{outline:3px solid #277064;outline-offset:3px}.presets,.actions,.inputs{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}.inputs label{font-size:13px;font-weight:600}input{display:block;width:95px;min-height:44px;padding:8px;font:inherit;color:inherit;background:#fff;border:1px solid #829e8b;border-radius:5px}.flow{display:grid;grid-template-columns:minmax(0,1fr) 18px minmax(0,1fr) 18px minmax(0,1fr);gap:5px}.card{min-width:0;background:#eef5ee;border:1px solid #9eb9a4;border-radius:8px;padding:13px 7px;text-align:center}.card.odd{background:#f8eedc;border-color:#b9a47c}.card.result{border:2px solid #277064}.card small,.card strong,.card span{display:block}.card small{font-size:10px}.card strong{font:600 22px/1.3 Georgia,serif;overflow-wrap:anywhere}.card span{font-size:12px}.operator{align-self:center;text-align:center;color:#277064}.equation{background:#edf5ef;border-left:3px solid #277064;padding:12px;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse}td,th{padding:10px;border:1px solid #bfcec0;text-align:center}.active{background:#dcefe3;outline:2px solid #277064;outline-offset:-3px}@media(max-width:620px){section{padding:14px}.flow{grid-template-columns:1fr}.card{padding:10px}}`]
})
export class QuotientExampleComponent {
  a = 1; b = 3; invalid = false;
  readonly pairs = ['EE','EO','OE','OO'];
  status = 'Change the representatives: the integer sum may change, but the result coset stays the same.';
  name(n: number) { return ((n % 2) + 2) % 2 ? 'O' : 'E'; }
  set(which: 'a' | 'b', raw: string) {
    const n = Number(raw);
    this.invalid = !/^-?(0|[1-9]\d*)$/.test(raw) || !Number.isInteger(n) || n < -9 || n > 9;
    if (!this.invalid) { this[which] = n; this.status = 'Choose different representatives from the same cosets to compare.'; }
  }
  choose(pair: string) { this.a = pair[0] === 'E' ? 0 : 1; this.b = pair[1] === 'E' ? 2 : 3; this.invalid = false; this.status = 'Now change representatives while keeping these two cosets.'; }
  changeRepresentatives() {
    const old = this.a + this.b;
    this.a += this.a <= 7 ? 2 : -2; this.b += this.b <= 7 ? 2 : -2;
    this.invalid = false;
    this.status = `Before: sum ${old} (${this.name(old)}). Now: sum ${this.a+this.b} (${this.name(this.a+this.b)}). The difference is even, so the result coset is unchanged.`;
  }
  reset() { this.a = 1; this.b = 3; this.invalid = false; this.status = 'O + O = E. Change the representatives to compare.'; }
}
