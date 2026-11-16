import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { MnistComponent } from './mnist/mnist.component';

const routes: Routes = [
  { path: 'mnist', component: MnistComponent }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class ProjectsRoutingModule { }
