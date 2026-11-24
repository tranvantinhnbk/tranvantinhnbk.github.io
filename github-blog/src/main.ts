import { provideZoneChangeDetection } from '@angular/core';
import { platformBrowserDynamic } from '@angular/platform-browser-dynamic';

import { AppModule } from './app/app.module';


// Angular 21 defaults to zoneless. Keep zone-based updates for the existing
// reader fetches and legacy async components during this dependency upgrade.
platformBrowserDynamic().bootstrapModule(AppModule, {
  applicationProviders: [provideZoneChangeDetection()],
})
  .catch(err => console.error(err));
