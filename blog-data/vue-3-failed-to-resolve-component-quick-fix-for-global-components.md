---
title: '"Vue 3:  "Failed to resolve component" -  Quick Fix for Global Components"'
date: '2024-11-08'
id: 'vue-3-failed-to-resolve-component-quick-fix-for-global-components'
---

```javascript
import { createApp } from 'vue';
import App from './App.vue';
import JournalCard from './widgets/journalCard.vue';

const app = createApp(App);
app.component('journal-card', JournalCard); 
const mountedApp = app.mount('#app');
```
