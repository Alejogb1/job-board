---
title: "Ultraviolet App: How to Automatically Proxy URLs for Clients?"
date: '2024-11-08'
id: 'ultraviolet-app-how-to-automatically-proxy-urls-for-clients'
---

```javascript
function encode(str) {
  if (!str) return str;
  return encodeURIComponent(
    str
      .toString()
      .split('')
      .map((char, ind) =>
        ind % 2 ? String.fromCharCode(char.charCodeAt() ^ 2) : char
      )
      .join('')
  );
}
```
