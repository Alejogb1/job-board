---
title: "Image Gaps Driving Me Nuts: How to Fix Responsive Percentage Sizing Issues"
date: '2024-11-08'
id: 'image-gaps-driving-me-nuts-how-to-fix-responsive-percentage-sizing-issues'
---

```css
.row .col {
  display: table-cell;
  width: 33.33333%;
}

.row .col.wide {
  width: 66.66666%;
}

.row .col:nth-child(2) img {
  position: absolute;
  top: 0;
  left: 0;
}
```
