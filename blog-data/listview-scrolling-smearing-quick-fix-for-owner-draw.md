---
title: "ListView Scrolling Smearing: Quick Fix for Owner Draw?"
date: '2024-11-08'
id: 'listview-scrolling-smearing-quick-fix-for-owner-draw'
---

```C#
public class DoubleBufferedListView : ListView
{
    public DoubleBufferedListView()
    {
        SetStyle(ControlStyles.DoubleBuffer | ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint, true);
        UpdateStyles();
    }
}
```
