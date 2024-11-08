---
title: "Download a List of Docs with CFDirectory: A Quick & Easy Solution"
date: '2024-11-08'
id: 'download-a-list-of-docs-with-cfdirectory-a-quick-easy-solution'
---

```cfscript
this.mappings["/guides"]="J:\guides";
```

```html
<cfdirectory action="list" directory="j:\guides" recurse="false" name="nameofqry" type="file" sort="datelastmodified desc" filter="*.docx">
<cfoutput query="nameofqry" maxrows="40">
    <li><a href="/guides/#nameofqry.name#" class="downloadlink" target="_blank">#nameofqry.name#</a></li>
</cfoutput>
```
