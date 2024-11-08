---
title: '"ENOENT" on Node.js? Quick Fix for "spawn ENOENT" Errors!'
date: '2024-11-08'
id: 'enoent-on-node-js-quick-fix-for-spawn-enoent-errors'
---

```javascript
(function() {
    var childProcess = require("child_process");
    var oldSpawn = childProcess.spawn;
    function mySpawn() {
        console.log('spawn called');
        console.log(arguments);
        var result = oldSpawn.apply(this, arguments);
        return result;
    }
    childProcess.spawn = mySpawn;
})();
```
