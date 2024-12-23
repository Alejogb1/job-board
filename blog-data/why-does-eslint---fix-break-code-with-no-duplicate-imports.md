---
title: "Why does `eslint --fix` break code with `no-duplicate-imports`?"
date: "2024-12-23"
id: "why-does-eslint---fix-break-code-with-no-duplicate-imports"
---

Alright, let's unpack this. I remember a particularly frustrating project a few years back, where a seemingly innocuous eslint configuration change triggered a cascade of code breakages. It involved exactly this: `eslint --fix` and the `no-duplicate-imports` rule going rogue. Let's get into why that happens, from a practical angle.

The `no-duplicate-imports` rule, at its core, aims to enforce a clean and maintainable import structure. It's designed to catch instances where the same module is imported multiple times within a single file, which can lead to confusion and potentially unnecessary resource consumption. When eslint encounters such duplication, and if `--fix` is enabled, it attempts to merge these imports. This merge operation is where the trouble often begins.

The primary problem isn’t the rule itself, but rather its interaction with JavaScript's module system and how `eslint --fix` interprets and resolves those import statements. Specifically, the automated fix often doesn’t account for named imports versus default imports, or the nuanced ways modules can be structured and re-exported. Additionally, problems arise when imports are aliased differently. This leads to cases where, in its attempt to 'fix,' eslint actually creates invalid or non-functional code.

To illustrate this, consider these three distinct scenarios I’ve encountered in past projects:

**Scenario 1: Default and Named Import Collision**

Often, developers mix default and named imports from the same module, either intentionally or unintentionally. Take a file that initially looks like this:

```javascript
// initial_code_1.js

import React from 'react';
import { useState, useEffect } from 'react';

function MyComponent() {
  // ... component logic using React, useState, and useEffect
}
```

`eslint --fix`, seeing two imports from 'react', might attempt to consolidate this into:

```javascript
//eslint_fixed_code_1.js

import React, { useState, useEffect } from 'react'; // **PROBLEM!**

function MyComponent() {
   // ... component logic using React, useState, and useEffect
}
```

This fixed code is now fundamentally broken. `React` is intended as the default import, representing the React library’s main export, while `useState` and `useEffect` are specific named exports. By combining them into a single import, `eslint --fix` has essentially tried to treat the default export as if it were a named export. This results in `React` being undefined or not usable as intended, creating a runtime error. The solution here isn’t just to suppress the warning – it's understanding the import types and fixing the imports correctly so that the default export is separated.

**Scenario 2: Aliased Imports and Merge Conflicts**

Another frequent issue I've observed arises from the practice of aliasing imports. Suppose we have code similar to this:

```javascript
// initial_code_2.js
import { add as addFunction } from './utils';
import { add } from './utils';

function calculateSum(a,b) {
    return addFunction(a,b) + add(a,b);
}
```

This situation, while potentially questionable practice, might not break code. The intention could be to potentially invoke different functions under the same umbrella (although it’s not advisable). The `eslint --fix` would see these imports from the same location and attempt a merge:

```javascript
//eslint_fixed_code_2.js

import { add as addFunction, add } from './utils';  //**PROBLEM!**
function calculateSum(a,b) {
    return addFunction(a,b) + add(a,b);
}
```

Now, you have a problem. The second `add` is now re-defining the `add` alias, effectively creating two aliases to the same thing and likely overwriting the previously aliased `addFunction`. Depending on the execution and compiler, this can lead to unpredictable results. While it doesn't necessarily throw an error, it invalidates the code's intent. The fix here would be to review the usage, and decide whether to consolidate under one alias or remove the duplicated import (along with renaming one of them), but that requires thought and cannot be done automatically.

**Scenario 3: Re-exported Modules and Incorrect Deduplication**

Finally, consider a scenario where a module exports things from another module, and we inadvertently import from both locations. Here's how that could look:

```javascript
// initial_code_3.js
// library_a.js
export const value = 10;

// module_b.js
export * from './library_a';

// my_component.js
import { value } from './library_a';
import { value as reExportedValue } from './module_b';

function useValues(){
    console.log(value, reExportedValue);
}
```

`eslint --fix` will see the multiple imports of value and likely deduplicate this, leading to:

```javascript
// eslint_fixed_code_3.js

import { value , value as reExportedValue } from './library_a'; //**PROBLEM!**
function useValues(){
    console.log(value, reExportedValue);
}
```

Now you have the same situation as before, and `eslint --fix` has broken the intent. We now have the same value twice – no longer differentiating the original and re-exported value. The fix in these scenarios requires a full understanding of the module structure, which automated tools often can't reliably infer. This may not error directly but will cause the code to act in an unexpected manner.

So, why does `eslint --fix` break code with `no-duplicate-imports`? It's because while the rule is well-intentioned, the automated fixing mechanism is overly simplistic. It lacks the contextual awareness to handle the complexities of import syntax, module aliasing, and re-export patterns. The tool assumes that any import of the same name or from the same module can simply be merged, overlooking the subtle distinctions that make code work.

The lesson I learned – painfully – was to *never* blindly apply autofixes, particularly on import rules. It's always a better strategy to understand the code and the underlying import patterns first, then apply fixes manually. If eslint finds errors of duplicate import, investigate the reasons for duplication first, and then decide the appropriate course of action. Do not leave this to autofix to decide for you, because that decision is often wrong. This isn't a bug in eslint; it's a limitation of automated code correction without understanding intent.

For further reading, I’d recommend diving into the "ECMAScript specification" for a deep dive on import syntax. Specifically, Section 15.2.3.6, “Import Declarations," provides the definitive guide. Also, explore “Effective JavaScript: 68 Specific Ways to Harness the Power of JavaScript” by David Herman to understand the nuances and best practices of structuring javascript. Understanding these finer details is crucial to avoid the common pitfalls caused by blindly applying `eslint --fix` rules, particularly with module import statements. The documentation from eslint itself regarding `no-duplicate-imports` rule should also be studied carefully.
