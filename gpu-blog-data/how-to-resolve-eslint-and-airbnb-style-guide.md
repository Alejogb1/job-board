---
title: "How to resolve ESLint and Airbnb style guide conflicts?"
date: "2025-01-30"
id: "how-to-resolve-eslint-and-airbnb-style-guide"
---
Airbnb's style guide, while rigorous and often beneficial, can frequently clash with ESLint's default rules, particularly after a project's initial setup or when integrating legacy code. The key to resolving these conflicts lies in strategically configuring ESLint to align with specific Airbnb rules while retaining the flexibility to deviate where necessary for project-specific considerations or existing codebase conventions. This requires a nuanced approach beyond simply extending the Airbnb configuration and adjusting a few settings. It demands a detailed understanding of both ESLint's and the Airbnb style guide's granular rule specifications.

My experience, particularly with a recent refactoring project, has involved several rounds of configuration refinement to achieve a balance between strict adherence to style and pragmatic coding practices. I've learned that a single configuration file modification rarely solves the majority of these conflicts. Instead, it requires careful examination of individual errors, understanding why ESLint flags them, and whether the Airbnb style guide's recommendations are truly beneficial or impose unnecessary rigidity.

The core of resolving these conflicts involves the `.eslintrc.js` (or similar config file like `.eslintrc.json` or within `package.json`) file. This file defines the linting rules that ESLint enforces. A typical initial setup, assuming you have `eslint-config-airbnb-base` installed would likely include this in the `.eslintrc.js`:

```javascript
module.exports = {
    extends: 'airbnb-base',
    rules: {
        // Your custom rules go here
    }
};
```

This configuration extends the base Airbnb configuration, effectively adopting its ruleset. This initial step is essential, however, the `rules` object is where you selectively disable, modify, or extend the Airbnb rules. Conflicts often arise with rules that enforce a very specific stylistic preference not universally applicable or those that impede productivity due to overly strict constraints.

Consider a common conflict: `import/no-unresolved`. This rule flags import statements when ESLint cannot resolve the module path. This occurs frequently with path aliases or dynamically generated module names. While the rule has merit for standard module resolution, in projects with complex folder structures or custom module loaders it generates a large number of false positives, hindering progress. The ideal resolution here isn't to completely disable the rule because valid resolution errors should be caught. Instead, selectively disable it for specific files or directories by using the `overrides` property within `.eslintrc.js`.

**Example 1: Overriding `import/no-unresolved` in specific directories**
```javascript
module.exports = {
    extends: 'airbnb-base',
    rules: {
         // Other rules
    },
   overrides: [
       {
         files: ["src/utils/**/*.js", "src/config/**/*.js"], // adjust the directories/file patterns as needed
         rules: {
           'import/no-unresolved': 'off',
         }
       }
   ]
};
```
Here, we use `overrides` to specify that `import/no-unresolved` should be turned `off`  (or disabled)  only for JavaScript files in the `src/utils` and `src/config` directories. Other files, not matched by this pattern, will still be subject to the default rule, preventing us from overlooking genuine resolution issues. This highlights the power of selectively modifying rules based on context instead of globally disabling them.

Another typical conflict involves formatting rules, specifically `max-len`. Airbnb’s style guide enforces a maximum line length, often 100 characters. While generally beneficial, excessively long strings or deeply nested function calls in configurations or JSX can break this rule causing  unnecessary code ugliness and reduced readability. Blindly enforcing the 100-character limit can lead to forced, unnatural code splits that decrease, not increase, maintainability.

**Example 2: Adjusting `max-len` to accommodate exceptions**

```javascript
module.exports = {
    extends: 'airbnb-base',
    rules: {
      'max-len': ['error', { code: 120 }], // Increased line limit
      'react/jsx-max-props-per-line': [1, { "maximum": 4 }], // Added react jsx rule
    },
};
```

Instead of disabling `max-len`, I’ve configured it to allow 120 characters, providing more flexibility without sacrificing readability. Note the `['error', { code: 120 }]` syntax. ESLint rules can accept an severity as their first argument, followed by an options object. A severity of `error` means violations will be reported as errors. Adjusting the `code` property changes the character limit. Furthermore, I added an additional rule, `react/jsx-max-props-per-line`,  to exemplify further custom rule modifications. This is an example showing that you can not only adjust existing rules but you can also add rules specific to your needs.

Finally, `no-unused-vars` from Airbnb’s configuration is a crucial rule, helping detect and eliminate dead code. However, it can sometimes flag function arguments that are not immediately used, such as in callbacks or placeholder functions which might be expected in the future. While you can disable the rule completely, this can lead to the accumulation of real unused variables. A better approach involves enabling specific options to refine the rule’s behavior.

**Example 3: Modifying `no-unused-vars` to accept unused arguments**

```javascript
module.exports = {
    extends: 'airbnb-base',
    rules: {
       'no-unused-vars': ['warn', { args: 'after-used', ignoreRestSiblings: true }],
    },
};
```

This configuration adjusts `no-unused-vars`  to report unused variables as a `warn`, instead of an `error` (while keeping it enabled). Crucially it includes `args: 'after-used'`. This tells ESLint not to flag unused arguments *if* a subsequent argument is used. It also includes `ignoreRestSiblings: true` to ignore siblings in rest patterns, which often have elements that might not be directly used. This approach maintains the spirit of the rule while accommodating common coding patterns.

In conclusion, resolving ESLint and Airbnb style guide conflicts is a iterative process requiring in-depth understanding of the project's conventions and each rule’s implications. Global modifications are rarely the answer. Focus on individual conflicts, assess their merits, and modify specific rules using the `rules` and `overrides` sections within `.eslintrc.js` . Always prioritize pragmatic, maintainable code over rigid rule enforcement. I suggest referring to ESLint’s documentation for a complete understanding of available rules and their configurations. The Airbnb style guide, while not formally documented as a rule list, can be examined by looking at the code in its GitHub repository. Also, reviewing the documentation of specific ESLint plugins (such as `eslint-plugin-import` or `eslint-plugin-react` ) is invaluable. By combining these resources, you can create an ESLint configuration that enforces a consistent style while still allowing for the necessary flexibility to create clean and maintainable code. Remember, configuration is not a one-time event, but an ongoing process throughout a project.
