---
title: "How can I run a formatter on save in VSCode without needing to manually save twice?"
date: "2024-12-23"
id: "how-can-i-run-a-formatter-on-save-in-vscode-without-needing-to-manually-save-twice"
---

Let's tackle this issue of formatting on save in vscode, a situation I've certainly encountered more than once in my development journey, particularly when dealing with diverse codebases and team preferences. It's definitely frustrating when a single save doesn’t trigger the formatting as expected, necessitating a second save, which introduces unnecessary interruption to the workflow. Let me share how I've tackled this, diving into the mechanics and configuration details with a focus on preventing that extra save.

The core of the problem lies in how vscode’s formatting mechanisms and its 'save' event interplay. Usually, when you press `ctrl+s` (or `cmd+s` on macOS), vscode detects this event, triggers its built-in or configured formatters, and then saves the formatted file. However, if this process isn't correctly set up, particularly with certain extensions or complex file structures, you can end up with the dreaded need for a double save. The culprit often isn't just one setting, but a confluence of factors, including the specific formatter you're using, how it’s integrated with vscode, and the presence of other extensions.

The first thing to address is ensuring that your chosen formatter is correctly configured and registered as the default formatter for the type of file you're working with. Let’s use prettier as a concrete example, it's a common choice and quite robust. You must have the prettier extension installed. Check your `settings.json` file (accessible via `ctrl+,` or `cmd+,` and then looking for the "open settings (json)" icon.) You should have, at least, something similar to the following:

```json
{
 "editor.defaultFormatter": "esbenp.prettier-vscode",
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[javascriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
   },
   "[typescriptreact]": {
      "editor.defaultFormatter": "esbenp.prettier-vscode"
   },
  "editor.formatOnSave": true,
  "prettier.requireConfig": true
}

```

Here’s what this configuration is doing:

*   `"editor.defaultFormatter": "esbenp.prettier-vscode"`: This line designates the prettier extension as the default formatter for the whole editor; other extensions will likely have their own identifying string that can be used here.
*   `"[javascript]": { ... }`, `"[typescript]": { ... }`, etc.:  These sections ensure that prettier is the formatter specifically for files of the given types. If you use other extensions, like ESLint with a formatter, this is where you would configure them.
*   `"editor.formatOnSave": true`: This is the crucial setting enabling formatting each time you save.
*   `"prettier.requireConfig": true`: This forces prettier to look for a config file (`.prettierrc`, `prettier.config.js`, etc.), making formatting behaviour consistent across environments and ensuring the rules are explicitly defined.

The key takeaway here is the explicit declaration for specific file types. If you only configure the default formatter, it may not always apply correctly, especially with extensions that use their own language server.

A common source of trouble is when other extensions interfere with the formatting process or register their own formatting capabilities which conflicts with your intended workflow. For example, an linter extension might apply some quick fixes on save that make the file 'dirty' again, which in turn doesn’t always trigger the formatter reliably on first save because vscode might think the file wasn't modified by the user. In these scenarios, it's crucial to carefully examine the configurations of each installed extension. This could mean disabling extensions one by one to isolate the culprit or configuring one extension to only provide linting and another to provide formatting.

Here’s a more complex case using a `settings.json` setup that includes both Prettier and ESLint for Javascript, which I’ve used when working on larger projects:

```json
{
   "editor.defaultFormatter": "esbenp.prettier-vscode",
    "[javascript]": {
        "editor.defaultFormatter": "esbenp.prettier-vscode",
        "editor.codeActionsOnSave": {
            "source.fixAll.eslint": true
        }
    },
    "[javascriptreact]": {
      "editor.defaultFormatter": "esbenp.prettier-vscode",
        "editor.codeActionsOnSave": {
           "source.fixAll.eslint": true
         }
    },
     "[typescript]": {
      "editor.defaultFormatter": "esbenp.prettier-vscode",
         "editor.codeActionsOnSave": {
           "source.fixAll.eslint": true
         }
      },
      "[typescriptreact]": {
        "editor.defaultFormatter": "esbenp.prettier-vscode",
        "editor.codeActionsOnSave": {
           "source.fixAll.eslint": true
        }
      },
    "editor.formatOnSave": true,
    "prettier.requireConfig": true,
    "eslint.format.enable": false
}
```

Notice the addition of `"editor.codeActionsOnSave"` within file-type specific settings. Here, we're instructing ESLint to fix all auto-fixable issues upon saving. This means that first prettier is applied, and after that ESLint auto-fixes any remaining formatting issues and reports the rest. Finally, `eslint.format.enable: false` ensures that eslint doesn’t attempt to format the code (since prettier is handling that part).

Another frequent cause for needing a double save is related to the `.gitattributes` file or related tooling. If you have differing line ending settings (`lf` vs `crlf`) between your local machine and the rest of the team and you have `autocrlf` enabled in git, each time you save, git may alter the line endings, which makes the file dirty, but only for git, not vscode, thus needing another save. These are often very specific to team preferences. Therefore, it's worthwhile to check if anything from git is making files seem dirty to vscode.

Lastly, there’s a rare instance where the formatter itself takes too long. While this isn't common with prettier, it can happen with more complex formatters or if you're working on particularly large files. When this occurs, vscode might save before the formatting is complete, necessitating another save to fully apply changes. Though not ideal, this can be managed by optimizing the formatter rules or considering a more incremental save and formatting approach in larger files, perhaps breaking the file down if it's large enough.

To delve deeper, I’d recommend the following resources:

*   **"Effective TypeScript" by Dan Vanderkam:** while focused on TypeScript, it often covers patterns that can improve overall settings and workflows, especially when dealing with linters and formatters.
*   **VSCode documentation for extensions:** The official vscode website has detailed information on how to configure the code editor, including specific documentation about settings and how extensions are integrated. Especially the page about code actions on save.
*   **Prettier documentation:** If using prettier, their official docs are the source of truth for configurations and integration.

By understanding the interaction between settings, extensions, and save triggers, and implementing the right configurations, the frustration of needing that second save can be avoided, leading to a much smoother and more efficient development experience. The examples provided and careful exploration of your configurations will likely resolve the issue, but more importantly will help you develop a more robust understanding of how vscode works under the hood.
