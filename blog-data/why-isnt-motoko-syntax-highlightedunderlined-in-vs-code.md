---
title: "Why isn't Motoko syntax highlighted/underlined in VS Code?"
date: "2024-12-23"
id: "why-isnt-motoko-syntax-highlightedunderlined-in-vs-code"
---

Alright,  The lack of syntax highlighting for Motoko in Visual Studio Code is a common frustration, and I’ve definitely been there. In the early days of my work with the Internet Computer, this very issue was a constant annoyance. It's not that VS Code actively dislikes Motoko; rather, the process of supporting a language with proper syntax highlighting and code analysis is a significant undertaking. There’s a fair bit going on behind the scenes, more than just a simple color change for keywords.

Essentially, VS Code relies on what are known as language extensions to provide that rich coding experience we've grown accustomed to. These extensions are, at their core, specific software packages that define how a given programming language should be handled within the editor. They provide not only syntax highlighting, but also things like autocompletion, error checking (linting), and often even debugging support. The extension for a language needs to be explicitly developed, tested, and maintained.

For a language like Motoko, which is still relatively new compared to established languages like JavaScript or Python, creating a robust and feature-complete VS Code extension is a non-trivial task. It requires a thorough understanding of Motoko’s grammar, its nuances, and its specific type system. It also requires dedicated time and resources from the developer or development team maintaining the extension. So, if you’re not seeing Motoko highlighted, it's primarily because the necessary extension is either not installed, outdated, or, in some cases, still in active development or doesn’t exist fully at all.

Let’s dig into the technical underpinnings. A VS Code language extension typically involves several components. One major component is a TextMate grammar file. These files are usually in JSON format and use regular expressions to match different parts of your code: keywords, identifiers, string literals, comments, and so on. Each of these matched parts is then assigned a different 'scope' which is the name used by a visual theme to determine the specific color or style to apply. That's how you get keywords in a language like `let` or `func` to appear a different color than variables or strings. A language extension will generally include these grammar rules.

Alongside the grammar definition, an extension often includes other helpful features such as code completion based on static analysis or even integration with a language server to provide real time checking and feedback. A Language Server Protocol (LSP) implementation is more sophisticated, allowing an external process (the language server) to handle parsing, semantic analysis, and code generation. The benefit is that you can share a language server among several editors and improve development time for language features because you only have to make changes in a single codebase.

Let me illustrate with a few examples, using simplified (and fictional) examples of how a grammar rule might look. Imagine, for a completely fictional language called 'fictLang':

**Example 1: Simple Keyword Highlighting (Fictional)**

Let's pretend the JSON configuration for a fictional language called "fictLang" included the following to highlight a keyword 'create':

```json
{
  "scopeName": "source.fictlang",
  "patterns": [
    {
      "name": "keyword.control.fictlang",
      "match": "\\bcreate\\b"
    }
  ]
}
```

In this example, if we had a file containing code like `create object`, the 'create' keyword would be highlighted according to the theme's "keyword.control" settings. In a real Motoko grammar, there would be a large number of such rules for all the Motoko keywords and other language constructs.

**Example 2: String Literal Highlighting (Fictional)**

Here's a fictional example of a rule to highlight string literals in our `fictLang`:

```json
{
  "scopeName": "source.fictlang",
    "patterns": [
      {
        "name": "string.quoted.double.fictlang",
        "begin": "\"",
        "end": "\"",
         "patterns": [
           {
             "name": "constant.character.escape.fictlang",
             "match": "\\\\."
           }
        ]
      }
    ]
}
```

This would make strings like `"hello, world"` appear in a different color. It also includes an example of how to handle escape characters (`\\`) within a string, highlighting them differently as well. Note how more complex cases require a nested structure.

**Example 3: Function Declaration (Fictional)**

Here’s how we might highlight a function declaration:

```json
{
  "scopeName": "source.fictlang",
  "patterns": [
    {
      "name": "keyword.control.fictlang",
      "match": "\\bfunction\\b"
    },
    {
      "name": "entity.name.function.fictlang",
      "match": "\\b\\w+\\b(?=\\s*\\()"
    }

  ]
}

```

Here we are using regular expressions to match "function" keywords and then matching a subsequent identifier if it is followed by an open parenthesis `(`. This example captures the common practice of highlighting the function's name.

These are, of course, highly simplified examples. A real Motoko grammar would involve significantly more complexity, capturing not only basic syntax but type definitions, actor-specific features, and more. The lack of proper syntax highlighting typically arises from the absence of a comprehensive and well-maintained grammar file or related language server integration.

So, what can you do while we await wider community support and mature extensions? First, check the VS Code Marketplace for any community-developed Motoko extensions. Sometimes these are under development, and the name may not be obvious. A search for `motoko`, `internet computer`, `dfinity` or similar is a good idea. If one exists, install it, and make sure it is updated regularly. Keep a watch on the Dfinity developer forums and release notes for any updates or official tooling.

Also, to truly understand the complexities involved, consider delving into resources like “TextMate Language Grammars” (available on the TextMate website or GitHub), or exploring the “Language Server Protocol Specification” for a deeper look at how these systems operate. These resources can significantly enhance your understanding of what’s involved in building a quality VS Code language extension. While it can be tedious to implement, the community always appreciates individuals who put in the effort to contribute to better tooling around the various languages we use.

Ultimately, the absence of robust syntax highlighting for Motoko isn't due to any inherent difficulty in VS Code but stems from the resource and time-intensive process of creating and maintaining a well-designed extension for a language like Motoko. This involves meticulously crafting grammar rules, potentially developing a language server, and adapting to the evolving language specification. Patience and a proactive approach to community resources and updates can mitigate the current lack of fully featured tooling.
