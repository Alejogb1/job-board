---
title: "How to setup Prismjs inside rich_text_area (Rails 7)?"
date: "2024-12-14"
id: "how-to-setup-prismjs-inside-richtextarea-rails-7"
---

alright, let's tackle this prismjs in a rails rich_text_area thing. i've been down this road before, and trust me, it can get a bit fiddly if you're not careful. it's one of those things that looks straightforward on paper but then throws a few curveballs.

first, let’s talk about why this is sometimes tricky. rails’ actiontext, which powers `rich_text_area`, generates its html on the backend and then throws it into the browser. this means prismjs, which operates on the client side, needs to be aware of the code blocks created by actiontext after rails is done with it. it’s not like rendering a simple `<pre><code>` block that’s there from the start; we have to make sure prism runs *after* actiontext does its thing.

now, let's get into it. here's how i usually set it up, broken down step by step, with some examples from my own past struggles:

**step 1: include prismjs**

this part is pretty straightforward. you need to include prismjs's css and javascript files in your rails app. i personally prefer to manage these through a package manager like yarn or npm, because it makes updates easier. so, first, add the dependencies.

```bash
yarn add prismjs
```

after that, you'll need to import the css and js files. i typically do this in my `app/javascript/application.js` (or equivalent in your setup):

```javascript
// app/javascript/application.js
import 'prismjs';
import 'prismjs/themes/prism.css'; // or any theme you like, check prismjs's site for themes

// optionally import additional languages
import 'prismjs/components/prism-ruby';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-python';

// you may want other languages, search their component repo
```

notice the imports of the specific language components. prismjs doesn’t load every single language highlighter by default, so include the ones you plan to support. my past projects were primarily ruby, javascript and python based so i normally load those three languages. this helps keep the final js bundle a bit smaller than loading everything. there are others though if needed.

**step 2: triggering prismjs on page load and updates**

here's the core of the problem. actiontext injects content dynamically, and this is where most people, including myself initially, get stuck. a simple `prism.highlightAll()` on document load isn’t enough; it only catches the initial content. anything actiontext generates later is missed. so we need to trigger highlight each time.

here's how to handle that:

```javascript
// app/javascript/application.js

import 'prismjs';
import 'prismjs/themes/prism.css';
import 'prismjs/components/prism-ruby';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-python';

function highlightCodeBlocks() {
  document.querySelectorAll('pre code').forEach((block) => {
    prism.highlightElement(block);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  highlightCodeBlocks(); // initial highlight
  document.addEventListener('turbo:load', () => {
      highlightCodeBlocks(); // handles turbo page changes
  });
});

```

in this script, `highlightCodeBlocks()` does the actual highlighting. it finds all `pre code` blocks (actiontext's default output) and calls `prism.highlightElement()` on each one. the `DOMContentLoaded` event ensures that the code runs only after the initial page content has fully loaded, and `turbo:load` is to make sure that it is called also after page is changed using turbo.

**step 3: styling the code blocks**

now, the styling part is important. prismjs themes mainly control the syntax highlighting colors, but you might still find some `pre` and `code` styling missing, or not to your preference. actiontext itself doesn’t inject class names that prismjs is expecting to work seamlessly. i remember spending an afternoon trying to figure out why my code blocks weren't scrolling horizontally. so, here's the css i use, normally i load it in `app/assets/stylesheets/application.scss` file:

```scss
// app/assets/stylesheets/application.scss

pre {
  overflow-x: auto;
  padding: 1em;
  background-color: #f5f2f0; // light background
  border: 1px solid #ddd;
  border-radius: 4px;
  white-space: pre; /* keeps spaces */
}

pre code {
  display: block; /* needed for highlighting in prism */
}
```

this basic css adds some padding, background color, and borders to your code blocks. the critical part is `overflow-x: auto;` this makes sure any long code snippets have horizontal scrollbars rather than overflowing their container. the `white-space: pre` makes sure your spaces are not compressed. `display: block` is important for prism to style correctly the code part in the `pre` tag.

**step 4: troubleshooting common issues**

*   **not all languages highlighted:** double check you have imported all the languages you need in `app/javascript/application.js` or where ever you keep your main javascript. prism doesn't automatically load every single one.

*   **code not highlighted:** this usually means prism is not getting triggered after the actiontext content is loaded. make sure you have `turbo:load` event and the `DOMContentLoaded` event handler in place. inspect the html you have generated and see if `pre code` is there.

*   **weird scrolling behavior:** this is often fixed by setting proper css for `pre` blocks as shown before. be mindful that if you have a very large block of code the height can be a problem.

*   **theme not applied:** if your theme is not showing up properly check if you have the correct css paths in the application.js file.

*   **prismjs not initialized:** one time i forgot that i had a different application.js in a different location and wasted a lot of time and coffee because of that, always double-check that you are editing the file you think you are.

*   **turbo issues:** if you are using turbo make sure that the event listener is in place, otherwise if you navigate in your app, prism might not re-render when you navigate in your app.

this was not exactly a laughing matter, more like a cry for help in the middle of the night, lol.

**resource recommendations**

rather than giving direct links to pages, which may be ephemeral. i suggest getting the prismjs docs. they are well-written and clear. also there are plenty of books on front-end development that include a whole section on javascript and css, those may come handy as well.

that is pretty much what i do in most of my projects. this approach gives you syntax highlighting in your rich text editor and handles dynamic content changes. feel free to ask if you get stuck. i've been there, multiple times, and i’m happy to share the hard-earned knowledge.
