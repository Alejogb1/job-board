---
title: "Is jsx support enabled for the Draftail getting started example?"
date: "2025-01-30"
id: "is-jsx-support-enabled-for-the-draftail-getting"
---
The Draftail getting started example, as provided in the official Wagtail documentation, does not inherently include JSX support within its text editor component. This is a crucial detail because, while React, the library that often uses JSX, powers Draftail, the default configuration expects plain JavaScript objects representing rich text content, not React components directly. My experience building custom rich text editors for a large content platform has highlighted the separation necessary between the data model Draftail uses and the potential for JSX-powered presentation later in a separate rendering process.

The core of Draftail lies in its block-based data structure, specifically the concept of ‘entities’ and ‘blocks’ that describe formatting (like bold, italics) and more complex elements (like links, images). When a user interacts with the editor, Draftail doesn't immediately create DOM elements with React; instead, it constructs and modifies a JavaScript object representing that content. This object is then stored and retrieved from the database. The rendering of this data into a visually rich display happens at a later stage, and it's this later stage where JSX might be involved, not within Draftail itself.

To illustrate, consider the simplest example of creating a Draftail text field. Here's a snippet demonstrating the basic setup as provided in the Wagtail documentation, devoid of any JSX integration:

```javascript
import React, { useState } from 'react';
import { DraftailEditor, BLOCK_TYPES } from 'draftail';

function MyDraftailEditor() {
  const [value, setValue] = useState(null);

  const handleEdit = (newValue) => {
    setValue(newValue);
    console.log(newValue); // View the data object structure
  }

  return (
    <DraftailEditor
      value={value}
      onEdit={handleEdit}
      blockTypes={BLOCK_TYPES.filter(type => type.label !== 'unstyled')}
    />
  );
}

export default MyDraftailEditor;
```

This code implements a basic Draftail editor, showing that the `DraftailEditor` accepts a value prop representing the current content as a JavaScript object; it doesn’t expect JSX directly. The `handleEdit` function displays the structure of the data as it is modified. Note the `blockTypes` prop, which dictates available editing options. This is essential for customization but doesn't affect whether JSX is within the editor's core logic. Observing the output within the browser console after editing reveals how content is stored as a set of nested JavaScript objects and arrays. This is the underlying data format, not JSX.

To introduce a custom functionality that _might_ involve JSX, one might consider using Draftail's `inlineStyles` or `entityTypes` functionalities. These allow for extension, but still don't provide a straightforward path for the usage of JSX directly within the editor. Here's an example of creating a custom ‘highlight’ style that changes text to yellow. Although not JSX within the editor, the data reflects this modification:

```javascript
import React, { useState } from 'react';
import { DraftailEditor, BLOCK_TYPES, INLINE_STYLES } from 'draftail';

const HIGHLIGHT_STYLE = {
  style: 'HIGHLIGHT',
  label: 'Highlight',
  type: 'style',
  icon: 'highlighter',
  description: 'Highlight text',
};


function MyDraftailEditor() {
  const [value, setValue] = useState(null);
  const handleEdit = (newValue) => {
      setValue(newValue);
      console.log(newValue);
  }

    const customInlineStyles = [
        ...INLINE_STYLES,
        HIGHLIGHT_STYLE
    ];

  return (
    <DraftailEditor
      value={value}
      onEdit={handleEdit}
        blockTypes={BLOCK_TYPES.filter(type => type.label !== 'unstyled')}
       inlineStyles={customInlineStyles}
    />
  );
}

export default MyDraftailEditor;
```

This extended snippet introduces a custom `HIGHLIGHT_STYLE` that can be used within the Draftail editor. When applied to some text, the resulting Draftail output, viewable via the browser console after editing, will now contain references to this 'HIGHLIGHT' style. While this shows how custom styles are encoded in data, it still underscores that the editor data itself is not JSX; it’s simply data that includes instructions, in this case, for the highlighting style.

Ultimately, the transformation from the Draftail data object to visually formatted text on the frontend is a separate, independent process. It's within that specific rendering step where one could potentially utilize JSX, if desired, for creating complex user interfaces or components. To illustrate this separation, let's look at a simplified example of converting the Draftail content to HTML, keeping in mind this is outside of the editor’s internal logic:

```javascript
import React from 'react';
import { render } from 'draftail';


function DraftailOutput({ rawContent }) {
   const htmlOutput = render(rawContent);

   return (
      <div dangerouslySetInnerHTML={{ __html: htmlOutput }} />
    );
}

export default DraftailOutput;
```

This `DraftailOutput` component receives the raw Draftail data and renders it as HTML. The `render` function transforms the data structure to HTML and then uses `dangerouslySetInnerHTML` to render the markup. It's important to note that while React (and JSX potentially) is used here, it's within a context completely separate from the Draftail editor itself. This emphasizes the core concept that the editor operates on a JavaScript data structure, whereas JSX is involved on how that data is displayed.

To summarize, Draftail does not inherently accept or process JSX; instead, it utilizes its internal data object structure for managing content. While React and, by extension, JSX, could be employed for the eventual display of Draftail content, that’s separate from Draftail’s actual input and storage process. Therefore, the getting started example, or Draftail itself, does not have JSX support enabled by default, nor should it in its data manipulation layer.

For those wanting to dive deeper into extending Draftail, it is imperative to consult:
*   The official Wagtail documentation provides comprehensive information on Draftail’s API and available customization options.
*   The Draft.js repository on Github, which serves as Draftail’s foundation, includes valuable technical details on the underlying editor data model.
*  Resources covering React and JSX lifecycle events can enhance understanding of frontend content rendering.
