---
title: "react pdf highlighter how to use?"
date: "2024-12-13"
id: "react-pdf-highlighter-how-to-use"
---

 so you're asking about how to use a React PDF highlighter right I've wrestled with this one before trust me. Its not always straightforward and you definitely can stumble down rabbit holes if you aren't careful. Been there done that got the t-shirt several times. I'm talking late nights debugging PDF parsers and getting intimately acquainted with coordinate systems. You know the drill.

First things first the React part is pretty standard. You'll be building components probably a parent component to handle the PDF rendering and maybe a child component to handle the highlighting itself. You can totally roll your own but let's be real here reinventing the wheel is a mug's game. There are a few libraries out there that do the heavy lifting so we don't have to write a PDF parser from scratch thank heavens for that. I've tried doing that once I don't recommend.

The most common library I've seen and the one I tend to gravitate towards is react-pdf. Its not perfect nothing is but it's pretty robust and well maintained. So I'd suggest starting there for the PDF rendering itself. You'll handle the PDF loading and page rendering using the `<Document>` and `<Page>` components and this gets you your base layer.

Now onto the juicy bit the actual highlighting. This is where things get tricky. You can't just slap a div over the text it won't work. The PDF is usually rendered on a canvas and you have to handle text selection programmatically and map that to pixel locations on the canvas and draw over this with some highlight rectangles. This means working with text layers and PDF coordinates. Think of it as a coordinate mapping nightmare a good headache is what I call it.

A good starting point is to look for selection libraries that interface nicely with the PDF text layer. There is a library that does a really good job with this but it doesn’t have a clear name that makes sense the library name is react-pdf-highlighter. Yeah its weird. It uses react-pdf to handle PDF loading and rendering and it then has components for handling text selection and drawing rectangles over the text. It works by using react-pdf's text layer to get the bounding boxes of selected text and then using those boxes to draw the highlight rectangles. You can install it using npm like this: `npm i react-pdf react-pdf-highlighter`.

Here's a very basic example. This assumes you've already got a react project running:

```javascript
import React, { useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { HighlightText } from 'react-pdf-highlighter';
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.js`;

function App() {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
  };

  return (
    <div>
      <Document
        file="/path/to/your/document.pdf"
        onLoadSuccess={onDocumentLoadSuccess}
      >
        <Page pageNumber={pageNumber}>
            {({ page, viewport }) => (
                <HighlightText page={page} viewport={viewport} onSelection={console.log} />
            )}
        </Page>
      </Document>
    </div>
  );
}

export default App;
```

In this basic example we are using Document and Page of the react-pdf library to render the pdf. Then, the `HighlightText` component of the `react-pdf-highlighter` library that takes the `page` and `viewport` from react-pdf and provides the user with the selection and highlight capability. The `onSelection` callback we are using here outputs the selected bounding boxes to the console. You can expand on this function to handle drawing rectangles on your pdf file.

Now things are not always this simple there will be some configurations that you need to do. The `react-pdf` library uses web workers to handle parsing PDF files. So you need to specify a `workerSrc` for pdf.js otherwise the library throws an error about the worker not being initialized. You'll also need to have PDF file in place for that code snippet to work. I am using `/path/to/your/document.pdf` in this case remember to change that.

You can then expand the basic example by storing the selections in the state and rendering the highlights programmatically. So let's look at a more elaborate example:

```javascript
import React, { useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { HighlightText } from 'react-pdf-highlighter';
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.js`;


function App() {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [highlights, setHighlights] = useState([]);

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
  };

 const handleTextSelection = (selection) => {
    setHighlights([...highlights, selection]);
  };

  return (
    <div>
      <Document
        file="/path/to/your/document.pdf"
        onLoadSuccess={onDocumentLoadSuccess}
      >
        <Page pageNumber={pageNumber}>
          {({ page, viewport }) => (
              <HighlightText
                page={page}
                viewport={viewport}
                onSelection={handleTextSelection}
                highlights={highlights}
                isSelectable
                enableAreaSelection
                highlightStyle={{backgroundColor: "yellow", opacity: "0.5"}}
              />
            )}
        </Page>
      </Document>
    </div>
  );
}

export default App;
```

Now we are actually drawing the highlights on our pdf file. We are storing all the selections and the `HighlightText` component takes these as an input and draws the highlights. I have also enabled `isSelectable` and `enableAreaSelection` for more advanced interaction and also added a basic style. You can change the style of your highlighted area with this `highlightStyle` prop. The `selection` object in the `handleTextSelection` function stores all the bounding box coordinates along with the selected text and page number.

Of course this is a simplified example. In real world applications you’ll probably have to handle multiple pages selection persistence saving highlights to a database or local storage etc. Think about how you'd handle large PDFs that's something that can trip you up. You might have to use virtualized rendering for page loading optimization especially with extremely large documents.

There's a lot of complexity hiding behind what seems like a simple UI element. The PDF format itself can be complex there can be a lot of variation between different PDF generators which makes it hard to maintain uniformity between the highlighted boxes and text alignment. Then there's the issue of different browsers and operating systems all handle rendering canvas differently which can lead to inconsistencies in rendering. I've spent hours debugging rendering issues across different platforms. Fun times for sure… kind of. I also had one incident where my PDF worker was having an internal philosophical debate and refused to cooperate. Yeah the issue was a missing slash in my worker path. Turns out computers do have a sense of humor after all just not a great one.

One thing you need to consider is that the highlight selection is very tightly coupled with the react-pdf library. If that library changes in a major way your highlighting functionality can get messed up so you need to be constantly updating and checking for such regressions. Version compatibility between the libraries also matters. For example I remember having a compatibility issue between the latest react-pdf library and the react-pdf-highlighter library which took me almost a day to figure out. You need to keep those in sync.

Let's talk about more advanced highlighting. You could create annotations using some kind of custom component and render them on the PDF. This will require a more intricate understanding of the PDF format itself and its annotation structure. You might need to explore PDF libraries that go beyond basic rendering if you want to have annotations that are part of the PDF document itself. There's more to the PDF format than meets the eye trust me.

And finally about saving the highlights. You will probably need to serialize the highlights in some format to be able to persist it in a database or local storage. You could just save the selections but if you need to recreate the whole pdf with these highlights you will need to store the page number along with the bounding box coordinates and selection content. You need to have a well-defined data structure for this process. Here's a example snippet of saving highlight selections to localstorage:

```javascript
const saveHighlights = () => {
  localStorage.setItem('pdf-highlights', JSON.stringify(highlights));
}

const loadHighlights = () => {
  const savedHighlights = localStorage.getItem('pdf-highlights');
  if (savedHighlights) {
    setHighlights(JSON.parse(savedHighlights));
  }
}

// Call this on component mount
useEffect(() => {
  loadHighlights();
}, []);

// Call this to store highlights on selection
const handleTextSelection = (selection) => {
  setHighlights([...highlights, selection]);
  saveHighlights();
};
```

I have only given a rough overview here. There is much more to react pdf and react-pdf-highlighter than what I could cover here. I suggest starting with the documentation of react-pdf and the example code of react-pdf-highlighter. I can recommend a few books and papers though. For a deeper understanding of the PDF structure I would recommend looking at the PDF reference documents released by Adobe. The PDF specification is your bible in that case. For a broad overview of document processing you can read “Document Engineering: Managing Documents Throughout Their Lifecycle” by Robert J. Glushko and Tim McGrath it has good information on Document formatting and different markup languages used for document processing. Finally “Web Application Architecture: Principles Protocols and Practices” by Leon Shklar and Richard Rosen provides a good overview of Web application structure which is useful when you are designing a system like this.

 that's it. Hope this gives you a good starting point. Remember to take it one step at a time and don't be afraid to dive deep when needed. Good luck and let me know if you have any other questions and maybe I will give some good answers. I probably would given the amount of time I’ve wasted figuring this stuff out.
