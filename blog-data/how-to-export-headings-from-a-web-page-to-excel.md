---
title: "How to export Headings from a Web Page to Excel?"
date: "2024-12-15"
id: "how-to-export-headings-from-a-web-page-to-excel"
---

i've been there, battling with the html structure just to extract some decent data. headings, specifically, are a common pain. been there, done that, got the t-shirt - and a couple of scars from messy dom traversals. what you're asking isn't a walk in the park, but it's definitely doable with some javascript and a dash of ingenuity.

so, let's break down how i'd tackle exporting those headings from a webpage to excel.

first off, the core idea is this: we're going to use javascript to grab all the heading elements (h1, h2, h3, and so on) from the webpage's dom (document object model). then, we'll extract the text content from each of these elements and stick them into a javascript array. finally, we'll craft a csv string from this array, which excel can readily open.

now, for the code. remember, this is all client-side, so we're assuming you can run this javascript code in a browser environment. most likely through the developer console.

```javascript
function extractHeadings() {
  const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
  const headingData = [];

  headings.forEach(heading => {
    headingData.push(heading.textContent.trim());
  });

  return headingData;
}

function createCsvString(data) {
  let csvString = '';
  data.forEach(item => {
    csvString += `"${item}"\n`;
  });
  return csvString;
}

function downloadCsv(csvString, filename = 'headings.csv') {
  const blob = new Blob([csvString], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}


const headings = extractHeadings();
const csvData = createCsvString(headings);
downloadCsv(csvData);


```

let's walk through this snippet.

*   `extractHeadings()`: this function is the workhorse. it uses `document.querySelectorAll()` to snag all the heading elements (h1 through h6). then, it iterates through these headings, using `forEach`, and pushes the text content of each one (after trimming any leading or trailing whitespace with `trim()`) into the `headingData` array. the resulting array containing the text of each heading is returned.

*   `createCsvString(data)`: this takes the array of heading texts generated in the previous step. it iterates through that array, wrapping each heading text in double quotes (that way commas within heading will not cause an issue) and adding a newline `\n` character after each one to separate it in different lines. this string is returned and it's what we'll eventually save to a file.

*   `downloadCsv(csvString, filename)`: this one constructs a download link, downloads the file, and cleans everything up after. it creates a `blob` (binary large object) that represents the csv string, using `URL.createObjectURL` to generate a temporary url for the blob, then creates an `<a>` tag to simulate a link that is clicked and then removed from the document.

now, this works fine for simple cases. but what if the webpage has some funky nested structures? like, you have headings inside a div that you don't want to extract, or maybe you want to extract only headings inside a specific element? we can modify the selector. the selector in `document.querySelectorAll` is a powerful tool.

for example, say you only want headings inside a specific element with the id `content-area`.

```javascript
function extractHeadingsInContentArea() {
  const contentArea = document.getElementById('content-area');
    if (!contentArea) {
      console.warn('no element with id "content-area" found.');
        return [];
    }
  const headings = contentArea.querySelectorAll('h1, h2, h3, h4, h5, h6');
  const headingData = [];

  headings.forEach(heading => {
    headingData.push(heading.textContent.trim());
  });

  return headingData;
}
const headings = extractHeadingsInContentArea();
const csvData = createCsvString(headings);
downloadCsv(csvData);

```

in this revised code, we first grab the element with the id `content-area`, and we return an empty array if that is not found to prevent errors. we then select the headings within it using the `contentArea.querySelectorAll()` method. note that we are now calling `querySelectorAll` on the content area object instead of the entire `document`.

it's really powerful when you can change the selectors to target specific parts of the page. if you've worked with css before this will feel natural.

let me tell you, i once had a client, who i am not naming as i want to protect their privacy, whose website was built like spaghetti. i'm pretty sure the div tags were reproducing at night. the html was a nightmare. extracting the data i needed was like navigating a minefield, but the idea of using specific selectors did the trick and saved me hours of pain.

now, what if you want not only the heading text, but also their levels (h1, h2, etc.). things can get a bit more interesting. here's how i'd do it:

```javascript
function extractHeadingsWithLevel() {
  const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
  const headingData = [];

  headings.forEach(heading => {
    const level = parseInt(heading.tagName.substring(1));
    headingData.push({
      level: level,
      text: heading.textContent.trim()
    });
  });

  return headingData;
}

function createCsvStringWithLevel(data) {
  let csvString = '"level","text"\n';
  data.forEach(item => {
    csvString += `"${item.level}","${item.text}"\n`;
  });
  return csvString;
}


const headings = extractHeadingsWithLevel();
const csvData = createCsvStringWithLevel(headings);
downloadCsv(csvData);

```

in this version, `extractHeadingsWithLevel` function now also extracts the heading level by parsing the tag name, and it pushes an object containing both `level` and `text` to the array.

the function `createCsvStringWithLevel` takes the objects in the array and transforms them into a csv string, prepending the headers `level` and `text`, that way the resulting file will show each column with it's associated header.

regarding resources, instead of specific links (which tend to break over time, not sure why) i would recommend the books "javascript: the definitive guide" by david flanagan, if you are not used to js, and if you want to master selectors read "css: the missing manual" by david sawyer mcfarland, selectors are crucial for getting specific elements when working with the dom. and for the understanding of how the browser works the book "how browsers work: behind the scenes of modern web browsers" by tal valenta could be very useful. these three books can definitely help you. there is also a myriad of free content available on the mdn web docs website which is always useful when working with web technologies.

one last thing, have you ever noticed how the same bug always seem to show up again and again? if your code works the first time then you probably made a mistake. it's a bit of a joke, but it's true.

so, there you have it. three snippets to get you going. always double check the content of the website you are scraping and always be sure that you have the permissions to do so. extracting data from the web can be tricky, but it's also very rewarding once you get it working. let me know if something isn't working as expected.
