---
title: "Why is an async jest test with glob stuck?"
date: "2024-12-15"
id: "why-is-an-async-jest-test-with-glob-stuck"
---

hey there,

so you're having trouble with an async jest test that's using glob and it's just hanging, huh? i've been there, trust me. this specific combo, async tests with file system operations (which is often what glob is used for), can get sticky pretty fast. let's break down what's probably going on and how i'd tackle it, based on some of my own past headaches.

first, the core of the issue is often how async operations interact with jest's test runner. jest expects tests to either complete synchronously or resolve a promise when they're async. when you're dealing with glob, which can trigger asynchronous file system reads, you can easily get into a situation where your test doesn’t explicitly signal it's done to jest. and the test just… sits there waiting.

let's imagine a scenario, and this is from my personal history with file i/o in tests, where we were trying to process a bunch of mock markdown files for a static site generator, the good old days. we were using glob to find all the `.md` files in a directory, then read them and perform some transformation on them. here's roughly what the problematic code looked like initially:

```javascript
const glob = require('glob');
const fs = require('node:fs/promises');

test('parses all markdown files', () => {
  const files = glob.sync('./mock_files/*.md'); // this was the first mistake, sync

  files.forEach(async (file) => {
     const content = await fs.readFile(file, 'utf-8');
     // some assertions here on the content, let’s say:
     expect(content).toContain('# my title');
  });
});
```

this code looks pretty simple but it has a major problem: `forEach` doesn't play well with async functions. it kicks off each async `readFile` and doesn't actually wait for any of them to finish before declaring the test complete. jest sees that your test function doesn't return a promise, so it thinks it's done, even though the file reads and assertions are still happening (or rather *not* happening). it’s similar to how that feeling when you try to use an old usb drive with a usb-c port for the first time, not fun!.

the test just hangs because jest never gets the signal that those async operations inside the `forEach` are finished. this is a classic async gotcha.

the fix here is to use `async`/`await` correctly with something that understands promises. we can use `promise.all` for that, collecting all our promises and wait for all them to resolve:

```javascript
const glob = require('glob');
const fs = require('node:fs/promises');

test('parses all markdown files', async () => {
  const files = glob.sync('./mock_files/*.md'); // still sync, but it’s not the issue anymore

  const promises = files.map(async (file) => {
     const content = await fs.readFile(file, 'utf-8');
     // some assertions here on the content, let’s say:
     expect(content).toContain('# my title');
    });
  await Promise.all(promises);
});
```

now, the test function is marked as `async`, and we await the resolution of all the promises returned by our file reading operations. jest will only consider the test done when all of the promises complete. this is a big improvement but we can do even better.

and this goes a bit deeper into the problem we face with glob, and my own experience using it for file crawling. `glob.sync` is simple, but it blocks the javascript thread while its executing the file crawling, making your tests slow and unresponsive, in addition to that, you miss the main reason you are using a async test in the first place. let's rewrite this using `glob` asynchronous interface:

```javascript
const { glob } = require('glob');
const fs = require('node:fs/promises');

test('parses all markdown files', async () => {
  const files = await glob('./mock_files/*.md'); // now we are async

  const promises = files.map(async (file) => {
     const content = await fs.readFile(file, 'utf-8');
     // some assertions here on the content, let’s say:
     expect(content).toContain('# my title');
    });
  await Promise.all(promises);
});
```

in this last version, we use glob's async interface, which is non blocking and much better. the key change here is the `await glob('./mock_files/*.md');` which makes sure that the test waits for the file list to be available before running the file read operations.

here's a summary of common reasons for these hang ups when using async tests with glob:

1.  **not returning a promise:** like we saw in the initial example, if your test function is async, you have to return a promise or use `async`/`await` to tell jest when it is done. any missed await or not returning a promise means the test will complete, not caring about async operations in between.

2.  **incorrect async with loops:** using `forEach` with async operations can lead to races and hanging tests, since it doesn't wait for each loop to finish. using `promise.all` is the better approach to use for multiple async operations inside a loop.

3.  **synchronous file ops on an async test:** even if you have an async test, `glob.sync` is still synchronous. it blocks the main thread. while this was not the root of our original problem it's still something you should avoid to keep your tests fast.

4.  **unresolved promises:** if you're creating promises that never resolve (maybe due to some error or a logic problem), jest won’t finish running the test, leaving it in a hung state. always make sure promises resolve or reject to avoid hanging.

for resources i'd recommend looking into the following:

*   **understanding promises:** "javascript promises" by jake archibald is a fantastic resource, or really any good resource to deeply understand javascript promises.
*   **async/await patterns:** "effective typescript" by dan vanderkam. not specifically about async, but a good overall resource for typescript, and most important it has a very detailed explanation on promises.
*   **jest documentation:** the official jest documentation has very good sections on async testing and debugging, that can explain in a much better way the concepts related to async testing with jest. pay a good attention on the timeout settings, sometimes you need a custom timeout for the tests.
*   **node.js documentation:** the nodejs.org website contains complete documentation about the `fs` module, with particular attention to the async operations.

testing async operations and external tools like glob can be tricky but with some practice it becomes just another tool to use. if you find yourself in any problems with this again, try to break it down, checking for every await and promise resolution. good luck, and happy testing!
