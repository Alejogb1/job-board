---
title: "Why is an async jest test with a glob stuck?"
date: "2024-12-15"
id: "why-is-an-async-jest-test-with-a-glob-stuck"
---

alright, so you're hitting that fun wall where your jest async test, using a glob pattern, decides to take an extended coffee break and never quite finish. i've been there, staring at the terminal, wondering if i’d accidentally summoned a time-dilation field in my test suite. it's a classic symptom of a few things, and it usually boils down to misunderstandings about how async operations, globbing, and jest all interact, especially when filesystems are involved, and i recall i was stuck with this issue back in the old days of node version 10, it feels like another lifetime.

first off, let’s clarify what might be going on. when you use a glob pattern – think something like `'./src/**/*.test.js'` – you're telling node's file system api to find all files matching that pattern. this operation itself is *synchronous*. the glob library returns a list of matching paths. it doesn't know if those files contain tests, or even if the files are completely loaded, its job is just to find files matching your string.

now, once jest starts running, it loads each of these files. if any of the tests within these files are asynchronous (which, given your question, i'm guessing they are), jest needs a way to know when each of those async tasks is complete so it can say whether that test passed or failed. this is usually where the problem occurs in your case if the test gets stuck.

if you're using async/await inside your tests, that's fine, jest understands that, but, what about the async operations *inside* the files you are globbing over. or if they are just promises that you're not resolving? jest won’t magically understand if your loaded test file is actually doing something async which never ends. for example consider a file that you have, and you are globbing over and loading:

```javascript
// my-async-test.js
const asyncTask = async () => {
  // this will never resolve unless you await or return it somewhere
  new Promise(() => {
      // some infinite operation here
  })
}

test('should hang', () => {
  asyncTask();
  expect(1).toBe(1); // this line passes but jest will still hang if you loaded this file
});
```

here, the test will actually pass, but the program will never finish as the promise will never resolve. therefore, if you loaded this with a glob pattern, and didn't await on that, jest will get stuck.

this example highlights how an unhandled promise within a test file that is included in your glob can lead to this stuck behavior. a similar issue arises if your asynchronous operation isn’t returning a promise or isn't resolving correctly. jest relies on the fact that promises returned from `async` test functions resolve at the end, or that the `done()` callback is called when you use that approach.

i remember when i had a similar issue. it involved a test that was simulating a network request. i used `node-fetch`, it worked as a normal test but when running a large glob, tests started not completing and jest timed out with no stack trace. i thought `node-fetch` was faulty. the thing was that somewhere, some tests were missing an `await`. node-fetch worked fine, i was calling an async operation in tests and not awaiting them. the files would be loaded, do their operation but never finish. it took me ages to find out this was the issue.

the most common cause i saw back then and even now is that you have an asynchronous operation that you've started but forgot to await on it, or return it. the most common example is calling an async function inside a test, or in an imported file, and forgetting to return the promise back to the jest test framework. another thing to look at is if you have callbacks that aren't called, which should be easily fixed.

a fix can be explicitly making sure that you have awaited the async operation inside the test or inside the files that are loaded from glob patterns. here is a good way of writing a test that you load with glob:

```javascript
// correct-async-test.js
const asyncTaskThatResolves = async () => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve('done!');
    }, 100);
  });
}

test('should finish', async () => {
  const result = await asyncTaskThatResolves();
  expect(result).toBe('done!'); // this is fine, test will pass and complete
});

```

in this case, the test will pass, and because we awaited `asyncTaskThatResolves()` it will correctly complete. the key here is making sure you have awaited and are returning the promise from your async functions.

a common way of making a test 'stuck' without realizing it is to have functions that have side effects. functions like `setImmediate` or `setTimeout` and forgetting to `clearTimeout` or `clearImmediate` in the test might result in jest not completing as the test execution context is not finished as those functions will keep the execution context open. consider this example

```javascript
// bad-async-test.js
test('should not finish', () => {
  const id = setImmediate(() => {
    // do something that will run but never finish
  });
  // there is no clearImmediate or awaiting on this, so jest never finishes
});
```
in this case, the `setImmediate` function will never complete the test, thus the test will keep running and jest will appear stuck and time out. the solution would be to use `clearImmediate` or if its a `setInterval` or `setTimeout` use `clearInterval`, `clearTimeout` respectively.

another thing i've noticed is that sometimes the issue is with the file system itself. if you have a huge number of files, the initial file system traversal that the globbing library does can take a while. even though the glob library is synchronous, it can be slow if you have thousands of files. therefore it can give a false positive on tests being stuck, if the globbing took some time to resolve and your tests have a short timeout time in jest. you would need to increase timeout to see if this is actually the issue. though it's not really a "stuck" test, but a very long processing time due to the glob pattern resolving many files. i always keep my tests in a smaller test folder structure to prevent this and i would recommend that too. i remember when i had my test files in the same folder as the source code, i thought that was fine, it was not and took me ages to realize this was also a source of performance problems, especially in CI/CD pipelines that had to pull and process a large tree of tests.

to debug this, i recommend starting with a small subset of your tests and then adding them back gradually to see which test files are causing the issue. you can reduce the glob pattern or comment out a number of tests, until the tests do not get stuck. that way you know which files are causing the issue, and then start debugging from there using node debuggers like `node --inspect-brk`.

if you are writing tests in files that you loaded with a glob pattern, always remember to return or await the promises from async functions, and make sure that any timers like `setImmediate`, `setTimeout` and `setInterval` are cleared. and i'll tell you a joke, it's a bit technical though, why did the async function break up with the promise? because it felt like it was always being awaited upon.

as for resources, you might want to check out "javascript: the definitive guide" by david flanagan, it's a good resource on understanding how promises and async operations work under the hood in javascript. for glob specific knowledge i recommend reading the documentation on the `glob` library you're using, understanding how it works under the hood can help a lot. the jest documentation also is a must have, if you haven't read the jest async documentation i recommend doing so to understand how jest treats async tests. finally for performance tips and tricks relating file system traversal consider books on node.js performance like "node.js design patterns" by mario casciaro and luciano mammino, it's great for these kind of issues. these resources helped me when i was beginning with these issues and i believe they can help you too. good luck.
