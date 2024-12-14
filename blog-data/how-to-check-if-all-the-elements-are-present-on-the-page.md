---
title: "How to check if all the elements are present on the page?"
date: "2024-12-14"
id: "how-to-check-if-all-the-elements-are-present-on-the-page"
---

alright, so you're asking about how to make sure all the expected elements are actually there on a webpage, right? i’ve been down that road, more times than i'd like to remember, and i've got some stuff to share. it’s a pretty common problem when you're doing any kind of front-end testing or scraping. it’s frustrating when a script breaks because an element didn't load or got renamed.

basically, there are a few ways to approach this, and the best one really depends on the specifics of your situation, and honestly, what tools you're already using. i'll give you a couple of examples using different techniques, going from straightforward to a little more involved. i’m assuming you are comfortable with javascript, since this is the way i did it always. i'm also assuming that you already have a way of interacting with the page, like using puppeteer, playwright, or selenium, or even vanilla javascript with a browser environment. i mostly use puppeteer, because it is fast.

first, the simplest method that i used for a long time when i had a side project where i was fetching data from a page, was just to use css selectors and check if the length of the results of those selector calls was equal to what i expected. it’s direct, it’s easy to read, and you can implement it quickly. here is an example:

```javascript
async function checkElementsSimple(page, selectors) {
    for (const selectorData of selectors) {
        const {selector, expectedCount} = selectorData;
        try{
             const elements = await page.$$(selector);
             if (elements.length !== expectedCount) {
              console.warn(`mismatch found for selector ${selector}: expected ${expectedCount} , found ${elements.length}`)
               return false;
            }
       } catch(error)
       {
         console.error(`error found for selector ${selector}: ${error}`)
         return false
       }
   }
    return true;
}


// example of usage:
// here, the page would be an instance of pupeteer page, but can be replaced with any implementation of an abstraction
// over the browser like playwright or similar
// this list should be adjusted to your case, with the selectors and expected number of elements
const selectorsToCheck = [
 {selector: '.product-card', expectedCount: 12},
 {selector: '#main-navigation li', expectedCount: 5},
 {selector: 'button.submit-button', expectedCount: 1},
];


(async () => {
  const allElementsPresent = await checkElementsSimple(page, selectorsToCheck);
  if (allElementsPresent) {
    console.log('all elements are present!');
  } else {
    console.error('some elements are missing or have unexpected counts.');
  }
})();
```

this method has served me well for many small scripts i wrote to check several websites daily, it worked and got the job done, and most of the time you don’t need much more than that. i remember when i was at university, when i used selenium for the first time, it took me 2 hours to run this kind of script on a very slow windows virtual machine, and i ended up spending more time debugging why it was so slow than actually coding the script itself. in reality it was just a matter of bad configurations with the virtual machine. but those are past times, i learned a lot from those mistakes.

the drawback with this method is that if the page has dynamically generated elements, things can get tricky. if the page uses javascript to render stuff after the initial load, those elements might not be available when you execute your `$$` calls. this is where waiting strategies can help. you can wait for a specific element to be present on the page before starting your checks. pupeteer offers several ways to do that, including `waitForSelector`. i have used the `waitForSelector` method multiple times and can say that it's reliable. but it's also a good idea to wrap that with a retry and timeout mechanism in case a selector fails. because sometimes javascript will not load the selectors as you expect. imagine if some javascript error in the page would prevent the rendering of some of the elements, this would cause your check to fail incorrectly and a timeout or retry would be really beneficial.

here's an example with a timeout and retry mechanism:

```javascript
async function waitForElementWithRetry(page, selector, timeoutMs = 5000, maxRetries = 3) {
  let retries = 0;

  while (retries < maxRetries) {
      try {
           await page.waitForSelector(selector, {timeout: timeoutMs});
           return true;
       } catch(error) {
         console.warn(`wait for selector ${selector} timed out, retrying... attempt ${retries + 1} of ${maxRetries}`);
          retries++;
          await new Promise(resolve => setTimeout(resolve, 1000)); // small delay between retries
       }
  }
  console.error(`max retries reached while waiting for selector: ${selector}`);
  return false;

}

async function checkElementsWithWait(page, selectors) {
    for (const selectorData of selectors) {
        const {selector, expectedCount} = selectorData;
        const elementPresent = await waitForElementWithRetry(page, selector);
          if(!elementPresent){
            console.error(`element selector ${selector} was not found.`)
            return false
          }
        try{
          const elements = await page.$$(selector);
           if (elements.length !== expectedCount) {
            console.warn(`mismatch found for selector ${selector}: expected ${expectedCount} , found ${elements.length}`)
            return false
          }
       } catch(error)
       {
         console.error(`error found for selector ${selector}: ${error}`)
         return false
       }
    }
   return true;
}


// example of usage:
// this list should be adjusted to your case, with the selectors and expected number of elements
const selectorsToCheck = [
 {selector: '.product-card', expectedCount: 12},
 {selector: '#main-navigation li', expectedCount: 5},
 {selector: 'button.submit-button', expectedCount: 1},
];


(async () => {
  const allElementsPresent = await checkElementsWithWait(page, selectorsToCheck);
  if (allElementsPresent) {
    console.log('all elements are present!');
  } else {
    console.error('some elements are missing or have unexpected counts.');
  }
})();

```

this version is more robust and can handle slow loading pages and it is what i usually use these days. using a simple retry strategy with a small timeout, usually is more than enough to handle most of the cases. this avoids random errors with your test, or with your scripts. i have seen many people simply throwing errors without retries and this is a big mistake in any scripting system, because it can cause headaches. i've learned that from trying to debug these situations.

now, for the final example, if you are dealing with complex situations where the elements are not just being dynamically loaded, but also their existence depends on the application's state, or there is complex conditional logic involved, you might need to use a more sophisticated strategy like an explicit wait with a condition. this is more involved but offers a lot of flexibility. in my experience, it's beneficial when you're testing a user interface with multiple components interacting between them. this is what i did at the beginning of my career when i worked with a bigger team of testing engineers.

```javascript
async function checkElementsWithCondition(page, selectorsWithCondition) {
  for (const selectorData of selectorsWithCondition) {
      const {selector, expectedCount, condition} = selectorData;
      try{
          await page.waitForFunction(condition, {timeout: 10000}); // wait until the condition is true, with a timeout
          const elements = await page.$$(selector);
          if (elements.length !== expectedCount) {
            console.warn(`mismatch found for selector ${selector}: expected ${expectedCount} , found ${elements.length}`)
            return false
          }
      } catch (error) {
        console.error(`error while waiting or checking selector ${selector}: ${error}`)
        return false;
      }
  }
  return true;
}


// example of usage:
const selectorsToCheck = [
{
    selector: '.product-card',
    expectedCount: 12,
    condition: () => document.querySelectorAll('.product-card').length > 0,
},
{
  selector: '#main-navigation li',
  expectedCount: 5,
  condition: () => document.querySelectorAll('#main-navigation li').length === 5,
},
{
  selector: 'button.submit-button',
  expectedCount: 1,
  condition: () => document.querySelector('button.submit-button') !== null &&
  document.querySelector('button.submit-button').disabled === false
},
];

(async () => {
  const allElementsPresent = await checkElementsWithCondition(page, selectorsToCheck);
  if (allElementsPresent) {
    console.log('all elements are present!');
  } else {
    console.error('some elements are missing or have unexpected counts.');
  }
})();
```

in this last example, the `condition` field is actually a javascript function. this gives you all the power you need to write very complex logic to determine when the element or elements should be present. this is useful because sometimes elements can be there but they are not ready or fully visible because of animations or other things. i once spent hours trying to figure out why my tests were randomly failing, and it turned out that the elements were not interactable even though they were technically present on the page, i had to use a `condition` to check if they were interactable. since that day, i never forget to add this kind of explicit waits with a condition. by the way, have you heard about the programmer who quit his job because he didn't get arrays? he couldn't bear the zero-based indexing. this is the only joke you are going to get from me today, because i'm not a funny person.

also, another tip that i can share with you, is to try to encapsulate your elements into functions or better yet classes, to have a good structure with your tests and scripts, since the tests and scripts tend to grow exponentially if you don't follow a good structure, i've learned that the hard way.

for resources, i recommend taking a look at "testing javascript applications" by lucas da costa and "automating with puppeteer" by matthew hargrove. those books really helped me understand the inner workings of testing and front-end automation. and also, the puppeteer documentation is fantastic, you should definitely read all of it. the same can be said about playwright.

so, that’s pretty much it. remember to adapt these techniques to your specific needs. no single approach is universally the best, it really depends on what you're doing, the size of the project, and the complexity of the page you are trying to check. good luck with it. i hope this helps you.
