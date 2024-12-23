---
title: "Why do I have a (Ferrum::Browser) simple browser.evaluate issue?"
date: "2024-12-15"
id: "why-do-i-have-a-ferrumbrowser-simple-browserevaluate-issue"
---

ah, ferrum and evaluate… yeah, i’ve been there, knee-deep in that particular pit of despair. it’s frustrating when you think you’ve got the page loaded and ready, only to have `evaluate` throw a tantrum. let me give you my take, based on some of the debugging sessions i’ve had with this beast.

first off, let's make sure we are talking about the same thing. when you say simple browser, i'm guessing you're using `ferrum` in ruby, right? just confirming, because there's always the chance it might be something completely different and we could be chasing ghosts here.

the problem, as i see it usually comes down to the timing, specifically, when `evaluate` is called relative to the page's state. think of it like this: the browser is a complicated machine. you tell it to go to a url, it fetches the html, then it has to parse that html, then fire all the javascript to make the page interactive. your `evaluate` call can run at any point during that, and if you call it too early you might just be evaluating on a mostly-empty or incomplete page. it is literally like trying to cook an omelette before you crack the eggs. the timing is everything, friend.

so, the first culprit to suspect is race conditions, plain and simple. the page is still loading, and the element you are looking for hasn’t fully rendered. `ferrum` might have connected, but javascript execution is still happening.

i remember this one project i had, a web scraper that pulled data from this ridiculously dynamic site. everything was javascript driven and it was a horrible mess. i kept getting these null return values when i tried to `evaluate` selectors. i was pulling my hair out, i swear. i could see the element on the actual page with my eyes, but not from the ferrum perspective.

i ended up spending hours adding `sleep` statements, which is just a terrible way to do things, i know, but when you're desperate, you will do almost anything to get things working. eventually i figured out there were specific elements that were loading after others. there were a lot of ajax calls happening in the background and i had to wait for those ajax calls to complete. i ended up abandoning the `sleep` strategy, it was unreliable and would slow things down terribly.

the fix was, instead of blindly waiting, to use something like this:

```ruby
require 'ferrum'

browser = Ferrum::Browser.new

browser.goto("https://some-dynamic-website.com")

# instead of sleep, wait for an element to appear
browser.wait_for_selector(".some-dynamic-element", timeout: 5)

# now evaluate!
result = browser.evaluate("document.querySelector('.some-dynamic-element').textContent")

puts result

browser.quit
```

this `wait_for_selector` method is a lifesaver. it polls the page and only continues when the specified selector is present. it’s infinitely better than just using sleep, that's one of the things i learnt the hard way. the `timeout` option makes sure you don’t get stuck forever if the element never appears.

another common problem occurs when the javascript you are trying to execute in `evaluate` has some dependency not available in the page's scope yet, or maybe the variable exists in the global space but only on certain events. for instance, if some javascript is only loaded after some button click, using `evaluate` before clicking that button will most likely fail. this happened to me on an old single page app, where certain parts were lazily loaded, only after a user triggered some event. it was a nightmare to debug back then. the site was a total mess of spaghetti javascript with no structure. that was the day i decided that i will be very selective on what projects to take. you should be selective too. do not let legacy code make you loose sleep.

another approach, related to the above, is to use the javascript console in the browser, or use `browser.js_eval`, then test the specific parts of javascript and selectors before executing them via `evaluate`. i usually do this when i am in the beginning phases of debugging something like this and it has helped me discover those issues related to scope or incorrect selectors.

```ruby
require 'ferrum'

browser = Ferrum::Browser.new

browser.goto("https://some-dynamic-website.com")

# first, check if the selector is correct
js_result_selector = browser.js_eval("document.querySelector('.target-element')")
puts "selector element exists? #{js_result_selector}"

# then, check if the code works in the console
js_result = browser.js_eval("document.querySelector('.target-element').textContent")
puts "js eval result: #{js_result}"

# finally, evaluate the code
result = browser.evaluate("document.querySelector('.target-element').textContent")
puts "evaluate result: #{result}"


browser.quit
```

this allows you to break things into small pieces, helping you isolate which part is not working correctly. the `js_eval` will allow you to see what results you would get in a real browser, which is essential for this type of debugging. sometimes the reason for the problem is more obvious than you would think.

there is also the case where your evaluate code is syntactically incorrect or has an error. ferrum will not give you much information on this. it will just fail or give you weird results. i had a problem like this once, where i missed a comma in the evaluate function. the result was really difficult to understand at first. i was really embarrassed when i discovered the issue. now i double check my syntax. i even use linters that are configured to highlight javascript syntax errors. it saves a lot of headache.

let’s say you have some data you want to retrieve, from different elements, it’s generally best to use a single evaluate call instead of multiple ones, since each call can have some overhead. also, this method simplifies your code and avoids potential issues with timing or scope. for complex scenarios, building the extraction javascript inside the evaluate method helps to keep the code cleaner. something like this:

```ruby
require 'ferrum'

browser = Ferrum::Browser.new

browser.goto("https://some-website.com/with-multiple-elements")

data = browser.evaluate(<<~JAVASCRIPT
  const elements = document.querySelectorAll('.data-item');
  const result = [];
  elements.forEach(element => {
      result.push({
        title: element.querySelector('.title').textContent,
        description: element.querySelector('.description').textContent
      });
  });
  return result;
JAVASCRIPT
)

data.each { |item|
  puts "title: #{item['title']}, description: #{item['description']}"
}

browser.quit
```
see how the javascript code is nicely included inside a multiline string literal. this keeps the code readable, instead of making one long, hard to read string. also, see how all the processing is done inside the browser, so only the final result needs to be returned.

i almost forgot, always, and i mean always, make sure that the browser is properly closed after use, using the `browser.quit` method. you don’t want to leave browser instances running in the background, especially when running a lot of tests or scrapers.

one final point. if your website uses a lot of iframes or shadow dom, that is another level of complexity you need to be aware of. `ferrum` does have methods to interact with these, but they deserve a whole another discussion by themselves. dealing with iframes and shadow dom is, let's say, a very good way to increase your blood pressure.

regarding resources, i recommend "javascript: the definitive guide" by david flanagan, for deeper understanding of javascript internals, "high performance javascript" by nicholas c. zakas if you have to deal with performance issues in the browser or want to write efficient javascript. but there are tons of good resources online.

debugging `ferrum` evaluate can be frustrating, but with careful timing, understanding page rendering, and verifying selector presence, you should be able to track the cause. the key is to be systematic and to break the problem down into small pieces. and never assume that it just works. there is always a reason for the behavior and it is up to us to find it. i know that, some times, that reason can be "user error", and that is  too. i myself have made so many mistakes that i have lost count of them. we have to embrace making mistakes, and we have to learn from them. we should, because that’s how we improve our work.
