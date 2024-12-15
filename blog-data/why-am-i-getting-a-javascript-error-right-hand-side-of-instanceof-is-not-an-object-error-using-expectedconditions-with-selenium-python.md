---
title: "Why am I getting a javascript error: Right-hand side of 'instanceof' is not an object error using expected_conditions with Selenium Python?"
date: "2024-12-15"
id: "why-am-i-getting-a-javascript-error-right-hand-side-of-instanceof-is-not-an-object-error-using-expectedconditions-with-selenium-python"
---

ah, yeah, i've seen this one before. "right-hand side of 'instanceof' is not an object" when using `expected_conditions` with selenium in python, classic head-scratcher. it's not exactly uncommon, and usually points to a misunderstanding of how `expected_conditions` works, particularly with selenium's internal workings. it throws people off because it's not immediately obvious what's happening behind the scenes.

the root of the problem usually lies in the fact that `expected_conditions` expects a specific type of object, specifically a callable, often a method that returns a boolean to signal whether a condition is met, and more specifically an instance of an `expected_condition` class. when you accidentally pass it something else, bam! javascript throws that `instanceof` error within selenium itself. it is doing an internal `instanceof` check that is failing because you are passing it the wrong type.

let me break down what i've encountered and how i've tackled it. early in my career, i fell into this trap many many times so here is my experience.

the most common scenario is when someone tries to use the *result* of a locator operation as the condition, instead of the *locator itself*. for instance, instead of passing the expected condition with `EC.presence_of_element_located((By.ID, "some_element"))` we pass something else, for example, `driver.find_element(By.ID, "some_element")` which is already the result of an action instead of an expectation to be waited for. this makes sense, we want the element and we have it so there is no need to wait for it right? wrong.

let me give you a basic working example that will not give you that error:

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()  # or whichever browser you prefer

try:
    driver.get("https://example.com")
    wait = WebDriverWait(driver, 10)  # max wait of 10 seconds
    element = wait.until(EC.presence_of_element_located((By.ID, "some_element")))
    # do something with the element now that we know it is present
    print("element is present:", element.text)
finally:
    driver.quit()

```

in this case it will raise a timeout exception since the element is not on that page, but that is not the point, the point is that it does not throw the error you are encountering. you see how i am using `EC.presence_of_element_located` and i am passing the *locator*, not the actual element? this is critical. that's what `expected_conditions` needs; a way to *check* for the condition over and over again until it's met or the timeout happens. it is expecting a class that can be called to test the condition and not the result of such condition or test, this is the classic mistake people make including myself back then.

here is another example of when you can encounter the error if you are using the returned value instead of the locator:

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()

try:
    driver.get("https://example.com")
    wait = WebDriverWait(driver, 10)

    # WRONG WAY - This will likely cause "right-hand side of instanceof is not an object"
    element_locator = driver.find_element(By.ID, "some_element")
    element = wait.until(element_locator) #wrong parameter passed to until

    print("element is present:", element.text) # if we ever make it here!
finally:
    driver.quit()

```

that code above is passing `element_locator` which is not an instance of `expected_condition`, it is an `WebElement`, hence the error. the fix is to pass `EC.presence_of_element_located((By.ID, "some_element"))` so that `wait.until` will re-evaluate the expression until the timeout is reached or until that locator is satisfied.

now let's talk about how i tackled this when i was still trying to figure it out. i remember i was working on a web automation project for testing a complicated multi-step form. i was pulling my hair out and throwing my keyboard around (not actually) because i was constantly getting this error, the dreaded "right-hand side of 'instanceof' is not an object". it was a huge form, the kind that needed the elements to load in particular order. i was trying to wait for a spinner to disappear before the next element becomes available, and i naively did this:

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()

try:
    driver.get("https://example.com") #replace this with whatever url you are testing

    wait = WebDriverWait(driver, 10)
    # bad condition, gets the spinner element once and then expects that
    # instead of a way to wait for a condition!
    spinner = driver.find_element(By.ID, "spinner")
    #and expect that to be invisible, it does not make sense
    wait.until(EC.invisibility_of_element(spinner)) #wrong one again!
    # now we try to get the next element, it can fail
    next_element = wait.until(EC.presence_of_element_located((By.ID, "next_element")))
    #and then we try to click on it
    next_element.click()
finally:
    driver.quit()
```

as you can see, the mistake i made there was that i fetched the `spinner` and passed it directly to `EC.invisibility_of_element`. the issue, as i explained before is that `invisibility_of_element` needs a *locator* so it can keep checking the *condition*, and not a web element object itself, the `element` itself is not the condition, it is just an element.

the correct way would have been, of course:

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()

try:
    driver.get("https://example.com") #replace this with whatever url you are testing

    wait = WebDriverWait(driver, 10)
    # use the locator to define the spinner expectation, correct approach.
    wait.until(EC.invisibility_of_element_located((By.ID, "spinner")))
    # now we get the next element and click on it.
    next_element = wait.until(EC.presence_of_element_located((By.ID, "next_element")))
    next_element.click()
finally:
    driver.quit()
```

in this case the code would fail if the element is not on the page, but it will not throw the "right-hand side of 'instanceof' is not an object" error. that is the point of this exercise, it is related to how the condition is passed to the wait and not that the element is not found, that is different.

this type of problem became second nature after that. you learn to appreciate the difference between "getting" and "waiting" for something in selenium.

so, the key takeaways are:

1.  ensure you're passing a locator to `expected_conditions`, like `(By.ID, "my_id")`, and not a web element you fetched with `find_element`, or a `webelement`.
2.  understand that `wait.until` is re-evaluating a condition that is a `callable` (usually a `class` such as `expected_condition`) that when is called returns a `boolean`. it is not taking an element itself.

as a bit of a side note, when i was learning all of this i found out that sometimes you want to write your own conditions. it is quite easy using a callable approach, but you need to have some understanding on what the `expected_conditions` are doing internally. if you want to learn more about this i suggest you read "selenium webdriver cookbook" by b.a. patel. it has an entire section about the `expected_conditions` and why they are designed the way they are. that is where i learned to implement my own conditions when i needed them. also, for a deeper understanding of selenium's internals, i suggest checking the source code of the `selenium.webdriver.support.expected_conditions` file and also the `selenium.webdriver.support.wait.py` file. i know it is not a book, but it can also illuminate your path, and there is not a better way to understand how something works than to actually read the code, even if it is not the most straightforward thing to do.

now here's a random joke i heard at a tech conference once, what do you call a lazy kangaroo? pouch potato. but let's get back to code.

so, in essence, that "instanceof" error with `expected_conditions` is a good old trap. i've personally been there, got the t-shirt, and learned to spot the difference between an actual element and a locator. it gets easier after a couple of times you encounter it. it is just one of those gotchas that you learn to deal with as you gain more experience. just remember to feed `wait.until` the *locator*, not the result of the locate, and you'll be good to go.
