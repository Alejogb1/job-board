---
title: "How to wait for multiple loading elements to disappear using Selenium?"
date: "2024-12-14"
id: "how-to-wait-for-multiple-loading-elements-to-disappear-using-selenium"
---

alright, so you're looking to handle multiple loading elements disappearing with selenium, i've definitely been there. it’s a common pain point when dealing with dynamic web applications, especially ones that love to throw up spinners or progress bars all over the place. basically you want your test to wait until all of those are gone before moving onto the next step, it's key for reliable automation.

i've had some frustrating debugging sessions because of this. early on in my career, like, way back when selenium webdriver was still the cool new thing, i was working on an e-commerce site. it had this habit of showing not one but three loading icons when you added an item to the cart. and they weren’t consistent either; sometimes two would go away instantly, and sometimes the last one would linger for a couple of extra seconds. my initial test scripts were a hot mess, flaky as all hell because they weren't waiting properly. they would jump the gun, try to interact with elements that were still covered, and the result? failed tests, and unhappy project managers. fun times.

my first attempt was a brute-force approach using `time.sleep()`, but that just led to slow test execution and still flakiness. it's like trying to catch a fly with a sledgehammer. you might get it sometimes, but it’s terribly inefficient and often ineffective. then i tried using `selenium.webdriver.support.ui.WebDriverWait` with simple `element_to_be_clickable` and `visibility_of_element_located` conditions. and yeah, while those are generally super helpful, they are geared for single element waits, and i had more than one loader to babysit. it is similar trying to herd cats with one piece of string. a recipe for disaster.

the solution, it turned out, wasn't that complicated once i got the hang of it. it basically involved creating a custom expected condition to check for the disappearance of all loading elements. and that’s where the `staleness_of` condition comes into play.

here is the logic, you grab all the loading elements initially using a selector that matches all of them. then use that list of elements in conjunction with a custom expected condition. inside of it use `all()` alongside a generator expression and `staleness_of()`.

here is the first code snippet that illustrates this approach. let's say all your loading elements have the class name 'loading-spinner':

```python
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By

def wait_for_loading_elements_to_disappear(driver, timeout=10):
    """
    waits until all loading elements become stale, meaning they are no longer attached to the dom.

    Args:
        driver: selenium webdriver instance.
        timeout: maximum wait time in seconds (default 10).
    """
    loading_elements = driver.find_elements(By.CLASS_NAME, 'loading-spinner')
    
    if not loading_elements:
      # no elements to wait for, so simply return.
      return
    
    wait = WebDriverWait(driver, timeout)

    wait.until(lambda driver: all(ec.staleness_of(element)(driver) for element in loading_elements))


# example usage
# assume you have a driver instance initialized as 'driver'
# ...code to trigger the loading elements...

wait_for_loading_elements_to_disappear(driver)

# now continue with your test logic as loading elements have disappeared
```

note the custom wait condition uses a lambda expression that iterates through the initial collection of elements. this ensures it checks the staleness of every loader individually. that is the key. `staleness_of` is ideal here because it returns true when an element is detached from the dom, meaning it’s no longer present or visible.

now, let's say some of your loading elements aren't so easily found by css classes. perhaps they have different ids, or maybe some use images instead of css based animations. in those scenarios you can also use xpath, which works equally well in this case, maybe you use a common parent and use child selectors, something like:

```python
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By

def wait_for_loading_elements_to_disappear_xpath(driver, timeout=10):
    """
    waits until all loading elements matched by xpath become stale.

    Args:
        driver: selenium webdriver instance.
        timeout: maximum wait time in seconds (default 10).
    """
    loading_elements = driver.find_elements(By.XPATH, '//div[@class="loading-container"]//*[contains(@class,"loader")]')
    
    if not loading_elements:
      # no elements to wait for, so simply return.
      return
    
    wait = WebDriverWait(driver, timeout)
    wait.until(lambda driver: all(ec.staleness_of(element)(driver) for element in loading_elements))

# example usage
# ...code to trigger the loading elements...

wait_for_loading_elements_to_disappear_xpath(driver)

# continue with the rest of your test logic
```
this approach gives you more flexibility when the loading elements are not defined in a uniform manner, it's a lifesaver on complex pages, i can tell you.

and what happens if some of your elements are not strictly 'stale', perhaps they simply become invisible or get their opacity set to zero, and they are still in the dom. in this case `staleness_of` won’t work, so we need to go with a slightly different approach. in this case you could implement a similar wait condition but instead of staleness you use the `invisibility_of` condition. this means you should be looking for the elements initially and then check until they are invisible.

```python
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By

def wait_for_loading_elements_to_become_invisible(driver, timeout=10):
    """
    waits until all loading elements become invisible.
    Args:
        driver: selenium webdriver instance.
        timeout: maximum wait time in seconds (default 10).
    """
    loading_elements = driver.find_elements(By.CSS_SELECTOR, '.loading-overlay,.loading-animation')
    
    if not loading_elements:
      # no elements to wait for, so simply return.
      return
    
    wait = WebDriverWait(driver, timeout)
    wait.until(lambda driver: all(ec.invisibility_of(element)(driver) for element in loading_elements))


# example usage
# ...code that triggers the appearance of loaders ...
wait_for_loading_elements_to_become_invisible(driver)

# ... continue your testing ...

```
the `invisibility_of` condition works like a charm in those scenarios. it’ll keep checking until the element's display is none or its opacity is zero, or something like that.

the key is to understand the way in which your loaders are being handled by the front-end code, are they being removed completely or simply hidden? choose the `staleness_of` or `invisibility_of` condition accordingly.

one last thing, if you're dealing with super-complex single page apps, you might find that your page updates trigger loaders with a brief delay. in this case, sometimes the loader appears, disappears, and reappears so quickly that the code will not see the first one so the wait will wait forever. in those scenarios, you can add a small wait at the beginning of the custom wait logic, this can make the test more robust. but do it with caution, you don’t want to add unnecessary `sleep`s everywhere, this is not a good practice but in very particular scenarios it might be necessary. you know, sometimes you have to embrace the dark side a little bit.

for more in-depth knowledge on this, i suggest taking a look at the 'selenium with python' documentation. they go really deep into expected conditions. or check the 'python testing with pytest' book by brian okun. it covers custom wait conditions in details in context with pytest, it's a very practical resource. also, 'effective selenium' by maurice bruen, is a great practical book, for more pragmatic usage of selenium.
