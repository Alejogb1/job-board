---
title: "How does Selenium 4's WebDriver manage implicit waits?"
date: "2024-12-23"
id: "how-does-selenium-4s-webdriver-manage-implicit-waits"
---

, let's talk about implicit waits in Selenium 4. From my experience, they are often a point of confusion, especially when you're dealing with dynamic web applications. I’ve certainly seen my share of flaky tests due to misunderstood implicit wait behavior. So, let's break it down.

The core idea behind an implicit wait is to tell the webdriver to pause for a specified duration when attempting to find an element that isn't immediately present in the document object model (dom). Instead of throwing a `NoSuchElementException` immediately, selenium will periodically poll the dom for the element until either the element is found or the specified wait time elapses. This mechanism provides a somewhat cleaner way of handling asynchronous loading compared to hardcoded `Thread.sleep()` calls, which are notoriously brittle.

Now, it’s crucial to understand that implicit waits are *global* in scope for the webdriver session. Once you set an implicit wait, it applies to *all* subsequent `find_element` and `find_elements` calls during that session. This is both its strength and potential pitfall. Its strength is that it reduces code verbosity and makes handling dynamic loads easier in the majority of situations. The pitfall, though, is that it can lead to unexpected delays and masking of actual errors if not used carefully.

For example, let’s say you set an implicit wait of, say, 10 seconds. If an element is present in the dom instantly, then selenium finds it instantly. There’s no waiting. However, if the element is not there at first, selenium will check regularly (polling) until 10 seconds has passed or the element is found. If after 10 seconds, it’s still not found, then, and only then, will a `NoSuchElementException` will be thrown.

I’ve seen teams get into situations where they stack different implicit waits across multiple tests, and it makes debugging hell. Imagine a situation where you set a 10 second wait in one part of your test setup, and then a 5 second wait in another part of the test, or even another fixture. The 10-second wait will dominate any 5-second setting that proceeds it. Furthermore, if your application suddenly has a performance hiccup, those waits become unpredictable. Sometimes a test will pass fine because things loaded quickly, sometimes tests will run for their full timeout. This leads to the type of flakiness no one wants to deal with.

Let's look at some code examples in python, which are easy to translate to other language bindings of selenium:

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

def setup_driver():
    options = Options()
    options.add_argument("--headless=new") # for running in headless mode
    driver = webdriver.Chrome(options=options)
    return driver


def test_implicit_wait_success():
    driver = setup_driver()
    driver.get("https://www.example.com")  # Use example URL, make sure it is quick
    driver.implicitly_wait(5)  # Set implicit wait to 5 seconds

    try:
       # Assuming the <title> tag exists, this should be very quick and not cause a delay.
        title_element = driver.find_element(By.TAG_NAME, "title")
        assert title_element is not None, "Element should be found quickly."
        print("Title found with implicit wait")
    except NoSuchElementException as e:
       assert False, "Error, element should have been found quickly, " + str(e)

    driver.quit()

test_implicit_wait_success()
```

This first example demonstrates a successful use of implicit waits. It navigates to a URL, sets an implicit wait, then quickly finds the title element, which exists right away, and it doesn't wait the full five seconds.

Now, let’s see an example where the element isn’t found within the implicit wait time:

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
import time

def setup_driver():
    options = Options()
    options.add_argument("--headless=new") # for running in headless mode
    driver = webdriver.Chrome(options=options)
    return driver

def test_implicit_wait_fail():
    driver = setup_driver()
    driver.get("https://www.example.com") # Use example URL, make sure it is quick

    driver.implicitly_wait(2)  # Set an implicit wait of 2 seconds
    try:
        # This non-existent class won't be found and will throw an exception after the wait.
        nonexistent_element = driver.find_element(By.CLASS_NAME, "nonexistent-class")
        assert False, "Error: This should have thrown NoSuchElementException"

    except NoSuchElementException:
        print("Element not found after implicit wait. Exception caught as expected.")
    driver.quit()

test_implicit_wait_fail()
```

In this scenario, the code attempts to find an element with a class that doesn't exist on the page. After 2 seconds of implicit wait, `NoSuchElementException` is correctly thrown. This highlights the behavior of an implicit wait: it’s a timeout mechanism and it will give up after the timeout is reached.

Finally, here’s a situation showcasing how subsequent waits will overwrite earlier ones, and how it can mask potential problems with a test setup.

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
import time

def setup_driver():
    options = Options()
    options.add_argument("--headless=new") # for running in headless mode
    driver = webdriver.Chrome(options=options)
    return driver

def test_conflicting_implicit_waits():
    driver = setup_driver()
    driver.get("https://www.example.com") # Use example URL, make sure it is quick

    driver.implicitly_wait(10)  # first implicit wait
    driver.implicitly_wait(2)  # this will overwrite the first implicit wait
    try:
      # The element doesn't exists, this will now wait 2 seconds, not 10
      element = driver.find_element(By.CLASS_NAME, "another-nonexistent-class")
      assert False, "Error: should have thrown exception"
    except NoSuchElementException:
        print("Exception caught after only 2 second wait, demonstrating override behavior")
    driver.quit()

test_conflicting_implicit_waits()
```

As you can see, the second `implicitly_wait(2)` call overrode the earlier setting of 10 seconds. This underscores the critical point that only the *last* implicit wait set on a driver instance is the effective setting. This kind of silent override can mask test setup issues or timing problems.

To manage this properly, I often advise using a combination of best practices:

1. **Prefer explicit waits:** Where possible, use explicit waits (e.g., `WebDriverWait` in python) for greater control. This allows you to wait for specific conditions, such as an element being clickable or visible, rather than relying on a fixed time. Explicit waits are much more granular and less likely to cause long waits.
2. **Be judicious with implicit waits:** If you do use implicit waits, keep the timeout relatively short and consistent across your project. Avoid setting implicit waits at arbitrary times within the test suite.
3. **Reset implicit waits after each test:** A common practice is to reset implicit waits to 0 at the end of each test or fixture. This ensures that one test's implicit wait doesn't affect subsequent tests.
4. **Don't mix explicit and implicit:** Avoid using both implicit and explicit waits on the same elements, as this can lead to unpredictable behavior and make debugging difficult. Pick one strategy and stick with it.
5. **Understand the polling interval:** While not directly configurable by the user, selenium polls every few hundred milliseconds, this is good to keep in mind when thinking about possible total time of the implicit wait.

For further reading, I'd highly recommend checking out "Selenium WebDriver 3 Practical Guide" by Boni Garcia. It provides a deep dive into selenium, and the way it approaches waits. Also "Test Automation with Selenium" by Matthew Walker is a good option for a broader view on how to think about test automation as well as selenium. The official selenium documentation, of course, is essential. Specifically, take the time to review the section on timeouts and waits, as it lays out very clearly how these mechanisms operate in selenium.

In conclusion, while implicit waits can be convenient, it's important to use them cautiously and with a solid understanding of their global scope and override behavior. Often explicit waits provide a much more robust and manageable alternative, especially in larger test suites and when trying to handle complex loading situations.
