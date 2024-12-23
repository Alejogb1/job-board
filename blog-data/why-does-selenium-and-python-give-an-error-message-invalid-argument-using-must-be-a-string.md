---
title: "Why does Selenium and Python give an Error: Message: invalid argument: 'using' must be a string?"
date: "2024-12-23"
id: "why-does-selenium-and-python-give-an-error-message-invalid-argument-using-must-be-a-string"
---

, let's tackle this one. I’ve definitely seen my share of "invalid argument: 'using' must be a string" errors when working with selenium and python, and it usually boils down to a fairly straightforward misunderstanding of how locators are defined within the selenium framework, coupled with some common python usage patterns that can, unintentionally, lead you astray. It's a classic example of where the apparent simplicity of the library can mask some of the underlying mechanics if you aren't paying close enough attention.

The core of the issue lies within the selenium's `find_element` (or `find_elements`) method. These methods, which you use to interact with web elements, require you to provide two crucial pieces of information: the *locator strategy* and the *locator value*. The 'using' argument the error message references specifically corresponds to that locator strategy, which tells selenium *how* to find the element. We're talking about things like finding by `id`, `class name`, `xpath`, `css selector`, and so on. Importantly, the "using" parameter must always be a string that maps to a valid selenium locator strategy.

Where things often go wrong, at least from my experience, is when a non-string type gets inadvertently passed as the 'using' argument. Python's dynamic typing can be incredibly convenient, but it can also be the root of these kinds of problems if not managed carefully. I remember, years back, leading a team that was migrating from an older testing framework to a selenium-based solution. One of the junior members, a bright individual but less familiar with selenium’s nuances, was passing a variable containing what he *thought* was the string "xpath" but it turned out to be a variable of type `locator.XpathStrategy` from a helper library they were also testing. The code looked fine at a glance, but the error popped up immediately during execution. This serves as an excellent lesson in the importance of scrutinizing variable types, especially when interfacing with third-party libraries.

Let me illustrate with three code examples that showcase different scenarios which might lead to the error, along with corrected versions:

**Example 1: Incorrect Locator Strategy Type**

```python
# Incorrect Code (example_1_incorrect.py)
from selenium import webdriver
from selenium.webdriver.common.by import By

def find_element_incorrect(driver):
    locator_strategy = By.XPATH  # Incorrect: This is not a string
    locator_value = '//*[@id="myElement"]'
    element = driver.find_element(using=locator_strategy, value=locator_value)
    return element

if __name__ == "__main__":
    driver = webdriver.Chrome() # Or any other browser driver
    driver.get("https://www.example.com")
    try:
        element = find_element_incorrect(driver)
        print("Element found:", element)
    except Exception as e:
        print("Error:", e)
    finally:
        driver.quit()
```

In the above case, `locator_strategy` is a *member* of the `By` enum, not the string itself. The correct way to do this is as follows:

```python
# Corrected Code (example_1_correct.py)
from selenium import webdriver
from selenium.webdriver.common.by import By

def find_element_correct(driver):
    locator_strategy = "xpath"  # Correct: Explicitly providing the string "xpath"
    locator_value = '//*[@id="myElement"]'
    element = driver.find_element(using=locator_strategy, value=locator_value)
    return element

if __name__ == "__main__":
    driver = webdriver.Chrome() # Or any other browser driver
    driver.get("https://www.example.com")
    try:
      element = find_element_correct(driver)
      print("Element found:", element)
    except Exception as e:
      print("Error:", e)
    finally:
      driver.quit()
```

Here, we've rectified the issue by directly assigning the string `"xpath"` to the `locator_strategy` variable. It may seem trivial, but this is an extremely common cause of the "invalid argument" error when using the selenium library, as it’s a common pitfall of attempting to use class constants where strings are explicitly required.

**Example 2: Incorrect Function Return Type**

Another situation I've come across was when the function meant to determine the locator was incorrectly returning a `list` of strings instead of a single string, which will cause an error. In the following code, the function should return "id", but actually, it is returning `['id']`, a list, causing the error:

```python
# Incorrect Code (example_2_incorrect.py)
from selenium import webdriver

def get_locator_type():
    return ['id']  # Incorrect, returns a list

def find_element_incorrect_2(driver):
    locator_strategy = get_locator_type() # This is now a list
    locator_value = 'myElement'
    element = driver.find_element(using=locator_strategy, value=locator_value)
    return element

if __name__ == "__main__":
    driver = webdriver.Chrome()
    driver.get("https://www.example.com")
    try:
        element = find_element_incorrect_2(driver)
        print("Element found:", element)
    except Exception as e:
        print("Error:", e)
    finally:
        driver.quit()
```

The fix, as always, is to return a string:

```python
# Corrected Code (example_2_correct.py)
from selenium import webdriver

def get_locator_type():
    return 'id'  # Correct: Returns a string

def find_element_correct_2(driver):
    locator_strategy = get_locator_type()
    locator_value = 'myElement'
    element = driver.find_element(using=locator_strategy, value=locator_value)
    return element

if __name__ == "__main__":
    driver = webdriver.Chrome()
    driver.get("https://www.example.com")
    try:
        element = find_element_correct_2(driver)
        print("Element found:", element)
    except Exception as e:
        print("Error:", e)
    finally:
        driver.quit()

```
The corrected code makes sure the `get_locator_type` returns a `string` which can be used with the selenium function.

**Example 3: Passing `None` or an Empty String**

Lastly, while seemingly obvious, it’s worth mentioning that sometimes, the “using” parameter may be accidentally assigned `None` or an empty string (``""``). The result is always the same error.

```python
# Incorrect code (example_3_incorrect.py)
from selenium import webdriver

def get_locator_info():
  return None, None

def find_element_incorrect_3(driver):
    locator_strategy, locator_value = get_locator_info() # This is None
    element = driver.find_element(using=locator_strategy, value=locator_value)
    return element

if __name__ == "__main__":
    driver = webdriver.Chrome()
    driver.get("https://www.example.com")
    try:
      element = find_element_incorrect_3(driver)
      print("Element found:", element)
    except Exception as e:
      print("Error:", e)
    finally:
      driver.quit()
```
The code will raise the error, as the value for 'using' is `None`. The fix in this case is to provide a valid locator type and locator value.
```python
# Corrected Code (example_3_correct.py)
from selenium import webdriver

def get_locator_info():
  return 'id', 'myElement'

def find_element_correct_3(driver):
    locator_strategy, locator_value = get_locator_info() # This is now an id
    element = driver.find_element(using=locator_strategy, value=locator_value)
    return element

if __name__ == "__main__":
    driver = webdriver.Chrome()
    driver.get("https://www.example.com")
    try:
        element = find_element_correct_3(driver)
        print("Element found:", element)
    except Exception as e:
        print("Error:", e)
    finally:
        driver.quit()
```

In summary, the "invalid argument: 'using' must be a string" error in selenium with python usually arises from passing a non-string value to the `using` parameter of the `find_element` or `find_elements` methods. Common causes include incorrect type assignments (e.g., using `By.XPATH` instead of `"xpath"`), functions returning incorrect types, or passing `None` or an empty string. Debugging usually involves meticulously verifying the type of the variable passed as the `using` argument, making sure that your locator strategy is expressed as a valid string. For a deeper understanding of selenium, the official Selenium documentation is invaluable. Also, "Web Scraping with Python" by Ryan Mitchell is a great practical resource for tackling real-world applications using the library. Finally, the “Selenium with Python” course by Dave Gray in Test Automation University is a great resource to solidify the basics. Addressing these common pitfalls, and being careful with variable types, will solve the issue most of the time.
