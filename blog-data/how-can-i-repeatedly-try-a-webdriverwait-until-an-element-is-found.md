---
title: "How can I repeatedly try a WebDriverWait until an element is found?"
date: "2024-12-23"
id: "how-can-i-repeatedly-try-a-webdriverwait-until-an-element-is-found"
---

Okay, let's tackle this. It's a common scenario when dealing with asynchronous web applications: you need to wait, repeatedly, for an element to become available. I've seen this trip up many a developer, and the solution, while conceptually straightforward, often requires some careful implementation to avoid infinite loops or brittle tests. My early days involved a particularly challenging project where dynamic content rendering was practically the norm; we relied heavily on precisely controlled waits to keep our automated tests stable. That experience cemented my understanding of the subtleties involved.

The core of the problem is that web pages, especially those using modern frameworks like React or Angular, often load elements dynamically using JavaScript. This means the element you're looking for isn't necessarily present in the initial page source. Instead, it might appear only after some period of time, triggered by a server response, user interaction, or some other event. We can't just blindly search for it once and give up if it's not there, leading to unreliable tests and frustration. The standard `WebDriverWait` class in Selenium provides a powerful mechanism for handling this, but using it correctly, particularly when dealing with elements that might not appear consistently and reliably, requires more than a basic understanding. The naive approach of waiting once and failing, simply doesn't scale.

What you need is a structure that repeatedly attempts to find the element, until either the element is found or a reasonable timeout is reached. This often boils down to encapsulating `WebDriverWait` within a loop with an appropriate exception handling strategy. The general pattern involves using a `try-except` block with an appropriate exception, most commonly `TimeoutException` when the element can't be found within the specified time.

Let's illustrate with a concrete example. Imagine we are looking for a "Success Message" element that appears after a form submission. It might not appear immediately, and we need to handle the case where the server takes longer than normal to respond.

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import TimeoutException
import time

def wait_for_success_message(driver, max_attempts=3, wait_time=5):
    """
    Repeatedly waits for a success message element to be visible.

    Args:
        driver: The Selenium WebDriver instance.
        max_attempts: Maximum number of times to retry.
        wait_time: Time to wait in seconds for each attempt.

    Returns:
        The WebElement representing the success message if found, None otherwise.
    """
    for attempt in range(max_attempts):
        try:
            print(f"Attempt {attempt + 1}: Waiting for success message...")
            element = WebDriverWait(driver, wait_time).until(
                ec.visibility_of_element_located((By.ID, "success-message"))
            )
            print("Success message found.")
            return element
        except TimeoutException:
            print(f"Attempt {attempt + 1} failed. Retrying...")
            if attempt == max_attempts - 1:
                print("Max attempts reached. Giving up.")
                return None
            time.sleep(1) # Optional brief pause to reduce CPU usage.

if __name__ == '__main__':
    driver = webdriver.Chrome()  # Or any other browser driver setup
    driver.get("your_page_url_here") # Replace with your actual url

    # Assuming the form submission is done here to trigger the message

    success_element = wait_for_success_message(driver)

    if success_element:
        print("Success! Message content:", success_element.text)
    else:
        print("Failed to find success message after multiple attempts.")

    driver.quit()
```

In this first example, I've used a `for` loop to control the retries. Inside the loop, the `WebDriverWait` is enclosed in a `try-except` block. If a `TimeoutException` is caught, we retry the wait. We’ve added a message to indicate which attempt number we are on for more feedback. Note the `time.sleep(1)` which is an optional way to reduce CPU load during wait periods, though its usage is more beneficial when the wait period is shorter.

Now let's consider a slightly different scenario. Imagine you need to repeatedly wait for a list of elements to be non-empty, perhaps a list of results that are populated by an AJAX call. This requires a different approach but builds on the same basic principle:

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import TimeoutException
import time


def wait_for_non_empty_list(driver, locator, max_attempts=3, wait_time=5):
    """
    Repeatedly waits for a list of elements to be non-empty.

    Args:
        driver: The Selenium WebDriver instance.
        locator: A tuple specifying the By strategy and locator value (e.g., (By.CSS_SELECTOR, ".results")).
        max_attempts: Maximum number of times to retry.
        wait_time: Time to wait in seconds for each attempt.

    Returns:
        A list of WebElements if the list becomes non-empty, None otherwise.
    """
    for attempt in range(max_attempts):
        try:
            print(f"Attempt {attempt + 1}: Waiting for non-empty list...")
            elements = WebDriverWait(driver, wait_time).until(
                lambda d: len(d.find_elements(*locator)) > 0
            )
            print("List is non-empty.")
            return driver.find_elements(*locator)  # Return the actual elements
        except TimeoutException:
            print(f"Attempt {attempt + 1} failed. Retrying...")
            if attempt == max_attempts - 1:
                 print("Max attempts reached. Giving up.")
                 return None
            time.sleep(1) #Optional brief pause

if __name__ == '__main__':
    driver = webdriver.Chrome()  # Or any other browser driver setup
    driver.get("your_page_url_here")  # Replace with your actual url

    # Assuming the process that populates results is done here

    results = wait_for_non_empty_list(driver, (By.CSS_SELECTOR, ".result-item"))

    if results:
        print(f"Found {len(results)} results.")
        for element in results:
            print("Result text:", element.text)
    else:
        print("Failed to find results after multiple attempts.")

    driver.quit()

```
This second example uses a custom `lambda` function within `WebDriverWait` to check if the length of the element list located using `find_elements` becomes greater than zero. This pattern is particularly useful when the expected condition is not directly available in `expected_conditions`, like checking for list size. Here, I have used `.find_elements` instead of `.find_element`, as I am waiting for the list to be non-empty. Also, I return the actual list of `WebElements` if found, rather than a `Boolean` value.

Finally, let's look at a case where the element might be present, but not necessarily visible or interactable initially, maybe it is initially hidden and then becomes visible after animation or transitions.

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import TimeoutException, ElementNotVisibleException
import time

def wait_for_interactable_element(driver, locator, max_attempts=3, wait_time=5):
    """
    Repeatedly waits for an element to become visible and interactable.

    Args:
        driver: The Selenium WebDriver instance.
        locator: A tuple specifying the By strategy and locator value (e.g., (By.ID, "submit-button")).
        max_attempts: Maximum number of times to retry.
        wait_time: Time to wait in seconds for each attempt.

    Returns:
         The WebElement if found and interactable, None otherwise.
    """
    for attempt in range(max_attempts):
        try:
            print(f"Attempt {attempt + 1}: Waiting for interactable element...")
            element = WebDriverWait(driver, wait_time).until(
                ec.element_to_be_clickable(locator) # Ensure clickable instead of just visible
            )
            print("Element is interactable.")
            return element
        except (TimeoutException, ElementNotVisibleException):  # Catch both conditions
            print(f"Attempt {attempt + 1} failed. Retrying...")
            if attempt == max_attempts - 1:
                print("Max attempts reached. Giving up.")
                return None
            time.sleep(1) #Optional brief pause


if __name__ == '__main__':
    driver = webdriver.Chrome()  # Or any other browser driver setup
    driver.get("your_page_url_here") # Replace with your actual url

    #Assuming the button's state changes here

    button = wait_for_interactable_element(driver, (By.ID, "submit-button"))

    if button:
        print("Button is ready to interact. Clicking...")
        button.click()
    else:
        print("Failed to find interactable button after multiple attempts.")

    driver.quit()
```

In this third example, instead of just `visibility_of_element_located`, I've used `element_to_be_clickable`. This makes sure the element is not only present but also visible and interactable. This is especially important for elements that may initially be hidden or covered by other elements. Also note I've included `ElementNotVisibleException` in the `except` clause to catch the case where the element exists, but is not yet visible.

For more in-depth knowledge of handling asynchronous operations and web testing, I would highly recommend “Selenium WebDriver: Book by Boni Garcia" as a fundamental resource to get started. Also, to deepen your knowledge around how web pages interact and the general asynchronous behaviour, the “High Performance Browser Networking” book by Ilya Grigorik goes into great detail about the underlying mechanisms and networking interactions, which is beneficial to fully grasp why these delays and asynchronous behaviour exist.

In conclusion, repeatedly trying a `WebDriverWait` involves encapsulating it within a loop that handles `TimeoutException` and other relevant exceptions. Always define your exit conditions clearly, such as a maximum number of attempts, and provide descriptive log messages to aid with debugging. Tailor the specific expected condition to the needs of your particular application (e.g., `visibility_of_element_located`, `element_to_be_clickable`, or a custom lambda function) as per requirement. This approach will significantly improve the reliability and stability of your automated tests.
