---
title: "Why is there a 'No signature of method: geb.navigator.NonEmptyNavigator.getJs()'?"
date: "2024-12-23"
id: "why-is-there-a-no-signature-of-method-gebnavigatornonemptynavigatorgetjs"
---

Ah, that "No signature of method: geb.navigator.NonEmptyNavigator.getJs()" error. I've seen it pop up a fair few times over the years, and each instance tends to point to a slightly different nuance of how Geb interacts with the underlying web driver. It’s never quite straightforward, is it? Let’s break down why this specific error arises and how we can effectively resolve it.

The core of the issue lies in how Geb’s `Navigator` objects, specifically those that are non-empty, handle JavaScript execution. The `getJs()` method, which you’re trying to invoke, is *not* a direct member of the `NonEmptyNavigator` class itself. Instead, it's accessed indirectly through a mechanism that's more concerned with the `Browser` instance and its associated web driver.

Think of it this way: `Navigator` objects, in their non-empty form, represent *elements* located on the web page. They’re essentially pointers to specific parts of the dom. The `getJs()` method, on the other hand, isn't designed to operate directly on *elements*. It's designed to interact with the browser context, allowing you to execute arbitrary JavaScript within the browser's scope and potentially obtain information about the page or manipulate its state.

The `NonEmptyNavigator`, therefore, represents *something* on the page. It isn't the browser itself, and that's the key distinction. When you attempt `someElement.getJs()`, Geb interprets this as, "I want to execute javascript relative to *this* element" when the getJs() API is not structured that way.

Let me give you some context based on my experience. Several years ago, I was working on an automated UI test suite for a large e-commerce platform. We used Geb extensively. We often found ourselves in situations where, after identifying elements, we needed to extract specific computed styles or attribute values not easily accessible through Geb's default `value()` or `attr()` methods. My team and I fell into the trap of attempting `element.getJs()`, mirroring what we had used with other tools, but ran head first into this exception. It was confusing until we understood that Geb's `getJs()` is meant to be called from the browser object, not the navigator.

So, where does that leave us? How do we correctly execute javascript and get the information we're looking for?

The solution revolves around using the `browser` object's `js` property directly, passing the `Navigator` (or the underlying element) to the javascript execution scope when necessary. Geb provides mechanisms to achieve this efficiently, using the `js.exec` and `js.eval` methods with an associated `Navigator`. This allows Geb to pass the wrapped element to the JavaScript, which can then be accessed within your script using the `arguments[0]` variable.

Here's a breakdown with examples:

**Scenario 1: Getting a Computed Style**

Instead of something like `$("div.myclass").getJs("return getComputedStyle(this).color")`, you'd have to do something like this using `js.eval`:

```groovy
import geb.Browser
import org.openqa.selenium.By

def browser = new Browser() // Assumed already configured for local execution
browser.go("data:text/html;charset=utf-8,<html><head></head><body><div class='myclass' style='color:red;'></div></body></html>")

def myElement = browser.$("div.myclass")

def computedColor = browser.js.eval(
    '''
      var element = arguments[0];
      return getComputedStyle(element).color;
    ''', myElement
  )

println "Computed color: ${computedColor}"
browser.quit()
```

In this case, the `js.eval` method passes `myElement` (a `Navigator`) to the javascript execution environment. Inside, `arguments[0]` contains the element, allowing us to extract the computed color.

**Scenario 2: Accessing an Element's Attribute**

Let's say you want to get the value of a custom data attribute:

```groovy
import geb.Browser

def browser = new Browser() // Assumed already configured for local execution
browser.go("data:text/html;charset=utf-8,<html><head></head><body><div id='mydiv' data-custom='somevalue'></div></body></html>")


def myDiv = browser.$("#mydiv")
def dataAttributeValue = browser.js.eval(
    '''
     var element = arguments[0];
     return element.getAttribute("data-custom");
    ''', myDiv
  )
println "Data Attribute: ${dataAttributeValue}"
browser.quit()
```

Here, the same logic applies. We pass the `Navigator` (referencing the div with id `mydiv`) to the `js.eval` method, accessing its `data-custom` attribute using native javascript.

**Scenario 3: Performing Actions on an Element Through JavaScript**

Now, let's perform some actions via JavaScript, such as changing an element's background color. Note, the js.exec function doesn't return a value, but executes arbitrary JS:

```groovy
import geb.Browser

def browser = new Browser() // Assumed already configured for local execution
browser.go("data:text/html;charset=utf-8,<html><head></head><body><div id='mydiv' style='background-color:white'></div></body></html>")

def myDiv = browser.$("#mydiv")

browser.js.exec(
  '''
    var element = arguments[0];
    element.style.backgroundColor = "red";
    ''', myDiv
)

def backgroundColor = browser.js.eval(
    '''
      var element = arguments[0];
      return getComputedStyle(element).backgroundColor
    ''', myDiv
)


println "Background color: ${backgroundColor}"

browser.quit()
```

This time we use `js.exec`, which returns nothing, to modify the background color, then use `js.eval` to retrieve the background color, all using the element we identified through Geb as an argument.

These examples should illustrate the core concept: `getJs()` is not directly available on `NonEmptyNavigator` instances. Instead, you use `browser.js.eval` or `browser.js.exec` and pass a `Navigator` as an argument.

For a deeper understanding, I would highly recommend delving into the official Geb documentation. It provides a comprehensive overview of its architecture and how JavaScript interactions are intended to be managed. You should also review the WebDriver API documentation, particularly regarding element interaction and javascript execution, since Geb relies on it heavily. Specifically, explore the methods pertaining to "executeScript" and related APIs.

Furthermore, consider reading "Selenium WebDriver 3 Practical Guide" by Boni Garcia. This book, while not specifically about Geb, provides a solid foundational understanding of how WebDriver works and, therefore, what Geb is built upon. This can be quite helpful in troubleshooting these type of issues and creating more robust, maintainable tests. In addition, "Test Automation Patterns and Implementation in Selenium WebDriver" by Pardeep Kumar could be highly beneficial to understand some common patterns in this type of test automation.

This `No signature` error, while initially confusing, serves as a good reminder of the separation of concerns that Geb employs and how it interacts with the underlying browser engine via WebDriver. It's a common hiccup, and once you understand how to effectively use `js.eval` and `js.exec`, the issue becomes quite manageable. Hopefully, these examples, along with the recommended resources, provide a clearer path forward.
