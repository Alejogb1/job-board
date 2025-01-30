---
title: "How can I retrieve iframe elements by tag name within a hidden (display:none) WebElement using Selenium Python?"
date: "2025-01-30"
id: "how-can-i-retrieve-iframe-elements-by-tag"
---
Selenium’s interaction with elements, particularly those nested within iframes and further complicated by `display: none`, presents unique challenges requiring careful navigation of the DOM structure and JavaScript context. Direct retrieval via `find_elements(By.TAG_NAME, 'iframe')` on a hidden parent element will likely yield an empty list. This occurs because Selenium's default element finding behavior considers visibility and interactability; hidden elements, while present in the DOM, are not typically considered for direct interactions or element location unless explicitly circumvented. My experiences building automated testing suites for complex web applications have led me to utilize a specific multi-step strategy involving script execution to reliably access iframes nested within hidden elements.

The core issue stems from the rendering pipeline within web browsers. An element with `display: none` is not rendered in the layout tree, even though it exists in the DOM. Selenium's default find mechanisms largely rely on the rendered layout. Consequently, the typical XPath or CSS selector based queries will often fail to return elements located within or associated with the hidden element. To effectively locate these hidden iframes, we must explicitly manipulate and query the DOM through JavaScript execution within the browser context. This approach bypasses Selenium’s rendering-based visibility checks.

We need to employ `execute_script()` to inject and run JavaScript code in the context of the browser. This allows us to retrieve all iframe elements within the hidden parent, irrespective of their rendering status. Once retrieved within the script, we pass these iframe elements back to Selenium, which can then be used for further manipulation.  This strategy effectively decouples the element location from Selenium's visibility constraints.

First, the JavaScript needs to obtain a reference to the parent element. This element, being hidden, needs to be identified through alternative methods. Suppose the hidden parent element has a specific ID such as 'hiddenContainer'. The JavaScript logic can locate this parent and subsequently all iframes within it. It will return an array of DOM `iframe` elements. Selenium's `execute_script` translates the array into a Python list of WebElement objects. These can then be directly interacted with like any other visible elements. This approach facilitates interactions such as switching to the retrieved iframes for element searches within them.

Here are three code examples illustrating this process:

**Example 1: Basic iframe retrieval within a hidden element.**

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Example setup with a placeholder HTML content.
html_content = """
<html>
<head><title>Example Page</title></head>
<body>
    <div id='hiddenContainer' style='display:none;'>
        <iframe id='iframe1' src='about:blank'></iframe>
        <iframe id='iframe2' src='about:blank'></iframe>
    </div>
</body>
</html>
"""

options = Options()
options.add_argument("--headless")
service = Service(executable_path="/path/to/chromedriver") #replace with your chromedriver path
driver = webdriver.Chrome(service=service, options=options)
driver.get("data:text/html;charset=utf-8," + html_content)

# Locate the hidden container element using its ID
hidden_container = driver.find_element(By.ID, "hiddenContainer")

# JavaScript to find iframes within the hidden container.
script = """
    var parentElement = arguments[0];
    var iframes = parentElement.getElementsByTagName('iframe');
    return Array.from(iframes);
"""

# Execute the script and retrieve iframe WebElement list.
iframes_list = driver.execute_script(script, hidden_container)

# Verify that two iframes are found.
print(f"Number of iframes found: {len(iframes_list)}")
assert len(iframes_list) == 2

# Interact with the first iframe (e.g. switching to it).
driver.switch_to.frame(iframes_list[0])

# Additional operations within the iframe can be performed here.
driver.switch_to.default_content()
driver.quit()
```

This example demonstrates how to first locate a hidden container by its ID, then uses JavaScript to retrieve all `iframe` elements within that container. The returned list of `WebElement` objects can then be used directly within Selenium.  Importantly, it uses the `arguments` variable of the Javascript `execute_script` function to pass the parent WebElement to the javascript context.

**Example 2:  Retrieving iframes with specific attributes from a hidden container.**

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Example setup
html_content = """
<html>
<head><title>Example Page</title></head>
<body>
    <div id='hiddenContainer' style='display:none;'>
        <iframe id='iframeA' class='target-iframe' src='about:blank'></iframe>
        <iframe id='iframeB' src='about:blank'></iframe>
         <iframe id='iframeC' class='target-iframe' src='about:blank'></iframe>
    </div>
</body>
</html>
"""

options = Options()
options.add_argument("--headless")
service = Service(executable_path="/path/to/chromedriver") #replace with your chromedriver path
driver = webdriver.Chrome(service=service, options=options)
driver.get("data:text/html;charset=utf-8," + html_content)

# Locate the hidden container.
hidden_container = driver.find_element(By.ID, "hiddenContainer")

# JavaScript to find iframes with class 'target-iframe'
script = """
    var parentElement = arguments[0];
    var iframes = parentElement.getElementsByTagName('iframe');
    var targetIframes = [];
    for (var i = 0; i < iframes.length; i++){
        if (iframes[i].classList.contains('target-iframe')){
            targetIframes.push(iframes[i])
        }
    }
    return Array.from(targetIframes);
"""

# Execute the script.
target_iframes = driver.execute_script(script, hidden_container)

# Verify correct number is found.
print(f"Number of targeted iframes: {len(target_iframes)}")
assert len(target_iframes) == 2

# Interact with the first iframe.
driver.switch_to.frame(target_iframes[0])

# Additional operations can be performed within the iframe.
driver.switch_to.default_content()
driver.quit()
```

This example extends the functionality by adding a filter within the JavaScript to retrieve only iframes with a specific class name, demonstrating selective retrieval. The Javascript includes a for-loop and checks if an element has the target class before adding it to the return array.

**Example 3:  Handling nested hidden containers and iframe retrieval.**

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Example setup
html_content = """
<html>
<head><title>Example Page</title></head>
<body>
    <div id='hiddenContainer1' style='display:none;'>
       <div id='hiddenContainer2' style='display:none;'>
          <iframe id='iframeX' src='about:blank'></iframe>
       </div>
    </div>
</body>
</html>
"""

options = Options()
options.add_argument("--headless")
service = Service(executable_path="/path/to/chromedriver") #replace with your chromedriver path
driver = webdriver.Chrome(service=service, options=options)
driver.get("data:text/html;charset=utf-8," + html_content)

# Locate the first hidden container using its ID
hidden_container1 = driver.find_element(By.ID, "hiddenContainer1")

# Locate the second hidden container within the first hidden container
script_container2 = """
    var parentElement = arguments[0];
    var container2 = parentElement.querySelector("#hiddenContainer2");
    return container2;
"""
hidden_container2 = driver.execute_script(script_container2,hidden_container1)

# JavaScript to find iframes within the second hidden container.
script_iframe = """
    var parentElement = arguments[0];
    var iframes = parentElement.getElementsByTagName('iframe');
    return Array.from(iframes);
"""

# Execute the script to retrieve iframes.
iframes_list = driver.execute_script(script_iframe, hidden_container2)

# Verify that one iframe is found
print(f"Number of iframes found: {len(iframes_list)}")
assert len(iframes_list) == 1

# Switch to the iframe.
driver.switch_to.frame(iframes_list[0])
driver.switch_to.default_content()
driver.quit()
```
This example demonstrates the retrieval of an iframe from within a nested set of hidden containers. This required two separate `execute_script` calls. The first script locates the second, nested hidden container, using `querySelector` within Javascript; this approach is preferred when the nested hidden element does not have a parent-element ID that could be used as the starting point of the selector.

For further exploration, resources such as the Selenium documentation, which details the `execute_script` method, and any introductory Javascript documentation which details the function `getElementsByTagName` would be beneficial. Also relevant are resources detailing `querySelector`, and DOM manipulation via Javascript. Understanding browser DOM structures will further clarify the need for specific approaches when dealing with hidden or dynamically rendered elements.
