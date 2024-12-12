---
title: "What are some key considerations when transitioning from scraping Sales Navigator URLs to public profile URLs? (Teaching point: Explains workflow adjustments and tooling changes.)"
date: "2024-12-12"
id: "what-are-some-key-considerations-when-transitioning-from-scraping-sales-navigator-urls-to-public-profile-urls-teaching-point-explains-workflow-adjustments-and-tooling-changes"
---

Okay lets break down moving from scraping Sales Navigator URLs to public profile URLs its a shift in data source for sure and it hits several points you need to nail down first url structure and access thats the big one sales navigator urls are basically gated behind a login and those urls are often session based they have a bunch of extra parameters and sometimes they’re dynamically generated public profile urls on the other hand are designed for open access they’re typically cleaner like a `linkedin.com/in/some-user` format this means your scraper needs to handle those formats explicitly your code’s logic for url construction or parsing needs to be updated you cant use the same regex or string operations you used for sales nav urls the second big change is authentication sales nav needs a logged in session often with cookies and headers public profiles are accessible without logging in this is both a blessing and a curse you dont have to worry about authentication which simplifies your code a lot but it also means your rate limiting is usually more apparent linkedin can easily throttle you if they see a bunch of requests coming from the same ip address because there’s no logged in user session to manage they can more aggressively block you the scraper itself will need some tweaks for example your code will need new selectors for parsing the html structure it's often different between sales navigator and public profiles the html structure is never consistent you know that sales nav often uses specific class names or ids and the public profile pages have different ones you’ll need to use browser tools like the developer panel to figure out which selectors get the data you need if you’re using a library like `beautifulsoup` or `scrapy` you need to adjust your code that uses `find_all` or `css` selectors also consider data format if you’re used to sales nav output expect that data from public profiles might be organized differently or might be missing the data you extracted from sales nav like how many connections an user has on sales nav you can find information on the public profile page like the headline of the user or the experience history and so on you’ll need to rewrite parts of your scraping code to work with the new format even if you extracted similar information from both sources the structure of this info can vary consider that your data validation logic might need adjustment too. so lets talk code

```python
# Example of parsing a public profile URL using Beautiful Soup
from bs4 import BeautifulSoup
import requests

def scrape_public_profile(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Example of extracting the users headline
        headline_element = soup.find('div', class_='text-heading-xlarge inline t-24 v-align-middle break-words')
        if headline_element:
            headline = headline_element.text.strip()
            return headline
        else:
            return None

    else:
        return None

public_url = 'https://www.linkedin.com/in/some-user'
headline = scrape_public_profile(public_url)
if headline:
    print(f"Headline: {headline}")
else:
    print("Couldnt retrieve the headline")

```
this piece shows how to grab data from the public profile url it’s not using selenium for browser rendering so it assumes you don’t need to deal with javascript heavy rendering. it’s a direct `http` request and extraction with `beautifulsoup`

if youre doing this at a large scale you might need to change the data pipelines your data might flow into a database or different reporting tools if your old scraper loaded everything into a csv that you then used for some analysis you might have to change the columns or data structures to fit the new info now you might get a completely different output structure or missing certain fields you used to extract from sales nav so your data analysis and visualization processes needs to adapt to the new format rate limiting as mentioned before is crucial you might have been running your scraper quite aggressively against sales navigator but you’ll hit throttling problems quickly with public profiles you might want to use techniques such as random delays between requests or implement an intelligent retry mechanism with exponential backoff its also good to spread out your requests over a longer period you might want to change the user agent as well to avoid immediate detection consider rotating ips by using a proxy server or a vpn to reduce the chance of getting ip banned this is critical in any web scraping operation it protects against network level blocking or temporary bans and speaking of detection you should be careful with your code to not look like a robot if you send requests too quickly without delays the server will likely detect it's a bot and block you using libraries like selenium or playwright can help you simulate a real browser and reduce your chances of getting blocked these libraries can execute javascript and simulate user interaction which is harder to detect the cost however is they are more complex to setup and use compared to basic `requests`

```python
#Example of using Selenium for extracting data from a public profile url
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.chrome.options import Options

def scrape_public_profile_selenium(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        # Example of extracting the users headline using a selenium selector
        headline_element = wait.until(ec.presence_of_element_located((By.CSS_SELECTOR, 'div.text-heading-xlarge.inline.t-24.v-align-middle.break-words')))
        headline = headline_element.text.strip()
        return headline
    except Exception as e:
        print(f"An error occured {e}")
        return None
    finally:
      driver.quit()


public_url = 'https://www.linkedin.com/in/some-user'
headline = scrape_public_profile_selenium(public_url)
if headline:
    print(f"Headline {headline}")
else:
    print("couldnt retrieve the headline with selenium")
```

This example shows how to use selenium it sets up a headless browser and uses css selectors again but it is using `selenium` instead of `beautifulsoup` notice the explicit wait strategy which makes sure that you only start parsing the data after it has been loaded into the dom it reduces the race conditions that often happen in dynamically loaded pages this is a very useful technique if you’re dealing with javascript rendered content you will also need to handle error cases some users profile might have different layouts or might have incomplete sections you might get `none` results from selectors so your code should be able to gracefully handle these missing cases you could return a `null` value or empty string or you could use try catch blocks to handle missing fields to avoid crashing or misinterpreting data which might affect the accuracy of your data analysis.

another point you should also be cautious about terms of service you should review the linkedin terms of service related to web scraping as you might want to avoid breaking any rules. although public data is available its still owned by linkedin and scraping it too aggressively could lead to legal issues consider consulting with a legal professional if youre unsure. also as you scrape monitor your data quality and validate frequently to catch changes in linkedin’s html structure they might change the layout of pages and elements on the website without any warning which will break your scraper you should have a plan to detect these failures and implement fixes and consider using a monitoring dashboard so that you have real time visibility of your scraping operation data sources change and web pages change and scraping will need constant adjustment

```python
#Example using a retry decorator to avoid sudden failures
import requests
import time
from functools import wraps

def retry(max_tries=3, delay=1, backoff=2):
  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      tries = 0
      while tries < max_tries:
          try:
            return func(*args, **kwargs)
          except Exception as e:
            tries += 1
            time.sleep(delay)
            delay *= backoff #Exponential backoff for retrying
            print(f"Retrying {func.__name__} attempt {tries} after {delay} seconds. Exception {e}")
      raise Exception(f"Max retry attempts reached for {func.__name__}")
    return wrapper
  return decorator

@retry(max_tries=3, delay=1, backoff=2)
def get_data_with_retries(url):
  response = requests.get(url)
  response.raise_for_status()
  return response.content

public_url = "https://www.linkedin.com/in/some-user"
try:
  html_content = get_data_with_retries(public_url)
  print("Successfully retrieved the page")
except Exception as e:
  print(f"Failed to get the data {e}")

```

this shows an example of using a retry decorator this decorator wraps around a function and automatically retries it if there is an exception adding a delay between the retries and an exponential backoff is helpful to avoid the server from getting overwhelmed you could find more about web scraping best practices from the “web scraping with python” by ryan mitchell or check out the “data science from scratch” book by joel grus for more on data handling. also i recommend the paper “large scale web data extraction” by zhao et al. for more technical details on large scale scraping also check out selenium documentation and playwright documentation for handling more dynamic sites and complex scenarios. always test your scraper thoroughly on a few pages before launching it on thousands of pages and keep an eye out for rate limiting errors or blocking
