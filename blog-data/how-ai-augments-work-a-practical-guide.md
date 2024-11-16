---
title: "How AI Augments Work: A Practical Guide"
date: "2024-11-16"
id: "how-ai-augments-work-a-practical-guide"
---

dude you will *not* believe this talk i just watched it's like a mind-blowing combo of spreadsheets and ai and it totally changed how i think about, well, everything.  the whole thing's about how ai can make our work lives way less painful and more efficient it's not just about robots taking our jobs it's about ai becoming a super-powered assistant.  the speaker, super chill btw, starts by talking about how accounting used to be a nightmare before spreadsheets.  i mean, imagine manually calculating everything and crossing things out with ink like it was some medieval manuscript.  


the whole setup is like "remember how awful things used to be?  now imagine ai making it *way* better."  he uses this as a springboard to talk about two main ways ai interacts with interfaces: automation and augmentation.  automation's the "doing stuff for you" part like automating those tedious calculations.  augmentation is more like giving you superpowers, enhancing your abilities, not replacing them.  


he makes a *killer* point though, he says augmentation is basically just a bunch of little automations stacked on top of each other.  to analyze data, you'll need to automate tasks like data aggregation and visualization before you can actually analyze anything meaningful.  it's like building a really cool Lego castle – you need to assemble all the individual bricks (automations) before you can stand back and admire the finished product (augmentation).


one visual cue that stuck with me was the image of those old, messy accounting ledgers. talk about a throwback!  another great visual was his demo of zooming in and out on a text using a large language model (LLM).  i mean it really made sense of his point of layering.  he even mentioned a specific example of the story arc in a book which made it even more relatable.


the main idea is this "ladder of abstraction" thing.  he uses google maps as a perfect example.  at the zoomed-in level, you see individual buildings but when you zoom out, you see streets, then highways, then states.  the level of detail changes depending on the task.  you don't need to see every single building to get from san francisco to los angeles.


he showed this amazing demo of Peter Pan using an LLM to create different levels of summaries. you could go from reading every word to seeing one-sentence summaries of paragraphs, or even one-sentence summaries of whole chapters! it was so cool.  the code behind something like that isn't trivial, but let me give you a *super* simplified python example using a hypothetical LLM API:


```python
import openai  # remember to install the openai library!

def summarize_text(text, level):
    prompt = f"Summarize the following text at level {level}:\n\n{text}"
    response = openai.Completion.create(
        engine="text-davinci-003",  # replace with your preferred model
        prompt=prompt,
        max_tokens=150,  # adjust as needed
        n=1,
        stop=None,
        temperature=0.5,  # adjust for creativity
    )
    summary = response.choices[0].text.strip()
    return summary

text = """ (insert Peter Pan text here) """
chapter_summary = summarize_text(text, "chapter")
paragraph_summary = summarize_text(text, "paragraph")
#etc.
print(f"Chapter Summary: {chapter_summary}")
print(f"Paragraph Summary: {paragraph_summary}")
```

this is a super simple version. a real implementation would involve clever prompt engineering and likely need to use some sort of state management and possibly some other things to deal with long texts.

another code example (again, simplified):  imagine generating different visualizations from data.  


```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  #for some calculations, probably

# sample data (replace with your actual data)
data = {'price': np.random.rand(50)*1000,
        'distance': np.random.rand(50)*10,
        'wifi': np.random.randint(0, 10, 50)}  # 0-10 scale
df = pd.DataFrame(data)


# different levels of visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)  #first plot
plt.scatter(df['distance'], df['price'], c=df['wifi'], cmap='viridis')
plt.title('Scatter Plot: Distance vs. Price')
plt.colorbar(label='Wifi Rating')

plt.subplot(1, 3, 2)  #second plot
plt.hist(df['price'], bins=10)
plt.title('Price Histogram')

plt.subplot(1, 3, 3)  #third plot, a boxplot for each level
df.boxplot(column=['distance', 'price'], by='wifi')  #a boxplot for each wifi rating

plt.tight_layout()
plt.show()

```
this shows how you could use the same data to generate different visualizations, each suited for a different task and level of detail, which the speaker discusses extensively.

and finally, here's an example relating to automating actions on a web page (think interacting with that Airbnb page):


```python
# this is a super simplified example and requires appropriate libraries
# and understanding of web scraping/automation.  do NOT use this for malicious purposes

from selenium import webdriver #selenium is for this one

# (initialize the webdriver - chrome, firefox etc - you'll need correct browser drivers)
driver = webdriver.Chrome()

# navigate to Airbnb listing page
driver.get("https://www.airbnb.com/some-listing-url")

# find and extract information (replace with actual selectors)
price = driver.find_element("xpath", "//span[@class='price']").text  #this part is VERY website specific
distance = driver.find_element("xpath", "//div[@class='distance']").text
wifi_reviews = driver.find_elements("xpath", "//div[@class='review-wifi']") #get all the reviews related to wifi

# perform actions (replace with actual methods)
# (you might need to use the LLMs to handle more complex data transformation and analysis here)
#book_button = driver.find_element("id", "book-button")
#book_button.click()  

# remember to close the browser!
driver.quit()
```


this is simplified.  real-world scenarios would involve much more complex interactions with the webpage, possibly using JS, error handling, and much more.

the resolution of the talk is that ai can make all this way easier. by automating small tasks, ai helps us focus on the bigger picture. by understanding the ladder of abstraction, we can use ai to build interfaces that let us quickly move between different levels of detail, getting exactly the information we need when we need it and doing things that we can do more readily and without undue effort. it’s not about replacing humans—it’s about becoming super-humans.  the speaker's company, adept, is all about building tools to make this a reality.  it's all about streamlining and making knowledge work less of a mental workout.  it's super exciting stuff.
