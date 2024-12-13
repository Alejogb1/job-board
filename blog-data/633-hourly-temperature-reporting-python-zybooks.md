---
title: "6.3.3 hourly temperature reporting python zybooks?"
date: "2024-12-13"
id: "633-hourly-temperature-reporting-python-zybooks"
---

Okay so you're wrestling with zybooks 6.3.3 hourly temperature reporting in python right Been there done that multiple times actually let me tell you it's one of those exercises that sounds simple but can trip you up if you're not careful 

I remember back in the day when I was first getting into python I got stuck on this exact problem for hours I think it was around 2018 or something and I was trying to optimize the script for some weird reason anyway it took me like 3 different iterations to finally nail it It's always the simple things that get you I swear

So what we're essentially dealing with is reading a list or a set of temperature readings and reporting the temperature for each hour of the day usually from some data source whether it's a file or some input I believe in zybooks they usually use stdin but yeah we need to iterate over that data and format it in a human-readable manner 

First the key thing is parsing the input properly zybooks usually just gives you the data and doesn't tell you how its formatted explicitly so you need to be careful You need to understand what the input looks like before even thinking about the code it's the classic garbage in garbage out situation

Let's say you receive the input as a space separated string of floats or integers which represent the temperature every hour of the day starting from hour 0 and going all the way to hour 23 Okay so here's the most basic solution I would say to tackle that

```python
def report_hourly_temperatures(temperature_data):
  temperatures = list(map(float, temperature_data.split()))
  
  for hour, temp in enumerate(temperatures):
    print(f"Hour {hour}: {temp:.2f}")

# Example usage (with stdin in zybooks)
# assuming input is on a single line
temperature_input = input()
report_hourly_temperatures(temperature_input)
```

Okay so this works fine right Its straightforward we just split the input string convert them to floats enumerate through it and print formatted output with 2 decimal places no rocket science here 

But what if you have more complex scenarios right Like what if you need to handle missing data or you want to filter the data for specific temperature ranges things can get a bit hairy pretty fast

For instance consider this example maybe we are handling data from some old weather station and it uses `-999` or a `NaN`  value to indicate a sensor malfunction in this case we want to filter or skip these readings because they will destroy the report's integrity right

Here's what you might do in that case

```python
import math

def report_hourly_temperatures_with_missing(temperature_data):
  temperatures = temperature_data.split()
  
  for hour, temp_str in enumerate(temperatures):
      try:
          temp = float(temp_str)
          if math.isnan(temp) or temp == -999:
            print(f"Hour {hour}: Data unavailable")
          else:
            print(f"Hour {hour}: {temp:.2f}")
      except ValueError:
        print(f"Hour {hour}: Invalid data") 

# Example usage
temperature_input = input() # Example:  22.5 23.1 NaN 24 -999 25.6 26 27 27.2 28 28.5 29 28 27 26 25 24 23 22 21 20 19 18 17
report_hourly_temperatures_with_missing(temperature_input)
```

This is still pretty simple but now we are using some try except block to handle potential non numerical data and a condition to check `NaN` or `-999` for special cases right We are using `math.isnan` to check for `NaN` in the input and handling those cases by reporting "Data unavailable" instead of blowing up the script with a traceback

I want to say that dealing with `NaN` is extremely common when you get your hands dirty with real world data So be aware of that

Another situation might involve working with very big dataset where storing the entire input in a list before processing might be inefficient This is common when we deal with large amount of data that cannot fit in memory

So let's say you want to process large data files and output a report without storing all data in memory first because why not It’s always good to improve your code

Here is a generator version of the solution that I actually used back in college that I still find useful

```python
def generate_hourly_temperatures(temperature_data):
    for hour, temp_str in enumerate(temperature_data.split()):
        try:
            temp = float(temp_str)
            if not math.isnan(temp) and temp != -999: # Filter out invalid data here as we did before
              yield f"Hour {hour}: {temp:.2f}"
            else:
               yield f"Hour {hour}: Data unavailable"
        except ValueError:
           yield f"Hour {hour}: Invalid data"

# Example usage
temperature_input = input()
for report_line in generate_hourly_temperatures(temperature_input):
    print(report_line)
```

Now this is the interesting part This version employs a generator function `generate_hourly_temperatures` instead of processing the entire data set at once This means that data is only processed when required and printed on each iteration not at the start It’s a much more efficient way to handle larger datasets that would otherwise cause memory errors

As you can see each approach has its advantages and disadvantages the right one depends entirely on the scenario and your data I always say the art of programming is choosing the right tool for the job

I know this looks easy enough but trust me there are tons of these little things that can be difficult to deal with when you start This is why I think zybooks or courses in general are important because they teach you how to think step by step and find the answer instead of just giving it to you 

For more in depth look into data handling and efficient processing of larger dataset there are a couple of really good resources you should check out one is "Fluent Python" by Luciano Ramalho I think it's an absolute must read for anyone who's serious about python Also look into "Python Cookbook" by David Beazley and Brian K Jones It’s a bible for solving common python programming problems They're lifesavers trust me

Also another good resource if you plan on going the data scientist or numerical programmer route is "Numerical Python" by Robert Johansson It is an amazing resource for understanding the inner workings of numpy a critical tool in the python ecosystem if you deal with numeric data a lot

I always get a chuckle when I see beginners use lists to store simple temperature data which can be handled with a generator function like in the last example it's like using a firetruck to put out a candle. You could just use a glass of water you know

Anyway I hope that helps and if you're stuck on something similar feel free to ask I'm always happy to help even if it's something as seemingly simple as hourly temperature reporting
