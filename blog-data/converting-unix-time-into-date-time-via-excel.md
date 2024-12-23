---
title: "converting unix time into date time via excel?"
date: "2024-12-13"
id: "converting-unix-time-into-date-time-via-excel"
---

 so you've got a bunch of unix timestamps and you need to wrangle them into something Excel understands aka date and time I've been there believe me it's like a rite of passage for anyone who's ever touched data that wasn't born inside an Excel spreadsheet

First off let's get this straight Unix time it’s not a calendar it’s a number the seconds since January 1 1970 0000 UTC that's your starting point Excel on the other hand thinks of dates as serial numbers days since January 1 1900 with time as the fractional part of the day So yeah you can see we're gonna need a translation layer

I remember this one time back in like 2015 I was scraping this ancient web API that just dumped data with unix timestamps all over the place I was building an internal dashboard and was struggling with it It was a mess I used to handle it with those online converters but it was so tedious after a while I had to build a process to do it on the fly This issue you are facing is exactly what I went through years ago

 so how do we go from this number to something we can actually read First thing you need is to understand the core concept Excel's date system has a base of January 1 1900 while Unix has January 1 1970 We need to account for that difference that is the starting point so you need to figure out the offset in days

We can accomplish this with a simple formula in excel You'll need to take that Unix timestamp and then convert it to days Add a fixed number of days for the epoch difference and then multiply by 86400 which is seconds per day and divide by it to get the fraction

So lets do it Excel's formula is pretty straightforward Let's say your unix timestamp is in cell A1 This is how the formula looks like

`=(A1/86400)+DATE(1970,1,1)`

That DATE function gives us Excel's January 1 1970 date and the whole thing together just gives us the right starting day You can format the cell as a date or date/time format to visualize it as human readable

Let's try a practical example Suppose you have a unix timestamp like `1678886400` that corresponds to March 15 2023 000000 UTC If you punch that number into the A1 cell and use this formula you will see Excel display 2023-03-15 or the proper date format

 let's say you are doing this programmatically maybe with VBA that's a bit more involved but still manageable The VBA approach is similar

```vba
Function UnixTimeToExcelDate(unixTime As Long) As Date
    UnixTimeToExcelDate = DateAdd("s", unixTime, "01/01/1970 00:00:00")
End Function
```
This function you just call it with any unix timestamp as a long and get back Excel date time variable This snippet is more concise because VBA has it's own date-time functions which take care of offset you can then assign to an excel cell for example `Range("B1").Value = UnixTimeToExcelDate(Range("A1").Value)` this will put the excel date-time output in the B1 cell given that cell A1 has the unix timestamp

The key here is the `DateAdd` function It takes the seconds from your unix timestamp and adds it to the excel date of "01/01/1970 00:00:00" this approach avoids direct calculation of offsets and it makes the code a bit more readable

Now what if you’re dealing with a whole column of these things Well you wouldn't want to manually copy and paste this formula everywhere would you? No we want automation

Let me show you a Python approach you could use Python with libraries like `openpyxl` or `pandas` to do batch conversions and updates to your excel spreadsheets It is the most elegant way to achieve it once you get the grasp of libraries and how to use python effectively I would start with pandas if I were you You can achieve it by doing something like

```python
import pandas as pd
from datetime import datetime

def unix_to_excel_date(unix_time):
    return datetime.utcfromtimestamp(unix_time)

# Sample data in a dictionary
data = {'unix_timestamp': [1678886400, 1678890000, 1678900000]} # example list of unix times

df = pd.DataFrame(data)

#apply the transformation to a new column
df['excel_datetime'] = df['unix_timestamp'].apply(unix_to_excel_date)

#save it to an excel file
df.to_excel("output.xlsx", sheet_name="Sheet1", index=False)

```

Here what happens first we take the data and put it in a pandas dataframe so we can easily do data processing then we apply a python function using the `apply` method on the whole series of timestamps This is extremely important because the function will get executed in each row and will make your life a lot easier It is efficient too and finally we write to excel and save the spreadsheet to disk This is a much more robust way of handling mass conversions if you have a lot of spreadsheets to process you can easily wrap this in a loop and you have a batch processing tool very fast

This python approach is more portable too because it doesn’t rely on Excel as much you can generate a excel sheet and then import it or email it as an attachment to someone else you can do it anywhere you have python installed It can become your go to solution if you get familiar with how to use it effectively in your process

Now some notes you want to be aware of When dealing with times remember that Unix timestamps are UTC times If your excel sheet needs to be in a local time zone you'll need to do a timezone conversion after the base conversion using date functions It's pretty important to get this step right unless you want to confuse everyone about their meeting times The timezone problem was a big headache of mine back in 2019 I was sending reports to clients in different countries and I had to build an extensive timezone converter that was really annoying to work on I feel like i should have done it with python instead of using excel functions... hindsight is 20/20

Now you might be asking “what if my timestamps are in milliseconds” Well then you just divide by 1000 before you do anything else because 1000 milliseconds is 1 second it is a simple math issue In addition check the format of the excel column so that is of type date and not other formats such as numeric or text

To deepen your knowledge I recommend you check out “Time Series Analysis” by James D Hamilton It is an amazing book you could also look into “Python for Data Analysis” by Wes McKinney if you want to use the python approach extensively it covers a lot of useful stuff You will have to do some research on your own because they do not focus on the exact issue but it will help you get a solid base in understanding more complex data manipulation scenarios These will give you a good background on the principles involved

Finally and this is something I always tell everyone dealing with data is to double check you are using correct data types because nothing is more frustrating than having to debug type errors in your data processing pipelines And speaking of pipelines did you hear about the programmer who got stuck in a loop? He just kept saying "break"

So in summary you can easily convert unix timestamps to excel datetime format using a simple formula or a VBA function or with python for batch processing with a little care with datatypes and timezones This should cover you and get you going on your data manipulation tasks
