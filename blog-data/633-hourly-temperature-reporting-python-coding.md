---
title: "6.3.3 hourly temperature reporting python coding?"
date: "2024-12-13"
id: "633-hourly-temperature-reporting-python-coding"
---

Alright so hourly temperature reporting with Python yeah I’ve been down that rabbit hole a few times Let me tell you it's not as straightforward as it sounds when you actually need to make it robust and reliable I've seen so many junior devs just slap together something that works for a few hours then crashes and burns when it hits a real world edge case

Ok so you're looking at hourly reporting right That means you need to read in temperature data from somewhere process it then output it hourly Simple right Not quite The devil as always is in the details I've learned this the hard way trust me I once had a sensor logging data every 5 seconds which was fine for testing until I got a thousand readings a minute and my system choked Turns out I needed some serious aggregation logic

First things first you're gonna need to grab temperature data Where is it coming from a CSV file a database an API or some kind of hardware sensor This makes a difference Let’s say you're dealing with a CSV file named `temperature_data.csv` with a basic structure like `timestamp,temperature_celsius` So we'll do CSV for the purpose of my explanation

```python
import csv
from datetime import datetime
from collections import defaultdict

def process_temperature_data(filepath):
    hourly_temps = defaultdict(list)
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # Skip the header
        for row in reader:
            try:
              timestamp_str, temp_str = row
              timestamp = datetime.fromisoformat(timestamp_str)
              temp = float(temp_str)
              hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
              hourly_temps[hour_key].append(temp)
            except ValueError as e:
                print(f"Error processing row {row}: {e}")
                continue #skip badly formed or problematic data

    return hourly_temps

def calculate_hourly_averages(hourly_temps):
    hourly_averages = {}
    for hour, temps in hourly_temps.items():
      if temps:
        hourly_averages[hour] = sum(temps) / len(temps)
      else:
        hourly_averages[hour]=None #if there is no data for the hour
    return hourly_averages

if __name__ == "__main__":
    filepath = "temperature_data.csv"
    hourly_temps = process_temperature_data(filepath)
    hourly_averages= calculate_hourly_averages(hourly_temps)
    for hour, avg_temp in hourly_averages.items():
        print(f"Hour: {hour}  Average Temp: {avg_temp} ")
```

Ok so this is a basic processing script It reads a CSV processes it and calculates hourly average temperatures We’re using a `defaultdict` which is awesome because it handles cases where a new hour is encountered without throwing KeyErrors It assumes your timestamp is in ISO format which is generally the safe bet It also adds some error handling to avoid crashing on invalid rows which believe me you need I've learned the hard way that data is rarely pristine.

Now remember that `datetime.fromisoformat()` only works if you provide iso format strings so you might want to change it if your date format is different I’ve spent hours debugging date formats in the past Trust me its boring

Also pay attention to the `try-except` block there It's a good practice especially when dealing with external data This is where I messed up once I had a sensor that suddenly started logging garbage data and I hadn’t handled it properly My whole reporting system crashed and I had a very angry manager to explain to

Next you're probably gonna want to store this data somewhere A simple CSV will do for some use cases but you may want to store it in a database for larger applications where you need to do more complex aggregations over time

Here’s an example on how you could insert data into a SQLite database this is for a simpler setup

```python
import sqlite3
from datetime import datetime

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hourly_temperatures (
                hour TEXT PRIMARY KEY,
                average_temperature REAL
            )
        """)
    except sqlite3.Error as e:
       print(e)
def insert_average_temperatures(conn, hourly_averages):
    sql = ''' INSERT INTO hourly_temperatures(hour,average_temperature)
              VALUES(?,?) '''
    try:
      cursor= conn.cursor()
      for hour, avg_temp in hourly_averages.items():
        cursor.execute(sql,(hour.isoformat(),avg_temp)) #convert to string for sqlite
      conn.commit()
    except sqlite3.Error as e:
        print(f"Error inserting data: {e}")

if __name__ == "__main__":
    database_file = "temperature_data.db"
    conn = create_connection(database_file)

    if conn is not None:
        create_table(conn)
        filepath = "temperature_data.csv"
        hourly_temps = process_temperature_data(filepath)
        hourly_averages = calculate_hourly_averages(hourly_temps)
        insert_average_temperatures(conn, hourly_averages)
        conn.close()
    else:
        print("Error cannot create database connection")
```

This script uses the sqlite3 library It creates a database and a table to store the hourly averages It also uses parameterized queries for data insertion which is absolutely crucial to avoid SQL injection vulnerabilities I once forgot this on a project and well it's not a fun story I wont get into it but if you see `cursor.execute(sql, values)` that is good and you are safe but If you ever see something like `cursor.execute(f"INSERT INTO table VALUES ({value})")` you will regret it trust me.

Now another approach would be if you are dealing with streaming data or you have a more complex setup You might want to use libraries like Pandas if it fits your use case

Here's a quick example using pandas to read and process CSVs:

```python
import pandas as pd

def process_temperature_data_pandas(filepath):
    try:
      df = pd.read_csv(filepath, parse_dates=['timestamp'])
      df['hour'] = df['timestamp'].dt.floor('H') # Truncate to hour
      hourly_averages = df.groupby('hour')['temperature_celsius'].mean().reset_index()
      return hourly_averages
    except Exception as e:
      print(f"Error: {e}")
      return None

if __name__ == "__main__":
    filepath = "temperature_data.csv"
    hourly_averages_df = process_temperature_data_pandas(filepath)
    if hourly_averages_df is not None:
        for index, row in hourly_averages_df.iterrows():
            print(f"Hour: {row['hour']}, Average Temp: {row['temperature_celsius']}")
```

Pandas is really convenient for data manipulation and it automatically handles missing data nicely It's especially useful if you are working with large datasets I mean we're talking millions of rows of data It can handle it like it is nothing And by the way I once had to handle a 10 gigabyte CSV file so yeah Pandas saved my day

If you are working with large data processing or data science you will probably be using it so learning this is not going to be a loss of time

Now you’ll notice that all these examples process past data In the real world you may want to make these processes run continuously and process data as it arrives This usually involves some kind of scheduler to periodically run the script or even using asynchronous programming if it is a live data stream

Here are some good resources I recommend if you want to deep dive into this I would start by checking the Python documentation itself is great for libraries like `csv` `datetime` `sqlite3` For data manipulation and analysis with python you should get a copy of "Python for Data Analysis" by Wes McKinney this book is a classic and pretty much a must read and if you want more complex data workflows check "Designing Data-Intensive Applications" by Martin Kleppmann it is a bit theoretical but it goes very deep into how the system works

One more thing about temperature sensors they’re usually inaccurate so it's good practice to have some kind of calibration routine I once spent 2 weeks debugging a sensor which was just off by 2 degrees It was not fun so try to not make the same mistake

So there you have it Hourly temperature reporting with python isn't rocket science but it requires careful consideration of the details you can easily find yourself down a rabbit hole if you don't take time to actually consider the specific requirements of the problem And remember always handle errors and log everything And most importantly remember not to debug without coffee.
