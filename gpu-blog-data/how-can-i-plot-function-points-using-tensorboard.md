---
title: "How can I plot function points using TensorBoard?"
date: "2025-01-30"
id: "how-can-i-plot-function-points-using-tensorboard"
---
TensorBoard, while primarily designed for visualizing machine learning model training metrics like loss and accuracy, can be adapted to plot function points, providing valuable insights into software development progress and productivity. This requires a custom logging mechanism integrated into the development process, which transforms function point data into a format consumable by TensorBoard. The core challenge lies in representing abstract function points as scalar values over time, enabling visualization as trends rather than inherent model performance.

The core strategy centers on utilizing TensorBoard's `SummaryWriter` to log scalar values corresponding to function points. Each log step ideally corresponds to a development time unit, such as a day, week, or sprint. Therefore, I've had to design a wrapper that maintains a record of function point totals and outputs the deltas into the event stream. The initial setup begins with establishing a mechanism for tracking function point completion. This entails defining a data structure or persistent store to maintain records of all planned function points, including estimated effort, actual effort, and completion status. I generally use a SQL database for this purpose.

To leverage this, function point data, obtained from a software development tracking system, needs to be serialized into a format compatible with TensorBoard. This involves mapping specific information, such as completed function points, to scalar values. The approach involves two primary steps: 1) creating a method to retrieve the relevant function point data, and 2) using `SummaryWriter` to log these extracted function points at each reporting interval. Let's look at an example.

**Code Example 1: Function Point Retrieval and Basic Logging**

```python
import sqlite3
from torch.utils.tensorboard import SummaryWriter
import datetime
import time

class FunctionPointLogger:
    def __init__(self, db_path, log_dir):
        self.db_path = db_path
        self.writer = SummaryWriter(log_dir=log_dir)
        self.last_logged_total = 0
        self.current_step = 0

    def retrieve_function_point_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(actual_effort) FROM function_points WHERE status='completed'")
        total_completed_effort = cursor.fetchone()[0] or 0  # Handle case where there's no data
        conn.close()
        return total_completed_effort

    def log_function_points(self):
        total_completed = self.retrieve_function_point_data()
        if total_completed > self.last_logged_total:
          delta = total_completed - self.last_logged_total
          self.writer.add_scalar('function_points/completed_since_last_log', delta, global_step=self.current_step)
          self.last_logged_total = total_completed
        self.writer.add_scalar('function_points/total_completed', total_completed, global_step=self.current_step)
        self.current_step +=1

    def close(self):
        self.writer.close()


# Example usage:
db_path = "function_points.db"  # Path to SQLite database
log_dir = "function_point_logs" # TensorBoard log directory

# create dummy database (not for production)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS function_points (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        actual_effort REAL,
        status TEXT
    )
""")
for i in range(10):
  cursor.execute("INSERT INTO function_points (actual_effort, status) VALUES (?, ?)", (i*2.5, "completed"))
for i in range(5):
  cursor.execute("INSERT INTO function_points (actual_effort, status) VALUES (?, ?)", (i*1.5, "in progress"))
conn.commit()
conn.close()


logger = FunctionPointLogger(db_path, log_dir)

for i in range(10):
  logger.log_function_points()
  time.sleep(1) # simulate time passing, the sleep is only for demo purposes

# additional data points
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
for i in range(3):
  cursor.execute("INSERT INTO function_points (actual_effort, status) VALUES (?, ?)", (i*3.0, "completed"))
conn.commit()
conn.close()
for i in range(5):
  logger.log_function_points()
  time.sleep(1)


logger.close()
```

In this first example, I’ve initialized `FunctionPointLogger` to handle logging to TensorBoard. The `retrieve_function_point_data` function fetches the total completed function point effort from an SQLite database (which is replaced with a robust data access later) by querying the `function_points` table. The `log_function_points` function then writes two scalar values to the TensorBoard log: total completed function point effort as well as the completed effort delta. The `global_step` parameter provides a time axis, ensuring the data is displayed correctly in TensorBoard's time series graphs. The example shows a basic structure, but this structure can be enhanced further.

**Code Example 2: Enhanced Logging with Multiple Metrics**

```python
import sqlite3
from torch.utils.tensorboard import SummaryWriter
import datetime
import time

class EnhancedFunctionPointLogger:
    def __init__(self, db_path, log_dir):
        self.db_path = db_path
        self.writer = SummaryWriter(log_dir=log_dir)
        self.last_logged_completed_total = 0
        self.last_logged_estimated_total = 0
        self.current_step = 0

    def retrieve_function_point_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(actual_effort), SUM(CASE WHEN status='completed' THEN actual_effort ELSE 0 END), SUM(CASE WHEN status!='completed' THEN actual_effort ELSE 0 END) FROM function_points")
        total_effort, completed_effort, remaining_effort = cursor.fetchone()
        conn.close()
        return total_effort or 0, completed_effort or 0 , remaining_effort or 0

    def log_function_points(self):
        total_effort, completed_effort, remaining_effort = self.retrieve_function_point_data()

        if completed_effort > self.last_logged_completed_total:
          delta = completed_effort - self.last_logged_completed_total
          self.writer.add_scalar('function_points/completed_since_last_log', delta, global_step=self.current_step)
          self.last_logged_completed_total = completed_effort
        self.writer.add_scalar('function_points/total_completed', completed_effort, global_step=self.current_step)
        self.writer.add_scalar('function_points/total_estimated', total_effort, global_step=self.current_step)
        self.writer.add_scalar('function_points/total_remaining', remaining_effort, global_step=self.current_step)
        self.current_step +=1

    def close(self):
        self.writer.close()


# Example usage:
db_path = "function_points_enhanced.db"  # Path to SQLite database
log_dir = "function_point_logs_enhanced" # TensorBoard log directory

# create dummy database (not for production)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS function_points (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        actual_effort REAL,
        status TEXT
    )
""")
for i in range(10):
  cursor.execute("INSERT INTO function_points (actual_effort, status) VALUES (?, ?)", (i*2.5, "completed"))
for i in range(5):
  cursor.execute("INSERT INTO function_points (actual_effort, status) VALUES (?, ?)", (i*1.5, "in progress"))
conn.commit()
conn.close()


logger = EnhancedFunctionPointLogger(db_path, log_dir)
for i in range(10):
  logger.log_function_points()
  time.sleep(1) # simulate time passing, the sleep is only for demo purposes

# additional data points
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
for i in range(3):
  cursor.execute("INSERT INTO function_points (actual_effort, status) VALUES (?, ?)", (i*3.0, "completed"))
conn.commit()
conn.close()
for i in range(5):
  logger.log_function_points()
  time.sleep(1)

logger.close()
```

In this improved example, `EnhancedFunctionPointLogger`, I've expanded data collection to include total estimated effort, completed effort, and remaining effort in addition to completed deltas. I retrieve these three values using a modified SQL query and then add each to the summary as its own scalar value. This allows for analysis not just of how much work has been done but how that compares to the original estimate and current remaining effort. Using a combination of time series, it is now possible to track the overall estimated work, progress over time, and total remaining work. This approach helps visualize project progress more holistically.

**Code Example 3: Integrating with a More Realistic Project Setup**

```python
import sqlite3
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import random

class RealisticFunctionPointLogger:
    def __init__(self, db_path, log_dir):
        self.db_path = db_path
        self.writer = SummaryWriter(log_dir=log_dir)
        self.last_logged_completed_total = 0
        self.last_logged_estimated_total = 0
        self.current_step = 0

    def retrieve_function_point_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(estimated_effort), SUM(CASE WHEN status='completed' THEN actual_effort ELSE 0 END), SUM(CASE WHEN status!='completed' THEN estimated_effort ELSE 0 END) FROM function_points")
        total_estimated_effort, completed_effort, remaining_estimated_effort = cursor.fetchone()
        conn.close()
        return total_estimated_effort or 0, completed_effort or 0 , remaining_estimated_effort or 0

    def log_function_points(self):
        total_estimated_effort, completed_effort, remaining_estimated_effort = self.retrieve_function_point_data()
        if completed_effort > self.last_logged_completed_total:
          delta = completed_effort - self.last_logged_completed_total
          self.writer.add_scalar('function_points/completed_since_last_log', delta, global_step=self.current_step)
          self.last_logged_completed_total = completed_effort
        self.writer.add_scalar('function_points/total_completed', completed_effort, global_step=self.current_step)
        self.writer.add_scalar('function_points/total_estimated', total_estimated_effort, global_step=self.current_step)
        self.writer.add_scalar('function_points/total_remaining', remaining_estimated_effort, global_step=self.current_step)
        self.current_step += 1
    def close(self):
      self.writer.close()


# Example usage:
db_path = "function_points_realistic.db"  # Path to SQLite database
log_dir = "function_point_logs_realistic" # TensorBoard log directory

# create dummy database (not for production)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS function_points (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        estimated_effort REAL,
        actual_effort REAL,
        status TEXT
    )
""")

# Function to generate random effort
def generate_random_effort(base_effort):
  return base_effort + random.uniform(-base_effort * 0.2, base_effort * 0.2)

for i in range(20):
  estimated = generate_random_effort(5.0)
  cursor.execute("INSERT INTO function_points (estimated_effort, actual_effort, status) VALUES (?, ?, ?)", (estimated, 0, "planned"))

for i in range(8):
  estimated = generate_random_effort(5.0)
  actual = generate_random_effort(estimated)
  cursor.execute("INSERT INTO function_points (estimated_effort, actual_effort, status) VALUES (?, ?, ?)", (estimated, actual, "completed"))

conn.commit()
conn.close()

logger = RealisticFunctionPointLogger(db_path, log_dir)
for i in range(20):
    # Simulate progress by randomly completing work:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # select 2 random planned entries and update them to completed
    cursor.execute("SELECT id, estimated_effort FROM function_points WHERE status = 'planned' LIMIT 2")
    rows = cursor.fetchall()
    for row in rows:
      id, estimated_effort = row
      actual_effort = generate_random_effort(estimated_effort)
      cursor.execute("UPDATE function_points SET status = 'completed', actual_effort = ? WHERE id = ?", (actual_effort, id))
    conn.commit()
    conn.close()
    logger.log_function_points()
    time.sleep(1)


logger.close()
```

In this final example, `RealisticFunctionPointLogger`,  I incorporated more realistic project dynamics.  The database structure now includes both estimated effort and actual effort.  The database is populated with planned function points, and these are then gradually changed to completed in the logging loop.  The logging itself uses a combination of estimated and actual effort to calculate remaining effort. This shows the value of both tracking estimated efforts, as these may change over time based on new data. This approach better reflects a real-world development scenario. I've added realistic estimates and a random simulation to show how the graph would behave in practice.

For further reading and information, I highly recommend reviewing the official PyTorch documentation regarding TensorBoard usage and the `SummaryWriter` class. This provides a robust understanding of the TensorBoard API, allowing for more advanced logging and customization. For more background in software engineering project management, there are multiple resources from reputable publishing houses that detail function points and their use in productivity tracking. Finally, to better understand database structure and usage, resources regarding SQL databases will be highly valuable. It is crucial to note that the example code relies on SQLite for simplicity, but production systems should consider more appropriate database solutions.  Adapting these principles and using appropriate data retrieval mechanisms will enable you to leverage TensorBoard's visualization capabilities effectively for function point analysis.
