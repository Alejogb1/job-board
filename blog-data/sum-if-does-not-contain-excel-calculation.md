---
title: "sum if does not contain excel calculation?"
date: "2024-12-13"
id: "sum-if-does-not-contain-excel-calculation"
---

 so you're asking about summing values in Excel-like data but with a condition specifically you want to sum only if a certain value is not present in a range right I've been down this rabbit hole before more times than I'd like to admit lets unpack this no nonsense style

First off lets clarify what I assume you're wrestling with. You've got something like a spreadsheet maybe its actually a CSV or a database table but the structure is essentially rows and columns one of your columns has numbers another one has say tags or categories that are strings and you want to sum the numbers only for rows where a specific string is missing in that tags column

I've seen this crop up countless times in data manipulation I remember once dealing with a massive user behavior dataset for an e-commerce site where I needed to filter out users who clicked on a particular "spam" tag and then calculate the total purchase value of the others. Nightmare fuel seriously I spent like half a day just untangling that query back then. Good times good times... I think

Anyway lets ditch the vague talk and go straight to the tech I'm going to give you a few ways to approach this in different environments just pick what suits your needs best I’ll assume you are not using a GUI but we are talking about excel data so I will go with something that can handle csv formatted data

**Python with Pandas**

Pandas is your friend for anything data related Seriously if you’re not already using it you are missing out I'm telling you here's how you'd do this with pandas its clean and readable:

```python
import pandas as pd

def sum_if_not_contains(data_path, sum_column, filter_column, excluded_value):
    """Sums values in a column based on a negative containment condition in another column.

    Args:
        data_path (str): Path to the CSV file.
        sum_column (str): The name of the column to sum.
        filter_column (str): The name of the column to filter on.
        excluded_value (str): The value to exclude from the filter column.

    Returns:
        float: The sum of values in the sum_column where the filter_column does not contain the excluded_value.
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error file not found at {data_path} make sure it exists")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error the CSV at {data_path} is empty")
        return None
    except Exception as e:
        print(f"An error ocurred: {e}")
        return None

    if sum_column not in df.columns or filter_column not in df.columns:
       print(f"Error the dataframe does not contain the required columns {sum_column} and {filter_column}")
       return None

    filtered_df = df[~df[filter_column].astype(str).str.contains(excluded_value,na=False)]

    return filtered_df[sum_column].sum()

# Example Usage
data_file = "mydata.csv"
sum_column_name = "price"
filter_column_name = "tags"
excluded_tag = "discount"

total_sum = sum_if_not_contains(data_file, sum_column_name, filter_column_name, excluded_tag)

if total_sum is not None:
    print(f"Sum where '{filter_column_name}' does not contain '{excluded_tag}': {total_sum}")

```

 a couple of things to note here. First I use a csv reader so you can replace the csv if you need a excel reader or any other reader I left it as simple as possible. Second thing is the `astype(str)` and `.str.contains` are key the `astype(str)` makes sure all your values are treated as strings before filtering. Third the `~` is the not operator in pandas. Lastly `na=False` makes sure all NAs or null values are automatically ignored. This is my personal style I like to program so that null values never cause exceptions unless you explicitly want them I have wasted too much time debugging with null values to be careful about it. This approach is super robust handles NaNs in data and is efficient.

**Command Line using `awk`**

If you are a more old-school kind of person like me and the data is not too big this is a great solution you might say this is overkill but I prefer this for simple quick stuff

```bash
awk -F, -v excluded_value="discount" '
    BEGIN { sum = 0 }
    {
        contains_excluded = 0;
        for (i=1; i<=NF; i++) {
           if ($i ~ excluded_value){
                contains_excluded = 1;
                break;
           }
        }
        if (contains_excluded == 0) {
            sum += $2; # Assumes sum column is the second column (index 2)
        }
    }
    END { print sum }
' mydata.csv
```

 lets get into this thing:

`-F,`: This tells `awk` that the delimiter is the comma a common delimiter for CSVs
`-v excluded_value="discount"`: This sets a variable called `excluded_value` that is equal to the value you want to exclude if you are looking for multiple values you can loop through a list of them in the code if needed.
`BEGIN { sum = 0 }`: This initializes a variable called `sum` to zero this is executed once in the beginning.
The next block ` { ... }`: Is executed for every row in the data file.

   `contains_excluded = 0`: Resets the flag that indicates if the row contains the excluded value
    `for (i=1; i<=NF; i++)`: Here `NF` represents the number of columns in the current row. The loop is used to go through every single cell in the row
    `if ($i ~ excluded_value)`: This compares the cell against the excluded value, if it matches `contains_excluded` is set to one and we break out of the loop
    `if (contains_excluded == 0)`: If `contains_excluded` is 0 means the row did not contain the excluded value then we add it to the sum `$2` here assumes your price/value is in the second column
    `END { print sum }`: This gets executed in the end and it prints the accumulated sum

This command line tool is very concise and fast for big data I have used it for gigabytes of data before. Just be careful with your column numbers make sure the column that you want to sum matches `$2` or edit the code

**SQL (If your data is in a Database)**

Now if you’ve got your data in a database you can skip this stuff and go with straight SQL it is better than anything else for data processing this is what I used back then with the massive ecommerce data.

```sql
SELECT SUM(price_column)
FROM your_table
WHERE tags_column NOT LIKE '%excluded_tag%';
```

This is pretty self-explanatory:

   `SUM(price_column)`: sums all the values from the column you want to sum
   `FROM your_table`: the table containing your data
   `WHERE tags_column NOT LIKE '%excluded_tag%'`: selects only the rows where the `tags_column` does not contain the `excluded_tag`

This is the easiest way if you know the basics of SQL.

**Important Considerations**

*   **Case Sensitivity**: If your filter is case sensitive you need to adjust the contains method in pandas or use a SQL database that cares about case sensitivity or just change the input before the search
*   **Data Cleaning**: Make sure your input data is clean before doing any of this there is nothing worse than getting unexpected results because of spaces in your input or different data types. I spent like 1 hour debugging a problem with pandas before just because a date had some space in the string.
*  **Performance**: If you are working with a very big data set pandas may start to be slow and awk and SQL will become much more performant and that is why you should use SQL or awk for those cases.

**Resources**

Here are some resource you might want to check out:

*   "Python for Data Analysis" by Wes McKinney if you want to master pandas
*   "The AWK Programming Language" by Aho Kernighan and Weinberger if you want to master awk.
*   Any good SQL query book should be enough for this but I liked "SQL for Dummies" if you are starting and want a simple book.

Look I hope this has been useful no funny business no strange analogies or metaphors just plain code and real examples I’ve dealt with this stuff before so yeah good luck and if you have other questions feel free to ask I’ll try to be as helpful as I can
