---
title: "How to load multiple files into multiple tables using SSIS for each loop?"
date: "2024-12-15"
id: "how-to-load-multiple-files-into-multiple-tables-using-ssis-for-each-loop"
---

alright, so you're looking at a classic etl challenge, loading data from multiple files into corresponding tables using ssis's for each loop container. i've been down this road more times than i care to remember. trust me, it’s a very common scenario. the devil, as usual, is in the details, but let's get to it.

first off, the for each loop is your friend here. it’s the workhorse that lets you iterate through a collection – in this case, a folder of files. the trick is setting it up correctly so each file lands in the correct table. here's how i generally approach it, having made a few mistakes along the way (mostly involving messy file paths, i assure you).

the key is variables. you absolutely need variables to hold the current file path, the table name, and potentially other information that you might need as you loop. it's all about making that dynamic connection for each iteration. a good practice to follow is to never hardcode paths, file names or table names, not even once. variables, variables, and variables.

i remember one time, back when i was just starting out, i hardcoded file paths in my ssis packages. i was convinced it was "faster" and more efficient (narrator: it wasn't). then i moved my dev environment to a new server, and everything broke. hours of debugging later, i understood the power and flexibility of parameters, oh the sweet sweet taste of parameters and configuration. never again, i told myself. never again. lesson learned, the hard way. you've been warned.

ok, so here’s a typical setup, broken down:

**1. the for each loop container configuration**

*   **collection:** you'll choose the 'for each file enumerator'. specify the folder path where your files are located. be sure to use a variable for this. never hardcode paths. remember my hard path experience? don't be me.
*   **files:** set the file mask to something like `*.csv`, `*.txt`, or whatever your file extensions are, or maybe if you want to get fancy, you can do `myfile_*.csv`.
*   **retrieve file name:** select 'name and extension', or just 'name' depending on if you need the extension. this part stores the full file name into an ssis variable.
*   **variable mappings:** create a string variable, say `@[user::currentfilefullpath]`, and map the output of the enumerator to it. this will store the full path to the current file in the loop. you also need a variable to store the file name without the path. say, `@[user::currentfilename]`, and that can be derived from `@[user::currentfilefullpath]` inside a script component, as you see below in point 2.

**2. deriving the table name:**

usually, there's some pattern or naming convention you follow, so you can use that pattern to derive the table name from the file name. this can often be done within a script component or expression. the idea is to extract the name that precedes the file extension and then, if necessary, do some processing on it. this gives you the name of the table to load the data into.

i’ve seen some complex table-naming schemes. it was a client that wanted a table name with the file name, date of processing and some weird hash. let's just say it was not good. since then, i've made sure table names are predictable. here’s an example of how you could do this in a script component (using vb.net – yes, i still sometimes use vb.net, don’t judge):

```vb.net
imports system
imports system.text.regularexpressions
imports microsoft.sqlserver.dts.runtime

<microsoft.sqlserver.dts.tasks.scripttask.scripttask(scripttasklogentry.none)> _
partial public class scriptmain
    inherits microsoft.sqlserver.dts.tasks.scripttask.vstaskscript

    public sub main()
		dim filePath as string = dts.variables("currentfilefullpath").value.tostring()
        dim fileName as string = system.io.path.getfilename(filepath)
        dim tableName as string = regex.replace(fileName, "\.[^\.]*$", "", regexoptions.none)

		dts.variables("currentfilename").value = filename
		dts.variables("currenttablename").value = tableName
        dts.taskresult = dts.taskresult.success
    end sub
end class

```

this script extracts the filename, and then removes everything from the last dot onwards (including the dot), this leaves you with just the filename to use as a table name for example, storing the result in the `@[user::currentfilename]` variable and then storing the result in the `@[user::currenttablename]` variable. you may want to validate the table name if your names are not simple enough and contains non supported chars for a table name. you can check against a whitelist of allowed chars, for example.

**3. the data flow task**

*   **flat file source:** here's the usual step. you need to configure a flat file source connection manager, but be careful! you *must* use an expression to dynamically define the connection string property, set to `@[user::currentfilefullpath]`. this way, each loop iteration will process a different file.
*   **oledb destination:** you need to setup a database connection to your database, but again, don't hardcode the table name. you need to set the `tablename` property using an expression and pointing it to the `@[user::currenttablename]` variable.
*   **column mappings:** this is probably the easiest part. just make sure your columns match up correctly, and data types as well.

**4. error handling**

i cannot stress this enough. error handling is not optional. it is fundamental. a simple catch on any step will save your hide some day. it's like leaving the doors open in your house, sooner or later you will have unwanted visitors. you should add error outputs from your source and destination to some error logging table, at least. don’t just let the package crash silently; log those errors. this part is key for your own mental health. it's better to be proactive and plan for errors rather than debug the aftermath.

you might also want to consider creating a logging table for the files processed: add at least the filename, the processing datetime, and if it was ok or not. this adds some real observability to your process.

here’s an example of a simple expression to get the table name (alternative to the script component from above) in case your naming convention is simple:

```sql
substring( @[user::currentfilename], 1, findstring(@[user::currentfilename], ".", 1) - 1)
```

this expression takes the variable currentfilename, extracts characters from the first character up to the position of the first dot, and removes that dot. in other words, the file name without the extension. the substring function will fail if there's no dot, this shows that this way has its limitations so be sure to test your code.

and a basic sql script for creating the logging table

```sql
CREATE TABLE dbo.files_processed (
    filename varchar(255),
    processdate datetime,
    status varchar(10),
    errormessage varchar(max)
)
```

you'd then use a `sql server destination` component to insert into this table.

**some extra tips, the experience talks:**

*   **file encoding:** be aware of file encoding! different systems use different character sets. if your files are not consistent, you could see some issues. i've spent hours figuring this out. there's a character encoding option in flat file connection manager; use it carefully.
*   **header row:** sometimes files have a header row and sometimes they don't. again, the flat file manager allows you to skip the header row, or you can deal with it using the conditional split component.
*   **performance:** if you're dealing with many files, you should look at increasing the `maxconcurrentexecutables` property on the `for each loop` to allow parallelism and increase the performance of your process, but do this carefully as this can overload your source system. keep in mind that there will be some overhead of this and it might affect your process overall. do some benchmarks if you are doing some serious data movement.
*   **transaction control:** i sometimes encapsulate the for each loop in a sequence container with its `transaction option` set to `required` or `supported` to handle exceptions at a higher level. it might be overkill if the process is simple.
*   **configuration:** you can externalize database connection strings, folder paths, and even your naming rules into configuration files or environment variables. this avoids having to republish packages anytime something changes. this also makes your packages less fragile to changes in the environments. a really important skill to learn.

so, to recap, you need a `for each loop` to iterate through your files. use variables for file paths and table names. you can use a `script task` to derive the table name, or an expression. the data flow task will load the data using those variables. don’t forget error handling.

if you really want to understand the ssis object model and how these components interact with each other, i recommend a deep dive into the book "microsoft sql server 2019 integration services cookbook", it's an oldie, but a goldie when it comes to ssis and will help you understand it in more depth.

and that’s how i do it. it might sound a little complex, but once you have the pattern down, it becomes second nature. and remember, always always always avoid hardcoding! seriously!

hope this helps and good luck, you've got this!
