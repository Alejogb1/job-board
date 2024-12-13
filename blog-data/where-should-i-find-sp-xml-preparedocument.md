---
title: "where should i find sp xml preparedocument?"
date: "2024-12-13"
id: "where-should-i-find-sp-xml-preparedocument"
---

Okay so you're wrestling with `sp_xml_preparedocument` huh I've been there it's like that one SQL Server stored procedure you never quite forget once you've dealt with it

Alright listen up because I've spent way too many late nights debugging this thing than I'd like to admit First things first `sp_xml_preparedocument` isn’t something you find just lying around in the database like a random table its a system stored procedure It lives within the SQL Server engine itself meaning its part of the core functionality So its not like you can `SELECT * FROM sp_xml_preparedocument` that's a no go

When people ask about finding it usually what they actually mean is how do they actually use it where do you even invoke this beast You're not going to be hunting for its physical file on your disk drive Its accessed by calling it with a specific set of parameters like any other stored procedure or system function

So here's how it works in a very simple example lets say you have an XML string This XML could be anything a config file some data you got from an API or even a user provided data stream

```sql
DECLARE @xmlDocument VARCHAR(MAX) = '<Root><Element1>Data1</Element1><Element2>Data2</Element2></Root>';

DECLARE @docHandle INT;

EXEC sp_xml_preparedocument @docHandle OUTPUT, @xmlDocument;

SELECT @docHandle AS DocumentHandle;

EXEC sp_xml_removedocument @docHandle;
```

See how that works We are declaring a variable for the XML and another integer variable to hold the "document handle" This document handle is a temporary identifier that `sp_xml_preparedocument` creates for your XML It basically parses the XML creates an internal representation of it and gives you a way to refer to it and then we call `sp_xml_removedocument` to clean it up once done

Now what people usually want to do after `sp_xml_preparedocument` is to query the XML data using `OPENXML` or XQuery I prefer XQuery myself but here is a basic `OPENXML` Example

```sql
DECLARE @xmlDocument VARCHAR(MAX) = '<Root><Element1 id="1">Data1</Element1><Element2 id="2">Data2</Element2></Root>';
DECLARE @docHandle INT;

EXEC sp_xml_preparedocument @docHandle OUTPUT, @xmlDocument;

SELECT
    X.id,
    X.data
FROM OPENXML(@docHandle, '/Root', 2)
WITH (
  id INT './@id',
  data VARCHAR(100) './text()'
) AS X;

EXEC sp_xml_removedocument @docHandle;
```

So what happens here we pass the docHandle which points to our XML along with the root node and other parameters to `OPENXML` then we define a schema what data to get based on XML node attributes and content then we get the data

But be warned `OPENXML` is an older method and is generally slower and less flexible than XQuery I've seen performance nightmares with huge XML files using `OPENXML` It's much better to use XQuery whenever possible if you have a newer version of SQL Server

Here’s a snippet using XQuery which is my preferred method

```sql
DECLARE @xmlDocument XML = '<Root><Element1 id="1">Data1</Element1><Element2 id="2">Data2</Element2></Root>';

SELECT
    T.X.value('./@id', 'INT') as id,
    T.X.value('./text()', 'VARCHAR(100)') AS data
FROM @xmlDocument.nodes('/Root/*') AS T(X);

```

Notice how I use XML datatype directly it makes things cleaner and simpler Also note we are not using `sp_xml_preparedocument` or `sp_xml_removedocument` with XQuery it handles the xml parsing and extracting of data on its own This is one of the reasons why we prefer it

Now when it comes to finding "where" `sp_xml_preparedocument` *is* think of it like asking "where is the `PRINT` function in SQL Server" It’s just there as a fundamental part of the engine It's not something you'd find in a specific file or database table You just use it Its part of system stored procedure so its defined in master system database

So here’s a bit of advice I learned the hard way never ever leave a `sp_xml_preparedocument` call without a corresponding `sp_xml_removedocument` call This creates memory leaks and believe me you don’t want to mess with that I had an app once which forgot that cleanup call and the SQL Server memory just went up up and away it needed a server restart to clear that mess It was a very bad day. So be careful you have been warned

If you want to really dive deeper into XML handling in SQL Server I'd recommend reading "SQL Server 2019 Query Performance Tuning" by Grant Fritchey it has a good section on XML data types and querying And the official Microsoft documentation is surprisingly good sometimes search for `sp_xml_preparedocument` or `OPENXML` to get the specifics of parameters and usage cases

Oh and by the way I always make sure to double check my XML formatting because if you have malformed XML `sp_xml_preparedocument` will throw errors and that’s always fun to debug. I once spent three hours debugging an issue only to find a missing closing tag. It was a character one character out of place three hours lost but thats life of a developer

Okay now you got the basics of `sp_xml_preparedocument` and some things around it Remember the important thing is to manage your document handles properly especially `sp_xml_removedocument` If you follow that simple rule you will be a much better programmer.
