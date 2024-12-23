---
title: "adding all articles to a publication using tsql?"
date: "2024-12-13"
id: "adding-all-articles-to-a-publication-using-tsql"
---

 so you wanna cram all the articles into a publication using tsql yeah I've been there let me tell you it's not always as straightforward as it seems trust me on this one

so you have a publication let's call it `MyPublication` and you want to add all articles currently floating around in your database into it right I get it I've been in that situation more times than I'd like to admit Back in the early days of my career I messed up a replication setup real bad it was a test environment thankfully but man oh man did I learn a lot from that particular train wreck lets just say that after the chaos settled my database was about as stable as a toddler on a sugar rush and that involved a lot of scripting to put back things in place That’s when I began writing these kinds of scripts to automate all of it so here is my take

First thing first you gotta know the basics publications in SQL Server they are basically containers for articles And these articles they are usually tables or sometimes views that you want to replicate or distribute to other servers or databases

Now when you say all articles I'm assuming you mean all existing tables in your database that haven't been added to your publication before If that's the case this is the most important part: you can't just waltz in and add every table blindly you gotta check which ones are already articles and that is where this script comes to play

```sql
-- Step 1: Find all tables in the database
SELECT
    TABLE_SCHEMA AS SchemaName,
    TABLE_NAME AS TableName
FROM
    INFORMATION_SCHEMA.TABLES
WHERE
    TABLE_TYPE = 'BASE TABLE'

EXCEPT

-- Step 2: Filter out tables that are already articles in the publication
SELECT
    art.schema_name AS SchemaName,
    art.name AS TableName
FROM
    sys.articles AS art
JOIN
    sys.publications AS pub
ON
    art.pubid = pub.pubid
WHERE
    pub.name = 'MyPublication'; -- replace with your publication's name
```
 so this code is like a two-part query the first part gets you all tables from your database using a system view information\_schema it's pretty standard it returns all the base tables that exist in your database then the second part gets you all tables already registered as articles within the publication so what we do is exclude the second part from the first part That’s what the except keyword does if you run this query you’ll get a list of all the new tables you can potentially use

This brings me to the next point now you have all those tables you want to add them to your publication you can’t do it all at once that would break your system it is an operation that can take time and is better suited to batches this is where dynamic tsql comes in it allows you to create and execute tsql scripts at run time and also is perfect for this scenario

```sql
DECLARE @SchemaName sysname
DECLARE @TableName sysname
DECLARE @PublicationName sysname = 'MyPublication' -- replace with your publication's name
DECLARE @SQL NVARCHAR(MAX);

-- loop through the tables
DECLARE table_cursor CURSOR FOR
    SELECT SchemaName, TableName FROM
    (SELECT
    TABLE_SCHEMA AS SchemaName,
    TABLE_NAME AS TableName
    FROM
        INFORMATION_SCHEMA.TABLES
    WHERE
        TABLE_TYPE = 'BASE TABLE'

    EXCEPT

    -- Filter out tables that are already articles in the publication
    SELECT
        art.schema_name AS SchemaName,
        art.name AS TableName
    FROM
        sys.articles AS art
    JOIN
        sys.publications AS pub
    ON
        art.pubid = pub.pubid
    WHERE
        pub.name = @PublicationName) as NewTables
OPEN table_cursor

FETCH NEXT FROM table_cursor INTO @SchemaName, @TableName

WHILE @@FETCH_STATUS = 0
BEGIN
    -- build the sql for each table
    SET @SQL = N'
        EXEC sp_addarticle
            @publication = N''' + @PublicationName + N''',
            @article = N''' + @TableName + N''',
            @source_owner = N''' + @SchemaName + N''',
            @source_object = N''' + @TableName + N''',
            @type = N''logbased'',
            @destination_owner = N''' + @SchemaName + N''',
            @destination_object = N''' + @TableName + N'''
    ';

    -- debug print
	PRINT @SQL

    -- execute the sql
    EXEC sp_executesql @SQL;

    FETCH NEXT FROM table_cursor INTO @SchemaName, @TableName
END

CLOSE table_cursor
DEALLOCATE table_cursor
```
 so what is going on here well first we declare a bunch of variables to hold the table schema name table name publication name and finally the dynamically built tsql script after that we select all the new tables using the same query as before that is a subquery this cursor will loop through each new table we will build a tsql script to add the table as an article to our publication and then execute it so after each iteration we fetch the next table to be added as an article to the publication. The loop will stop when all the new tables are registered as articles in the target publication

`sp_addarticle` is the system stored procedure you use to add an article to a publication and it takes a lot of parameters like the names of the objects owner etc This example assumes you're using a standard log based replication which is the most common scenario but if you are using something different make sure to tweak that part

I used to make a silly mistake when creating these kinds of dynamic scripts I always forget to add the N before the string which is necessary for unicode string literal especially when you are dealing with non-ascii characters but one day I added this before the string the N is there always now it was a happy day well not so happy since I realized that after spending hours trying to figure out what was wrong I guess it is one of those moments that you can’t be proud of anyway... moving forward.

 so lets recap the above steps basically this will go through each table that isn't already an article and add it to your publication It's pretty straightforward but you have to be aware of some things like the types of replication you're using the amount of data your tables contain and the impact it could have on your server

Also you may want to check if your tables have primary keys or not because most types of replication require tables to have one if not you’ll need to add that first or your replication setup will just fail and then there is another challenge if your replication is transactional then you’ll need to add a snapshot to get everything synchronized

```sql
-- generate snapshot
EXEC sp_startpublication_snapshot @publication = 'MyPublication' -- replace with your publication's name
```
So yeah `sp_startpublication_snapshot` this is the command you use to generate the snapshot and this needs to be done after the articles are added to the publication it takes a publication as an argument and after you execute it a snapshot is generated for the new tables that you added as articles

I know this was a bit of a dump but honestly I think that covers the basics You are going to have to do a lot of trial and error and read the docs carefully if you want to understand the whole process but these scripts should work as a good starting point

As for recommended resources I would say check "SQL Server 2019 Administration Inside Out" by William R. Vaughn and also check the Microsoft official documentation on replication specifically the section about sp\_addarticle and other replication related system stored procedures those helped me a lot and I am sure they will help you too.
