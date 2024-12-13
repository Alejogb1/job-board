---
title: "66000/3 sap b1 error code?"
date: "2024-12-13"
id: "660003-sap-b1-error-code"
---

Okay so you're wrestling with the dreaded 66000/3 error in SAP Business One huh Been there done that got the t-shirt and probably a few stress-induced grey hairs trying to figure this one out let me tell you I've seen this error pop up more times than I care to remember and it's always a fun little puzzle to solve

So first off 66000 generally speaking in SAP B1 usually points to a problem with data access a sort of generic "something went wrong fetching this" kind of error you know kind of the catch all for all things database related The /3 part specifically is where things get a little more interesting it's more often than not a permissions issue within the database itself that SAP B1 is using

Back in the day during my early dabbling with SAP B1 I remember spending hours debugging this exact error for a client they had a custom report that was hitting multiple database views and tables and bam 66000/3 every single time The funny thing though is that their standard SAP reports were working perfectly fine That's when I learned the painful lesson that SAP user permissions aren't the only kind of permissions you have to worry about with B1 It's a permissionception nightmare you have permissions on top of permissions

We initially suspected the usual suspects the user account not having access the linked tables not being public but everything seemed legit user had all the authorizations assigned for accessing modules and everything that was in the permissions menu It was a head scratcher I was ready to throw my keyboard out the window I swear

Turns out the problem was actually at the SQL database server level the database user SAP B1 was using didn’t have the proper read rights for all the underlying database objects that were referenced in their custom query This was especially true for views which sometimes are not that easily accessible to all users

So here's the kind of thing you'll want to check first:

**1 Database User Permissions**

You'll need to hop into the SQL server management studio or whatever your DB client is and look at the database user that SAP B1 is using usually it's something like `SBODEMO` or `B1ADMIN` or something customer created Ensure that this user has the `SELECT` permission specifically for all tables and views involved in your SAP B1 transaction or process that's throwing this error This often gets missed because people just focus on SAP permissions and forget the database level side of things

Here is a basic SQL example of how you would go about granting permissions the database user `B1USER` for a specific table called `OITM`:

```sql
USE [YourDatabaseName];
GO
GRANT SELECT ON OITM TO B1USER;
GO
```

And if you're dealing with views you need to explicitly grant permissions on those too:

```sql
USE [YourDatabaseName];
GO
GRANT SELECT ON YourViewName TO B1USER;
GO
```

Remember to replace `YourDatabaseName` `B1USER` and `YourViewName` with your actual database user database name and view name respectively Also it's a good practice to run this in a test environment first before doing this in a production database you might break something you never know

**2 The Linked Server Problem (If Applicable)**

If your SAP B1 system is pulling data from another database using linked servers it's another common place for these 66000/3 errors to happen The database user that SAP B1 uses needs not only permissions on the local database but also on the linked database on the other server You might need to set up proper database user mapping on the linked server side for this to work

Here's what you have to check on the linked server using SQL management studio:

1 First make sure your linked server is configured correctly under Server Objects Linked Server section in the SQL Server.
2 Make sure the Security tab of your linked server properties shows the `Be made using this security context` option is selected.
3 Then under the `Remote Login` enter a username and password combination that has database read access in the other server. If you are using a service user remember to give it full database permissions on the other server.

Make sure you test the connection by right clicking the linked server and select `Test Connection`. Make sure you see the `Successfully connected to the server`.

**3 Transactional Data Issues**

Sometimes and I mean sometimes this error may come up when you're trying to update or create a record with a field that’s either improperly formatted or violates a data constraint It could be as simple as having a text field that is too long or a field that should be a number but you are trying to put in text For instance you get this error with the code I show below:

```sql
INSERT INTO OITM (ItemCode, ItemName, FrgnName, IntrnlCode)
VALUES ('TESTITEM', 'Test Item', 'Test Foreign Name', 'ThisTextIsTooLong');
```

This code would throw an error if `IntrnlCode` field in `OITM` table is set to maximum string length of for instance 10 characters in the database. This is not necessarily related to permissions but its still a related cause for 66000/3 so that's why I mention this too.

So you would rather make sure the data is in correct format and that all the required fields are filled with valid values

**4 Things to Double Check and Debug**

- Check the SAP B1 logs they can sometimes provide more granular details than just the error code itself The logs can usually tell you what part of the process or query was throwing the error
- Try to isolate the problem If the error is happening in a specific report or form try running simpler queries to identify whether the error is related to the database or is it a more complex issue that relates to your business logic.
- Consult the SAP B1 documentation they have tons of information there but sometimes things are buried deep within their documentation this is why having resources such as the **SAP B1 SDK documentation** is a good idea this can sometimes give you a clue on the possible errors of their API but it won't tell you directly that this specific error is database permissions error. The information there is mostly about how their APIs interact with database but it's a nice place to begin with.
- If you are developing custom addons look at the trace logs to see what was the query that caused the error. For this there are tools such as **SQL Server Profiler** it’s very useful in debugging and finding out what SQL code is running in the background when you operate on SAP B1. The profiler will show you exactly the SQL query that is causing the error if it is an SQL error of course.
- If you are using external data sources via custom queries try to use SQL Management Studio and try to execute the query there using the same user that SAP uses to connect to the database to check if it will throw the same errors and to isolate the problem.
- Remember this is not a user authorization problem so do not look there for a solution

**Where To Learn More**

- **SQL Server Books Online**: This is your best friend for anything SQL related. It details all aspects of SQL Server including permissions database security and linked servers.
- **SAP Business One help files and SDK documentation**: While this doesn't go into database level details you'll find information on permissions requirements for the SAP B1 application itself and its APIs.
- **"SQL Server 2019 Administration" by Robert L. Davis:** It's a great practical book for learning how to manage your SQL server database including user permissions and security.
- **Your company's internal IT or SAP B1 support team if you have one:** Don't hesitate to ask them because you might not be the only one facing the same problem.

I hope this is helpful and at least gives you a place to start your debugging journey with this notorious error I know how frustrating this one can be Just keep digging and you'll find the answer eventually It's usually something pretty simple you just need to know where to look for it. Happy debugging!
