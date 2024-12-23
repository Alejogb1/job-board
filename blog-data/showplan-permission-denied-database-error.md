---
title: "showplan permission denied database error?"
date: "2024-12-13"
id: "showplan-permission-denied-database-error"
---

 so "showplan permission denied database error" classic right I've seen this more times than I care to admit Let me break it down from my experience and how you can likely get this sorted it's a permission thing usually straightforward but the devil's always in the details

So basically you're hitting this error probably when you're trying to get the query execution plan right That's the plan SQL Server uses to figure out how to fetch your data The SHOWPLAN permission is what allows you to view that plan without running the actual query and this permission problem is pretty common especially when you're not the database administrator or you're working with complex permission setups It's a little bit like asking to see the chef's recipe but you're not allowed in the kitchen

I remember the first time I encountered this way back when I was still a junior dev I spent hours banging my head against the keyboard not understanding what was happening I'd copy-pasted some query from StackOverflow (yes even I've been there) and tried to get the plan to understand why the query was so slow I had no permissions to use showplan in that SQL Server instance I was working on I kept getting the "SHOWPLAN permission denied" message the error was frustrating but a good learning moment After digging around for far too long I realized it wasn't the query it was me and my permissions Then after a good amount of reading and asking I found out it was a permissions problem Not the query it was definitely a permission problem and well the obvious part was that I was not the database admin and therefore the permissions were not configured to allow my access to the showplan I had to ask someone more senior to me to help me out with the permission grants

Here's the thing there are a few ways this can happen You might be trying to use commands that implicitly require the SHOWPLAN permission like SET SHOWPLAN_ALL or SET SHOWPLAN_TEXT or you might be trying to view the estimated plan in SQL Server Management Studio which is doing something similar under the hood So let’s jump into some common cases and how to fix them and remember the root cause of this is almost always the lack of the required permission which is not really a database error in itself but an access error

First thing first the actual grant statement it looks something like this

```sql
GRANT SHOWPLAN TO YourUser;
GO
```

Replace `YourUser` with your actual SQL Server user account Or If you have specific roles you may want to grant permissions to that role instead of a user I also had to do that on a project when several developers required access to the showplan and you don't want to grant permissions to each and every one of them Instead you create a role and assign the needed permissions and you can then assign every user to this newly created role for example using the below

```sql
CREATE ROLE developer_role;
GO

GRANT SHOWPLAN TO developer_role;
GO
ALTER SERVER ROLE sysadmin ADD MEMBER your_user
ALTER ROLE developer_role ADD MEMBER your_user;
GO
```
This script first creates a role called `developer_role` then the next line adds the showplan permissions to this role and finally the user gets assigned to this role this is more scalable than grant permissions to single users

You'll need someone with sufficient permissions to run this usually a database administrator or someone with server admin privileges This is a one-time setup per user or role though If you are the database admin and for some reason you are getting the permission error then you need to first check that you actually have the admin rights on the SQL Server instance and you are really logged in with that account there are instances where people have more than one user accounts on their local machine and if they do not pay attention they are logged in into the instance with a different account than the one they think they are working with

After the permissions have been granted you may want to try to test it for example by using `SET SHOWPLAN_TEXT ON` command which is a classic way to test if showplan is working for you

```sql
SET SHOWPLAN_TEXT ON;
GO

SELECT * FROM YourTable WHERE YourCondition;
GO
SET SHOWPLAN_TEXT OFF;
GO
```
Replace `YourTable` and `YourCondition` with your actual table name and WHERE clause you will notice a textual representation of the query plan as a result of using `SET SHOWPLAN_TEXT ON` command if it is working properly this is an indication that showplan permissions have been granted correctly so this command will help you verify everything is set up correctly and you are good to go

If you're using SQL Server Management Studio (SSMS) and trying to view the execution plan the shortcut is Ctrl+L and also make sure that you are logged in with the user that has the correct permissions When you get this error it is probably because you are trying to get the estimated execution plan If you want to see the actual execution plan which is something different you need to actually run the query and check the plan you get this plan by selecting `Display Actual Execution Plan` which is also under the query menu as an icon and also has a shortcut of `Ctrl + M` This usually gives you an idea how SQL Server executes your query but make sure you are working on a development environment if it is a big query as the query will be executed and can affect the system performance

I also remember once on a legacy SQL Server instance I was fighting this very same issue for days that ended up to be that the user was part of some weird group on the server that was denying the SHOWPLAN permission even though the user himself had the permissions granted to him In those cases the server groups take precedence and they act as a final layer of security check make sure you investigate all security groups and configurations if the previous steps did not work out correctly This one took me a long debugging session because in this server the permissions had an inverted logic to the normal logic

Also if you are dealing with a lot of stored procedures and are working with a lot of databases and SQL Server instances it is a good practice to create dedicated users per database and also per environment like development and production in order not to mix the permissions and also try to use dedicated roles that can be assigned to the different users It's easier to manage access this way than to deal with individual permissions everywhere and also reduces your chances to introduce any human made mistakes and make sure you document all permissions and also user access this is critical for compliance and also for auditing

One final thing if the "SHOWPLAN permission denied" error pops up unexpectedly it might mean that there is some kind of trigger running that is trying to showplan or there is some other automated process running that is trying to access the showplan without having the right permissions and in such a case it might be a good idea to investigate all the database processes running by checking the server logs and also reviewing any scheduled tasks or jobs on the server and if that is the case it might be related to some kind of security issues or some automation that is incorrectly configured

So to recap it’s mostly about ensuring the user has `SHOWPLAN` permission. Grant the permission to your user using the `GRANT SHOWPLAN` command, try out some simple `SET SHOWPLAN_TEXT ON` to verify if the permissions are working and if you're using SSMS make sure to log in with the right user that you granted this permission to also if you are getting it while using SSMS it could be related to the estimated execution plan instead of the actual one.

Finally make sure to use good resources to learn more in depth about SQL Server permissions I would suggest "SQL Server 2019 Administrator's Guide" it is pretty detailed and cover everything about server and database administration or if you prefer something more academic I would suggest "Database System Concepts" by Silberschatz et al this book is a bit more theoretical but it gives you great foundational knowledge about database permissions and other concepts related to query optimization and management

Remember in tech we always have to keep digging that's what makes the job fun and also infuriating sometimes but hey that's part of it and in this case you are not alone I've been there many times and now you know the drill you will figure it out
