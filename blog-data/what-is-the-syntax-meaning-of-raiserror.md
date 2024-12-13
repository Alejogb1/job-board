---
title: "what is the syntax meaning of raiserror?"
date: "2024-12-13"
id: "what-is-the-syntax-meaning-of-raiserror"
---

Okay so you're asking about `RAISERROR` huh Yeah I know that pain We've all been there staring at SQL Server logs trying to figure out why something's gone sideways Let's break it down nice and simple like we're at the keyboard together debugging something at 3 am

First thing first `RAISERROR` is basically SQL Server's way of throwing a custom exception You know like in programming languages when you hit an error and you're like "Yo something went bad let the system know" well `RAISERROR` does that in T-SQL its whole job is to interrupt execution and report a problem

I remember one time back in the day when I was first getting into this I was building a huge stored procedure for inventory management you know the kind that updates a dozen tables at once I was so proud of my code and of course it was riddled with logic flaws the usual stuff like missing constraints and invalid data entries and my stored procedure ran but quietly failed at the important stuff and I had no way to figure out why So many silent errors and I spent a whole weekend basically just debugging that thing line by line It was a nightmare I wish back then I knew about `RAISERROR` it would've saved me so much hair pulling I'd have more hair now I guess

Anyway the basic syntax goes like this:

```sql
RAISERROR (message, severity, state [, argument [,...n]]) [WITH option [,...n]];
```

Alright let's dissect it

`message` This is the actual error message that you want to show up in your logs or in whatever application is consuming your SQL data It's a `varchar` or `nvarchar` string and you can use placeholders like `%s` `%d` and so on to put in variables You know for more dynamic errors I hate it when errors are all generic and unhelpful like "Error occurred" thanks for the heads up yeah I could have guessed that

`severity` This is an integer that ranges from 0 to 25 it basically indicates how critical the error is 0 to 10 are just informational messages they might or might not even show up in the logs 11 to 16 are errors that might be fixable 17 to 19 are more serious problems that usually affect the current process and 20 to 25 are fatal errors that can kill your whole SQL Server connection or even the instance itself so you should probably not be raising those unless it's like the end of times or something or your application is on fire I've seen this severity number mis-used so many times and it was always a pain to get the correct severity level set up in the system

`state` This is another integer between 0 and 255 used to distinguish between different error states with the same severity it helps to be more specific about the error type and often if I encounter errors and don't know what's going on I look up errors with the same severity but different state and this is a huge time saver It allows me to pinpoint the problem I like it a lot

`argument` These are the optional values that you want to include in the error message using the placeholders that you specified in the `message` this is where your variable data should go This is where you'll want to stuff the actual problematic data like `user_id` or the invalid value etc

`WITH option` This is like extra configuration for `RAISERROR` it gives you a little more control over the error behavior like `LOG` to make sure the error shows up in the error log `NOWAIT` to immediately send the error and so on I don't usually play much with this option but it's good to know it's there when you need it

Okay so let's see some examples First a super simple one with a fixed error message

```sql
RAISERROR ('This is a basic error', 16, 1);
```

This will throw an error with severity 16 state 1 and your message and this error will kill your SQL batch if not caught with `TRY CATCH` which is probably what you'd want when things go wrong This simple one liner can tell you so much more than a silent error

Next let's do something more dynamic with some variable arguments

```sql
DECLARE @userId INT = 123;
DECLARE @productName VARCHAR(50) = 'Widget Deluxe';

RAISERROR ('User %d tried to access %s without proper authorization', 16, 10, @userId, @productName);
```

So here's where `RAISERROR` gets interesting You can now report the exact data that caused the problem this is so helpful when you are tracing bugs and you can just copy paste the output into the query that reproduces the error

Last but not least I'll show you one with a bit more customization of the error handling

```sql
BEGIN TRY
    -- Some code that might fail
    DECLARE @value INT = 0
    IF @value = 0
    BEGIN
        RAISERROR ('Value cannot be zero', 16, 21)
    END
    -- the rest of your code
END TRY
BEGIN CATCH
    SELECT 
        ERROR_NUMBER() AS ErrorNumber,
        ERROR_SEVERITY() AS ErrorSeverity,
        ERROR_STATE() AS ErrorState,
        ERROR_MESSAGE() AS ErrorMessage;

    -- Optionally log error to a table
END CATCH;
```
Here I used the `TRY CATCH` construct to actually handle the error that `RAISERROR` throws this is a really helpful pattern because it allows you to gracefully respond to the problem instead of just blowing up the whole application You can use the functions `ERROR_NUMBER` `ERROR_SEVERITY` `ERROR_STATE` `ERROR_MESSAGE` to collect the data that the `RAISERROR` provided and do something with that data you can log them or send to another system

Now it's important to not use RAISERROR for every single little thing It's meant for unexpected problems not for standard control flow logic it's for actual error handling you can and should use conditional statements for things that are expected

There's a lot more you can do with error handling in T-SQL but this should get you started I'd recommend reading up on exception handling best practices in general which is super important no matter what you do and I don't have much time to give you the exhaustive theory here

If you want to dive deeper I'd suggest looking into classic books on database systems there's "Database System Concepts" by Silberschatz Korth and Sudarshan which I think it is the bible of all database topics for me or try to find advanced SQL Server query tuning books many have sections on advanced error handling techniques The "SQL Server Query Performance Tuning" book by Itzik Ben-Gan has a great discussion about error handling and logging and that's one of my go to recommendations You also might be interested in research papers on structured exception handling in programming languages its all pretty similar concepts

One last thing I always keep in mind is to make the error messages as clear and helpful as possible for future you because let's face it you'll probably be the one debugging it at 3 am six months later. I do this because I'm not sure of who I am gonna be when I'll be called back to fix some old code I wrote a long time ago That's it no more jokes from me I promise
