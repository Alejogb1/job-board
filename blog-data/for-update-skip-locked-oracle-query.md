---
title: "for update skip locked oracle query?"
date: "2024-12-13"
id: "for-update-skip-locked-oracle-query"
---

 so you're asking about `UPDATE SKIP LOCKED` in Oracle yeah I get it I've been there more times than I care to admit it’s a real headache when you’re dealing with concurrency

Let me tell you I’ve wrestled with this particular beast back in my days at DataCorp we had a system that processed financial transactions think stock trades and payments stuff that absolutely needed to be spot on with no screw-ups it was a multi-threaded java app hitting a massive Oracle database and if one thread got stuck on a row then everything else would just bottleneck and back up it was a total nightmare

Initially we were using your standard `SELECT ... FOR UPDATE` and if one process had locked the row others would just hang there waiting and waiting I mean we had timeouts but it just meant lots of failed jobs and angry calls from the trading floor. We needed a way to just skip over rows that were locked instead of just sitting there like a stunned muppet and that’s when we stumbled upon `SKIP LOCKED` it was a real game changer let me tell you

 so for the uninitiated `SKIP LOCKED` essentially tells the database “hey if this row is already locked by someone else just move on don’t wait for it” It’s particularly useful in scenarios like the one I just described where many processes are trying to update the same table and you can’t afford for one to get stuck and halt the whole operation

It’s not a magical fix-all solution though you need to think carefully about how you apply it because there are implications of potentially skipping updates.

Here's a simple example imagine you have a table called `transactions` and you want to update the status of all pending transactions to `PROCESSING`

Without `SKIP LOCKED` this is how we were doing it (and what failed miserably mind you)

```sql
UPDATE transactions
SET status = 'PROCESSING'
WHERE status = 'PENDING'
FOR UPDATE;
```

This version will hang as soon as it encounters a locked row which is like playing a game of chess with one hand tied behind your back pretty dumb right?

Now with `SKIP LOCKED` it gets much better

```sql
UPDATE transactions
SET status = 'PROCESSING'
WHERE status = 'PENDING'
FOR UPDATE SKIP LOCKED;
```

See the difference that one little phrase `SKIP LOCKED` changes everything it skips the locked ones and only updates the ones available meaning it does not wait for any locks and it proceeds to update every single row that it can and it skips ones that are already locked by another process it is like the opposite of a lazy programmer.

Now let's get into some real-world scenarios lets say you want to process messages from a queue and you are using Oracle for your datastore you can update the message status to `PROCESSING` so no one else can process the same message. If a message is locked skip it and move on to the next one.

```sql
UPDATE messages
SET status = 'PROCESSING'
WHERE message_id IN (
    SELECT message_id
    FROM messages
    WHERE status = 'PENDING'
    FETCH FIRST 100 ROWS ONLY
)
FOR UPDATE SKIP LOCKED;
```

This query updates up to 100 pending messages to processing status it uses `FETCH FIRST` to process them in chunks and it skips any rows that are locked you see it's doing the work and not getting stuck.

You need to remember though the `SKIP LOCKED` is a lock specific thing so you can't use it if you are trying to use other lock mechanisms like optimistic locking using version numbers you would need a different method to approach the problem like polling for the locks and retrying your operation. It is not going to make your code magically become concurrent.

One of the biggest mistakes I made when first using it was not logging skipped rows. In the beginning we thought we were golden just skip locked rows and everything was fine until we found out that sometimes there were transactions which never got processed because of a lock conflict which we did not notice because we were blindly skipping locked rows and everything looked happy from the outside.

So the lesson is log everything you need to know which rows are being skipped and you need to understand why this is happening. Otherwise you will be chasing ghosts in your database and it’s not a good look believe me.

Also keep an eye on the potential for starvation If you have a process that’s constantly locking rows it might prevent other processes from accessing them even with `SKIP LOCKED` In rare cases that could become a real pain point and it’s not nice when it happens.

Concurrency is never easy and you will probably run into many issues. There is a reason why there are entire books dedicated to concurrency and databases. You should familiarize yourself with these concepts to better understand what is going on in your application.

For further reading I would recommend "Database System Concepts" by Silberschatz Korth and Sudarshan it’s a classic and you will find a lot of what you need there another great resource is "Concurrency Control and Recovery in Database Systems" by Bernstein Hadzilacos and Goodman.

These books will give you a much better understanding of how databases handle concurrency and you will be able to better tackle more complex problems in the future. And avoid dumb mistakes that I did myself when starting.

So yeah that’s pretty much it `UPDATE SKIP LOCKED` in a nutshell it's a powerful tool if you know how to use it but if you don't you will end up in the weeds chasing issues and bugs that were your own fault always remember that with great power comes great responsibility or in this case with SQL power comes SQL mistakes so be careful out there.

Also one last thing do not skip your error handling for any reason if there are errors make sure they get caught and properly logged or you will have a even worse time than you expect
