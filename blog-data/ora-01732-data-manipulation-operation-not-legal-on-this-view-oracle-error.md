---
title: "ora-01732 data manipulation operation not legal on this view oracle error?"
date: "2024-12-13"
id: "ora-01732-data-manipulation-operation-not-legal-on-this-view-oracle-error"
---

Alright so you're hitting the classic ORA-01732 error a "data manipulation operation not legal on this view" error in Oracle right I've seen this one more times than I'd like to admit it's a real head scratcher when it first hits you but it becomes almost second nature after a while you're trying to do an insert update or delete on a view and Oracle's telling you no bueno

I've been there man trust me I remember this one project back in the day we were building this internal reporting tool and we had this super complex view it pulled data from like five different tables joined on all sorts of conditions it was a beast of a query you know the kind that makes your database server sweat a little we were all proud of it we even named it the "Data Titan" yeah yeah I know cheesy but we were young

Anyway this Data Titan was supposed to be a quick way for the data analysts to update certain fields instead of digging through the underlying tables so we thought lets create a view update it through there simple peasy right nah oracle was having none of it we tried an update BAM ORA-01732 it was like a cold bucket of water in the face and that's when I started doing the deep dive into Oracle's view update rules and all the limitations

So lets break it down why this error happens basically oracle views aren't always directly updatable they are logical representation of data they aren't actual storage locations so when you try to update a view oracle has to figure out which base table the updates should be applied to this is where things get complicated if the view is too complex oracle won't be able to resolve the update target and just throws this error

The biggest gotcha is when your view has joins especially outer joins or aggregate functions like `sum count avg` or `group by` clauses or `distinct` clauses these kind of views are typically not updatable because oracle cant figure out which base tables and columns are being targeted for modifications you can use `rownum` also but it will probably give you issues as well it’s a big red flag for the optimizer in general and should be avoided also views with `union` `union all` `intersect` or `minus` are also mostly not updatable this can be a real source of pain because this stuff comes up quite often

Another reason can be that your view is selecting data from columns that are not part of a primary key or unique constraint in the base table which makes it difficult for oracle to update a specific row

Its important to remember that this is the database being protective of itself if it lets you mess with a view it could potentially lead to inconsistencies and data corruption so its doing its job kind of a grumpy but effective one. It's like that friend who won't let you go out in the rain without an umbrella they might be annoying but they're looking out for you I always thought this should be a meme but ok

Now lets get into what you can do here are few things you can try first you can simplify your view if possible this is often the best approach if you remove things like the outer joins the aggregations the distinct etc the chances are higher that the view will become updatable so instead of updating through the view you can always create the update statement to update the base tables directly this is the most reliable solution for those complex queries

However sometimes you have no choice other than to update through a view which means we have to work with the restrictions oracle provides its time for the `instead of triggers`

`Instead of triggers` are a godsend here they essentially tell oracle "hey when someone tries to update this view don't try to figure it out yourself just execute this trigger instead" this trigger then becomes responsible for performing the appropriate update delete or insert on the base tables this means you have full control over how your updates are handled

Here’s an example of an `instead of trigger` for a simple view let's say we have a view called `employee_view` that combines data from `employees` and `departments` tables and for simplicity we will have a basic example and we assume that it is a simple view that we want to be able to update.

```sql
CREATE OR REPLACE VIEW employee_view AS
SELECT
    e.employee_id,
    e.first_name,
    e.last_name,
    d.department_name
FROM
    employees e
JOIN
    departments d ON e.department_id = d.department_id;

--instead of update trigger
CREATE OR REPLACE TRIGGER employee_view_instead_of_update
INSTEAD OF UPDATE ON employee_view
FOR EACH ROW
BEGIN
    UPDATE employees
    SET
        first_name = :NEW.first_name,
        last_name = :NEW.last_name
    WHERE employee_id = :OLD.employee_id;
    -- you can do other additional things here like logging

    --If you want to update the department table
    --You will need to select it from base tables and update accordingly

    --Example would be like this but depends on the logic of your view
    -- UPDATE departments
    -- SET
        -- department_name = :NEW.department_name
    -- WHERE department_id in (select d.department_id from employees e join departments d on e.department_id = d.department_id where e.employee_id = :OLD.employee_id);
END;
/
```

Now this trigger is listening for updates to `employee_view` when you try to update a row in `employee_view` the trigger grabs the new values you're providing in `:NEW` and the old values `:OLD` and then updates the appropriate rows in the `employees` table. Notice the comment section if you want to update another table like `departments` you will need to select the `department_id` using the `employee_id` because this is how the view is constructed

Its important that the trigger does not only update the employee table but if it's necessary it should also update the other tables involved in the view, otherwise the update wont be reflected. This is an `instead of update` example.

Let's see a similar `instead of insert` trigger example this time we're going to assume that we have a view with the same name with only the employee table columns it's a simple view that we also want to insert into.

```sql
CREATE OR REPLACE VIEW employee_view AS
SELECT
    e.employee_id,
    e.first_name,
    e.last_name
FROM
    employees e;


CREATE OR REPLACE TRIGGER employee_view_instead_of_insert
INSTEAD OF INSERT ON employee_view
FOR EACH ROW
BEGIN
    INSERT INTO employees (employee_id,first_name, last_name)
    VALUES (:NEW.employee_id, :NEW.first_name, :NEW.last_name);

END;
/
```

This one is more straightforward we are taking the data from the view insertion and then we are inserting it into the base table. The last example would be `instead of delete` and it's again more straightforward.

```sql
CREATE OR REPLACE VIEW employee_view AS
SELECT
    e.employee_id,
    e.first_name,
    e.last_name
FROM
    employees e;


CREATE OR REPLACE TRIGGER employee_view_instead_of_delete
INSTEAD OF DELETE ON employee_view
FOR EACH ROW
BEGIN
    DELETE FROM employees
    WHERE employee_id = :OLD.employee_id;
END;
/
```

So the instead of delete will grab the `:OLD` columns that it needs to delete from the base table in the end those triggers will allow to update insert or delete views regardless of how complex it is but you'll need to implement them yourself carefully and test them before deploying.

A good resource for understanding the theory of relational databases and view updatability is "Database System Concepts" by Silberschatz, Korth, and Sudarshan. While not Oracle specific it has solid background on these concepts and should help you a lot to grasp the concept of updating views in general. Another book that will be helpful is "Oracle Database SQL Language Reference" this is more oracle specific and will help you with syntax and with all of the options available in Oracle SQL. And if you are looking into database performance "Oracle Database 12c Performance Tuning Recipes" by Darl Kuhn is a really good pick. This book will teach you a lot about Oracle internals and its important to know how things work under the hood.

These books should give you a really good foundation on both the theoretical and practical sides of the issue with updates and views.

So yeah ORA-01732 isn't fun but it's definitely manageable once you understand the rules Oracle is imposing. Hope this helps and good luck with your view updating adventures.
