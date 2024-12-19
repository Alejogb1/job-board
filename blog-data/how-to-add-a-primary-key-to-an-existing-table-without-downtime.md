---
title: "How to add a primary key to an existing table without downtime?"
date: "2024-12-15"
id: "how-to-add-a-primary-key-to-an-existing-table-without-downtime"
---

alright, so you're looking at adding a primary key to an existing table without causing a major outage, right? i've been there, trust me. it's a situation that can make even the most seasoned database admins sweat a little. i remember back in the day, at my first real job, we had this beast of a production database. it was a monolithic thing, with a table that grew like crazy. someone forgot to put a primary key on it when they set it up, and naturally, it became a problem later on. trying to add one then felt like walking a tightrope. let's get into the details of how i would approach this, and how i learned the hard way that there are ways to do it without bringing the whole system down.

the core issue here is that adding a primary key usually requires a table lock, which stops all reads and writes. in a production system, that's generally unacceptable. a short lock might be manageable during a maintenance window, but ideally, we want to make the change completely online. that is to say, without locking out clients.

the basic approach i've come to rely on involves adding a new column, populating it with unique values, and then promoting it to be the primary key. it's not exactly instantaneous, but it allows the database to keep working during the process.

here is a general recipe i usually use:

**step 1: add a new column and populate with unique values**

first, we need to add a column that will eventually become our primary key. i typically prefer using uuid generation for this. this way you avoid problems with sequential ids. also, it makes the whole thing more parallelizable and also works very nicely when you want to migrate a big table to a more modern database.

the following sql would perform the operation. note that the table name is fictional, change it to the table you are working with:

```sql
alter table my_table add column new_id uuid;

update my_table set new_id = uuid_generate_v4();
```

the first line adds the `new_id` column of type `uuid`. then, the second line populates each row with a randomly generated uuid. in postgresql the function `uuid_generate_v4` can be used. other databases might have similar functions. remember you need to have this function installed in your database beforehand. this might require enabling extensions or modules in the database.

this step is crucial, as it generates the unique values we need. it does not use locks, but it will generate load on the database because it updates a lot of rows. i recommend doing this during off peak hours, this is a lesson i learned a long time ago. we once attempted this during peak hour, it didn't end well. it was not fun explaining why the application went down. after that, we added metrics and alerts in place and that prevented future disasters.

**step 2: create a unique index**

after the new column has been populated with unique values we must ensure no duplicates can appear later. for this, we add a unique index. it's important we don't set this as a primary key at this stage. we want to avoid locking the table for now.

the sql command to create the index looks like this:

```sql
create unique index idx_my_table_new_id on my_table (new_id);
```

this will create an index on `new_id` and enforce that values in that column are unique. you might think this will lock the table. usually the database engine is smart enough to create this index online. if that is not the case, you should be able to specify that you want to create this index concurrently. in postgresql you would use the `concurrently` keyword. read your database docs for this.

this index will allow faster searches and will prevent duplicate new_id values being added. you could think of it as a pre-primary key constraint.

**step 3: promote new column to be the primary key**

now we are ready to add the actual primary key. i usually like to do this in steps. first, i add a primary key constraint and then i drop the old one. sometimes it makes sense to rename the column and leave the primary key constraint untouched. however, in this case, i will go for the simpler route where we create a new primary key.

```sql
alter table my_table add constraint pk_my_table_new_id primary key (new_id);
alter table my_table drop constraint <old_primary_key_constraint_name>;
```

in the first line we add the primary key constraint on the `new_id` column. then, in the second line, we are dropping the old primary key constraint. you will need to find the name of your old constraint. you can do that using database introspection tools. that depends on your database engine. this step is the most sensible one because the primary key will typically require an exclusive lock.

**important considerations**

*   **monitoring**: throughout this whole process, keep an eye on your database performance. you will be doing quite a few operations. you need to track your database resource consumption in terms of io and cpu usage, and see if the database is still performing adequately. there are plenty of monitoring tools available that are useful for this.
*   **downtime:** depending on the size of the table, you might need to perform this change during off-peak hours. even though the database will be mostly available, adding the primary key index can use a substantial amount of resources and that could slow down the application.
*   **data integrity**: if you have existing foreign key relationships in other tables that reference the old primary key, you will need to modify those to point to the new one after adding the new primary key. remember that we are adding the primary key constraint to the `new_id` column, that means that your foreign keys will need to point to that one and not to the old one. we also need to consider that data in the foreign keys might not be unique. this makes migrations harder. the safest approach is to add another column in the dependent tables and then rewrite the logic. after that, drop the old foreign keys. it's a very intricate problem. it's not funny how complicated that can be if you have a large application. but it's very rewarding when you get it done.
*   **testing**: before performing these operations on production, make sure to perform them on a test database that has the same structure and data as your production one.
*   **alternative approaches**: some database engines have built-in online schema change features. such as in mysql. those are worth exploring if the standard approach is too complex for your database.
*   **uuid versions:** there are different versions of uuid generation algorithms. i recommend you understand the differences between them and choose wisely depending on the database you have. for instance, some older versions of mysql do not handle efficiently uuids that are totally random, and that might cause issues with index generation. the database index algorithms are optimized for sequential data. totally random uuids can impact performance significantly in this case. that is why you might find that some applications use uuids that are generated based on a timestamp. it's a complicated topic. if you want to understand this better i recommend reading the "designing data-intensive applications" by martin kleppmann.
*   **backup**: before you perform any operations, ensure your backup procedures are in order and you have taken a full database backup.

this entire process involves careful planning and execution. it's not a task you want to rush through. i've seen the consequences of rushing and they are always painful. always make sure to plan in advance and test on a production-like environment.

to understand the inner working of databases, i really recommend "database system concepts" by abraham silberschatz, henry f. korth, and s. sudarshan. that will give a lot of insight into how primary keys work and the internals of indexes.

i hope this helps you get your primary key added smoothly. just take it one step at a time, monitor closely, and don't be afraid to double-check every step. you'll get there.
