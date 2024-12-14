---
title: "Do we require to do DML commands like insert in liquibase scripts of Corda state schema during migration from v1 to v2?"
date: "2024-12-14"
id: "do-we-require-to-do-dml-commands-like-insert-in-liquibase-scripts-of-corda-state-schema-during-migration-from-v1-to-v2"
---

hey there,

so, you're asking about needing dml commands, specifically inserts, in liquibase scripts when migrating a corda state schema from v1 to v2. i get it, this is a common pain point, and i've definitely been down that rabbit hole myself. let me break down what i've learned through some hard-won experience.

first off, the short answer is: it depends, but generally, **no, you shouldn't need to use insert statements directly in your liquibase scripts for basic schema evolutions**. liquibase is really designed to handle structural changes: adding, removing, or modifying columns, creating tables, indexes, and constraints – the ddl side of things. it is not really meant to be a dml tool for your data migration, in most normal cases.

the core idea of liquibase is to manage the database schema, not the data within it. when you move from v1 to v2 of your corda application, you are normally changing your state schema, often by adding new fields, renaming them or changing how the data is managed under the hood. this often involves alterations to table definitions. liquibase is exceptionally good at this. it keeps a record of all changes applied and it keeps the target schemas in sync. for example, adding a new column to an existing table: that's a textbook liquibase use case and is typically done with a `<addColumn>` change tag in your xml. you don't need to manually insert existing data into that new column. if you want you can create that column with a default value.

now, where things can get a bit tricky is when you have substantial data changes *alongside* the schema changes, when you are for instance splitting a single column into multiple ones. that is where we start moving into more complicated scenarios that are beyond just the automatic column transformations.

let's say, for example, in v1 you had a state that stored an address as a single text field:

```sql
-- v1 schema (simplified)
CREATE TABLE address_data (
    id INT PRIMARY KEY,
    full_address VARCHAR(255)
);
```

then, for v2, you decide to split that into separate street, city, and postcode fields:

```sql
-- v2 schema (simplified)
CREATE TABLE address_data (
    id INT PRIMARY KEY,
    street VARCHAR(255),
    city VARCHAR(255),
    postcode VARCHAR(20)
);
```

in this scenario, liquibase alone won't automatically know how to populate the `street`, `city`, and `postcode` columns from the existing `full_address` column. you'll be left with those columns empty, that would require to write some extra sql for this migration using a `sql` tag on liquibase if you were using liquibase for that.

this is the sort of situation where you might *think* you need insert statements but actually, that is not what you want. you don't need to perform a manual insert on the new fields, what you need is to copy or transform existing data to the new columns using a query. in other words, you would run a sql update command, using a select statement for the specific columns using string operations for example if you need to split the address column on the example, this is often achieved with `<sql>` tags.

here's how a liquibase change set to handle such a migration using a select query would look, assuming you are using xml:

```xml
<changeSet id="v1-to-v2-address-split" author="me">
    <addColumn tableName="address_data">
        <column name="street" type="VARCHAR(255)"/>
        <column name="city" type="VARCHAR(255)"/>
        <column name="postcode" type="VARCHAR(20)"/>
    </addColumn>
    <sql>
        UPDATE address_data
        SET street = SUBSTRING(full_address, 1, POSITION(',' IN full_address) - 1),
            city = SUBSTRING(full_address, POSITION(',' IN full_address) + 1, POSITION(',' IN SUBSTRING(full_address, POSITION(',' IN full_address) + 1) ) -1 ),
           postcode = SUBSTRING(full_address, POSITION(',' IN SUBSTRING(full_address, POSITION(',' IN full_address) + 1))+POSITION(',' IN full_address)+1);
    </sql>
    <dropColumn tableName="address_data">
        <column name="full_address"/>
    </dropColumn>
</changeSet>
```

note the `sql` tag, that is where the magic happens. this is a simplified example and the `SUBSTRING` usage might be different based on your specific sql engine, also, this is an example where the address is comma separated and it could be way more complex to parse, but the overall point is the same, do not do inserts on your migration scripts, you need to update the existing records.

a real-world scenario i faced involved a corda state that had a serialized json blob for storing extra parameters. we decided to move those into their own columns. it was a total mess because we had to do a full database transformation using a script that read the old data and insert into the new columns, and the script was full of `json_extract_path_text` and other similar functions that made the code not very readable. this was a big pain, and we later decided that this was not a good pattern to have and if we would have known this, we would have structured the database in a better way. we ended up building an ad-hoc script that used a jdbc connection, which was way faster than using liquibase for this task due to the amount of data we were dealing with (more than 1 million records), if we would have used liquibase we would be there till tomorrow morning, this can lead to timeouts or even worse, inconsistent states on your data if for example the server restarts in the middle of the update. i wouldn't recommend you try that, as it can lead to data loss or corrupt your database.

another time, i was migrating a state that used a enum stored as a string. we decided to store that as a numeric id mapping that to a new enum table. this was also a complex migration that required a custom update script, as the enum was not fixed and can change from version to version.

so, let's recap. liquibase is primarily for:

*   creating tables
*   adding, removing, or altering columns
*   adding indexes and constraints
*   other ddl tasks

liquibase is *not* usually for:

*   complex data transformations
*   data migrations that require substantial logic
*   bulk data updates

if you have complex data transformations like the ones i mentioned, avoid doing them with liquibase, instead:

*   write an ad-hoc script or application that uses jdbc to directly interact with the database. this is often faster and gives you more control for complex changes.
*   use a dedicated data migration tool if the migration is a one-off operation, some tools support these kind of complex tasks, but be careful as they require special knowledge to use them properly and might cause downtime.
*   if possible try to do changes in a way where you don't need these complex data transformations if possible by changing how your application models data.

here’s another example of a simple liquibase migration using `<changeSet>` and a `sql` tag to add a new column and populate it with a default value:

```xml
<changeSet id="add-status-column-with-default" author="me">
    <addColumn tableName="some_table">
        <column name="status" type="VARCHAR(20)" defaultValue="PENDING"/>
    </addColumn>
</changeSet>
```

and another example of a more complex one where we are splitting a single name column into `first_name` and `last_name`:

```xml
<changeSet id="split-name-column" author="me">
    <addColumn tableName="users">
        <column name="first_name" type="VARCHAR(255)"/>
        <column name="last_name" type="VARCHAR(255)"/>
    </addColumn>
    <sql>
        UPDATE users
        SET first_name = SUBSTRING(name, 1, POSITION(' ' IN name) - 1),
            last_name = SUBSTRING(name, POSITION(' ' IN name) + 1);
    </sql>
    <dropColumn tableName="users">
      <column name="name"/>
    </dropColumn>
</changeSet>
```

when dealing with migrations, documentation like "refactoring databases" by scott ambler and pramod sadalage or "evolutionary database design" by martin fowler can give you some ideas on better strategies on how to make changes that require less effort and that is something i have learned during my experience.

a common mistake i've seen is trying to stuff too much into liquibase scripts, especially those complex data updates, the golden rule is: *liquibase changes are not data migration scripts*. try to keep them focused on schema changes as much as possible, it will save you a lot of time and headaches down the road, and if you try to force liquibase to do data migration that is not how it is supposed to work you might end up with a broken system, and nobody wants that, if it doesn't fit, don't use it.

i hope this helps clear up the landscape a bit and that my pain is your lesson, haha, good luck!
