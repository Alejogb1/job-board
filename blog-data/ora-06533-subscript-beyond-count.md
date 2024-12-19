---
title: "ora-06533 subscript beyond count?"
date: "2024-12-13"
id: "ora-06533-subscript-beyond-count"
---

Okay so you're banging your head against the `ORA-06533 subscript beyond count` error right Been there done that got the t-shirt believe me This error pops up in Oracle PL/SQL when you're trying to access an element in a collection either a nested table or an array using an index that's out of bounds Basically the index you're using is either negative zero or bigger than the actual size of the collection

I've seen this thing rear its ugly head in various contexts and you’d think after 20 years I'd have it all figured out but nope it's a sneaky little devil Always crops up when you least expect it like when you're in the middle of a big deploy and all eyes are on you Yeah had that happen a few times It’s like when your code decides to throw a tantrum at 3 am on a sunday.

Alright lets break this down First things first let's be clear about what a collection is in PL/SQL Think of it like a container holding multiple values of the same data type You can have nested tables which are dynamic meaning their size can change at runtime and you can have arrays which are fixed size So you'll hit this `ORA-06533` if you try to get at an element in a nested table or array that's not actually there

The most common culprit I’ve found is bad looping logic When you're looping through a collection and the loop's counter goes past the last valid index boom `ORA-06533` I’ve personally debugged so many for loops its not even funny.

Here's a basic example of how this can happen

```sql
DECLARE
  TYPE my_table_type IS TABLE OF VARCHAR2(50);
  my_table my_table_type := my_table_type('item1', 'item2', 'item3');
  i NUMBER;
BEGIN
  FOR i IN 1..4 LOOP -- Incorrect loop boundary
    DBMS_OUTPUT.PUT_LINE(my_table(i));
  END LOOP;
END;
/
```

This will throw `ORA-06533` because the table only has three elements numbered 1 2 and 3 but the loop tries to access index 4

The fix here is obviously to make sure your loop ends before going out of range Always good to remember that the first index of a plsql collection starts from 1 not 0 so remember that if you’re coming from python or something.

So how do we prevent this from happening I always recommend adding checks before accessing elements like using the `COUNT` method which returns the number of elements in a collection or by checking if the index exists using `EXISTS` I know that sounds basic but you’d be surprised how much time it saves you.

Here is an example using count

```sql
DECLARE
  TYPE my_table_type IS TABLE OF VARCHAR2(50);
  my_table my_table_type := my_table_type('item1', 'item2', 'item3');
  i NUMBER;
BEGIN
  FOR i IN 1..my_table.COUNT LOOP -- Correct loop boundary using COUNT
    DBMS_OUTPUT.PUT_LINE(my_table(i));
  END LOOP;
END;
/
```
Here I am looping up to the my\_table count so if someone adds more elements to the table the loop will adjust automatically. This prevents me from changing the loop logic when someone changes the table data later on.

And here's one using `EXISTS` which is useful if you have a sparse collection some indices might be missing.

```sql
DECLARE
  TYPE my_table_type IS TABLE OF VARCHAR2(50);
  my_table my_table_type;
  i NUMBER;
BEGIN
  my_table := my_table_type();
  my_table(1) := 'item1';
  my_table(5) := 'item2';

  FOR i IN 1..5 LOOP
    IF my_table.EXISTS(i) THEN
      DBMS_OUTPUT.PUT_LINE(my_table(i));
    END IF;
  END LOOP;
END;
/
```

In this example only 1 and 5 are initialized so if you try to do an ordinary access you'll fail but here it will not error since I check if an index is available before I access it.

Another area I've seen this `ORA-06533` is when working with multi dimensional collections Like nested nested tables. The index management there gets complicated fast I have had to debug a 6 levels deep collection before it was an absolute nightmare I tell you.

One time I was dealing with a huge dataset from a legacy system and I was trying to transform it using nested tables for processing A slight error in the indexing logic caused this error to be thrown inside a procedure. It took half a day of debugging and tracing to realize that I had mixed the outer and inner loops index vars and that was throwing the error I’m telling you this whole job would have been simpler to convert into a relational format but the customer wasn't too keen on that.

So what to do next when you encounter this error My approach is usually to start by checking the loop boundaries then I print the size of the collection and then print out the index being used at every access point I add trace logging in the code and if possible in the database if you have access to the dbms output log that also helps.
If the collection is populated from a database make sure that the query returns what you expect. Sometimes it's not about the loops but the source of the data.

For resources I'd highly recommend books like "Oracle PL/SQL Programming" by Steven Feuerstein it is a classic for a reason and its my go to for a refresher on obscure stuff like this or checking up on the syntax when it’s being weird also the oracle official documentation is essential and it has more details and stuff.

Also remember to always have error handling in place. Even after you’ve debugged the issue itself and have a fix it’s good practice to prevent it in the future. In a procedure you can have a catch all error handling block. This won’t make the error disappear but it will at least make the error more graceful and it won’t break everything upstream.

Remember no code is perfect it has bugs in it its just a matter of time and practice before you fix it all and be the best coder you can be. Happy debugging and hopefully no more `ORA-06533` errors for you for a long time.
