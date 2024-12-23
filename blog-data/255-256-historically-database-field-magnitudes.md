---
title: "255 256 historically database field magnitudes?"
date: "2024-12-13"
id: "255-256-historically-database-field-magnitudes"
---

so you're asking about the historical evolution of database field sizes particularly the shift from 255 to 256 and why that happened I've been neck deep in databases for what feels like a geological era so let me break it down in my usual no-nonsense style

 so back in the day early database systems were seriously resource-constrained storage was a premium CPU cycles were like gold dust every byte counted and developers had to think very carefully about how to use the resources given to them At that time the value 255 for variable length fields especially strings or text fields was pretty much the norm this is because it fit perfectly into a single 8-bit byte if you want to think about it in simple way 8 bits give you 2^8 different possible values which is exactly 256 but because they are often used as length or number of something they start counting from zero hence the number goes up to 255 That’s why you’d see things like `VARCHAR(255)` quite often even the first versions of MySQL used that a lot

The limitation was that the length of the variable-length string field had to be stored somewhere usually at the beginning of the field before the actual character data so that the database software knew where the string ended and the next field began that length needed to be stored using the smallest amount of storage space possible And the smallest amount of space that can store a number from 0 to 255 is a single 8-bit byte

Now I remember in one of my first jobs we were using an ancient database system I kid you not it was something running on a mainframe I had never seen before it was a mess to say the least we had a table with user information and the user’s last name field was like that VARCHAR(255) I kid you not we had users with last names that were very long think Polish and German surnames and we had to literally truncate those and we got constant complaints I had to write a bunch of scripts just to flag this issue and create a manual fix process for this it was crazy and that’s why when I first saw databases supporting 256 I felt like I was in heaven

So why the shift from 255 to 256? It wasn't a sudden jump rather it was a gradual change as systems became more powerful and memory less of a problem the need to maximize storage of the field length for variable-length strings began to diminish

You see the number 255 is a historical quirk not really a fundamental technical limitation 256 is the natural outcome when you think of a single 8-bit byte the whole idea of storing only up to 255 was because we needed to allocate one byte to store length but the byte itself can store 256 values The real change here was actually going from storing the length of the field in one byte to two bytes or more this change gave us the capability to have variable length fields that can be much longer than 255 characters

The transition to more flexible sizes became more common so now the number 255 became more a reminder of the past than anything else. Modern databases can handle significantly longer strings and large binary objects (BLOBs) as well without even batting an eye

Take for instance databases like PostgreSQL or newer versions of MySQL they can handle variable length fields much larger then 255 bytes no problem You just need to specify the required size when creating the table for example `VARCHAR(1000)` or even `TEXT`

Here’s a quick example of SQL DDL to give you some context

```sql
-- Example using VARCHAR(255) - old school
CREATE TABLE users_old (
    id INT PRIMARY KEY,
    username VARCHAR(50),
    last_name VARCHAR(255)
);

-- Example using larger VARCHAR - modern times
CREATE TABLE users_new (
    id INT PRIMARY KEY,
    username VARCHAR(50),
    last_name VARCHAR(1000)
);

-- Example using TEXT for arbitrary length text
CREATE TABLE posts (
    id INT PRIMARY KEY,
    title VARCHAR(200),
    content TEXT
);

```

See how much more flexible it is?
The underlying storage mechanisms for these fields are more advanced usually using pointers or offsets to indicate where the string data starts and ends and how much space is allocated for them this means the size is not constrained by a single byte anymore You can even store multi megabyte text fields like that without issue in a normal relational database

Now let’s dive a little bit into some internal implementation details most database systems will store variable length fields in an efficient manner for example in relational databases when you define `VARCHAR(n)` n is not necessarily the exact upper limit of the field size but rather a maximum or nominal size that you specify and the system will allocate a reasonable size to store it behind the scenes the database might be using pointers or other internal data structures to manage memory dynamically as the data changes

Think about a `TEXT` field which can potentially store enormous amount of textual data It’s not like the database sets aside a giant chunk of memory for that field in advance no no no. Most databases use a combination of techniques like storing the data in separate blocks or utilizing a linked list like structure to keep track of where the text data is actually stored the field itself might just contain a pointer to the actual data location when the database needs to retrieve it

Here is some rough conceptual pseudocode to give you the idea

```cpp
struct StringField {
    int length;   // Actual length of string
    char *data;  // Pointer to string data
};


// For TEXT or BLOB which can be large
struct LargeTextField {
  long long length; // Actual length
  void* data_chunks; // Pointer to another data structure that manages memory
}
```

It may look simplistic but it kind of represents how most databases internally deal with strings

And I will tell you this database designers spend a lot of time optimizing this because performance and storage efficiency is very important it's also one of the trickiest areas of database design

So this change wasn’t about "oh let’s use 256 now" it was the realization that the one-byte length field was a limiting factor and we needed more flexibility and in doing so the artificial 255 cap simply melted away and we gained a bunch more space for our texts

I do think we need a little bit of humor here so I will tell you this in those days debugging a database issue was basically trying to solve a mystery using only the tools you could find in your dad's garage if your dad was a computer engineer from 1970s I wish that were not the case but it was

Now about resources to read this is something that is not really discussed in depth in many places but here are some of the things that could help you deepen your knowledge about this

**Books**:

*   **Database System Concepts** by Abraham Silberschatz Henry F Korth and S Sudarshan This is pretty much the bible for database systems covering a broad spectrum of topics including storage and indexing. It does not focus on this exact problem but has detailed info about it that can help you deduce how this evolved
*   **High Performance MySQL** by Baron Schwartz Peter Zaitsev and Vadim Tkachenko is great for getting hands-on knowledge on how MySQL works and it will give you a good insight of practical storage techniques
*   **Architecture of a Database System** by Joseph M. Hellerstein and Michael Stonebraker: while not directly covering the historical aspect this paper explains concepts used in modern databases that can give you insights in storage handling techniques

**Papers**:

*   Look for papers on database storage engines specifically the ones on variable length data structures and memory management techniques this will allow you to find more about this historical aspect of the topic as well
*   Research papers and articles on early relational database systems like System R and Ingres are also very useful you may find information there even if not directly addressing the problem

In short the shift from 255 to 256 in database field sizes isn't about the number it’s about moving past the limitations of fixed-size length fields and accommodating larger more flexible data types it's a classic example of how historical constraints shape software design and how the constant push for more efficient resources allocation has led to more modern and flexible systems If you really understand this you can understand almost anything about databases

I hope this helps and if you have more questions hit me up
