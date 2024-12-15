---
title: "Why does Gorm does not autogenerate model ID?"
date: "2024-12-15"
id: "why-does-gorm-does-not-autogenerate-model-id"
---

so, you're hitting that classic gorm gotcha, eh? the one where you expect your model ids to magically appear and instead you're left staring blankly at a zero or some other default value. i've been there, man, more times than i care to count. trust me, it’s a pretty common pitfall, and it stems from how gorm, or rather databases, handle primary keys and auto-incrementing values. it isn't really gorm’s fault at the end of the day.

let's break down why this happens and how to fix it. the core issue is that gorm, by default, assumes you're handling the generation of your model ids yourself. it doesn't automatically decide that an `id` field should be an auto-incrementing primary key. it needs explicit instructions. this behaviour is quite deliberate, it’s not a bug, but it can be very frustrating when you're starting out.

a little bit of my past, when i started exploring go and orms i remember spending hours, at the peak of my frustration, trying to understand how to make these ids work. i had a project, a simple api for managing books i think it was, where i just kept using `gorm.model` structure assuming that was all needed and that it was supposed to handle the ids generation itself. i kept adding data to the database and instead of having all ids automatically created i kept getting only 0 as id. that's when i realized that i needed to use tags. the `gorm.model` struct is useful to have your basic struct but it does not magically solve the issue of id generation. i ended up needing to specify tags for that. it wasn't intuitive at first but reading the documentation carefully and looking at some examples made me understand.

the key is the struct tags. when defining your model, you need to tell gorm that the `id` field is indeed the primary key and that the database should automatically generate its value. most databases use auto-increment for this, but it's not a given. you explicitly have to specify that behaviour.

here's the most basic way to do it using gorm struct tags. let's say you have a simple user model:

```go
type user struct {
    id        uint   `gorm:"primaryKey;autoIncrement"`
    name      string
    email     string `gorm:"unique"`
    password  string
}
```
here, the `gorm:"primaryKey;autoIncrement"` tag is the magic that tells gorm: “hey, this `id` field is my primary key and the database should handle auto-incrementing its values”. that's the minimum you need for a normal auto-generated id.

another detail that a lot of people miss is that if your id is not named exactly `id` or `ID` gorm might have issues in understanding that's a primary key. in those cases, you can use `gorm:"primaryKey"` in any field name. in my previous api project i remember renaming the id from `id` to `book_id` at some point and things suddenly stopped working, i had to use the primary key tag to fix it.

```go
type book struct {
  book_id   uint   `gorm:"primaryKey;autoIncrement"`
  title     string
  author    string
  isbn      string
}
```
the other way to define the primary key is using the `gorm.model` and then defining the tags. you can always add the tags to the fields if you don’t want gorm to autogenerate them.

```go
import "gorm.io/gorm"

type User struct {
	gorm.Model
    Name string
    Email string `gorm:"unique"`
	Password string
}
```
in this case, the `gorm.model` adds the default `ID` field and the fields `createdat`, `updatedat`, and `deletedat`. if you don't want to use these fields then you have to define your fields.

the database itself also matters. gorm is flexible enough to work with different databases, and each one has its nuances when it comes to auto-increment. postgres, mysql, sqlite—they all might have very similar implementations, but they are not exactly the same. gorm abstracts much of it, but knowing your database's peculiarities is very beneficial in more complex use cases. i usually start with sqlite for small projects and then move on to a better database for scalability reasons.

there's one common trick that might trip you up is that if you’re manually setting the id of an element, even if it's 0, gorm assumes you know what you’re doing and it will use that id instead of generating one. that’s the usual scenario i encounter if i'm importing data for instance.

if you're trying to force a specific id, gorm won't overwrite it with the auto-generated value. for instance, if you have an auto-increment id but you are inserting the data with some seed and you want to force a particular id, you will find that that id will be used. i remember creating the initial data for a test database and i was trying to set the ids but the ids were ignored, and later i realized i had the auto-increment tag, when i deleted it the ids started working fine.

now, let's talk about resources for more in-depth understanding. instead of throwing random links, i would suggest diving into some good database design books. "database system concepts" by silberschatz, korth and sudarshan is a classic. if you want more about database implementations and theory, there is "designing data-intensive applications" by martin kleppmann. these resources will provide you a solid theoretical base and help you with more complicated problems. also always check the official gorm documentation they have very useful information there. i know it might not sound very useful now, but you will be grateful when you encounter more advanced scenarios.

one more important tip is about data migration. if you change your model, gorm will attempt to update the database structure. while gorm migration is useful, is better to use tools like liquibase or flyway for database migration management. it's safer and gives you more control. i remember one time when i changed a column and gorm dropped and recreated a table losing some valuable data, which was quite an unpleasant experience. it's always better to be safe than sorry.

finally, let me add a small piece of humor: debugging this issue is like trying to find the missing sock in the laundry; you know it’s supposed to be there, but it’s just hidden behind something unexpected. it always seems to happen at the most inopportune times, like when you are about to give a demo to an important client and you're getting zero ids.

so, to recap:
*   make sure your `id` field has the `gorm:"primaryKey;autoIncrement"` tag.
*   be aware of how your database handles auto-increment.
*   check if you are accidentally forcing the id when adding data.
*   use external libraries for database migration
*   read good books on database design.

hope this clarifies things. let me know if you have more questions, i've probably tripped on it before and i might be able to help.
