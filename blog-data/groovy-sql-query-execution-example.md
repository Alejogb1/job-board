---
title: "groovy sql query execution example?"
date: "2024-12-13"
id: "groovy-sql-query-execution-example"
---

Okay so you're asking about groovy and sql execution right? I've been there man believe me. I've wrestled with JDBC drivers and weird SQL dialects more times than I care to remember. It's a classic case of "it works on my machine" until it totally doesn't.

Alright lets dive into it I've got some scars from past battles I can show you.

First off you need that JDBC driver. Seriously make sure you've got the correct one for your database I've spent hours debugging a simple connection because of that mistake. Been there done that I'm not proud. After that you need groovy. The good news is Groovy makes things way less painful than Java directly. No crazy verbose boilerplate I'm looking at you Java I love you but some times you are a pain.

Let's assume you've got that sorted and you're ready to start executing some queries.

Here's a basic example of getting a connection and doing a select I'm assuming you just want a quick run to see it in action. You can adapt this:

```groovy
import groovy.sql.Sql

def dbUrl = "jdbc:your_database://your_host:your_port/your_db"
def dbUser = "your_username"
def dbPassword = "your_password"
def dbDriver = "your.driver.class.name"

def sql = Sql.newInstance(dbUrl, dbUser, dbPassword, dbDriver)


try {
    sql.eachRow('SELECT * FROM your_table') { row ->
        println "Got a row with id: ${row.id} name: ${row.name}" // change this of course
    }
} finally {
    sql.close() // ALWAYS close your connection you don't want connection leaks trust me
}
```

Okay so quick breakdown. `groovy.sql.Sql` is your friend. You need to set that up properly the first time with your database information. The `eachRow` method is nice because it iterates over the results one by one.  I tend to prefer this over trying to grab a whole list of results at once especially if we're talking big tables.

Now a quick word on those `your_database` `your_host` `your_port` `your_db` `your_username` `your_password` and `your.driver.class.name`. Obviously you need to replace these with your specifics. I've lost count how many times i've left them with placeholder values. One time in college i left the placeholders on a project that had to be presented for a grade. Lets just say I learn my lessons.

Another thing I'd like to point out is closing the connections with `sql.close()` is very important in order to avoid leaks. I've also had the bad habit of leaving the connection open and then my app would stop working. I'm still learning this stuff like the rest of us.

So that's a `SELECT *` which you might use as a starter but obviously in real world scenarios you are probably going to want a specific query. Let's take a look at something a bit more refined. Maybe you want to pass parameters to your query instead of hardcoding values into it.

```groovy
import groovy.sql.Sql

def dbUrl = "jdbc:your_database://your_host:your_port/your_db"
def dbUser = "your_username"
def dbPassword = "your_password"
def dbDriver = "your.driver.class.name"

def sql = Sql.newInstance(dbUrl, dbUser, dbPassword, dbDriver)
def searchId = 123 //let's say you are looking for id 123

try {
    sql.eachRow('SELECT name, email FROM users WHERE id = ?', [searchId]) { row ->
        println "User with id ${searchId}: Name = ${row.name}, Email = ${row.email}"
    }
} finally {
    sql.close()
}
```

Notice the `?` in the query string and the `[searchId]` in the arguments to eachRow. This prevents SQL injection and cleans up your code. Never concatenate strings to construct SQL queries please I've learned this the hard way. Once i had a whole application exposed because someone was able to add a '; DROP TABLE' at the end of the query. I had to start over.

Also notice we're selecting only name and email. Good practice to avoid selecting more than you need to. I hate when people return an object that contains 20 fields when they are just going to use two.

One of the more annoying things I have run into is handling updates and inserts. I always mess them up when it involves more than a few columns.

Here's how you can do an update query and it's pretty straight forward

```groovy
import groovy.sql.Sql

def dbUrl = "jdbc:your_database://your_host:your_port/your_db"
def dbUser = "your_username"
def dbPassword = "your_password"
def dbDriver = "your.driver.class.name"

def sql = Sql.newInstance(dbUrl, dbUser, dbPassword, dbDriver)
def userIdToUpdate = 42
def newEmail = "newemail@example.com"

try {
    int rowsAffected = sql.executeUpdate('UPDATE users SET email = ? WHERE id = ?', [newEmail, userIdToUpdate])
    println "${rowsAffected} row(s) updated"
    if(rowsAffected == 0){
        println "No user with id $userIdToUpdate found"
    }

} finally {
    sql.close()
}
```

The `executeUpdate` method returns the number of rows affected. So you can actually check to see if your update went through as expected. I often find that is very helpful when doing debug. It helps to know when you messed up and the data you expected to be there is not there.

Alright a couple things about the examples above. These are pretty basic and you should consider using a connection pool in any serious application. You do not want to create a new connection every time you need to query your database. Connection pools can really improve performance so do consider that.

Also error handling is very limited in this examples. You should have more proper try-catch and exception handling in your production code. You will get errors on your database. Just a question of when not if.

Now instead of linking to a specific tutorial or something I'd suggest you take a look at "Database System Concepts" by Silberschatz Korth and Sudarshan for a deeper understanding on how databases work. And on the Groovy side "Groovy in Action" by Dierk König is excellent. Those are your bible on these topics I've learned most of the things I know about SQL and Groovy by reading them.

Oh and one last thing you know why I always close my SQL connections? Because I heard they can sometimes hang around like that annoying relative who overstays his welcome ha!

Alright that's the gist of it. It's a pretty wide topic but if you have any specific questions don't hesitate to ask! I've been there. I know the struggle.
