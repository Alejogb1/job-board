---
title: "How do I create hyperlinks with a formula or rollup?"
date: "2024-12-14"
id: "how-do-i-create-hyperlinks-with-a-formula-or-rollup"
---

Okay so you wanna create hyperlinks using a formula or rollup huh Been there done that trust me Its a pretty common thing you run into when you’re dealing with any kind of database like system I’ve spent countless nights staring at a screen trying to get this exact thing to work so I can sympathize lets dive in

First lets tackle why you would want to do this A static link is fine if you have like one thing but if you have a list of records and those records have fields that dynamically create the destination you need a formula or a rollup. Simple as that This often happens when you need to construct urls from unique ids or record names or a combination of things You know like product ids on an ecommerce site or a specific user profile on a internal tool

So here’s the deal direct hyperlinks in these kinds of systems usually dont work with simple concatenations you need to use the system’s syntax to tell it this is a link and not just plain text You need a specific formatting often using special characters or functions This took me a while to understand early on I was like whats the point of it not being able to treat them like regular text but security and functionality I get it now

Let’s start with the simplest case you have a url and maybe a display name for the link Lets say you have a column called “RecordID” containing strings like “REC123” and you want to create links to your company’s internal wiki page for these records.

Here’s how it will likely look using a formula function common in most no code tools and databases where formulas are used. I will just use a generic formula notation that should translate for all platforms

```
FORMULA:
"[" & "View Record " & RecordID & "](" & "https://wiki.yourcompany.com/record/" & RecordID & ")"
```

Breaking this down `“["` and `”]”` are used to create the display text or title of the link which will appear on the screen so “View Record REC123” then you concatenate a url that actually works with the hyperlink formatting so in this case you combine `“https://wiki.yourcompany.com/record/“` and `RecordID` and you wrap it into parenthesis

The idea is this will produce something like this if the RecordID was REC123: `[View Record REC123](https://wiki.yourcompany.com/record/REC123)` and it will get converted into a clickable link on the client side when viewed on your screen in a browser or an application

The thing about these kind of systems is that they dont just output plain HTML so it needs to know this is not just text and thus it will create the hyperlink as it is meant to be. So it will often interpret this as it’s own formatting to show a user a real hyperlink

Now here’s where things get a little more complicated. Let's say you have more complex urls Maybe you need query parameters or multiple parameters or you have parameters that might be null and you dont want the hyperlink broken when these are missing or are not defined.

Here’s where you gotta use a bit more formula logic. I remember this one time I had a client and they were importing CSVs with external references and we needed to generate urls to external sites but sometimes the external id was missing we needed the formula to handle this like a pro

```
FORMULA:
IF(
  NOT(IS_EMPTY(ExternalID)),
    "[" & ExternalID & " Site Link](" & "https://externalsite.com/item?id=" & ExternalID & IF(NOT(IS_EMPTY(OptionalParam)), "&param="& OptionalParam, "") &")",
  "No External Link"
)
```
In this example we use the IF function which checks if the field `ExternalID` is not empty It uses another function `IS_EMPTY` to achieve this. If it isn’t empty we create the hyperlink same as before but this time we have another check `IF(NOT(IS_EMPTY(OptionalParam)) ...)` If `OptionalParam` has something in it we add the `&param=OptionalParam` string to the url otherwise we dont append anything This means if its null or empty we wont get `?param=` at the end making the URL look broken

Otherwise if `ExternalID` is empty we output `No External Link`. This also makes your data more user friendly when you see the data itself in a spreadsheet like view.

Rollups are a different beast altogether because they deal with multiple records in another table or the same table. I once spent 3 hours trying to understand this whole rollup thing I swear it was like 2 am and my brain just stopped functioning but anyways here is how its typically done:
Lets say you have a ‘Projects’ table and a ‘Tasks’ table and you want to display the links to all the tasks for each project in the projects table as a single field using a rollup. So in your tasks table you should have an identifier for the project id such as a column with the name `ProjectID`.

```
ROLLUP:
  JOIN(
    MAP(
      Tasks,
      "[" & TaskName & "](" & "https://yourtasksite.com/task/" & TaskID & ")"
    ),
    ", "
  )
```

Here `Tasks` is the related tasks from the other table we’re rolling up from. First `MAP()` is used to iterate over all of the tasks Then for each task a new link string is generated same as the previous examples. `TaskName` and `TaskID` are columns that exist in the `Tasks` table.

Then `JOIN()` takes all the links generated by the `MAP` function and combines them into a single string using `, ` as a delimiter. So you get something like `[Task 1](url1), [Task 2](url2), [Task 3](url3)` for a given project with three tasks

Now I know you probably want to just copy and paste this to your no code system and start using it right away but I have to say that you’re going to find slight differences between systems so the concept is always the same but the syntax differs slightly. For example you might have a `CONCAT` or similar function to join strings or an `ENCODE_URL` function to make the url more clean and prevent breakage or similar

If you want to deep dive more into this I would recommend looking into “Database System Concepts” by Abraham Silberschatz, Henry F Korth, and S. Sudarshan, especially the sections on data modeling and query languages This can help you really understand how these systems work and what they are capable of. Also papers on functional programming particularly the ones that are about map reduce can help you understand the thought process behind `MAP` and `JOIN` functions.

Also if you want to create something more robust you need to think about edge cases. What happens if the title is empty? What happens if the external id has special characters that break the link. These are not theoretical questions these are very real problems I've personally faced when dealing with client data that’s not clean. Sometimes I swear the data I receive is like a toddler let loose in a coding convention it does whatever it wants. I once saw a name field which had an email in it and it made my day.

So remember this when generating links make sure your data is clean and always test always test always test. This can save you a lot of headaches and sleepless nights. Believe me I've been there and back I've spent hours debugging issues due to bad data or incorrect formulas. So if I can impart one last thing on you from all my years of fighting with formulas its to make sure your data is clean and your formulas are robust to prevent a meltdown. Good luck with your links I hope it works well and if it doesnt come back here and we can figure it out.
