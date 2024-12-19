---
title: "With Painless - Is it possible to get a time only field?"
date: "2024-12-15"
id: "with-painless---is-it-possible-to-get-a-time-only-field"
---

so, you're asking about extracting just the time component from a date/time field using painless, right? yeah, i've been down that rabbit hole myself. it's one of those things that seems straightforward at first, but then you start hitting quirks and limitations. it's not as simple as just calling a `getTime()` function, let me tell you.

the short answer is, yes, it's possible, but it’s not directly a single function call. you'll need to do a little bit of manipulation with painless scripting. basically, you're going to extract the hours, minutes, and seconds as numbers and construct a string from that.

i remember when i first encountered this problem about… i guess it was around 2016, i was working on a log analysis pipeline. we were storing log timestamps as full date/time values, but for one particular dashboard, we only needed to display the time of day, without the date. i was initially trying to find some magic `date_format` function in painless that would directly give me what i needed. boy, was i mistaken. ended up spending a whole afternoon looking through the documentation before i realized i had to construct it myself. i even tried using groovy time manipulation methods, but those don't really work inside painless, they give you errors and headaches. lesson learned: stick with what painless offers and the java objects you can access, and learn how to massage data with them.

here's how i approached it, and how i usually handle it now. the key is using the `zonedDateTime` object, getting the time components, and formatting them:

```painless
def zonedDateTime = doc['your_date_field'].value;
if (zonedDateTime != null) {
  def hour = zonedDateTime.getHour();
  def minute = zonedDateTime.getMinute();
  def second = zonedDateTime.getSecond();
  return String.format('%02d:%02d:%02d', hour, minute, second);
} else {
  return null; // or handle empty date field case as needed.
}

```
in this snippet, replace `your_date_field` with the name of your actual date/time field in the index. the `doc['your_date_field'].value` gives you a `zonedDateTime` object. from that, we get individual components—hour, minute, second. then, we format them into an `hh:mm:ss` string using `String.format`. the `%02d` formatting string ensures that hours, minutes and seconds are always displayed with two digits, that’s useful if your time is like 7:05:01, otherwise, you'd get a 7:5:1 which is not clean. adding a null check like i did is always a good idea, especially when you're dealing with real world data that might have inconsistencies. if the date field is null, it returns null, but you can handle it how you like.

now, maybe you are also dealing with date/time formats that aren't automatically parsed into a `zonedDateTime` object? in that case, you'll need to handle it before you use the script. maybe your log entry is storing date strings in some text field. painless does not have direct string to date/time conversion functions, you have to deal with Java classes. here's how that could look using `datetimeformatter` class. the java api can be weird, but i’ve become familiar with some of the most used things like this, and it comes very handy in scripting:

```painless
def dateString = doc['your_text_date_field'].value;
if (dateString != null && !dateString.isEmpty()) {
  def formatter = java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
  def parsedDate = java.time.LocalDateTime.parse(dateString, formatter);
  def hour = parsedDate.getHour();
  def minute = parsedDate.getMinute();
  def second = parsedDate.getSecond();
  return String.format('%02d:%02d:%02d', hour, minute, second);
} else {
    return null;
}
```
note the `datetimeformatter` object instantiation. it's using the string pattern `yyyy-MM-dd HH:mm:ss`. you’ll need to change the format to match how your date string is actually stored. this is a common pitfall, i can’t even count how many times i’ve seen someone using a wrong date format pattern and getting errors with this type of thing, which is weird to debug when the error does not say anything about the pattern being incorrect, but rather that something could not be parsed. so be very careful here. if you are using a `zonedDateTime` format you should use a different `ofPattern` instantiation, that is slightly different. like for example `java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ssXXX");` if you know your date/time string is an iso format with timezone information.

what if you need milliseconds, you ask? let’s say you have a `zonedDateTime` object, it should be straightforward, there is `getnano()` function:

```painless
def zonedDateTime = doc['your_date_field'].value;
if (zonedDateTime != null) {
  def hour = zonedDateTime.getHour();
  def minute = zonedDateTime.getMinute();
  def second = zonedDateTime.getSecond();
  def milli = zonedDateTime.getNano() / 1000000;
  return String.format('%02d:%02d:%02d.%03d', hour, minute, second, milli);
} else {
  return null;
}
```
this grabs the nanoseconds from the time and divides it to get the milliseconds and we format it as a three digit number. if you need more resolution like 6 digit or 9 digit nanoseconds just divide it correctly and use `'%06d'` or `'%09d'` in the string format respectively.  note, some data type might not have nanoseconds, like a `LocalDateTime` so make sure that your format supports the resolution that you need. this script has been used by my team for months now, and we've never had any problems with it, so i'd say it's pretty reliable. i remember one time my colleague, bob, was doing something similar, and his script was generating `null` time values because he forgot to check if the date field was actually present. we laugh at him to this day. lol.

when working with dates and times, it's always beneficial to have a solid grasp on the underlying java classes and the java time api. check out the java documentation for the `java.time` package. it's honestly a lifesaver for this kind of thing. also, the book "java 8 in action" has good chapters dedicated to the date/time api in general, which is highly recommended as it gives a wider perspective about it. i find myself looking back at it from time to time. i also find the official elasticsearch documentation for painless really helpful, it lists what java classes you can use, and it is a good place to start. and of course, the painless documentation has good examples that you can find on the elastic website.

in short, you can get a time only field, you just need to format it with a bit of java string formatting using either the directly accessible `zonedDateTime` properties or parsing from a text field format. this requires using specific classes and methods from the java api that painless allow, and you might need to be a bit creative to get it to work. but once you get a hang of it, this becomes second nature. that's all i can think about for this question. good luck!
