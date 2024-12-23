---
title: "invalid char between encapsulated token and delimiter in apache commons csv libr?"
date: "2024-12-13"
id: "invalid-char-between-encapsulated-token-and-delimiter-in-apache-commons-csv-libr"
---

 so you're banging your head against the wall with that pesky `Invalid char between encapsulated token and delimiter` error in Apache Commons CSV right I feel your pain man I've been there done that got the t-shirt and the stack trace burnt into my retinas

This error basically screams that you've got some funky character lurking between your quoted field and the comma or whatever you’re using as a delimiter It’s usually spaces or tabs sometimes even some weird invisible unicode gremlin messing up your day

Let’s get down to brass tacks I’ve spent way too much time debugging this exact issue back in my days working on a big data project processing those messy legacy files they had all the encodings you could think of and then some it was a nightmare trust me

First things first the obvious thing that I usually check is if you’re even quoting your values correctly and are using the quote char correctly Sometimes people mess this part up even after a lot of experience with CSV you know how it is Here is some Java code example I will give more code example later but lets start with the basic one

```java
CSVFormat format = CSVFormat.DEFAULT
    .withQuote('"')
    .withDelimiter(',')
    .withRecordSeparator('\n');
```

Make sure that your `quoteChar` and delimiter are what you expect them to be `DEFAULT` settings are using double quote `"` but you can see that in the code example that I provided It seems very simple but this is the one of the first things to check

And also double check that you're not mixing different quotes like using single quotes in some fields and double in others it happens more than you think I’ve seen it my own eyes

So if we assume the quotes and delimiters are fine we move on to the actual character between them and I am not talking about characters like `'a'` I am talking about like a real single space or a tab or maybe an non breaking space Let me give you some example data and I will show you the issue

let's assume you have this type of data

```csv
"field1", "field 2" ,"field3"
"value1", " value2  ", "value3"
```

See that single space between the second field ` "field 2" ` and the second comma there is also multiple spaces before and after the `value2` value? That's the gremlin we are hunting This space is usually the problem That’s the invalid character between encapsulated token and delimiter

The thing is that commons CSV is pretty strict about this and is correct in reporting it as incorrect data. Usually real world data isn't always clean though right?

So you have to tell commons CSV to ignore those spaces This can be done with the `withTrim()` method Let me give you a complete example of code where you can see it in action and see the output

```java
import org.apache.commons.csv.*;
import java.io.*;
import java.util.List;

public class CsvParser {

    public static void main(String[] args) throws IOException {
        String csvData =
                "\"field1\", \"field 2\" ,\"field3\"\n" +
                        "\"value1\", \" value2  \", \"value3\"\n" +
                        "\"val4\",   \"val 5\", \"val6\"";

        CSVFormat format = CSVFormat.DEFAULT
                .withQuote('"')
                .withDelimiter(',')
                .withRecordSeparator('\n')
                .withTrim(); // <--- the magic

        try (Reader in = new StringReader(csvData);
             CSVParser parser = new CSVParser(in, format)) {

            List<CSVRecord> records = parser.getRecords();
            for (CSVRecord record : records) {
                for(String value : record) {
                    System.out.print("[" + value + "] ");
                }
                System.out.println();
            }

        }
    }
}
```

This is how the output would look like:

```
[field1] [field 2] [field3]
[value1] [value2] [value3]
[val4] [val 5] [val6]
```

See now all the values in the CSV are parsed correctly even with leading spaces and multiple spaces within a single value This is the most basic approach and most likely the solution to your problem `withTrim()` is really one of the most important settings to use in most cases when dealing with CSV files where you do not control the data source

Now let’s assume that your problem was not a space but some other character what I am talking about are things like UTF BOM characters or weird ASCII characters that can also cause this issue

If it is not a space then you'll need a different approach Sometimes these weird characters are hidden and really hard to catch So you have to actually go deeper and start looking into the actual bytes representing the characters in your data.

I’ve also had that one time where the source system produced a bunch of CSV files with different encodings because the developers did not understand the importance of the character encoding standards and used whatever they had and no standards it was a mess This will also give this error of invalid character between encapsulated token and delimiter because you would use a different encoding than the one that was written in that file and a non encoded character is interpreted as a different character that can produce this error

To investigate what character is the issue you might want to go to the very basics print the character as a hex or even as a decimal value to understand what character you are dealing with I will give you the code example on how to do that

```java
import org.apache.commons.csv.*;
import java.io.*;
import java.util.List;

public class CsvParser {

    public static void main(String[] args) throws IOException {
        String csvData =
                "\"field1\",\u00A0\"field 2\", \"field3\"\n" +
                        "\"value1\", \"value2\", \"value3\"\n";

        CSVFormat format = CSVFormat.DEFAULT
                .withQuote('"')
                .withDelimiter(',')
                .withRecordSeparator('\n');


        try (Reader in = new StringReader(csvData);
             CSVParser parser = new CSVParser(in, format)) {


            List<CSVRecord> records = parser.getRecords();

            for (CSVRecord record : records) {
                for(String value : record) {

                     for (int i = 0; i < value.length(); i++) {
                        char c = value.charAt(i);
                        System.out.print("Character: " + c + " (Hex: " + Integer.toHexString(c) + ", Dec: " + (int) c + ") ");
                    }
                     System.out.print(" | ");
                }
                System.out.println();
            }

        }
    }
}
```

In this example I've used a non breaking space character `\u00A0` as the character between the comma and the quote character you can use your specific data to detect the offending character This example shows all the values of your CSV row with the hex and decimal representation of each character which can be very useful to see the character that is causing the problem

You will probably need to do some more string manipulation to completely filter out those characters before parsing this is not a perfect solution but this is a very good first step in understanding what character is the one causing the problem This step can be cumbersome but there are a lot of resources online for example stackoverflow itself that can help you with that particular step

If you're dealing with a large file and are using a buffered reader that can also mess up the characters make sure that you use buffered reader that supports the character encoding you want

I was working on one project one time and the error was very strange because it did not happened in the dev environment but in production the issue was that a different locale on the linux servers made the java library use a different encoding by default and then bam the `invalid char between encapsulated token and delimiter` happened I lost a full day looking at it just to find out that it was a dumb mistake a silly locale setting which I should have seen before I am also a human so I commit errors too

Also and this is very important make sure you know the real encoding of your CSV files before trying to parse them not all files are UTF-8 sometimes you have to use `ISO-8859-1` or other types of character encodings This is not related directly to the error but is another thing that can mess up your parsing and that is very important to keep in mind

For example in this example we use UTF-8 encoding

```java
import org.apache.commons.csv.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.List;

public class CsvParser {

    public static void main(String[] args) throws IOException {
        String csvData =
                "\"field1\", \"field 2\" ,\"field3\"\n" +
                        "\"value1\", \" value2  \", \"value3\"\n" +
                        "\"val4\",   \"val 5\", \"val6\"";

        CSVFormat format = CSVFormat.DEFAULT
                .withQuote('"')
                .withDelimiter(',')
                .withRecordSeparator('\n')
                .withTrim();

        try (Reader in = new InputStreamReader(new ByteArrayInputStream(csvData.getBytes(StandardCharsets.UTF_8)), StandardCharsets.UTF_8);
             CSVParser parser = new CSVParser(in, format)) {

            List<CSVRecord> records = parser.getRecords();
            for (CSVRecord record : records) {
                for(String value : record) {
                    System.out.print("[" + value + "] ");
                }
                System.out.println();
            }

        }
    }
}
```

Note that I am using `new InputStreamReader(new ByteArrayInputStream(csvData.getBytes(StandardCharsets.UTF_8)), StandardCharsets.UTF_8)` which is specifying that the data is UTF-8 encoded

In conclusion to solve this `invalid char between encapsulated token and delimiter` you have to ensure correct quoting use `withTrim()` for spaces and investigate any other non visible characters you have in your data if it is not spaces And always always be careful about character encodings.

Some good resources for further reading are “Understanding Character Encoding” by O’Reilly which is a book that goes deeper into the topic of character encoding and “The Unicode Standard” which is the formal reference for all things Unicode If you want to dig deeper into CSV you can check the RFC 4180 which defines the CSV format

That should cover it hopefully now you don’t need to bang your head anymore
