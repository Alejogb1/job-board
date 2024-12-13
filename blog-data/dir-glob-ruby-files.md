---
title: "dir glob ruby files?"
date: "2024-12-13"
id: "dir-glob-ruby-files"
---

Okay I see this question dir glob ruby files right so let me tell you I've been there done that more times than I can count it's like a right of passage in the ruby world especially when you're dealing with project structures that aren't exactly how you want them

So the core issue is you want to grab a list of file paths that match a certain pattern right that's what globbing is all about and you want to do it in Ruby I assume from the question that you want to use it with the `Dir` class since you mentioned `dir`

Let me tell you a bit of my history with this I had this project back in my early days maybe 8 9 years ago where I was writing this script to process some log files. These logs were generated every hour and they had a filename like `log-2015-10-26-13.txt` `log-2015-10-26-14.txt` and so on my job was to find all the log files for a specific date and then aggregate the data. I tried a few things I even fell into the pitfall of trying to use `ls` from within my ruby code yes that was a huge mistake don't ever do that that's a recipe for a disaster and portability hell I know that now but hey you learn from your mistakes

First things first we need to understand what `Dir.glob` actually does it’s a function that’s built into Ruby's standard library and takes a string pattern as an argument This pattern uses wildcard characters like `*` and `?` to match files and directories and it returns an array containing the paths of everything matching your pattern

So the most basic example would be like this let's say you want to grab all the ruby files in your current directory you'd use

```ruby
ruby_files = Dir.glob("*.rb")
puts ruby_files
```

This code will simply print an array that contains the file paths of all ruby files in the directory you are executing the code from the `*.rb` pattern means match anything that ends with `.rb` pretty straightforward right

Now lets say you wanted to be a bit more specific you only want files inside a specific folder a common mistake that beginners commit is trying to concatenate file paths and making errors using string manipulations when ruby provide path tools and glob can use them perfectly so don't do that

Okay so if your files were all inside a directory called "scripts" that was in the same directory as the ruby script you are running it would look something like this

```ruby
scripts_dir = "scripts"
ruby_files_in_scripts = Dir.glob(File.join(scripts_dir, "*.rb"))
puts ruby_files_in_scripts
```

This is better first we defined a string called `scripts_dir` then we used `File.join` which is the right tool for constructing file paths using the correct path separators for the current OS this is very important specially if your application is expected to be multi platform then we use glob exactly like the previous example but this time inside our defined directory

Let's make it a bit harder so now you want all the files with any extension that starts with `t` and you also want to recursively search in subdirectories let's add some complexity to this problem

```ruby
require 'find'

files_starting_with_t = []
Find.find(".") do |path|
  if File.file?(path) && File.basename(path).start_with?('t')
    files_starting_with_t << path
  end
end
puts files_starting_with_t

```

This example is not using Dir.glob as requested and you may find someone saying hey you can do that with `Dir.glob('**/*')` that is a true statement but if the question is specifically asking for how to recursively find ruby files it is not the best solution that example is going to iterate all the files from the directory and it can get slow if the project contains a lot of files we avoid that with the `Find` library

The code uses the Find module a built in ruby module that will recursively scan the directory provided it calls the block for each file and directory we test if the path is a file using `File.file?(path)` and we also make sure the name starts with the character we want using `File.basename(path).start_with?('t')` and if the conditions are met we added it to our `files_starting_with_t` array remember that this is just an example and in a real application we usually need to do operations with the file path not just simply printing it

Now let's talk about some other things `Dir.glob` can do besides `*` and `?` there are more powerful patterns you can use For example `**` which is like saying match anything recursively but it can be slow if used everywhere so use with caution you can also use character classes like `[abc]` which will match any character within the bracket and `[!abc]` which is the negation version I've used character classes countless times when dealing with messy project structures you wouldn't believe how messy some project folders can get

One time i was cleaning a project a very old project and I found the documentation folder in the root folder I was wondering why the documentation wasn't on a documentation server and also why it was on html files instead of markdown this was before markdown became popular so I had to convert all the html files manually you can only imagine how annoying that was

There are other options too the `glob` function also receives a second optional argument that is a flag that can alter the way the function works one example is `File::FNM_CASEFOLD` if you add that to your arguments then the glob operation will be case insensitive which can be very useful in some circumstances

But also be aware of the limitations of glob for example it doesn't support regular expressions natively and some patterns can get complicated so if you find yourself trying to do extremely complicated patterns then maybe you want to investigate other libraries specifically designed for parsing complex patterns for a few minutes that I don't recommend it here because the question is about Dir.glob

Alright let's talk resources if you want to dive deeper here are some recommendations first the official ruby documentation for the Dir class it's the bible of the ruby world for a reason you'll find the documentation in the ruby-doc website search for `Dir` and you will find everything you need to know about the `Dir` class and all the available methods

Then if you want to understand how the glob function works under the hood then you can check the POSIX standard specifically the section about file globbing it's a bit dry but you will understand how the patterns actually work the standard is the IEEE Std 1003.1-2017 you can check the standard if you are interested in how these patterns are implemented in different operating systems

And another more general resource is the book "The Ruby Programming Language" by David Flanagan and Yukihiro Matsumoto (Matz) it has an excellent explanation of the `Dir` class along with many other useful things about the language it's a classic for a reason

One last thing when doing file operations on a server make sure you are handling the security issues related to that that's a very common mistake people commit when starting developing backend applications remember that is the same as not locking the doors of your house a common mistake I made was writing a code where a user could specify the path that they were processing that was a terrible idea I will never do that again

Also just as a random bit I think my hard drive is starting to feel old it's been spinning for quite a while lately I think I'm going to need to get a new one sooner or later

Okay I think that's it for now let me know if you have more questions I've spent a lot of time dealing with files and patterns and I'm glad to share what I've learned along the years
