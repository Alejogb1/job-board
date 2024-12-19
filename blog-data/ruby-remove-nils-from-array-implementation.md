---
title: "ruby remove nils from array implementation?"
date: "2024-12-13"
id: "ruby-remove-nils-from-array-implementation"
---

Okay so you're asking about removing nils from a Ruby array right yeah I've been there done that many times in my career It's a pretty common task actually especially when you're dealing with data that might have missing or incomplete values Lets just jump right in

I remember this one project back in my early days building this inventory management system you know classic crud stuff. We were getting data from various sources some APIs some from csv files and some god knows where. The problem was that the data wasn't always clean. Sometimes we'd get fields that were supposed to be numbers or strings but they'd come as nil. Which is a common occurrence I know right It was messing up our calculations and views causing all sorts of UI headaches.

So at first I tried the basic approach a loop you know the kind of thing you first learn in coding school. Something like this

```ruby
def remove_nils_loop(arr)
  result = []
  arr.each do |item|
    result << item unless item.nil?
  end
  result
end

my_array = [1, nil, 2, nil, 3]
clean_array = remove_nils_loop(my_array)
puts clean_array.inspect # Output: [1, 2, 3]
```
It works right but it felt kinda clunky even then and I thought to myself there's gotta be a more Ruby way to do this. You know Ruby is famous for its concise and elegant syntax

So I looked it up on the internet back then and I found that Ruby has this fantastic method built-in that does exactly that for you the `compact` method. It’s like they knew we would have this exact problem and it's pretty straightforward. So instead of a loop we could write the same thing like this

```ruby
def remove_nils_compact(arr)
  arr.compact
end

my_array = [1, nil, 2, nil, 3]
clean_array = remove_nils_compact(my_array)
puts clean_array.inspect # Output: [1, 2, 3]
```

Much cleaner right I mean it's just one line. It's so much more readable and less code and usually less code means less potential for bugs which was my goal back then and is my goal nowadays also. `compact` simply returns a new array with all `nil` elements removed. It doesnt change the original one

But I was working on this other project it was for parsing configuration files and these files were massive. Like multiple megabytes of text it was a nightmare So while the `compact` method was great I noticed it was creating a new array which is fine most of the time but in the parsing project when we had arrays with thousands or millions of elements it was creating lots of objects which sometimes led to memory pressure issues and performance slow downs. I mean we were processing these configurations on the fly and every millisecond mattered. If we could avoid creating a copy of the array we would speed up the application.

So I had to figure out another way to do this but without creating a new copy of the array. Luckily Ruby provides a `compact!` method which modifies the array in-place. The bang operator the exclamation mark in the method name is the convention for methods which change the original object. It's a bit like using `git push --force` but for arrays I guess you can say that

Here is how I used it that time
```ruby
def remove_nils_compact_inplace(arr)
  arr.compact!
  arr
end

my_array = [1, nil, 2, nil, 3]
clean_array = remove_nils_compact_inplace(my_array)
puts clean_array.inspect # Output: [1, 2, 3]
puts my_array.inspect # Output: [1, 2, 3]
```
Notice how the original array `my_array` is also changed. Now that's important to remember. If you don't want to modify the original you should make a copy before using `compact!`. I have been bitten by this several times in the past always forgetting to make a copy it's like a rite of passage for Ruby developers haha.

So that's basically it. When you need to remove nils from a Ruby array you've got a couple of options:
- Use `compact` if you want a new array without the nils
- Use `compact!` if you want to modify the original array in place which is useful when you are in high performance applications with limited resources.

Now which one to use depends entirely on your needs and if you can change your original data structures or not. So choose wisely

A good resource to learn more about these kinds of array methods in Ruby would be the "Programming Ruby" book also known as the "Pickaxe Book" it covers many more methods like these. Also Ruby documentation on the website is also a valuable source of information. And if you wanna know more about memory management and garbage collection specifically in Ruby I'd suggest doing some research on papers about Ruby's Garbage Collector or look for talks from conferences like RubyConf. These are much more helpful than just googling it trust me. Hope that helps someone
