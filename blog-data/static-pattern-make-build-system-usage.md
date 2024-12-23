---
title: "static pattern make build system usage?"
date: "2024-12-13"
id: "static-pattern-make-build-system-usage"
---

 so you're asking about using static patterns in build systems right Specifically how they can be useful I get it I've been there trust me I've wrestled with build scripts more times than I care to remember Let me break it down from my experience it's not always sunshine and rainbows but it gets the job done

First off "static pattern" lets be clear we're talking about predefined rules patterns basically templates to generate build instructions based on predictable file structures Think like if you have a bunch of source files all ending in `.cpp` you might have a static pattern to say "for every file that ends in `.cpp` compile it to an object file that ends in `.o`". That's a very basic static pattern right There's no magical AI or anything just simple string matching and replacement.

Now why would you use that? Well imagine this you're on a project and you've got hundreds of source files spread all over the place You could write explicit rules for each and every single file in your build system but that is going to be really really slow and error prone and maintenance nightmare believe me I know this one time I had a project and the Makefile had like 2000 lines for no reason and changing one source file required 20 mins to figure out where to change the rule and another 20 mins to make sure it worked well it is a nightmare You'd be there all day just editing build files. Static patterns save your sanity in that type of scenario they are basically "for loops" for build systems.

So how do they work in practice? Well in most build systems it's a combination of some type of string matching functionality alongside automatic target generation. The way you define them varies between build tools but the goal is the same define a pattern once and apply it to a whole bunch of files automatically. Let's see some examples real quick.

Here is how we could use GNU Make for example:

```makefile
SRC_DIR = src
OBJ_DIR = obj
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	g++ -c $< -o $@

all: $(OBJ_FILES)

clean:
	rm -f $(OBJ_DIR)/*.o
```

What's happening here? So we're using the `patsubst` function to replace the `src/*.cpp` string with `obj/*.o` This effectively gives us the output objects which we then use as dependencies. The `%.o` pattern rule is a static pattern It means "for anything that ends in `.o` if we go to the corresponding source file and find a `.cpp` file there do this action. And the actions is simply compile. Now this is a very simple example mind you but it scales really well you just add files in src and Make figures it out. Trust me it is a bless.

Next let's look at CMake:

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SRC_FILES "src/*.cpp")

add_executable(my_executable ${SRC_FILES})

target_include_directories(my_executable PUBLIC "include")
```

CMake takes a slightly different approach it's not exactly the same type of pattern matching as Make but the `file(GLOB_RECURSE)` command essentially does that it finds all the `.cpp` files in a tree structure recursively and then it uses them to create a target. No explicit rule making here but the effect is the same everything is automatic and declarative. So for new source files you won't need to add to the CMake lists file unless you start creating new projects or add new libraries and so on. Which is always a pain.

one more example let's see Meson it's a bit more modern and Pythonic you know:

```meson
project('my_project', 'cpp')

src = files(
    'src/main.cpp',
    'src/module1.cpp',
    'src/module2.cpp',
)

executable('my_exe', src, include_directories: 'include')

test('unit_tests', 'tests/test_main.cpp', dependencies : dependency('gtest'))
```

Meson uses a more explicit approach You need to explicitly list out the source files in the `files()` function but you can also use wildcards and create custom patterns based on it but I don't want to include a code sample of this to not overly complicate things. The build rules are then set using the functions such as `executable()` for executable programs or `library()` for libraries the cool thing is Meson handles different operating systems gracefully it does not really care if you have MacOS Linux or Windows it will compile anyway it's a feature. It is also much easier to understand than other systems. I think it is a great system. I don't know what people think though.

The key thing to remember about static patterns is this they are based on naming conventions they work best when your files are organised in a predictable way. If you start throwing source files all over the place static patterns can become really tricky to manage You need consistency when using static patterns. That's why project structure is so important when you deal with big projects.

I need to say one thing also that static patterns are not a substitute for understanding your build system. If you write some mess of static pattern rule and have no clue what it does then I am sorry but you are in trouble. They make things easier but you still need to understand what's going on under the hood. It's kinda like saying that driving a car can get you from point A to point B but it's still important to understand how to fill your gas tank.
I know a person who did not understand and one day his laptop started to compile something that was not supposed to be compiled and it took him a week to figure out what happened. Not so fun.

And now to something a bit controversial but it's true a lot of people I have seen do this they over-rely on static patterns they just create everything through them it is a bad practice if you ask me. This one time I saw a build system where literally every file in the project was a result of a static pattern. I mean not just C++ files also data files config files literally everything. It was so confusing to work with and you had to do so much work to avoid generating a file in the wrong directory. It is usually better to use explicit rules for files that are not related to compilation of source files. Think about a configuration file that can be located in a different folder based on a condition or a file that needs custom build steps. Static patterns are cool but don't overdo it.

 so where can you learn more? I’d say don’t look for “static patterns” specifically cause that term isn’t really standardized. Instead look at the documentation of the build tools directly:

*   **GNU Make:** Start with the official Make manual it's a classic I'd say its good to read it because it will give you some of the history and why Make was made in the first place but that may be very boring to some people
*   **CMake:** Read the CMake documentation and the CMake tutorial it's very well written and covers all the basics
*   **Meson:** Meson has a great documentation which is easy to read and understand so make sure to start there
*   **Autotools:** While it is older and less popular it is still used but it's very different so make sure to check it out. It can also be helpful to understand some history of build systems. The documentation can be a bit hard to read though but if you like a deep dive then have a go.

There are also a bunch of books out there on build systems but I am not sure that they will cover this in detail but it's worth it to check out the following books if you are interested in general build process:

*   "Making Software: What Really Works, and Why We Believe It" by Andy Oram - This books explores the general process of building software so it is beneficial in any type of case.
*   "Effective CMake" by Daniel Pfeifer - If you are into CMake this book may be a great purchase for you.

Static patterns aren’t some magic bullet but they sure can be a lifesaver they help you manage complex builds more efficiently and they are pretty flexible.
Just keep in mind they are simple string processing rules that need consistent structure and explicit configuration for special cases.

Hope this helps out. Let me know if you got any more questions. Good luck with your project!
