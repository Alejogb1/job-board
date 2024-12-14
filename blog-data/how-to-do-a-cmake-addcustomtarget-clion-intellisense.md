---
title: "How to do a CMake add_custom_target() CLion intellisense?"
date: "2024-12-14"
id: "how-to-do-a-cmake-addcustomtarget-clion-intellisense"
---

i've been there, pulling my hair out over cmake and clion intellisense not playing nice, especially with `add_custom_target`. it feels like the ide just decides to ignore that part of your build setup. been there, done that, got the t-shirt – and probably spilled coffee on it while frantically trying to figure this out.

let's talk about the issue. `add_custom_target` in cmake is powerful, lets you run arbitrary commands, but it’s essentially an opaque box to clion's intellisense. clion parses cmake to understand your project structure, source files, and dependencies, but it doesn't automatically infer what's going on inside that custom target. clion doesn't execute it, it just sees it as a black box. so when you have a custom target generating files that your project uses, intellisense has no idea they exist. this leads to red squigglies, "cannot find" errors, and general code navigation misery. this experience is so widespread it is almost a right of passage in cmake.

the core issue isn’t really cmake itself. it's that clion's intellisense needs help to understand the output of the custom target. we need to explicitly tell clion about these generated files. the fundamental solution i've found is all about making those files visible to clion's indexing process.

here are the techniques i've found effective, and i'll throw in some stories from the trenches of my past projects:

**1. explicitly listing generated sources using source_group() and file()**

this approach essentially involves making your custom target generate files in a known location and then explicitly telling cmake (and by extension clion) that these files are part of your source tree. i used this when i had a custom target that was generating configuration files from templates. i had tons of red squigglies popping up in my code every time i referenced those configs. it was maddening.

```cmake
add_custom_target(generate_config
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/config
    COMMAND some_template_generator -i template.conf.in -o ${CMAKE_CURRENT_BINARY_DIR}/config/generated_config.conf
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tools
)

# make cmake aware of the generated sources for intellisense
source_group(TREE ${CMAKE_CURRENT_BINARY_DIR}/config FILES ${CMAKE_CURRENT_BINARY_DIR}/config/generated_config.conf)
file(GLOB_RECURSE generated_config_files
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/config/*.conf
)

add_dependencies(my_main_target generate_config)
target_sources(my_main_target PRIVATE ${generated_config_files})
```

here, the `add_custom_target` named `generate_config` creates a `config` folder within your build directory and then runs a `some_template_generator` (replace with your actual generator). now, the magic. `source_group` tells cmake to treat the `config` directory (and the generated file) as a real source directory. using `file(glob)` we pick up all `.conf` files from the config directory and make them available as source files to `my_main_target`.  the `add_dependencies` makes sure that `generate_config` runs before `my_main_target`, and we add this as `PRIVATE` so only `my_main_target` is dependent of it and not anything else. this will make sure intellisense finds it.

**2. using `message(STATUS)` to print out files generated**

this is an alternative or addition to the previous strategy. i used this tactic once when i had a code generation step that was super complex and difficult to glob files. basically, i'd have the custom target print the generated file paths to the console, which clion picks up, allowing it to understand the project’s structure. it is a bit of a cludge but it works.

```cmake
add_custom_target(generate_code
    COMMAND some_codegen -i model.proto -o ${CMAKE_CURRENT_BINARY_DIR}/gen_code
    COMMAND echo "generating files"
    COMMAND find ${CMAKE_CURRENT_BINARY_DIR}/gen_code -name "*.h" -print
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tools
)

add_dependencies(my_app generate_code)
```

in this case the `some_codegen` generates `.h` files in the `gen_code` directory. the `find` command lists all header files and `print`s them to the output. the key here is that clion parses the output of your cmake configuration. it sees these paths in `message` and considers those files part of your project. this works but it's a bit brittle, if clion changes how it reads the output then this method fails.

this technique does not fix code navigation with it is more of a "hey, i am aware of this file now" that will get you rid of errors. this technique has some severe limitations and should only be used as last resort or in complement of strategy 1. i prefer strategy 1.

**3. using cmake variables and file(write)**

the previous method is a bit brittle but we can improve it. we can have the custom command write out the list of generated files and then we parse that list with cmake variables. this removes the dependency of clion parsing cmake output messages.

```cmake
add_custom_target(generate_data
    COMMAND some_data_gen -i input.data -o ${CMAKE_CURRENT_BINARY_DIR}/generated_data
    COMMAND find ${CMAKE_CURRENT_BINARY_DIR}/generated_data -type f -print0 | xargs -0 -I {} echo {} >> ${CMAKE_CURRENT_BINARY_DIR}/generated_files.txt
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tools
)

file(READ ${CMAKE_CURRENT_BINARY_DIR}/generated_files.txt generated_files_var)
string(REPLACE "\n" ";" generated_files_list "${generated_files_var}")

add_dependencies(my_service generate_data)
target_sources(my_service PRIVATE ${generated_files_list})
```

here the `some_data_gen` will generate a bunch of data files that we need to read. then, the `find` command outputs the generated files into a text file called `generated_files.txt`. then we read this list and make a `generated_files_list` cmake list variable that we can use with `target_sources`.

**a few additional tips based on my pain:**

*   **make sure custom target dependencies are correct.** i once spent hours troubleshooting a situation where my custom target was creating sources. i was getting red squigglies and weird behaviours because the generation step was happening *after* the main target was compiled, or not in time for cmake to consider them when configuring the project. `add_dependencies` is crucial for ordering. make sure you get the dependencies right or you are going to be confused.

*   **clean build is your friend.** when you change cmake stuff, clion’s intellisense can get confused. try deleting your build directory or using clion's "reload cmake project" option. i lost a good chunk of my hair not realizing a clean build would do the trick, and just kept running and running the project expecting things to magically work.

*   **cmake configure dependencies.** in very complicated projects, sometimes the source generation is not only dependent on an input file but also dependent on other parts of the code. make sure that the custom target is triggered when those dependent sources or files change. you can use `configure_file()` with `CONFIGURE_DEPENDS` to detect changes that might trigger a regeneration. sometimes this is not enough and custom code is required to detect the changes.

*   **check the cmake output window in clion.** look for any warnings or errors during cmake generation. it can give you hints if things go awry. sometimes errors during configuration are silenced or difficult to spot, keep an eye in the output.

*   **be extra careful with generated paths.** make absolutely sure the paths you give to source_group and other functions are exactly right. even a single character mismatch can cause intellisense to fail, and cause you a lot of pain. copy and paste these paths around and you'll reduce mistakes. this is the main reason why i prefer the third method, since it is less dependent on writing absolute paths directly.

i know it can feel frustrating, but once you get a hang of these techniques, you’ll find the balance between the power of cmake custom targets and clion's intellisense is manageable. these methods worked for me in many many situations. when you feel frustrated just relax and know that even the most experienced devs pull their hair out with cmake from time to time, it's the nature of the beast. i have lost tons of hair figuring out cmake and i'm sure i'll loose tons more in the future. it is what it is... i've heard that a lot of new hair growth products are being developed. not that i need them, yet.

**further reading**

for a deeper dive, i would recommend:

*   **"professional cmake: a practical guide" by craig scott**: this book is comprehensive and covers everything from the basics to advanced topics, and has a good section about generating code and how to deal with this in the context of a complex project. a must have for anybody who does serious cmake.

*   **"effective modern cmake" by daniel pfeifer**: another great book that focuses on modern cmake practices. very practical and covers most daily issues and offers a more modern approach compared to the "classic" books. very worth reading.

*   **the cmake documentation itself**. it might seem obvious, but sometimes rereading the official documentation of the functions you are using helps. specially the `add_custom_target` page and the `file` related pages.

hopefully, these stories from my personal cmake battles will help you out and get clion to behave. good luck, and may your builds be clean and your intellisense happy.
