---
title: "How to undo 'Exclude from Coverage Results'?"
date: "2024-12-15"
id: "how-to-undo-exclude-from-coverage-results"
---

alright, so you're in a spot where you've accidentally marked something to be excluded from coverage results, and now you need it back in the mix. i get it, been there, done that a bunch of times. it’s one of those things that seems simple enough in theory but can lead to a little head scratching when you're knee deep in a project.

let me walk you through it, and share some of the lessons i learned the hard way. i've spent more years than i care to count staring at code coverage reports, and i’ve definitely made my fair share of marking mistakes. it’s a common slip up, especially when you're in a hurry to get things running or you’re experimenting with new areas of the codebase.

first things first, let's be clear: the exact way to undo this "exclude from coverage" action depends heavily on the specific tool or platform you are using for code coverage. it's not a one-size-fits-all kind of deal. we could be talking about something integrated into your ide like intellij or visual studio, it could be a dedicated coverage tool like jacoco for java, or even a command-line thing like gcov for c++. the general principle is usually similar, but the specific buttons and configurations differ.

in my experience, when i first stumbled upon this particular headache, i was working on a java project using jacoco. we were experimenting with some new microservice architecture, and our test suites were a bit… let’s just say they were still in development. i'd mistakenly excluded a crucial package from coverage. initially, i didn’t notice the impact, but it became apparent later when the overall coverage percentage looked suspicious lower than usual.

let's start with the scenario where you're using an ide, something like intellij. usually, these ides will let you configure coverage on a project-wide, module, or package basis. the 'exclude' action tends to be a context menu option when you see the source file or directory within the coverage results view. the reversal will be the other way around usually, selecting the same thing in the same context menu and choosing something like 'include in coverage results'. in my old setup, i had to right click the package, go to the "coverage" section, and then uncheck that exclusion, usually, it is named 'exclude from coverage' and you must do the reverse of that.

the important part here is to be careful with your selections, because some ides could get a bit tricky and not display what you expect. for example, in a maven project you may have multiple modules, and you can exclude on each of those. the most important thing to make sure is that you have 'the right' module selected. the ide will usually 'remember' your choice, so if you made a mistake, it is usually there.

now, for the folks using jacoco, like me in that past project, the configuration is slightly different. jacoco usually picks up which classes to instrument based on the ant or maven configuration (i was using maven in my case). when you specify that you want to skip certain classes from jacoco, you have to do so usually in your pom.xml file, using configuration like this (snippet one):

```xml
<plugin>
   <groupId>org.jacoco</groupId>
   <artifactId>jacoco-maven-plugin</artifactId>
   <version>0.8.8</version>
   <executions>
    <execution>
       <goals>
         <goal>prepare-agent</goal>
       </goals>
    </execution>
   <execution>
      <id>report</id>
         <phase>prepare-package</phase>
         <goals>
             <goal>report</goal>
         </goals>
    </execution>
   </executions>
 <configuration>
   <excludes>
       <exclude>com/example/excludedpackage/*</exclude>
   </excludes>
 </configuration>
</plugin>

```

to bring the excluded code back, you would have to remove that `<exclude>com/example/excludedpackage/*</exclude>` line entirely. remember to trigger a clean build after you change this pom configuration to make sure it gets picked up by maven. maven will usually output to console something like `jacoco:prepare-agent` which is a good way to know if you are using the right configurations. jacoco is very powerful when configured correctly, and it takes time to get used to it.

also, if you are using a newer version of jacoco, you might see some different settings in the configuration, but the exclude tag is usually similar. the core idea is that you're either explicitly excluding things or not. it’s a toggle of sorts, and it is usually under some sort of <configuration> tag in maven or gradle.

let's move onto command line tools. if you're working with something like gcov for c++, you're probably dealing with flags during compilation. typically, you would compile with `-fprofile-arcs -ftest-coverage` to generate coverage information and in that process exclude using `-fno-profile-arcs` or similar. to 'undo' that, you need to recompile without that exclusion flag (snippet two):

```bash
# compilation to include coverage
g++ -fprofile-arcs -ftest-coverage -o myprogram myprogram.cpp

# to exclude use something like this, for this example, lets assume it is file2.cpp
g++ -fno-profile-arcs -o myprogram myprogram.cpp file2.cpp

# to get coverage again you just need to recompile without -fno-profile-arcs
g++ -fprofile-arcs -ftest-coverage -o myprogram myprogram.cpp file2.cpp
```

so the way to 'undo' the exclusion in this specific example of gcov is to recompile the file again without the exclusion flag, which in this example is `-fno-profile-arcs`. as i said, the logic is usually simple, but the way to get there is sometimes a bit obscure, especially in these compiler tools.

also, for the java people, you can also configure jacoco to use agent-based coverage. in this scenario, you have to specify which packages or classes you want to exclude using the java command line when you are running the application. for example (snippet three):

```bash
java -javaagent:/path/to/jacocoagent.jar=destfile=jacoco.exec,includes=com.example.* -jar myapp.jar

# if you want to exclude something
java -javaagent:/path/to/jacocoagent.jar=destfile=jacoco.exec,includes=com.example.*,excludes=com.example.excludedpackage.* -jar myapp.jar
```

in the examples above, `/path/to/jacocoagent.jar` is the path to your jacoco agent jar, usually it will be available in your `.m2` directory or downloaded from some repo like maven central or similar. when you are using the agent, you'll notice you use the `includes` property to include stuff to the jacoco coverage and the `excludes` property to exclude stuff from jacoco coverage. in order to 'undo' the exclusion of `com.example.excludedpackage` you just need to remove that from the excludes property and re-run the program.

i remember one time i spent an entire morning trying to figure out why a particular service was not being covered. i had a rather complicated gradle setup and after hours i realized i had an exclude regex in the build script that was a little too eager in excluding files. it’s funny in retrospect; it is like one of those jokes where the answer is right in front of your face, but you cannot see it, but it definitely wasn't funny at the time.

i would also recommend, in general, that you make sure you are not excluding things from the coverage by accident in the first place. this usually comes with experience and knowing your tooling. also, having some sort of standard exclusion configurations that you reuse across projects also helps.

as for resources, if you’re diving deep into code coverage, i highly recommend “effective unit testing” by maurizio aniche, if you are interested in java, it gives you some nice insights on code quality and coverage. also, for jacoco, the official jacoco documentation is very complete, with everything you need from configuration to all the flags available. and for gcc, it’s a good idea to check the gcc documentation itself, usually a search in the google search engine with 'gcc documentation' brings it up. reading those, will certainly give you a deep understanding on how code coverage works and how to configure and not mess it up in the first place.

so, to recap, the process of undoing an exclusion from coverage is mostly to reverse the action you took previously. it’s usually a setting or configuration, sometimes a command line flag and it varies a lot from tool to tool. be mindful when using the 'exclude' feature, and always double-check your configuration when you run into coverage issues. once you have a better grasp of it, it becomes much less of a headache. and trust me, you will be making a lot less mistakes as time passes.
