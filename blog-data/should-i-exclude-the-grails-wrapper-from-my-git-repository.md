---
title: "Should I exclude the Grails Wrapper from my git repository?"
date: "2024-12-15"
id: "should-i-exclude-the-grails-wrapper-from-my-git-repository"
---

hey there,

so, you're pondering the grails wrapper and its place in your git repo, huh? i get it. it’s one of those seemingly small decisions that can have ripple effects down the line, especially when you start working in teams or have multiple machines involved in the build process. i've been there, tangled in similar dependency management webs, and let me tell you, there are good arguments on both sides.

first off, let's talk about what the grails wrapper actually *is*. it's essentially a script (typically `gradlew` on unix-like systems or `gradlew.bat` on windows) and a properties file (`gradle/wrapper/gradle-wrapper.properties`) that are included in your grails project. they allow you to execute gradle builds *without* having gradle installed globally on your system. this means that anyone working on the project, regardless of their local setup, will use the specific gradle version declared by the project itself. it's a great way to enforce consistency and avoid those "works on my machine" moments.

now, the question of including it in git? well, the general consensus is: *yes, you should*. and i’ll tell you why, drawing from some personal battle scars i’ve accumulated over the years.

i remember back in 2014, before project wrappers were ubiquitous, i was working on a grails 2.x app. we had developers using different gradle versions, and it was, shall we say, *interesting*. builds would pass on one machine and then spectacularly fail on another. hours were spent chasing down dependency conflicts and subtle incompatibilities. it was a real mess. the root cause, more often than not, was that gradle versions were not aligned across machines. we were using some sort of shared project setup with maven which didn't prevent this and made it worse. it felt like working with a time machine that kept sending you back to the 90's where a bad config is almost impossible to find. when wrapper came around it was a big relief. since then, i always recommend it to any team.

the primary advantage of version controlled wrappers is deterministic builds, this is key. when you check the wrapper files into your repository, everyone working on the project is forced to use the exact same gradle version. this eliminates one huge area of potential conflicts that tend to introduce unexpected behaviours. it also provides repeatability in the builds. if someone were to accidentally update the version, it's much easier to rollback via git history. it’s a small step but the impact in teams is big.

think of it this way: you're not just checking in source code; you're checking in the environment necessary to build that source code. the wrapper is as much part of your application as the files. leaving the wrapper out is almost like storing your source code but not the `application.yml` file, which sounds like a nightmare right? it's the same idea but with your build tool.

now, i know some might have reservations about including binary files in the git repo. in the grand scheme of things, the wrapper script files are tiny. the benefits that they provide far outweigh the negligible increase in repository size. the `gradle-wrapper.jar` inside the `gradle/wrapper` directory is another question, some people ignore this jar file via `.gitignore` and let it download every time they perform a `gradlew build`. this has also its advantages. but, personally, i always prefer to keep the full wrapper in git for consistency, you never know if your network will not work when downloading dependencies, it has saved me more than once. the best approach is to commit everything under `gradle/wrapper`.

here's a simple `.gitignore` example showing the minimum you would typically need. not to exclude, but for the other stuff you should exclude like build files, etc:

```gitignore
# gradle build related files
.gradle
build/
**/build/
!gradle/wrapper/
gradle-wrapper.jar # optional if you don't want to commit the jar.
.idea
*.iml
```

notice how `!gradle/wrapper/` makes sure the directory `gradle/wrapper` and its contents are included even if `build/` is being ignored. without the `!` it would not work. and i must specify that i'm not ignoring the `.jar`, because i don't want to download it all the time.

to illustrate this further, let’s look at how the `gradlew` script uses the properties file:

```properties
#Tue Oct 10 17:06:37 CEST 2023
distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
distributionUrl=https\://services.gradle.org/distributions/gradle-8.4-bin.zip
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists
```

this `gradle-wrapper.properties` file tells the wrapper script which gradle distribution to download (if needed) and how to store it. notice the line: `distributionUrl=https\://services.gradle.org/distributions/gradle-8.4-bin.zip`. this is what ensures that everyone will use gradle 8.4 in this case. also, it's worth noting that gradle provides a mechanism for checksum verification, which prevents anyone from tampering with the downloaded files. which is cool.

here’s a basic example of how you'd run a build:

```bash
./gradlew clean build
```

this command will execute gradle tasks using the wrapper script and the version defined in your `gradle-wrapper.properties`, regardless of any gradle version that might be installed in the system. if you have not added the `gradle-wrapper.jar` this will take a bit more time, and will slow down any pipeline, that's why i don't like this approach.

there are a number of good reads on the subject if you want to really understand how gradle works internally. the "gradle in action" book is great, and the gradle official documentation is also extensive. there are a couple of papers in the past, but they are not really worth it nowadays. reading the documentation of the tool you use is usually the best investment. the only thing i’ll say is: be careful with outdated content. things change fast in the tech world and reading something from 2018 about gradle may not be valid anymore.

also, a bit of a side note, but make sure that when using a CI/CD pipeline you are using the same wrapper approach, especially if you want deterministic builds. don't make the mistake of installing gradle in the CI system and assume that will work, using the wrapper is always the way.

so, to wrap it up (pun intended), i strongly recommend including the grails wrapper in your git repository. it provides repeatable, reliable, and consistent builds. it helps to avoid many headaches down the road. trust me, i’ve learned this the hard way. and that is a good reminder, never underestimate the power of small details. i remember one time my team spent three days trying to find out why one new build was failing. eventually, we noticed that one developer was using a gradle version that was not correct. the wrapper prevented this since.

one funny anecdote was that after implementing the wrapper, a junior developer commented that now it felt like magic. and I had to tell him that it was just good software engineering practice, not magic. but i guess it is indeed magic if you have not experienced chaos without it.

i hope this is helpful. feel free to ask if you have more questions!
