---
title: "Why can't I run a Gradle-compiled JAR file due to a missing main class?"
date: "2024-12-23"
id: "why-cant-i-run-a-gradle-compiled-jar-file-due-to-a-missing-main-class"
---

Okay, let's tackle this. It's a common head-scratcher, and I've certainly spent my share of evenings debugging this exact issue. You’ve compiled a JAR, seemingly without errors, yet when you try to execute it with `java -jar your_application.jar`, you're greeted with that frustrating `no main manifest attribute, in your_application.jar` or, worse, a `ClassNotFoundException` that implies a missing main class. The root cause, in my experience, typically boils down to one of a few scenarios, each relating to how the manifest is built, or indeed *not* built, during the gradle compilation process.

First off, let's understand the purpose of the manifest. The manifest file (usually named `manifest.mf` within the jar's `META-INF` directory) is essentially a small file that holds metadata about your JAR archive. One crucial piece of this metadata is the `Main-Class` attribute. The java virtual machine (jvm) uses this attribute to know which class contains the `main` method where execution should begin. Without this specification, the jvm doesn't know where to start, and thus you get the 'no main manifest attribute' error.

The most common reason for this oversight is the omission of the appropriate configuration within your `build.gradle` file. You need to explicitly tell Gradle which class is the main entry point for your application. Typically, this is a class containing the signature `public static void main(string[] args)`. Often, I’ve seen developers mistakenly assume that gradle will automatically figure this out, especially when they only have one main method in their project, but that's not how it works; the configuration needs to be explicit.

Let me illustrate this with a couple of examples from my past experiences. The first scenario I encountered was a very vanilla setup: a single module project with a standard directory structure. My `build.gradle` file looked something like this initially, missing the crucial configuration:

```gradle
plugins {
    id 'java'
}

group 'com.example'
version '1.0-SNAPSHOT'

repositories {
    mavenCentral()
}

dependencies {
    testImplementation 'org.junit.jupiter:junit-jupiter-api:5.8.1'
    testRuntimeOnly 'org.junit.jupiter:junit-jupiter-engine:5.8.1'
}

test {
    useJUnitPlatform()
}
```

This configuration compiles the project, creates a JAR, but crucially lacks any information about the main class. The fix here was fairly simple. I added a `jar` task configuration, explicitly specifying the main class:

```gradle
plugins {
    id 'java'
}

group 'com.example'
version '1.0-SNAPSHOT'

repositories {
    mavenCentral()
}

dependencies {
    testImplementation 'org.junit.jupiter:junit-jupiter-api:5.8.1'
    testRuntimeOnly 'org.junit.jupiter:junit-jupiter-engine:5.8.1'
}

test {
    useJUnitPlatform()
}

jar {
    manifest {
        attributes(
                'Main-Class': 'com.example.MyApplication'
        )
    }
}
```

Here, I'm using a `jar` task which allows me to customize manifest attributes. The `'Main-Class'` attribute directs the jvm to the `com.example.MyApplication` class, assuming this is the fully qualified name of your main class. After this addition and a rebuild, the jar would run perfectly. The key here was understanding that you need to explicitly define where the program's execution begins. The `com.example.MyApplication` class must contain the public static void main(String[] args) method.

Another scenario I encountered involved multi-module projects, and this one was a bit more subtle. The configuration was seemingly correct for all the subprojects, but the root project's jar task had a default setting. Each subproject had its own jar task configuration with `Main-Class` defined, but the root project was also producing a JAR, and it, without a specified `Main-Class` attribute, was the one I was attempting to execute, hence the error.

The `build.gradle` for a root project might have looked like this, producing an empty JAR by default:

```gradle
plugins {
   id 'java'
}

subprojects {
   apply plugin: 'java'

    group 'com.example'
    version '1.0-SNAPSHOT'

    repositories {
        mavenCentral()
    }

    dependencies {
        testImplementation 'org.junit.jupiter:junit-jupiter-api:5.8.1'
        testRuntimeOnly 'org.junit.jupiter:junit-jupiter-engine:5.8.1'
    }

    test {
        useJUnitPlatform()
    }

    jar {
        manifest {
            attributes(
                   'Main-Class': 'com.example.subproject.MySubprojectApplication'
            )
        }
    }
}
```

While subprojects would generate runnable JARs, the root project produced one without the needed entry point. In this case, I had to either disable the generation of the jar artifact at the root project level or configure the root jar task similar to subproject ones:

```gradle
plugins {
   id 'java'
}

subprojects {
   apply plugin: 'java'

    group 'com.example'
    version '1.0-SNAPSHOT'

    repositories {
        mavenCentral()
    }

    dependencies {
        testImplementation 'org.junit.jupiter:junit-jupiter-api:5.8.1'
        testRuntimeOnly 'org.junit.jupiter:junit-jupiter-engine:5.8.1'
    }

    test {
        useJUnitPlatform()
    }

    jar {
        manifest {
            attributes(
                'Main-Class': 'com.example.subproject.MySubprojectApplication'
            )
        }
    }
}


jar {
   enabled = false //disable creation of jar artifact at root level
}

```

This example is significant; it shows that you must be aware of which jar artifact you intend to execute. You may have multiple jars being generated, and one mistake that I've made in the past, is to try and run the root jar when it is not configured. Alternatively, I could have added a similar `manifest` section to the root project's `jar` task, however disabling is usually cleaner.

Finally, another case I've seen, although less common, is when the `Main-Class` attribute is correct, but class loading fails for other reasons. This often arises in more complex scenarios involving dependencies. If the necessary classes are not available during runtime, the jvm will complain, even if the main class is correctly specified in the manifest. Usually this is manifest in the form of a `ClassNotFoundException` rather than no main manifest attribute. It's not directly related to the missing `Main-Class`, but important to consider when troubleshooting. In this case, ensuring that the appropriate libraries are included in your jar is crucial. Gradle has built-in methods to address this using mechanisms to pack all the dependencies within a single jar, often called fat jars or uber jars. I won't include the code for this, as it goes beyond the core problem of this query, however, it is important to be aware that such problems can also manifest similar issues, even if your `Main-Class` is correctly set.

To deepen your understanding of these topics, I recommend exploring the official Gradle documentation, particularly the sections regarding the `jar` task and manifest manipulation. Also, the book "Effective Java" by Joshua Bloch offers great insights into best practices for Java, including packaging and application setup. Further, look into the java virtual machine specifications specifically regarding how class loaders and the main method are handled which can help to understand what happens at a lower level. "Java virtual machine specification" is the official and authoritative resource for this kind of information.

In conclusion, encountering this issue generally means your compiled JAR lacks the information on where the program should start. Explicitly specifying the `Main-Class` attribute in your `build.gradle`'s `jar` task will resolve this problem in most cases. Remember to always pay attention to the context of your project, especially in multi-module builds, and ensure that the jar you are attempting to run is the one with the correct configuration. Finally, consider issues around class loading and dependencies if the main-class configuration appears correct.
