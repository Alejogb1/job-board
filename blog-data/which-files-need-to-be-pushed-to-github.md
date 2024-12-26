---
title: "Which files need to be pushed to Github?"
date: "2024-12-23"
id: "which-files-need-to-be-pushed-to-github"
---

, let's unpack this. It's a question that seems simple on the surface, but the devil, as always, is in the details. Over the years, particularly during a stint building a distributed microservices architecture, I've seen firsthand the chaos that can erupt from improperly managed git repositories. The short answer is "it depends," but let's get beyond that and into a more precise explanation. When you ask what _files_ to push to GitHub, you're really asking about what constitutes the _source_ of your project—and what contributes to reproducible builds.

Generally, you need to push anything that is required to reconstruct your project and ensure its consistent behavior. This includes, but is not limited to:

1.  **Source Code:** This is the core of your project. Any files you write or modify—whether it’s `*.py`, `*.java`, `*.js`, `*.cpp`, `*.go` etc—should definitely go into the repository. These files define the logic of your application. Not including them is like trying to bake a cake without the recipe.

2.  **Configuration Files:** Think of `*.yaml`, `*.json`, `*.properties`, and other configuration files that specify your application's settings. Database connection details, service endpoints, API keys (ideally, those should be managed more carefully, which I will discuss below) and build configurations are all crucial. They dictate how your code behaves in various environments (dev, staging, production). Omitting them leads to unpredictable behavior.

3.  **Build Scripts/Tools:** If you use build tools like Make, Gradle, Maven, npm, or Dockerfiles, these files need to be part of the repository as well. They're essentially instructions on how to convert source code into executable applications. Without them, you’re left without a consistent and automated build process.

4.  **Dependencies/Requirements:** Files like `requirements.txt` for Python, `pom.xml` for Java, or `package.json` for Node.js should absolutely be tracked in the repository. These files list all the libraries and modules your project relies on. Ignoring these leads to dependency mismatches, versioning problems, and runtime errors.

5.  **Database Migrations/Schemas:** Any scripts or files that set up or modify the structure of your databases should be tracked. These ensure that your database schema is in sync with the version of the code. Without this, you might face data compatibility issues with different versions of your code.

6.  **Documentation:** While you might prefer to store more extensive project documentation externally (for example, on a wiki), any documentation critical to understanding and running the code—readme files, API specifications—needs to be tracked in the repository.

On the other hand, what you generally should _not_ push includes:

1.  **Sensitive Information:** API keys, passwords, security certificates, and any other credentials must be kept out of the repository. These are often managed using environment variables or secure vault services. Committing these sensitive items puts your system at huge security risk.

2.  **Build Artifacts:** Compiled binaries (e.g. `.class`, `.exe`), node modules, and various other temporary or output files that are generated from the build should not be pushed. These can be automatically generated using the build scripts from source code. Tracking them makes the repository unnecessarily large and results in potential merge conflicts and versioning issues.

3.  **Personal IDE Configuration Files:** Files created by your IDE or text editor (`.idea`, `.vscode`) can usually be excluded. They are specific to your environment and are usually not needed for other developers in your team.

4.  **Temporary Files/Cache:** Such files like local caches or intermediate process files should not be pushed into the repository.

Now, let me show you with a few code examples.

**Example 1: Python Project with Requirements**

Suppose you're working on a Python project. You'd have something like this structure:

```
my_project/
    ├── src/
    │   ├── app.py
    │   └── utils.py
    ├── requirements.txt
    └── .gitignore
```

The contents of `requirements.txt` might look something like this:

```
requests==2.25.1
numpy==1.20.3
```

Here's an example of an `.gitignore` file to avoid pushing unwanted files:

```
*.pyc
__pycache__/
.env
venv/
```

In this case, you would definitely push `src/`, `requirements.txt` and the `.gitignore` to your git repository.

**Example 2: Node.js Project with npm**

Now, consider a Node.js project:

```
my_node_app/
    ├── src/
    │    ├── index.js
    │    └── helper.js
    ├── package.json
    ├── package-lock.json
    ├── .gitignore
    └── server.js
```

The `package.json` file could have entries like this:

```json
{
  "name": "my-node-app",
  "version": "1.0.0",
  "dependencies": {
    "express": "^4.17.1",
    "axios": "^0.21.1"
  },
  "scripts": {
    "start": "node server.js"
  }
}
```

The contents of `.gitignore` would be something like:

```
node_modules/
.env
```

In this scenario, `src/`, `package.json`, `package-lock.json`, `server.js` and `.gitignore` must all be tracked in your repository, while `node_modules` is generated through the `npm install` command.

**Example 3: Java Project with Maven**

And finally, an example using Maven for a Java project:

```
my_java_app/
    ├── src/
    │   └── main/
    │        └── java/
    │             └── com/
    │                  └── example/
    │                       └── Main.java
    │   └── resources/
    │        └── application.properties
    ├── pom.xml
    └── .gitignore
```

A portion of the `pom.xml` file might be:

```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-java-app</artifactId>
    <version>1.0-SNAPSHOT</version>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
            <version>2.7.3</version>
        </dependency>
    </dependencies>
    <build>
       <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
       </plugins>
    </build>
</project>
```

And `.gitignore` in this case would look like:

```
target/
*.iml
```

Here you will include all contents of `/src`, `pom.xml`, and `.gitignore`.

**Recommendations for further reading:**

To get a deeper understanding of version control using Git, I strongly recommend going through the official Git documentation, specifically the _Pro Git_ book by Scott Chacon and Ben Straub. It’s available for free online and is excellent. You might also want to review material on CI/CD pipelines, particularly regarding how automated builds are created from version control. In terms of application security, books like "Secure by Design" by Dan Bergh Johnsson and Daniel Deogun, provide practical guidance on how to handle sensitive data in development and deployment, especially avoiding the commitment of secrets. Moreover, keep reviewing and updating your `.gitignore` file based on the languages and build systems you employ.

In summary, choosing which files to include in your repository involves understanding what files are needed to recreate your project and what are not. It also involves ensuring security by keeping your sensitive information out of source control. It’s not always a straightforward, one-size-fits-all answer, but with experience and good practices, you’ll develop a good sense of what to push and what to leave behind.
