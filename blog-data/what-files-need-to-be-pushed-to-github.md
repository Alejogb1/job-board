---
title: "What files need to be pushed to Github?"
date: "2024-12-16"
id: "what-files-need-to-be-pushed-to-github"
---

Let's jump straight into this; the question of "what files need to be pushed to GitHub" isn't as simple as just dumping everything in your project directory. Over the years, I’ve seen plenty of colleagues, and even myself in earlier days, make costly mistakes by either pushing too much, or too little. It's a balance, really, and understanding the rationale behind *what* to commit and *what* to leave out is crucial for a clean, efficient, and collaborative development workflow. I recall a time working on a large-scale distributed system; one developer, let's call him "Bob," pushed an entire directory of build artifacts – nearly a gigabyte – into the repository. It created chaos with repository sizes, slow cloning, and headaches for everyone else on the team. That's a mistake we learned from, fast.

So, let’s break down the types of files you *should* include and those you definitely *shouldn’t*, and why. The general philosophy is: push the source, not the results. Focus on the essence of your project, the code and configuration files, and exclude anything that is generated or specific to your local environment. This approach ensures that the repository contains the core components necessary to rebuild the project on any machine.

First, the essentials – files you *must* push:

1.  **Source code:** This includes all your `.py`, `.java`, `.js`, `.cpp`, `.go`, or whatever source files you use. This is the heart of your project, the instructions that define what it does. If you are working on a web application, HTML, CSS and JavaScript files are also part of this category.

2.  **Configuration files:** Think `.yaml`, `.json`, `.ini`, `.properties`. These files dictate how your application should behave. If you have database configuration files or environment-specific settings, handle them carefully (more on that later). Files like `.env`, which may contain sensitive information should absolutely be excluded.

3.  **Build system configuration files:** Files like `pom.xml` (for Maven), `build.gradle` (for Gradle), `package.json` (for npm/Node.js), `requirements.txt` (for Python), `Cargo.toml` (for Rust) are essential for building and managing your project's dependencies.

4.  **Version control configuration:** Specifically, the `.git` folder itself (though often this won’t be explicitly pushed and is managed by the git tooling) and the `.gitignore` file which specifies files and patterns that should *not* be tracked by git.

5.  **Tests:** Including unit, integration, and any other test code is non-negotiable. Tests are a key part of the project and need to be versioned alongside the source.

Now, what should you definitely *not* push?

1.  **Build artifacts:** These include compiled executables (.exe, .jar, etc.), object files (.o), generated libraries, and anything that results from the build process. These files are specific to your build environment and are not needed in the repository. These can bloat your repository and are largely unnecessary as they can be easily re-generated on different platforms from your source code.

2.  **Local configuration files:** This category includes configuration files that contain sensitive information, such as database passwords, API keys, or any secret tokens. These should be kept outside of version control; typically, these are handled using environment variables or secure configuration management tools. `.env` files are a classic example here; they should be listed in your `.gitignore`.

3.  **Operating system specific files**: things like desktop.ini files for Windows, or .DS_Store files for Mac, these are specific to your system and offer little to no value to others so should be excluded.

4.  **Personal IDE files**: IDE-generated files, such as .idea, .vscode, .eclipse files; while convenient, are usually specific to your local development environment and do not need to be shared. It can lead to conflicts if multiple developers are using different IDE’s.

5.  **Large media files**: it is generally considered poor practice to store large media files such as video, or high-resolution image files. GitHub is not designed for large binary file storage. Consider using services like cloud storage if you need version control over such files.

To help clarify, here are a few code examples that illustrate how to manage file exclusion using `.gitignore`:

**Example 1: Basic exclusion of build artifacts:**

Let's say you're working on a java project using maven. Your `.gitignore` would look something like this:

```
target/
*.class
*.jar
*.war
*.ear
```

In this snippet, we're excluding the entire `target` directory which typically holds the compiled java class files. We're also excluding any files ending with `class`, `jar`, `war`, or `ear`, which are commonly the results of a java build process.

**Example 2: Excluding IDE-specific files and local configurations:**

```
.idea/
.vscode/
*.log
*.swp
*.env
.DS_Store
```

This `.gitignore` entry will ignore files specific to IntelliJ IDEA (`.idea`), Visual Studio Code (`.vscode`), log files, swap files, environment configuration files, and MacOS specific files.

**Example 3: Specific exclusions within a directory, while keeping some files:**

Let's assume we have a `data` directory and want to keep some configuration files but exclude any generated results files:

```
data/*.results
data/config/*.config
!data/config/application.config
```

In this example, all files with `.results` extension inside the `data` folder will be excluded. All files ending with `.config` within the `data/config` subdirectory will also be excluded, *except* for `data/config/application.config`. The `!` means 'not' so it makes it an exception to the rule.

Now, about environment-specific configurations, in a prior project, I recall using a combination of environment variables and configuration files that were version-controlled, but the actual values were injected during the build process and at runtime. This approach allows for different environments (development, testing, production) to be configured without compromising sensitive information or the repository's consistency.

For deeper understanding of best practices around version control, I would highly recommend looking into "Pro Git" by Scott Chacon and Ben Straub, it's a comprehensive guide to Git that covers these topics in great detail. For understanding build tools, specifically, the documentation for Maven and Gradle are invaluable resources. As for configuration management, there are many resources available, such as "Infrastructure as Code" by Kief Morris, which details best practices when working with systems configuration.

In conclusion, pushing to GitHub requires a level of critical thought. Be mindful about what you commit. Keep your code repository clean and manageable, avoiding bloat from generated files or accidentally including sensitive data. It's about pushing the essence, the source, not the consequences of that source. The `.gitignore` file is your friend; use it wisely and effectively. A clean, well-maintained repository benefits everyone on the team and makes your workflow significantly more efficient.
