---
title: "sonarcloud long-lived branches pattern not reflecting changes?"
date: "2024-12-13"
id: "sonarcloud-long-lived-branches-pattern-not-reflecting-changes"
---

Okay so you're having issues with SonarCloud not picking up changes on your long-lived branches right I've been there man believe me This is a classic source control and analysis sync headache and it bites deep if you don't get it sorted early On a project way back when I was neck deep in Java microservices a good 5 years ago we had this same exact issue Long-lived branches were just not getting updated analysis even though we were pushing code like crazy It was infuriating so I get the pain Let's dissect this and I'll try to give you the real meat and potatoes of what I've seen work not some fluffy marketing nonsense

First off let's make sure we understand the core concept SonarCloud is looking at your source code and comparing it against what it last saw on that particular branch It uses the commit history to track changes and identify what's new If it misses changes it usually means the analysis is not getting triggered correctly or is not correctly picking up the latest commits I've personally found this can have multiple culprits

The most common mistake I see especially early on is incorrect branch naming patterns in SonarCloud Project settings Look I've wasted hours on this I'm not even kidding Go double check this thing and triple check it because it is the most likely culprit If you use `feature/*` for your branches and SonarCloud is configured to only recognize `feature-.*` or anything else you're out of luck This is all text based regular expressions under the hood and any difference will mess it up. It's case-sensitive so be extra diligent.

Second the analysis trigger mechanism If you're using a CI/CD pipeline make sure the scanner is correctly configured to analyze your long-lived branches and is passing the branch name properly to SonarCloud via the `-Dsonar.branch.name` parameter This parameter is critical I can't stress that enough. It tells SonarCloud what branch you're talking about If it's not there or if it's wrong SonarCloud will not update your long lived branch. It will create a new one or update the short-lived ones. So it's a classic problem of talking to the wrong resource.

Third lets consider the commit history itself Sometimes a really messed up rebase or a force push can confuse SonarCloud It relies on the Git history so if your Git history is a trainwreck your analysis might be a trainwreck as well Try running a forced analysis from the pipeline it is the equivalent of "turning it off and on again" to the scanner. Let's move on to see the specifics

**Example configuration of a CI/CD pipeline using a Sonar scanner**

Here's an example of how you might set up your CI/CD pipeline assuming you use something like GitHub Actions that uses bash for its commands or a similar system:

```bash
  - name: Set up SonarScanner
    uses: SonarSource/sonarcloud-github-action@v2
    env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
    with:
      args: >
        -Dsonar.organization=your_organization
        -Dsonar.projectKey=your_project_key
        -Dsonar.sources=.
        -Dsonar.host.url=https://sonarcloud.io
        -Dsonar.branch.name=${GITHUB_REF#refs/heads/}
```

This snippet uses the SonarSource action for GitHub It's setting up the scanner with your organization and project key And importantly the branch name is set dynamically from the GitHub ref using a bash string replacement thingy to get rid of the refs/heads/ part It’s the same in Gitlab or other CI providers you just need to get the current branch name and set the right argument.

Now this is important The `-Dsonar.sources=.` option tells it to scan everything from the current directory This might be OK in some cases but can be an issue if you want to limit the scope of your analysis or use specific module folders or you know need to speed things up a little I would try to be specific here later on once the issue is solved.

**Example for a direct scanner run**

Another example if you directly use the scanner or want to use it in a local environment you would do something like this. Assuming you have the sonar-scanner downloaded and configured correctly in your local path:

```bash
sonar-scanner \
  -Dsonar.projectKey=your_project_key \
  -Dsonar.sources=. \
  -Dsonar.host.url=https://sonarcloud.io \
  -Dsonar.login=your_sonar_token \
  -Dsonar.branch.name=$(git rev-parse --abbrev-ref HEAD)
```

Here we are doing something similar using the command-line scanner directly with a Git command to derive the branch. Make sure your sonar login token is set as an environment variable in the CI/CD context or set it directly as I did here but do not commit your secrets to a repo.

**Example for setting branch patterns in your sonar-project.properties**

You can also configure the branch patterns by directly adding them to a `sonar-project.properties` file in the root folder of your project This is another way to configure the scanner if you dont use parameters for every scan. You can set it to be included into the scan and the values in there will be used to find the patterns

```properties
sonar.projectKey=your_project_key
sonar.sources=.
sonar.host.url=https://sonarcloud.io
sonar.login=your_sonar_token
sonar.branch.pattern=^(main|master|release\/.*|develop|hotfix\/.*|feature\/.*)$
```
Here we are explicitly setting branch patterns that SonarCloud should understand so when a commit happens in such a branch it will take the patterns and use that for its logic. Its always a good thing to define it once and not in many places it can be a source of confusion.

If you are using a configuration file like that remember to commit it. You will be surprised at how many developers forget this simple fact and struggle to see the analysis working or having the same issue across development and CI/CD. I know it happened to me at least once or twice.

Now I have to be honest one time I was banging my head on the wall with this for hours and it turned out to be a typo in the project key It was so obvious once I found it that I almost threw my keyboard at the wall haha It is the classic "did you try turning it off and on again" kind of issue.

**Advanced Considerations and Troubleshooting**

If you're still running into issues even after checking all of that consider the following

*   **Caching Issues:** Sometimes the scanner or CI runner may have a cached version of the Sonar scanner and not the latest one or some kind of cached data that is interfering with the analysis. Try to always use the latest version of the scanner.

*   **Git Submodules:** If you use Git submodules ensure those are fetched correctly during the build process SonarCloud also needs to analyze them if that's where your code lives in a submodule. Also if you are building them as part of your CI/CD you should also account for those in the CI/CD setup.

*   **Large Monorepos:** Large monorepos especially those with specific module structures need care to be taken to not re-analyze the same code twice If you have a mono-repo consider using sonar includes and excludes to avoid this type of situation You can also use a dedicated scanner per module to speed things up. This is only for very large codebases or very high scanning times.

*   **SonarCloud Webhooks:** Check your SonarCloud webhook settings and make sure that the repository is properly setup. If the hooks are not set up correctly you are missing some crucial data and that may create issues down the road.

*   **SonarCloud Logs:** Check SonarCloud logs to find a reason or a cause for an error It's also useful to look for the scanner logs in your CI/CD logs if you have any issue with the scan itself.

**Recommended resources**

For a deep understanding of how Git works specifically the commit history and branch concepts you can check "Pro Git" by Scott Chacon and Ben Straub its free online. For information on SonarQube the documentation is actually good you can check the official website. Also, you should check the documentation of your CI/CD provider to see the best practices when using them in conjunction with the SonarCloud scanners for their best practices. They are not very hard to grasp but it might take a while to get the specifics for your particular case.

Let me know if any of that helps and tell me which part you have more trouble with I have seen some really strange problems in the past so I am willing to help if needed It’s a common problem and you are not alone in this type of issue.
