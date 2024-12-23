---
title: "How can Jenkins pipelines deploy artifacts to custom JFrog repository folders named after the build number?"
date: "2024-12-23"
id: "how-can-jenkins-pipelines-deploy-artifacts-to-custom-jfrog-repository-folders-named-after-the-build-number"
---

Okay, let's tackle this. I've encountered similar requirements numerous times throughout my career, often when managing complex microservice deployments and needing meticulous artifact versioning. Setting up Jenkins to deploy artifacts to JFrog Artifactory, using a custom folder structure derived from the build number, requires a bit of pipeline scripting and a solid understanding of Artifactory's API. Let me walk you through the process.

Firstly, the fundamental idea is to dynamically generate the target repository path within the pipeline itself. We'll extract the Jenkins build number, then construct the Artifactory target folder. For this, we'll primarily leverage Groovy's string manipulation capabilities within a Jenkins declarative pipeline. Crucially, ensure your Jenkins server has the JFrog Artifactory plugin installed, as that facilitates the necessary interactions with Artifactory.

Let's start with a basic pipeline structure. We'll use a declarative pipeline, which makes things fairly straightforward for this type of task:

```groovy
pipeline {
    agent any
    stages {
        stage('Build and Package') {
            steps {
                script {
                   // Imagine your build steps are here
                   echo "Building the application..."
                   def buildArtifact = "my-application.jar" // Replace with your actual artifact name
                   sh "touch ${buildArtifact}"
                }
            }
        }
        stage('Deploy to Artifactory') {
            steps {
                script {
                    def buildNumber = env.BUILD_NUMBER
                    def repoKey = "my-local-repository" // Replace with your actual repository key
                    def targetPath = "my-artifacts/${buildNumber}/${buildArtifact}"

                    def server = Artifactory.server('my-artifactory-server') // Configure your server in Jenkins
                    server.upload(buildArtifact).to(repoKey, targetPath)

                    echo "Artifact deployed to ${repoKey}/${targetPath}"
                }
            }
        }
    }
}
```

Here's a breakdown of this basic example:

*   **`agent any`**: This directive tells Jenkins to use any available agent.
*   **`stage('Build and Package')`**: This stage simulates the build process where your application is packaged. The `touch` command is just a placeholder to create a dummy file for demonstration.
*   **`stage('Deploy to Artifactory')`**: This stage handles the deployment to Artifactory.
*   **`def buildNumber = env.BUILD_NUMBER`**: This retrieves the current Jenkins build number as an environment variable.
*   **`def repoKey = "my-local-repository"`**:  This variable holds the key to your desired Artifactory repository.
*   **`def targetPath = "my-artifacts/${buildNumber}/${buildArtifact}"`**: This string interpolation constructs the target path in Artifactory.
*   **`Artifactory.server('my-artifactory-server').upload(...).to(...)`**: This is the crucial part for deploying. The `my-artifactory-server` identifier corresponds to a configured Artifactory server in Jenkins. The `upload()` function uploads the artifact, and `to()` designates the repository key and the generated target path.

This snippet gets the core idea across, but we could enhance it with error handling and perhaps incorporate metadata to enrich the artifacts. Let's explore how to do that.

```groovy
pipeline {
    agent any
    stages {
        stage('Build and Package') {
            steps {
                 script {
                    // Your actual build logic.
                    echo "Building the application..."
                    def buildArtifact = "my-application.jar"
                    sh "touch ${buildArtifact}"
                 }
            }
        }
         stage('Deploy to Artifactory') {
            steps {
                script {
                    def buildNumber = env.BUILD_NUMBER
                    def repoKey = "my-local-repository"
                    def artifactName = "my-application.jar"
                    def targetPath = "my-artifacts/${buildNumber}/${artifactName}"
                     def server = Artifactory.server('my-artifactory-server')
                    try {
                        server.upload(artifactName).to(repoKey, targetPath)
                        server.addProperties(repoKey, targetPath, [buildNumber: buildNumber])
                        echo "Artifact deployed and tagged with build number ${buildNumber} to ${repoKey}/${targetPath}"

                    } catch (Exception e) {
                         echo "Deployment failed: ${e.getMessage()}"
                        error("Artifactory deployment failed") // Fail the pipeline
                    }
                }
            }
        }
    }
}
```

Here, we’ve added a `try-catch` block around the upload operation to capture any exceptions, such as network errors or insufficient permissions. We’ve also introduced `server.addProperties()` to attach build number metadata directly to the artifact in Artifactory, making it easily searchable. The use of `error("Artifactory deployment failed")` ensures that the pipeline fails if the upload fails, a common practice.

Finally, let's consider more sophisticated scenarios. Suppose you have a multi-module project and want to upload several artifacts.

```groovy
pipeline {
  agent any
  stages {
    stage('Build and Package') {
        steps {
            script {
                 // Your actual build logic to build multiple modules
                echo "Building multiple modules..."
                sh "touch module-a.jar"
                sh "touch module-b.war"
                def buildArtifacts = ["module-a.jar", "module-b.war"]
                env.ARTIFACTS = buildArtifacts.join(',')
            }
        }
    }
    stage('Deploy to Artifactory') {
      steps {
        script {
           def buildNumber = env.BUILD_NUMBER
           def repoKey = "my-local-repository"
           def server = Artifactory.server('my-artifactory-server')
           env.ARTIFACTS.split(',').each { artifactName ->
            def targetPath = "my-artifacts/${buildNumber}/${artifactName}"
            try {
                server.upload(artifactName).to(repoKey, targetPath)
                 server.addProperties(repoKey, targetPath, [buildNumber: buildNumber])
                echo "Artifact ${artifactName} deployed and tagged with build number ${buildNumber} to ${repoKey}/${targetPath}"
             } catch (Exception e) {
                echo "Deployment of ${artifactName} failed: ${e.getMessage()}"
                error("Artifactory deployment failed for ${artifactName}")
              }
           }
        }
      }
    }
  }
}

```

This final example uses a `for` loop to iterate through a list of artifacts stored as a comma-separated string. It demonstrates how to scale up the deployment process to handle multiple artifacts, adding robustness and more closely aligning with a real-world scenario. Importantly, each artifact deployment attempt is enclosed within its own `try-catch` block.

**Resource Recommendation**

For further understanding of these concepts, I would highly recommend:

*   **The "Jenkins: The Definitive Guide" by John Ferguson Smart:** This is a comprehensive resource for Jenkins and its pipeline capabilities.
*   **The official JFrog Artifactory documentation:** This resource contains detailed information on Artifactory's REST API and plugin usage.
*   **"Groovy in Action" by Dierk König et al.:** As Jenkins pipelines utilize Groovy, familiarity with its syntax is crucial.

By employing the above techniques, you'll have a robust, auditable, and maintainable deployment process that meets your requirement of organizing artifacts by build number within Artifactory. Remember to adapt these examples to match your specific project needs and naming conventions.
