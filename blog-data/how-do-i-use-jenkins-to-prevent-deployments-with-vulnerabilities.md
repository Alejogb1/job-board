---
title: "How do I use Jenkins to prevent deployments with vulnerabilities?"
date: "2024-12-16"
id: "how-do-i-use-jenkins-to-prevent-deployments-with-vulnerabilities"
---

Let's tackle that thorny issue of vulnerability prevention in deployments using Jenkins. It’s a critical concern, and I’ve seen firsthand how a lapse in this area can quickly snowball into significant headaches. Back in my days working with a large e-commerce platform, we had a near miss that really underscored the importance of integrating security directly into our ci/cd pipelines, rather than treating it as an afterthought. We almost pushed a build to production that contained a known log4j vulnerability—that taught us a thing or two, believe me.

Essentially, what we're aiming for here is to shift security left—integrating security checks early and often in the software development lifecycle. Jenkins, being a highly versatile automation server, is a fantastic tool for this. It’s not just about building and deploying code; it’s about creating a secure and reliable software delivery mechanism. It’s not a silver bullet of course, but when properly configured, it will stop a good chunk of obvious issues before they hit the production environment.

Fundamentally, the approach involves embedding security scanning tools and policies within your Jenkins pipeline. This means your pipeline should not just build and test your code for functionality but also scan it for vulnerabilities. There are several categories of vulnerabilities we can cover with various tools: static application security testing (sast), dynamic application security testing (dast), and software composition analysis (sca), just to name the most common. Each addresses a different facet of the security posture of our software.

Let me give you some actionable approaches with code examples. Consider a straightforward pipeline setup using declarative pipeline syntax in Jenkins. For demonstration purposes, I'll focus on utilizing `sonar-scanner` for static code analysis, `npm audit` for sca of node.js applications, and a hypothetical custom script for more advanced checks.

**Example 1: Implementing SonarQube Scanning with `sonar-scanner`**

This is for a Java-based application, where you use Maven as your build tool, and SonarQube as your sast server.

```groovy
pipeline {
    agent any
    tools {
      maven 'maven-3.8.1'
      jdk 'jdk-17'
    }
    stages {
        stage('Checkout') {
            steps {
                git url: 'https://github.com/yourorg/your-java-project.git', branch: 'main'
            }
        }
        stage('Build') {
            steps {
                sh 'mvn clean install'
            }
        }
        stage('SonarQube Analysis') {
            steps {
                script {
                  def scannerHome = tool 'sonar-scanner'
                  sh "${scannerHome}/bin/sonar-scanner -Dsonar.projectKey=your-project-key -Dsonar.sources=. -Dsonar.host.url=http://your-sonarqube-server:9000"
                }
            }
        }
        stage('Quality Gate') {
            steps {
              script {
                def qgStatus = sh(script: "curl -s http://your-sonarqube-server:9000/api/qualitygates/project_status?projectKey=your-project-key", returnStdout: true)
                def status = readJSON text: qgStatus.trim()
                if (status.projectStatus.status != 'OK'){
                  error "SonarQube Quality Gate Failed: ${status.projectStatus.status}"
                } else {
                  echo "SonarQube Quality Gate passed."
                }
              }
            }
        }
        stage('Deploy'){
          //Only runs if quality gate passed
          when {
            expression { return true } //simplified; implement proper conditions later.
          }
            steps {
              echo "Deployment steps here." // Replace with actual deploy steps.
            }
        }
    }
}
```

This script does the following: checks out the code, builds it with maven, initiates a scan with the `sonar-scanner`, and then checks the quality gate status on the SonarQube server. The build fails if the quality gate indicates that there are vulnerabilities that do not match a configured baseline. The `tool` directive points to a pre-configured installation of Sonar Scanner within Jenkins, which means we don’t need to manually download and configure the scanner in our pipeline. You’ll need to install the SonarQube Scanner Plugin, as well as configure the scanner tool location in the Jenkins global configuration.

**Example 2: Using `npm audit` for Node.js Applications**

For Node.js projects, `npm audit` is a quick and effective way to uncover vulnerabilities in dependencies.

```groovy
pipeline {
  agent any
  tools {
      nodejs 'nodejs-18.16.0'
  }
  stages {
      stage('Checkout') {
        steps {
            git url: 'https://github.com/yourorg/your-node-project.git', branch: 'main'
        }
      }
      stage('Install Dependencies') {
          steps {
              sh 'npm install'
          }
      }
      stage('NPM Audit') {
          steps {
              script {
                  def auditOutput = sh(script: 'npm audit --json', returnStdout: true)
                  def audit = readJSON text: auditOutput.trim()
                  if (audit.metadata.vulnerabilities.total > 0) {
                      echo "npm audit found vulnerabilities:"
                      audit.vulnerabilities.each { dep, vulnerability ->
                          if(vulnerability){
                              echo " - ${dep} : ${vulnerability.severity}"
                          }
                      }
                    error 'NPM audit failed due to vulnerabilities.'
                  } else {
                    echo 'npm audit passed with no vulnerabilities.'
                  }
              }
          }
      }
      stage('Deploy') {
        when {
          expression { return true } //simplified; implement proper conditions later.
        }
          steps {
              echo "Deployment steps here."
          }
      }
  }
}

```

Here, we checkout the code, install dependencies using npm, run `npm audit`, and if any vulnerabilities are detected, the pipeline fails, stopping the deployment process. It extracts the vulnerabilities and reports them to the console for further inspection. Note how the pipeline gracefully handles the case where `npm audit` does not find any vulnerabilities. The use of `readJSON` greatly simplifies parsing the output of the audit tool.

**Example 3: Custom Vulnerability Check with a Bash Script**

For situations where specialized tools aren’t available or where specific company policies dictate custom checks, we can incorporate bash scripts directly into our pipeline.

```groovy
pipeline {
  agent any
  stages {
      stage('Checkout') {
          steps {
                git url: 'https://github.com/yourorg/your-project-with-custom-checks.git', branch: 'main'
            }
      }
      stage('Custom Vulnerability Check') {
          steps {
              script {
                  def customCheckResult = sh(script: '''
                      #!/bin/bash
                      # Example: Check for a specific hardcoded password in config files.
                      if grep -q 'password=supersecret' config.properties; then
                        echo "Found hardcoded password, failing build!"
                        exit 1
                      else
                        echo "No hardcoded passwords found."
                      fi
                      exit 0
                  ''', returnStdout: true)
                  if (customCheckResult.contains("failing build!")) {
                    error 'Custom vulnerability check failed.'
                  } else {
                      echo customCheckResult
                  }

              }
          }
      }
      stage('Deploy') {
          when {
              expression { return true } //simplified; implement proper conditions later.
          }
          steps {
              echo "Deployment steps here."
          }
      }
  }
}
```

This demonstrates a very basic custom check. In real-world scenarios, this script might involve complex logic to verify configurations or check for other security flaws that aren’t easily captured by existing scanners. The exit code of the script is crucial in determining if the `sh` step will fail the build.

For deeper understanding of security scanning techniques, I’d strongly recommend exploring "Static Program Analysis" by Anders Møller and Michael I. Schwartzbach for sast details. For sca, resources like “The Phoenix Project: A Novel About IT, DevOps, and Helping Your Business Win" by Gene Kim, Kevin Behr, and George Spafford, while not strictly technical, provides insights into the importance of integrating security practices into the overall software delivery chain. For dynamic analysis, "The Web Application Hacker's Handbook: Finding and Exploiting Security Flaws" by Dafydd Stuttard and Marcus Pinto offers excellent perspectives.

In practice, this involves continuous iteration. You'll need to constantly refine your checks and update your tools to address new vulnerabilities. The key is to treat security not as a one-time event but as an ongoing process, integrated seamlessly within your Jenkins pipelines. Security is not something that can be achieved through any single step, and vigilance is key to maintaining a strong security posture. Remember, the most secure systems are those that are designed with security in mind from the very start, not as an afterthought.
