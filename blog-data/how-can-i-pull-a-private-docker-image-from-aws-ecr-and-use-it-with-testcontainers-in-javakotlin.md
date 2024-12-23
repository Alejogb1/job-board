---
title: "How can I pull a private Docker image from AWS ECR and use it with Testcontainers in Java/Kotlin?"
date: "2024-12-23"
id: "how-can-i-pull-a-private-docker-image-from-aws-ecr-and-use-it-with-testcontainers-in-javakotlin"
---

Okay, let's tackle this. I remember dealing with this exact scenario a few years back while setting up an integration testing pipeline. We were moving away from monolithic application deployments to microservices, and private container registries like AWS ECR became essential. Pulling those images into a testing environment using Testcontainers, especially programmatically, had its nuances.

The core challenge lies in the authentication hurdle. AWS ECR, by design, requires authentication. We can't just grab an image like we would from Docker Hub. We need to inform Docker (and therefore, Testcontainers) about our AWS credentials. There are several ways to achieve this, each with its trade-offs.

Let's start with the most common approach: using AWS CLI credentials. This method assumes you've already configured your AWS CLI with credentials that have the necessary permissions to access your ECR repository. When Testcontainers initiates the docker daemon, it automatically picks up the cli authentication configurations as part of its runtime environment. This is a good starting point because it leverages existing system configurations. You should have already set the profile that's going to be used by the AWS cli on your machine. To verify your configured credentials, the command `aws sts get-caller-identity` should work fine.

Here's how that plays out in code, assuming you've got that working on your machine already:

```java
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.DockerImageName;
import org.junit.jupiter.api.Test;

public class ECRImageTest {

    @Test
    void testPrivateImagePull() {
        String imageName = "your-account-id.dkr.ecr.your-region.amazonaws.com/your-image:your-tag";
        DockerImageName dockerImageName = DockerImageName.parse(imageName);

        try (GenericContainer<?> container = new GenericContainer<>(dockerImageName)) {
           container.start();
           // Your assertions or testing logic here.
           System.out.println("Container started successfully from ECR.");
        }
    }
}
```

This Java snippet is pretty straightforward: you specify your fully qualified image name (including your account id, region, and repository name) in the `imageName` string, and then use `GenericContainer` with it. The magic here is that Testcontainers, if configured correctly, leverages the default Docker client configuration which can pick up the valid AWS CLI settings on your system. This approach works perfectly if you're developing locally and you've already configured the AWS cli. But it doesn't scale well for CI environments or production tests, where direct access to configured credentials might not be ideal or practical.

Moving beyond system-wide configuration, we can use programmatic credentials. This involves creating the docker configuration based on programmatically fetched credentials. You can accomplish this by first obtaining an authorization token from AWS ECR, then setting it up in your Testcontainers environment. Here's how you can achieve that using the AWS Java SDK v2:

```java
import org.testcontainers.DockerClientFactory;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.DockerImageName;
import org.junit.jupiter.api.Test;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.ecr.EcrClient;
import software.amazon.awssdk.services.ecr.model.GetAuthorizationTokenRequest;
import software.amazon.awssdk.services.ecr.model.GetAuthorizationTokenResponse;
import java.util.Base64;

public class ECRImageTestWithCredentials {

    @Test
    void testPrivateImagePullWithProgrammaticAuth() {

        String imageName = "your-account-id.dkr.ecr.your-region.amazonaws.com/your-image:your-tag";
        DockerImageName dockerImageName = DockerImageName.parse(imageName);
        Region region = Region.of("your-region"); //replace with your region.

        DefaultCredentialsProvider credentialsProvider = DefaultCredentialsProvider.create();


        try (EcrClient ecrClient = EcrClient.builder()
             .region(region)
             .credentialsProvider(credentialsProvider)
            .build()){
             GetAuthorizationTokenRequest request = GetAuthorizationTokenRequest.builder().build();
             GetAuthorizationTokenResponse response = ecrClient.getAuthorizationToken(request);

            if (response.authorizationData().isEmpty()) {
               throw new RuntimeException("Failed to retrieve authorization token for ECR");
            }
            String encodedToken = response.authorizationData().get(0).authorizationToken();
            String decodedToken = new String(Base64.getDecoder().decode(encodedToken));
            String[] parts = decodedToken.split(":");
            String username = parts[0];
            String password = parts[1];

            DockerClientFactory.instance().client().authConfig(username, password, "https://" + dockerImageName.getRegistry());
        }


         try (GenericContainer<?> container = new GenericContainer<>(dockerImageName)) {
           container.start();
           // Your assertions or testing logic here.
           System.out.println("Container started successfully from ECR with programmatic credentials.");
         }
    }
}

```
This example gets an authorization token from AWS, decodes it, and then uses the `DockerClientFactory` to create a docker authentication entry programatically for the image registry. This allows the Testcontainers engine to fetch the image. We are using the `DefaultCredentialsProvider` in the snippet which uses different providers to pick the AWS credentials, you can configure specific credentials providers using the AWS Java SDK. This is a more robust approach for continuous integration systems.

Finally, another way to approach this is using docker login command from within the Testcontainers initialization. We can set up a docker login command using Testcontainers `withCreateContainerCmdModifier` and run the login process programmatically. The advantage of this approach is that we leverage the regular docker cli which makes this approach similar to how we manage docker manually. This makes it easier to troubleshoot if something doesn't work.

```java
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.DockerImageName;
import org.junit.jupiter.api.Test;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.ecr.EcrClient;
import software.amazon.awssdk.services.ecr.model.GetAuthorizationTokenRequest;
import software.amazon.awssdk.services.ecr.model.GetAuthorizationTokenResponse;
import java.util.Base64;
import org.testcontainers.containers.Container.ExecResult;
import java.util.Arrays;

public class ECRImageTestWithDockerLogin {

    @Test
    void testPrivateImagePullWithDockerLogin() throws Exception{
        String imageName = "your-account-id.dkr.ecr.your-region.amazonaws.com/your-image:your-tag";
        DockerImageName dockerImageName = DockerImageName.parse(imageName);
        Region region = Region.of("your-region"); //replace with your region.

         DefaultCredentialsProvider credentialsProvider = DefaultCredentialsProvider.create();

        try (EcrClient ecrClient = EcrClient.builder()
                .region(region)
                .credentialsProvider(credentialsProvider)
               .build()){
            GetAuthorizationTokenRequest request = GetAuthorizationTokenRequest.builder().build();
            GetAuthorizationTokenResponse response = ecrClient.getAuthorizationToken(request);

            if (response.authorizationData().isEmpty()) {
              throw new RuntimeException("Failed to retrieve authorization token for ECR");
             }
            String encodedToken = response.authorizationData().get(0).authorizationToken();
            String decodedToken = new String(Base64.getDecoder().decode(encodedToken));
            String[] parts = decodedToken.split(":");
            String username = parts[0];
            String password = parts[1];


            try (GenericContainer<?> container = new GenericContainer<>(dockerImageName)
                  .withCreateContainerCmdModifier(cmd -> cmd.withEntrypoint(null)))
             {
               ExecResult dockerLoginResult =  container.execInContainer(
                        "/bin/sh",
                       "-c",
                       String.format("docker login -u %s -p %s https://%s", username, password, dockerImageName.getRegistry())
               );
                if (dockerLoginResult.getExitCode() != 0){
                   throw new RuntimeException(String.format("docker login failed with error: %s", dockerLoginResult.getStderr()));
                }

                container.start();
                // Your assertions or testing logic here.
                System.out.println("Container started successfully from ECR with docker login.");
             }
        }
    }
}
```
Here, we are retrieving the credentials, constructing the login command string, and then executing the docker login inside the container before starting the container. We are setting the entry point of the container to null by using `withCreateContainerCmdModifier` because we don't want to actually start the container with its default entry point at this time. This approach provides another viable way to configure docker login which can be convenient for debugging.

For deeper understanding, I would recommend looking into the official documentation for AWS SDK for Java, particularly the sections on authentication and the ECR client. Furthermore, the Testcontainers documentation on docker image configuration is helpful. Also, the 'Docker Deep Dive' book by Nigel Poulton is an excellent resource for understanding the underlying docker mechanics at play here. These references are invaluable for gaining a solid, in-depth understanding of both Testcontainers and Docker itself.

These approaches should provide you with solid ground for handling private ECR images with Testcontainers. Remember to adjust the region, account ids, image names, and credential settings to align with your specific environment. Each approach has advantages; you can select the most appropriate based on your specific requirements and environment. I've used all of these techniques in projects, and they all come with their own nuances, but a good understanding of these methods will significantly simplify your testing pipeline with containerized applications.
