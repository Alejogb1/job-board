---
title: "Does Testcontainers/LocalStack support DynamoDB Streams KCL 1.x?"
date: "2025-01-30"
id: "does-testcontainerslocalstack-support-dynamodb-streams-kcl-1x"
---
My experience configuring containerized AWS services for integration testing has revealed a nuanced compatibility landscape between Testcontainers, LocalStack, and the Kinesis Client Library (KCL) 1.x when processing DynamoDB Streams. While a seemingly straightforward objective, several architectural and configuration specifics demand careful attention. The short answer is: yes, it *can* be made to work, but not without potential pitfalls.

The primary challenge resides in the way KCL 1.x discovers and communicates with stream shards. KCL 1.x relies heavily on explicit AWS SDK interactions, particularly with the DynamoDB service itself to discover the stream and its associated shards. LocalStack, on the other hand, provides an emulated AWS environment, which, while largely compatible, deviates subtly in areas like resource ARNs, endpoint management, and implicit IAM contexts. These deviations, while beneficial for local development, can cause KCL 1.x to falter if not correctly configured. Moreover, the lack of native support for the KCL within LocalStack means developers must leverage a separate process to host the consumer application, adding complexity to integration setups. Testcontainers effectively wraps this process, providing a convenient way to deploy and orchestrate containers; however, this doesn't implicitly guarantee KCL 1.x compatibility with LocalStack’s implementation of DynamoDB streams.

The fundamental problem stems from the fact that KCL 1.x expects the environment it operates in to match, as closely as possible, the real AWS infrastructure it intends to connect to. Specifically, it uses `DescribeStream` and other calls which return specific ARNs and resource endpoints that need to correlate with what LocalStack offers. Because LocalStack is an emulator, it may not return ARNs in the format KCL 1.x expects, or the endpoints for the streams themselves might not directly correspond to the client process's networking configuration.

Here are some challenges I've encountered, and the solutions I found:

First, the KCL 1.x initialization process involves constructing a worker object. This worker requires a suitable `DynamoDB` client instance configured to target LocalStack's endpoint and credentials.

```java
import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.client.builder.AwsClientBuilder;
import com.amazonaws.services.dynamodbv2.AmazonDynamoDB;
import com.amazonaws.services.dynamodbv2.AmazonDynamoDBClientBuilder;
import com.amazonaws.services.kinesis.clientlibrary.lib.worker.InitialPositionInStream;
import com.amazonaws.services.kinesis.clientlibrary.lib.worker.KinesisClientLibConfiguration;
import com.amazonaws.services.kinesis.clientlibrary.lib.worker.Worker;

public class KCLWorker {

    public static void main(String[] args) {

        // LocalStack configuration
        String localstackEndpoint = "http://localhost:4566";
        String region = "us-east-1";
        String accessKey = "test";
        String secretKey = "test";

        AmazonDynamoDB client = AmazonDynamoDBClientBuilder.standard()
                .withEndpointConfiguration(new AwsClientBuilder.EndpointConfiguration(localstackEndpoint, region))
                .withCredentials(new AWSStaticCredentialsProvider(new BasicAWSCredentials(accessKey, secretKey)))
                .build();


        String streamName = "my-stream"; // Your DynamoDB stream name
        String applicationName = "my-kcl-application"; // Your application name

        // KCL configuration
        KinesisClientLibConfiguration kclConfig = new KinesisClientLibConfiguration(
                applicationName,
                streamName,
                new AWSStaticCredentialsProvider(new BasicAWSCredentials(accessKey, secretKey)),
                "my-worker-id"
        )
                .withInitialPositionInStream(InitialPositionInStream.TRIM_HORIZON)
                .withDynamoDBClient(client) // Inject configured DynamoDB client
               .withRegionName(region);


        Worker worker = new Worker(new RecordProcessorFactory(), kclConfig);

        // Start the worker
        worker.run();

    }
}
```

In this example, it’s crucial to configure the `AmazonDynamoDB` client with the specific LocalStack endpoint. If you miss this, KCL 1.x will attempt to connect to the real AWS endpoint, causing connection errors. Furthermore, the `withRegionName` method is not merely cosmetic. KCL 1.x uses it to locate the correct endpoint, and omitting it can cause unexpected behaviour as it might incorrectly query an AWS endpoint, even with an injected dynamodb client, due to internal caching or defaulting logic within the KCL libraries. The access and secret keys provided are merely placeholders and should be treated as insecure for production environments.

Second, the KCL 1.x heavily relies on consistent DynamoDB state management to track stream progress and lease acquisition between worker instances.  LocalStack's implementation of DynamoDB might exhibit slight variances in timing and eventual consistency. It is imperative to utilize a robust `DynamoDB` table for lease coordination. The naming and configuration of this table are also critical for correct behaviour and should mirror the AWS best practices for KCL 1.x. Here's how this is set up programmatically when the table isn't created externally:

```java
import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.client.builder.AwsClientBuilder;
import com.amazonaws.services.dynamodbv2.AmazonDynamoDB;
import com.amazonaws.services.dynamodbv2.AmazonDynamoDBClientBuilder;
import com.amazonaws.services.dynamodbv2.model.AttributeDefinition;
import com.amazonaws.services.dynamodbv2.model.CreateTableRequest;
import com.amazonaws.services.dynamodbv2.model.KeySchemaElement;
import com.amazonaws.services.dynamodbv2.model.KeyType;
import com.amazonaws.services.dynamodbv2.model.ProvisionedThroughput;
import com.amazonaws.services.dynamodbv2.model.ScalarAttributeType;

public class DynamoDBTableCreator {

    public static void main(String[] args) {
        String localstackEndpoint = "http://localhost:4566";
        String region = "us-east-1";
        String accessKey = "test";
        String secretKey = "test";
        String leaseTableName = "my-kcl-lease-table";


        AmazonDynamoDB dynamoDB = AmazonDynamoDBClientBuilder.standard()
                .withEndpointConfiguration(new AwsClientBuilder.EndpointConfiguration(localstackEndpoint, region))
                .withCredentials(new AWSStaticCredentialsProvider(new BasicAWSCredentials(accessKey, secretKey)))
                .build();

        // Construct create table request
        CreateTableRequest request = new CreateTableRequest()
                .withTableName(leaseTableName)
                .withAttributeDefinitions(
                        new AttributeDefinition("leaseKey", ScalarAttributeType.S)) // Key attribute
                .withKeySchema(
                        new KeySchemaElement("leaseKey", KeyType.HASH)
                )
               .withProvisionedThroughput(new ProvisionedThroughput(1L, 1L)); // minimal for local testing
        try {
            dynamoDB.createTable(request);
            System.out.println("Created table: " + leaseTableName);
        }
        catch (Exception e) {
            System.err.println("Could not create table " + leaseTableName + ": " + e.getMessage());
        }

        // Wait for the table to become active
        // (Implementation of a basic waiting loop omitted for brevity)
    }
}
```

The key takeaway here is the table name consistency and using a unique table for your KCL application for correct lease management within the local environment. KCL 1.x, by default, will use the application name to generate the lease table name, therefore, this must be handled carefully to prevent collisions between independent KCL processes.

Lastly, when using Testcontainers, correctly exposing the LocalStack container's ports is crucial. The KCL application needs network access to both LocalStack’s DynamoDB endpoint and the DynamoDB streams endpoint. The following example illustrates a basic setup using Testcontainers with a Java application:

```java
import org.testcontainers.containers.localstack.LocalStackContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;
import org.junit.jupiter.api.Test;

import static org.testcontainers.containers.localstack.LocalStackContainer.Service.DYNAMODB;

@Testcontainers
public class LocalStackKCLTest {

    @Container
    private static final LocalStackContainer localStackContainer =
            new LocalStackContainer(DockerImageName.parse("localstack/localstack:latest"))
            .withServices(DYNAMODB)
             .withExposedPorts(4566);


    @Test
    void testKCLIntegration() {
       // Use the KCL configuration demonstrated above, but configure the client to use
       //   localStackContainer.getEndpointOverride(DYNAMODB).toString() as the endpoint.
       // Then launch the KCL worker here.
       // The test might do further integration, like creating a stream and then writing to
       // it while the KCL worker is running, which would require access to the dynamodb client.
    }

}
```

Here, `withExposedPorts(4566)` is critical. This ensures that the host machine can reach LocalStack on the standard endpoint, allowing the KCL worker to connect correctly. It is advisable to use the dynamically generated endpoint from the container instance through  `localStackContainer.getEndpointOverride(DYNAMODB)`, as demonstrated by the comment in the `testKCLIntegration` method. Hardcoding "localhost" or "127.0.0.1" might fail in various test environments like CI, where the container's address is not the host machine's. This dynamically obtained endpoint is also essential to provide a correctly formed endpoint for the AWS SDK to connect to localstack, which might not be on your local system.

In conclusion, while Testcontainers and LocalStack can be used to test KCL 1.x applications against DynamoDB Streams, it requires specific attention to the endpoint configuration, lease management, and network settings. Directing the client towards the LocalStack endpoint and ensuring the lease table is configured correctly are vital. Ignoring any of these will lead to test failures.

For further information, I would recommend consulting the official AWS documentation for KCL 1.x and DynamoDB Streams. The documentation for LocalStack itself also contains valuable insights, though specifically about KCL implementation they are limited. Finally, exploring the Testcontainers documentation for LocalStack integration is highly advised. All of this can provide deeper context.
