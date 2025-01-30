---
title: "Why are there no resources in the SAM template?"
date: "2025-01-30"
id: "why-are-there-no-resources-in-the-sam"
---
The absence of resource definitions directly within a typical AWS SAM template, specifically the `template.yaml` file, stems from a fundamental design principle: the separation of infrastructure concerns from application concerns. SAM (Serverless Application Model) is primarily focused on defining the *application* elements – functions, APIs, layers – while abstracting away the underlying infrastructure. My experience deploying several serverless applications has consistently reinforced this design decision, where infrastructure is best managed separately for reasons of reusability, consistency, and flexibility.

The `template.yaml` file, therefore, is primarily an *application deployment* manifest, detailing the resources specific to the business logic being deployed. These are generally defined by AWS Serverless specification extensions of CloudFormation syntax. Instead of declaring VPCs, subnets, security groups, and other low-level resources directly in `template.yaml`, SAM leverages the inherent capabilities of CloudFormation to handle these infrastructural components. CloudFormation stacks can be managed as independent units of deployment. In my work, I've utilized nested stacks and CloudFormation Modules to define infrastructure building blocks that can then be imported into application stacks.

This separation leads to multiple benefits. First, infrastructure becomes reusable. Defining a network stack once allows it to be consumed by multiple application stacks. This reduces redundancy and promotes consistency across deployments. I've personally witnessed how this prevents inconsistencies that can occur if infrastructure is defined ad-hoc with each application deployment. Second, infrastructural modifications can be handled independently of application changes. Changes to, say, routing or security policies can be enacted on the infrastructure stack without forcing a redeployment of the application code. Lastly, it affords greater flexibility; developers can concentrate on their applications while infrastructure can be tailored to organizational needs by specialized teams using appropriate patterns, for example, using Infrastructure as Code best practices.

The central concept here is that SAM leverages CloudFormation, with the `template.yaml` being a condensed CloudFormation template that includes serverless-specific transformations. When deploying a SAM template, SAM first transforms its contents into a standard CloudFormation template and then submits this CloudFormation template for execution. The transformed CloudFormation template might use existing infrastructural components by referencing their outputs.

Consider the following SAM template snippet that defines a simple Lambda function:

```yaml
  MyFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: src/my_function
      Handler: app.handler
      Runtime: python3.9
      Policies:
        - AWSLambdaBasicExecutionRole
      Environment:
        Variables:
          MY_VARIABLE: "my_value"
```

In this example, you will notice no infrastructure such as subnets, VPCs, or security groups are defined. Instead, the function is declared using the `AWS::Serverless::Function` resource type. When the SAM template is processed, this is transformed by SAM into lower-level CloudFormation components, such as an IAM role (derived from the `Policies` property) and the actual Lambda function itself. Critically, the underlying network configuration and security context will be obtained through the default VPC or through parameters defined in an external infrastructure stack, not directly in this template.

Now, let's imagine a scenario where I needed to deploy a database along with the Lambda function. The SAM template would still focus on the function, but I would configure access using environment variables or other mechanisms. The database creation, VPC configuration, and all infrastructure-level details would reside in an independent CloudFormation template. Here's how the function template might look:

```yaml
  MyFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: src/my_function
      Handler: app.handler
      Runtime: python3.9
      Policies:
        - AWSLambdaBasicExecutionRole
        - arn:aws:iam::aws:policy/AmazonRDSDataFullAccess
      Environment:
        Variables:
          DATABASE_HOST: !Sub '${DatabaseStack.Outputs.DatabaseEndpoint}'
          DATABASE_USER: !Ref DatabaseUser
          DATABASE_PASSWORD: !Ref DatabasePassword
```

Here, you can see that the environment variables `DATABASE_HOST`, `DATABASE_USER` and `DATABASE_PASSWORD` are set by retrieving the necessary information from another CloudFormation stack, represented here as `DatabaseStack`. The actual definition of this `DatabaseStack` would occur elsewhere. Specifically, I use `!Sub` and `!Ref` to obtain parameters from different stacks or parameter inputs in the stack. I would not define the actual database resource in this SAM template. This demonstrates that even when integrating with other resources, the responsibility of creating those resources belongs to separate stacks.

Lastly, consider deploying the function in a specific VPC instead of the default one. Again, the VPC resource definition will not be defined in the application SAM template. Instead, I would specify the VPC and subnet using a `VpcConfig` property.

```yaml
  MyFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: src/my_function
      Handler: app.handler
      Runtime: python3.9
      Policies:
        - AWSLambdaBasicExecutionRole
      VpcConfig:
        SubnetIds:
          - !ImportValue 'InfraStack-PrivateSubnet1'
          - !ImportValue 'InfraStack-PrivateSubnet2'
        SecurityGroupIds:
          - !ImportValue 'InfraStack-LambdaSecurityGroup'
```
Here, the Lambda function will be deployed to the subnets defined in another CloudFormation stack 'InfraStack'. `!ImportValue` retrieves the output value from the other stack. The actual definitions of the subnets and security groups remain in that infrastructure stack. This further clarifies that the separation extends to network configurations, with the SAM template referencing infrastructural elements defined elsewhere.

For learning more about infrastructure as code with CloudFormation and SAM, I recommend focusing on resources that delve into CloudFormation best practices, particularly nested stacks and CloudFormation modules. The official AWS documentation provides extensive guides on these topics. Furthermore, exploring the concept of AWS Control Tower can greatly expand one's understanding of managing multiple infrastructure stacks across an organization. Learning about resource tagging strategies can also enhance your experience in managing these resources. Lastly, exploring use cases and deployment strategies using CI/CD pipelines will further enhance your experience.

In conclusion, the exclusion of direct resource definitions within SAM templates is a design choice that prioritizes modularity, consistency, and flexibility. This allows for the reuse of infrastructure components across applications and promotes a separation of concerns that simplifies the maintenance and evolution of both application and infrastructure. The SAM template's role is to focus solely on the serverless application's resources, deferring the infrastructure setup to CloudFormation and well-architected patterns.
