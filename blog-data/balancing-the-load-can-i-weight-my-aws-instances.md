---
title: "Balancing the Load: Can I Weight My AWS Instances?"
date: '2024-11-08'
id: 'balancing-the-load-can-i-weight-my-aws-instances'
---

```
# Create an Application Load Balancer
aws elbv2 create-load-balancer --name my-app-lb --subnets subnet-1234567890abcdef0 subnet-1234567890abcdef1 --security-groups sg-1234567890abcdef0 --type application

# Create Target Groups
aws elbv2 create-target-group --name my-tg1 --protocol HTTP --port 80 --vpc-id vpc-1234567890abcdef0
aws elbv2 create-target-group --name my-tg2 --protocol HTTP --port 80 --vpc-id vpc-1234567890abcdef0

# Register Targets to Target Groups
aws elbv2 register-targets --target-group-arn arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/my-tg1/1234567890abcdef0 --targets Id=instance-1234567890abcdef0,Port=80
aws elbv2 register-targets --target-group-arn arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/my-tg2/1234567890abcdef0 --targets Id=instance-1234567890abcdef1,Port=80

# Create a Listener Rule with Weighted Target Groups
aws elbv2 create-listener-rule --listener-arn arn:aws:elasticloadbalancing:us-east-1:123456789012:listener/app/my-app-lb/1234567890abcdef0/1234567890abcdef0 --priority 1 --conditions PathPattern=/ --actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/my-tg1/1234567890abcdef0,Weight=50
aws elbv2 create-listener-rule --listener-arn arn:aws:elasticloadbalancing:us-east-1:123456789012:listener/app/my-app-lb/1234567890abcdef0/1234567890abcdef0 --priority 2 --conditions PathPattern=/ --actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/my-tg2/1234567890abcdef0,Weight=50

```

**Explanation:**

1. **Create an Application Load Balancer:** We create an Application Load Balancer with a name, subnets, security groups, and type.
2. **Create Target Groups:** We create two target groups, one for each backend instance.
3. **Register Targets:** We register our backend instances to their respective target groups.
4. **Create a Listener Rule with Weighted Target Groups:** We create a listener rule for our load balancer that directs traffic based on a weighted policy.  In this example, traffic is split evenly (50/50) between the two target groups.

**Important Notes:**

* Replace the placeholders (subnet IDs, security group IDs, instance IDs, etc.) with your actual values.
* This example uses a 50/50 weight for the target groups. You can adjust these weights to your desired distribution.
* You can create additional listener rules and target groups to support more complex load balancing scenarios.
* Be sure to update your DNS records to point to the newly created Application Load Balancer.

This concise code example provides a solution for achieving weighted load balancing using the AWS Application Load Balancer. 

