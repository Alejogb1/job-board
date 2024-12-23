---
title: "How can Airflow be deployed on ECS Fargate tasks?"
date: "2024-12-23"
id: "how-can-airflow-be-deployed-on-ecs-fargate-tasks"
---

Alright, let’s tackle this one. Deploying Apache Airflow on ECS Fargate is a task I’ve seen crop up quite a few times in my experience, and it's definitely got its nuances. The traditional approach using EC2 instances for Airflow tends to become cumbersome, especially when scaling. Fargate offers a great alternative: it’s serverless, it scales well, and it fits nicely into a microservices architecture. But it's not always straightforward. Let's get into the details.

The core challenge revolves around translating Airflow's multi-component architecture—the scheduler, webserver, worker(s), and database—into the Fargate paradigm, which is designed for stateless, single-container deployments. We need to decouple these components and ensure they can communicate effectively, even as Fargate scales them independently. In previous projects, I've found that failing to decouple components effectively leads to all sorts of problems, most notably bottlenecks, which render Fargate's scalability advantages pointless.

The first step is to break down Airflow into its distinct components and containerize them. I typically use Docker for this, and the official Airflow images are a great starting point. However, those images are often built for single-instance deployments, so we need to make some modifications. The primary challenge here is configuration management: how do you tell each container about the shared resources they need – especially the metadata database? I often prefer environment variables for this. This keeps the containers portable and configurable.

For the metadata database, it’s generally best to rely on a managed service like Amazon RDS (PostgreSQL is often the most popular choice for Airflow). This eliminates database management concerns and allows your Fargate tasks to focus on Airflow execution itself.

Now, let’s get into the specifics.

**Scheduler & Webserver**

The scheduler and webserver can be bundled into a single container, but be mindful of resource allocations. In previous projects where I bundled them, I observed increased resource utilization in the combined task, so consider splitting them in high-load situations. These components primarily need connectivity to the metadata database and access to the dag files. In our configuration, we must set the necessary database connection strings via environment variables and ideally mount a shared EFS volume or use S3 to store the DAG files, ensuring all scheduler and worker containers have access to them.

**Workers**

The workers are where the actual task executions occur. These will require similar database access and DAG file access. However, these are generally run as independent tasks, and multiple workers can run simultaneously to scale processing capacity.

Here are a few code snippets to illustrate:

**Snippet 1: Dockerfile for the Scheduler/Webserver Container**

```dockerfile
FROM apache/airflow:2.8.0

# Install any additional python packages if needed
# RUN pip install ...

# Configure Airflow for Database connection
ENV AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="postgresql+psycopg2://airflow:airflow@your_rds_endpoint:5432/airflow_db"
ENV AIRFLOW__CORE__EXECUTOR="CeleryExecutor"
ENV AIRFLOW__CELERY__BROKER_URL="redis://your_redis_endpoint:6379/0"  # For Celery
ENV AIRFLOW__CELERY__RESULT_BACKEND="redis://your_redis_endpoint:6379/0" # For Celery
ENV AIRFLOW__CORE__DAG_DIR="/opt/airflow/dags"

# Copying DAGS to the correct directory. If using EFS volume, you may not need this section
# COPY dags /opt/airflow/dags

USER airflow
```

**Snippet 2: Task Definition JSON for Scheduler/Webserver Fargate Task**

```json
{
  "family": "airflow-scheduler-webserver",
  "containerDefinitions": [
    {
      "name": "airflow-scheduler-webserver",
      "image": "your-account-id.dkr.ecr.your-region.amazonaws.com/your-airflow-image:latest",
      "cpu": 512,
      "memory": 1024,
      "portMappings": [
        {
          "containerPort": 8080,
          "hostPort": 8080
        }
      ],
      "environment": [
        {"name":"AIRFLOW__DATABASE__SQL_ALCHEMY_CONN", "value": "postgresql+psycopg2://airflow:airflow@your_rds_endpoint:5432/airflow_db"},
        {"name":"AIRFLOW__CORE__EXECUTOR", "value": "CeleryExecutor"},
        {"name": "AIRFLOW__CELERY__BROKER_URL", "value": "redis://your_redis_endpoint:6379/0"},
        {"name": "AIRFLOW__CELERY__RESULT_BACKEND", "value": "redis://your_redis_endpoint:6379/0"}
      ],
      "mountPoints": [
        {
          "sourceVolume": "efs-volume",
          "containerPath": "/opt/airflow/dags"
        }
      ],
      "logConfiguration": {
          "logDriver": "awslogs",
           "options": {
               "awslogs-group": "/ecs/airflow-logs",
               "awslogs-region": "your-region",
               "awslogs-stream-prefix": "airflow-scheduler-web"
             }
       }
    }
  ],
    "volumes": [
        {
           "name": "efs-volume",
           "efsVolumeConfiguration": {
              "fileSystemId": "fs-xxxxxxxxxxxxx",
              "rootDirectory": "/airflow_dags"
            }
        }
    ],
    "requiresCompatibilities": [
      "FARGATE"
    ],
    "networkMode": "awsvpc",
    "cpu": "512",
    "memory": "1024",
    "executionRoleArn": "arn:aws:iam::your-account-id:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::your-account-id:role/ecsTaskRole"
}
```

**Snippet 3: Task Definition JSON for Worker Fargate Task**
```json
{
  "family": "airflow-worker",
  "containerDefinitions": [
    {
      "name": "airflow-worker",
      "image": "your-account-id.dkr.ecr.your-region.amazonaws.com/your-airflow-image:latest",
      "cpu": 512,
      "memory": 1024,
      "environment": [
          {"name":"AIRFLOW__DATABASE__SQL_ALCHEMY_CONN", "value": "postgresql+psycopg2://airflow:airflow@your_rds_endpoint:5432/airflow_db"},
          {"name":"AIRFLOW__CORE__EXECUTOR", "value": "CeleryExecutor"},
           {"name": "AIRFLOW__CELERY__BROKER_URL", "value": "redis://your_redis_endpoint:6379/0"},
          {"name": "AIRFLOW__CELERY__RESULT_BACKEND", "value": "redis://your_redis_endpoint:6379/0"}
         ],
        "mountPoints": [
         {
            "sourceVolume": "efs-volume",
             "containerPath": "/opt/airflow/dags"
           }
         ],
      "logConfiguration": {
          "logDriver": "awslogs",
           "options": {
               "awslogs-group": "/ecs/airflow-logs",
               "awslogs-region": "your-region",
               "awslogs-stream-prefix": "airflow-worker"
             }
       }
    }
  ],
    "volumes": [
        {
           "name": "efs-volume",
           "efsVolumeConfiguration": {
              "fileSystemId": "fs-xxxxxxxxxxxxx",
              "rootDirectory": "/airflow_dags"
            }
        }
    ],
    "requiresCompatibilities": [
      "FARGATE"
    ],
    "networkMode": "awsvpc",
    "cpu": "512",
    "memory": "1024",
    "executionRoleArn": "arn:aws:iam::your-account-id:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::your-account-id:role/ecsTaskRole"
}
```

Note that these examples are skeletal and would need to be extended to include specific requirements, like secrets management, custom network configurations, and resource constraints. Also, for production environments, it's crucial to set up proper monitoring, alerting, and logging. Amazon CloudWatch is generally a good place to start for logging and basic metrics.

As for deeper dives into the technology, I'd recommend the official Apache Airflow documentation as the go-to place. It's exceptionally detailed and will provide you with most answers on configurations and customizations. Additionally, the AWS documentation on ECS, Fargate and EFS is essential for this specific deployment. Look at the official AWS documentation surrounding ECS task definitions for full details.

Finally, consider how you will handle secrets. Storing passwords in environment variables is usually not the most secure solution, so look into solutions like AWS Secrets Manager to handle sensitive information.

In conclusion, deploying Airflow on ECS Fargate requires a strategic separation of the core Airflow components into independent containerized tasks, proper management of shared resources, and a good understanding of container networking. By carefully configuring the task definitions, utilizing managed services for the database, and storing dag files effectively, one can reap the benefits of Fargate's scalability and cost-effectiveness.
