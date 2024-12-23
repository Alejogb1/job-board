---
title: "How to upgrade Airflow using a Helm chart?"
date: "2024-12-23"
id: "how-to-upgrade-airflow-using-a-helm-chart"
---

Alright, let's delve into this. Upgrading Airflow via Helm, while often appearing straightforward on the surface, can definitely present its share of subtle challenges, and I've certainly navigated a few of those over the years. It's not simply a matter of bumping the chart version; there are several aspects to consider to ensure a smooth and reliable transition. I recall one particularly memorable incident at a previous company where a misconfigured configuration setting during an upgrade caused a cascading failure with the scheduler, and it definitely provided a valuable lesson on preparation. Let’s explore how I’ve typically approached these upgrades, outlining my process and including a few practical examples.

The core principle, for me at least, is that upgrades should always be approached incrementally and with a solid understanding of the changes being introduced. A "big bang" approach, where you leap several versions forward at once, is frankly, asking for trouble. So, let’s say we're using the official Apache Airflow Helm chart, which is usually the go-to. Before we modify anything, the critical first step involves assessing the release notes of both the chart version and the Airflow version we are targeting. This is absolutely non-negotiable. Ignoring the detailed changelogs can lead to unexpected compatibility issues, particularly with database migrations, which are often present in upgrades.

Specifically, what we must be aware of are potential schema changes in the metadata database (usually Postgres or MySQL) and configuration changes in the airflow.cfg file or within the values.yaml that the chart utilizes. I've had instances where a previously working configuration flag became deprecated, leading to scheduler instability after the upgrade. Carefully documenting and testing each step is essential. For resource purposes, the documentation on the official Apache Airflow website, as well as the changelogs on the chart repository, are fundamental reads. Don't rely solely on third-party blogs; the source is always the most accurate. Also, the *Database Internals* book by Alex Petrov can be invaluable for those wanting a detailed understanding of database migrations, which often go hand-in-hand with upgrades.

, let's solidify this with a practical approach. Typically, we'll follow a three-stage plan:

1.  **Staging Environment Upgrade:** This is where all the initial testing takes place. We deploy the new chart version, targeting the upgraded Airflow version to a non-production Kubernetes namespace. We’ll use a distinct copy of our production database, with realistic data, so we can thoroughly validate functionality.
2.  **Production Canary Deployment:** Once satisfied with our staging results, we start by upgrading a limited subset of our production resources. Think of it as a controlled blast radius—we aim to validate the changes in a live environment without affecting the entire service.
3.  **Full Production Upgrade:** Upon successful canary deployment, we roll out the upgrade to the rest of our production environment.

Here's a first code snippet that illustrates how we might initiate an upgrade in the staging environment:

```bash
# Assuming you have helm configured to point to your kubernetes cluster
# Let's say current version is 1.12.0 and we are moving to 1.13.0
helm upgrade airflow-staging \
   --namespace airflow-staging \
   --reuse-values \
   --set airflow.image.tag=2.7.2 \
   --version 1.13.0 \
   apache-airflow/airflow
```

In this example, `airflow-staging` is the name of our release, we're deploying into the `airflow-staging` namespace, reusing existing values, updating the airflow image tag to `2.7.2` (assuming that is what’s required for chart 1.13.0) and upgrading the helm chart to version `1.13.0`. `--reuse-values` is particularly crucial to retain any specific customization you have implemented earlier and prevents us from having to redefine everything. The `apache-airflow/airflow` is the helm chart name. The image tag you will need might be different depending on the chart, the important aspect is to check compatibility.

After the staging deployment is successful, and we've validated the application’s behavior, we move to the canary phase, targeting only a small portion of our production environment initially. This often involves using label selectors, so we can control exactly which pods we upgrade. It's important to note that in addition to changing the chart version and the `airflow.image.tag`, you also might need to update other parameters defined in your values.yaml file.

Let's examine how we can perform a canary deployment using label selectors. Assume we have some deployments or StatefulSets tagged with `role=airflow-worker` or similar. We would need to create a similar deployment or StatefulSet, but with different labels, for instance `role=airflow-canary`. Then, we configure the canary version to point to this new deployment using helm:

```bash
helm upgrade airflow-prod-canary \
   --namespace airflow-prod \
   --set airflow.workers.podLabels.role=airflow-canary \
   --set airflow.image.tag=2.7.2 \
   --version 1.13.0 \
   --values values-canary.yaml  \
   apache-airflow/airflow
```

In this instance, we're setting `airflow.workers.podLabels.role` to `airflow-canary`. We are also passing a custom `values-canary.yaml` file which will contain only customizations for the canary instance, this would usually include the new image tag and other relevant changes. After a successful canary run, we'd proceed to the full production deployment. For resource purposes, the book *Kubernetes in Action* by Marko Luksa provides an excellent understanding of Kubernetes deployments, including label selectors, which can aid in this process.

Finally, here's an example of performing the full production upgrade, where we update all resources:

```bash
helm upgrade airflow-prod \
  --namespace airflow-prod \
  --set airflow.image.tag=2.7.2 \
  --version 1.13.0 \
  --reuse-values \
  apache-airflow/airflow
```

Here, `airflow-prod` is our production release, we are in `airflow-prod` namespace and, again, we reuse the values to retain our existing configurations, update the airflow image tag and the helm chart version.

This process, though detailed, is essential. It is not simply a single command; it is a deliberate, multi-staged approach, and the commands above should be modified to reflect the specific needs of your environment. After each upgrade step, it is extremely important to thoroughly check the logs and monitors, not just within Airflow, but also at the Kubernetes layer, looking for anomalies or inconsistencies that could point to a configuration problem.

Crucially, remember that each upgrade is unique. The specific changes necessary, the flags to tweak, and the troubleshooting steps will vary depending on your initial setup, the complexity of your workflows, and the target version. Therefore, a solid pre-upgrade assessment, careful testing, and a controlled rollout are always the safest options. The time invested in these preliminary stages will undoubtedly save you a great deal of time, effort, and frustration down the line. This iterative approach to upgrades, while seemingly more cumbersome, has proven to be the most reliable, and is something I have consistently found to be the most successful.
