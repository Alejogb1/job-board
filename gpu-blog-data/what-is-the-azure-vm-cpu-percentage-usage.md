---
title: "What is the Azure VM CPU percentage usage?"
date: "2025-01-30"
id: "what-is-the-azure-vm-cpu-percentage-usage"
---
The Azure Virtual Machine (VM) CPU percentage usage represents the proportion of available processing capacity that a given virtual machine is actively consuming at a specific moment or over a defined interval. It's a crucial metric for monitoring performance, identifying resource bottlenecks, and optimizing infrastructure costs. I've spent the last several years managing cloud deployments, including a substantial Azure footprint, and have observed that accurate interpretation of this metric often requires careful consideration of factors beyond a simple percentage value.

Fundamentally, the reported CPU percentage is derived from the ratio of CPU time spent executing instructions relative to the total CPU time available. This calculation can differ slightly depending on the tool or method used for observation, but the core principle remains constant. Within Azure, this metric isn't a direct reflection of the host machine's physical CPU, instead, it reflects the allocation and consumption by the assigned virtual cores or vCPUs. Understanding this distinction is paramount; a VM utilizing 100% of its assigned vCPUs doesn't necessarily equate to 100% utilization of the underlying hardware.

Several factors influence the observed CPU percentage. First, the size and type of the VM play a significant role. Different VM sizes are allocated varying numbers of vCPUs, each with a distinct amount of processing power. This ranges from general-purpose VMs suitable for average workloads, to compute-optimized VMs designed for high-performance tasks. The underlying CPU architecture, including generation and core frequency, also contributes to the total available capacity. Secondly, the workload being executed on the VM directly dictates the CPU utilization. CPU-bound applications, such as complex simulations or data processing tasks, will naturally register higher percentages compared to applications with significant I/O or waiting times. Furthermore, background processes, operating system services, and monitoring agents all contribute to CPU consumption, although usually at lower percentages.

Azure provides multiple avenues for retrieving and analyzing CPU percentage data, each with specific advantages and use cases. Within the Azure Portal, the Metrics Explorer provides a graphical interface for visualizing CPU utilization over various time ranges. This can offer immediate insights for troubleshooting or high-level trend analysis. The data is pulled from the Azure Monitor service, the platform's core monitoring system.

Beyond the portal, programmatic access to this data is often required for automation or integration into larger monitoring pipelines. This can be achieved through the Azure CLI or PowerShell using REST APIs, retrieving raw data which can then be manipulated as needed. These options allow for complex queries, analysis, and integration with custom dashboards or alert systems. When working in my previous role, a critical issue involved a spike in processing during business hours. Using Azure CLI scripts, I quickly isolated the problem to a database query that could be optimized. In addition to these native tools, several third-party monitoring applications can pull and interpret this information as well, sometimes providing additional insights.

Let's examine three specific examples illustrating different methods of acquiring CPU percentage data. The first demonstrates the use of PowerShell to query the average CPU usage over the last hour:

```powershell
$resourceGroupName = "MyResourceGroup"
$vmName = "MyVM"
$timeWindow = "PT1H"  # Last 1 hour
$metricName = "Percentage CPU"

$metric = Get-AzMetric `
    -ResourceGroupName $resourceGroupName `
    -ResourceId "/subscriptions/$(Get-AzContext | Select-Object -ExpandProperty Subscription)./resourceGroups/$resourceGroupName/providers/Microsoft.Compute/virtualMachines/$vmName" `
    -MetricName $metricName `
    -TimeWindow $timeWindow

$averageCpu = ($metric.Data | Where-Object {$_.Total -ne $null} | Measure-Object -Property Total -Average).Average

Write-Host "Average CPU utilization for $vmName in the last hour: $($averageCpu)%"

```

This PowerShell script initially defines key parameters such as the resource group name, VM name, time window, and metric name. It utilizes the `Get-AzMetric` cmdlet to retrieve the performance data from Azure Monitor. The `Where-Object` clause removes null values, often present at the beginning and end of a metric query, to avoid misleading results. Finally, `Measure-Object` calculates the average CPU percentage and outputs the result. This type of query is beneficial for quick historical checks and can be incorporated into simple automation scenarios.

Next, consider a scenario using the Azure CLI to retrieve the same data. This provides a cross-platform alternative and is suitable for shell scripting environments:

```bash
resource_group="MyResourceGroup"
vm_name="MyVM"
time_window="PT1H"
metric_name="Percentage CPU"
subscription_id=$(az account show --query id -o tsv)

result=$(az monitor metrics list \
    --resource "/subscriptions/$subscription_id/resourceGroups/$resource_group/providers/Microsoft.Compute/virtualMachines/$vm_name" \
    --metric $metric_name \
    --time-window $time_window \
    --query "value[0].timeseries[0].data[].average" \
    -o tsv)

average_cpu=$(echo $result | awk '{total += $1; count++} END {print total/count}')

echo "Average CPU utilization for $vm_name in the last hour: $(printf "%.2f" $average_cpu)%"
```

This script, written for a Bash environment, defines the relevant variables. It uses `az monitor metrics list` to query Azure Monitor. The `--query` argument extracts the average value from the nested JSON output. `awk` is used to perform the final average calculation over the result set and outputs the result to two decimal places. This example illustrates how command-line tools can efficiently extract specific metrics, a frequent requirement when integrating with external monitoring systems.

Finally, consider a more comprehensive approach involving the Azure Monitor REST API, useful for creating custom tools or dashboards:

```python
import requests
import json
import datetime

subscription_id = "your-subscription-id"
resource_group = "MyResourceGroup"
vm_name = "MyVM"
time_window = datetime.timedelta(hours=1)
metric_name = "Percentage CPU"
now = datetime.datetime.utcnow()
start_time = now - time_window
end_time = now

headers = {
    'Authorization': 'Bearer YOUR_ACCESS_TOKEN',
    'Content-Type': 'application/json'
}


url = f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Compute/virtualMachines/{vm_name}/providers/microsoft.insights/metrics"

params = {
    'api-version': '2018-01-01',
    'metricnames': metric_name,
    'timespan': f"{start_time.isoformat()}Z/{end_time.isoformat()}Z",
    'interval': 'PT1M'
}


response = requests.get(url, headers=headers, params=params)
response.raise_for_status() # Raise an exception for bad status codes

data = response.json()

total_cpu = 0
data_points = 0

if 'value' in data and data['value']:
    for timeseries in data['value'][0]['timeseries']:
        for datapoint in timeseries['data']:
            if datapoint['average'] is not None:
                total_cpu += datapoint['average']
                data_points += 1
if data_points > 0:
    average_cpu = total_cpu / data_points
    print(f"Average CPU utilization for {vm_name} in the last hour: {average_cpu:.2f}%")
else:
    print("No data points found for the specified time range.")
```

This Python code, using the `requests` library, showcases how to retrieve the same CPU metric data using the Azure Monitor REST API. It requires an access token for authentication.  The code constructs the API endpoint and request parameters, then parses the returned JSON to extract the data points. The average CPU utilization is calculated by summing all data points and dividing by the number of data points. The `response.raise_for_status()` call prevents silent errors from API failures. Using APIs like this allows integration with other services or the creation of complex data processing workflows. This is the approach I've found most flexible, though more complicated to implement.

For further information on Azure monitoring, the official Azure documentation on Azure Monitor provides a solid foundation. Additionally, the Microsoft Learn platform offers detailed learning paths and modules covering various aspects of Azure resource monitoring. In-depth books focusing on Azure infrastructure and operations management can be very valuable, particularly for deeper understanding of system-level performance metrics.

Ultimately, the Azure VM CPU percentage metric is a critical indicator of a VM's performance and resource utilization. Successfully leveraging this data requires a holistic understanding of how it is calculated, factors influencing it, and the tools available for access. My experience consistently shows that proactive analysis and monitoring are key for optimal cloud resource management.
