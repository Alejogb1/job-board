---
title: "How do I retrieve the volume ID of a terminated EBS volume?"
date: "2024-12-23"
id: "how-do-i-retrieve-the-volume-id-of-a-terminated-ebs-volume"
---

Alright,  It’s a question that’s come up more than once in my career, especially during incident response and infrastructure audits. Dealing with the aftermath of terminated resources can be a bit tricky, but fortunately, the aws ecosystem provides the mechanisms, albeit often a little less obvious than we might like. You're asking about retrieving the volume id of a *terminated* ebs volume, and the key thing to understand is that once an ebs volume is completely terminated, it's no longer an active resource, and its volume id isn’t readily available through typical describe calls.

My experience comes from those hairy days working with a microservices architecture where we had frequent turnover of instances and, occasionally, some unfortunate accidental deletions. In one such instance, an automation script had an oversight, and we inadvertently terminated a bunch of instances, along with their associated ebs volumes. we needed to piece together what was lost, and the standard describe calls were no help. what we ended up relying on was the trail left by cloudtrail and, to some degree, the snapshot system.

Here’s the breakdown of how we typically go about this, and what the different approaches look like.

First off, cloudtrail logging is your primary source for this information. If configured correctly (and it absolutely should be for audit purposes), cloudtrail captures every api call made to your aws account, including the termination of ebs volumes. the ‘deletevolume’ event will include the volume id in its details. this is generally the fastest and most reliable method provided cloudtrail logging was enabled *before* the deletion.

```python
import boto3
import json
import datetime

def get_deleted_volume_id_from_cloudtrail(volume_id_filter=None, region_name='us-east-1', event_name='DeleteVolume'):
    """Retrieves volume ids of deleted ebs volumes from cloudtrail logs."""
    cloudtrail = boto3.client('cloudtrail', region_name=region_name)
    now = datetime.datetime.utcnow()
    start_time = now - datetime.timedelta(days=7) # Look back at last 7 days

    paginator = cloudtrail.get_paginator('lookup_events')
    response_iterator = paginator.paginate(
        LookupAttributes=[
            {
                'AttributeKey': 'EventName',
                'AttributeValue': event_name
            },
        ],
        StartTime=start_time,
        EndTime=now
    )
    deleted_volume_ids = []
    for page in response_iterator:
      for event in page.get('Events', []):
        event_details = json.loads(event['CloudTrailEvent'])
        if 'requestParameters' in event_details and 'volumeId' in event_details['requestParameters']:
            volume_id = event_details['requestParameters']['volumeId']
            if volume_id_filter is None or volume_id == volume_id_filter:
                deleted_volume_ids.append(volume_id)
    return deleted_volume_ids

#Example usage:
deleted_ids = get_deleted_volume_id_from_cloudtrail()
print(f"found the following deleted volume ids: {deleted_ids}")
# for example to search for a specific id:
# specific_deleted_ids = get_deleted_volume_id_from_cloudtrail(volume_id_filter="vol-xxxxxxxxxxxxxxxxx")
# print(f"found the specific deleted volume ids: {specific_deleted_ids}")


```

This python script leverages the boto3 library, which you need to have installed ('pip install boto3'). it filters cloudtrail events for the 'deletevolume' event. it parses through the event details, and if it finds the volume id, it adds it to a list and returns it.  it also gives an example of how to filter the results if you already know a volume id. This is generally the preferred method since it gets the volume ids as they were prior to deletion.

Now, a secondary method is to analyze any existing snapshots. snapshots created of the deleted volume will include its volume id, even if the volume no longer exists. you can use this method as a failsafe if you are unsure if cloudtrail was enabled correctly or for confirmation.  However, it's critical to understand this only works if snapshots were taken of the volume *before* deletion.

```python
import boto3
def get_volume_ids_from_snapshots(region_name='us-east-1'):
    """Retrieves volume ids from existing snapshots."""

    ec2 = boto3.client('ec2', region_name=region_name)
    response = ec2.describe_snapshots(OwnerIds=['self'])
    volume_ids = set()
    for snapshot in response['Snapshots']:
        if 'Description' in snapshot and "Created for volume" in snapshot['Description']:
              volume_id = snapshot['Description'].split("volume ")[1].split(".")[0]
              volume_ids.add(volume_id)
        elif 'VolumeId' in snapshot:
            volume_ids.add(snapshot['VolumeId'])
    return list(volume_ids)

# Example usage:
snapshot_volume_ids = get_volume_ids_from_snapshots()
print(f"found the following volume ids from snapshots: {snapshot_volume_ids}")

```

The python code here iterates through available snapshots, extracting volume ids from snapshot descriptions or the `VolumeId` metadata. Note that older snapshots might use a specific description text to reference the volume, and this might vary depending on your snapshot policies, so the code takes that into account. the `set()` is used to ensure that we don't have duplicate entries, because the same volume id can have multiple snapshots.

A third, albeit less direct, method relies on examining ebs volume backups created via aws backup. the backup metadata will reference the volume id at the time of the backup.

```python
import boto3

def get_volume_ids_from_backups(region_name='us-east-1'):
    """Retrieves volume ids from existing backups using aws backup."""

    backup = boto3.client('backup', region_name=region_name)
    response = backup.list_recovery_points_by_resource(ResourceType='EBS')
    volume_ids = set()
    for recovery_point in response.get('RecoveryPointsByResource', []):
      if 'ResourceType' in recovery_point and recovery_point['ResourceType'] == 'EBS':
          resource_arn = recovery_point['ResourceArn']
          volume_id = resource_arn.split('/')[1] #Extract volume id from ARN
          volume_ids.add(volume_id)
    return list(volume_ids)

# Example usage:
backup_volume_ids = get_volume_ids_from_backups()
print(f"found the following volume ids from backups: {backup_volume_ids}")

```

This snippet uses the aws backup service to locate recovery points of ebs volumes. It then extracts the volume id from the resource arn and adds them to the set. this method works as long as backups of the deleted volumes existed.

Remember that if snapshots and backups were not configured *prior* to the deletion of your volume, your most reliable method remains the cloudtrail logs. The accuracy of snapshot metadata and backup configuration should always be independently validated as well, to ensure data integrity and accessibility.

For deep dives into these areas, i would recommend reading “aws certified solutions architect study guide” by ben piper and david clark which does into detail about using cloudtrail and different forms of monitoring in the aws environment. “mastering aws networking” by samuel j. carpenter and michael l. weinhardt provides a detailed look into the networking aspects of cloud infrastructure, which, while not directly related to volume ids, provides foundational knowledge that can aid in understanding the holistic landscape of cloud resources. lastly, the aws documentation itself is your most authoritative source, so i'd recommend going through the documentation on cloudtrail, ec2 snapshots, and aws backup services for the most accurate information. They’re constantly updated, and you’ll find the latest specifics there.

I hope this detail helps you in your troubleshooting or audit efforts. It's essential to stay proactive with monitoring and logging, so you're well-equipped when these sorts of situations arise. let me know if there’s anything more specific you’d like to know about this!
