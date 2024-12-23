---
title: "How can I restore a domain from a deleted Google Cloud project?"
date: "2024-12-23"
id: "how-can-i-restore-a-domain-from-a-deleted-google-cloud-project"
---

, let’s tackle this one. I've seen this scenario a few times, usually in the wake of a late-night "oops" or someone cleaning house a bit too enthusiastically. The process of recovering a domain associated with a deleted Google Cloud project isn't exactly intuitive, but it's generally achievable with a bit of careful maneuvering. It's not a single button click, unfortunately, and it involves understanding a couple of key concepts about how google cloud handles domain registration and project deletion.

First off, it's vital to clarify that deleting a google cloud *project* is different from deleting a *domain registration* within that project (if the domain was registered through google domains linked to that project, for example). If you deleted the domain registration directly, that's another beast altogether and often requires working with support, which will be briefly mentioned. However, I’m assuming we're talking about the more common situation: you deleted the entire project, including a domain that was linked to it.

The core challenge here stems from how Google handles resource deletion. When a project is deleted, its resources, including attached services and configurations, are marked for deletion. They don’t disappear instantly but go into a "soft-deleted" state for a limited period – typically 30 days. Crucially, this grace period doesn’t automatically restore domain names. Google's design here favors preventing accidental permanent deletions, hence the soft-delete approach.

The critical steps to focus on are:
1. Project Recovery: We must restore the deleted project first and foremost.
2. Domain Reconfiguration: Once the project is recovered, we'll need to re-associate the domain with the appropriate services, if not already the case due to the recovery process.

Let's go through each step in more detail.

**1. Project Recovery**

The first thing you must understand is that the soft-delete period has a hard limit. If you're beyond the 30-day grace, recovery becomes significantly more complex, often involving direct contact with Google Cloud support. If you *are* within that window, though, the good news is that this is something you can do on your own. Here’s how we typically proceed:

* **Using the Google Cloud Console:** Navigate to the "Manage Resources" page in the cloud console. Look for a "Deleted Projects" filter or section. There, you should see a list of recently deleted projects that can be restored. Select your project and click "Restore." The cloud console typically handles the rest. The recovery process might take a few minutes (sometimes longer, depending on the project size).

* **Using the gcloud CLI:** The command-line interface can be faster and can be included in automated scripts. Here's an example of the process:
    ```bash
    gcloud projects list --filter="lifecycleState:DELETE_REQUESTED"

    # This shows the list of deleted projects
    # Locate the project ID you want to restore
    # Assuming the project ID is 'your-project-id', run this

    gcloud projects undelete your-project-id

    # This command will restore the project

    # You may want to check the status to ensure that the restore was successful
    gcloud projects describe your-project-id
    ```
   This command sequence will first retrieve a list of deleted projects, allowing you to identify the specific project to restore. Then, it will initiate the recovery process, and check the status of the restore after the command to ensure it was successful.

   *Note*: I’ve occasionally found that if a project had a highly complex setup (think multiple interwoven services and configurations), the recovery can be a little less straightforward. Sometimes, specific resources require manual re-configuration or re-linking even after the initial project recovery, but that rarely involves a domain name.

**2. Domain Reconfiguration**

Once the project is recovered, the domain should, theoretically, be attached again as it was originally configured. However, there are cases where you might need to double-check or perform some additional configuration. If the domain name was hosted with google domains, it's usually seamlessly restored with the project.

* **Verifying Domain Status:** Navigate to the relevant service where your domain was being used, whether it's App Engine, Cloud Run, or another service using the domain. Check the settings for custom domains or related configurations. If needed, you may need to re-establish the DNS records or certificate registrations that link to the service after the project restore, especially if the restore process didn't fully re-establish the service links.

* **Using gcloud CLI for Domain Mapping (App Engine Example):** If the domain is used with App Engine, it might require a manual re-mapping. Example:
    ```bash
    gcloud app domain-mappings create your-domain.com --service=your-service

    # Replace 'your-domain.com' with your domain, and 'your-service' with the appengine service

    # If you are using HTTPS, you might also have to manage certificates

    # For example, to view existing SSL certificates:
    gcloud app ssl-certificates list

    # To update a certificate, if it has expired, you need to upload a new one or reissue
    # This part can vary, and will probably require you to understand ssl certificates and configuration
    ```
   This example shows how you'd re-add the domain to the mapping in App Engine, ensuring the app is accessible using the custom domain. Depending on the specific application you are using and how you have it configured, you may need to use a different set of gcloud commands.

* **DNS Settings:** It’s highly unlikely the dns configuration will need to be touched directly, but it's best practice to verify your dns records after recovering a project and its domain name.

   *Note:* I’ve had times when DNS propagation takes some time after restoring the project, so it isn't a bad idea to use a DNS lookup tool to check for that as well.

* **What if the Domain Registration Was Deleted Directly?** This is where things get trickier. If the actual domain registration (not just the project) was deleted from Google Domains, it's often necessary to contact Google Cloud support. There's no guarantee you'll get it back, as the domain might have been released and become available for others to register, so it becomes a race.

**Important Resources**

To delve deeper into these topics, I suggest a few authoritative sources. The official Google Cloud documentation is the most reliable place to start and always contains the most up-to-date information. However, a couple of references are generally helpful to solidify this knowledge:

*   **"Google Cloud Platform in Action" by JJ Geewax**: This book provides a comprehensive overview of Google Cloud, including project management, service deployment, and troubleshooting. It's quite practical and covers the details of resource recovery.
*   **"Designing Data-Intensive Applications" by Martin Kleppmann**: While not specific to Google Cloud, the sections on distributed systems and resource management are incredibly useful in understanding the underlying complexities of these platforms. This book might provide a deeper understanding into why these systems behave as they do.

**Final Considerations**

The key to a smooth domain recovery lies in understanding Google’s soft-delete mechanism and acting swiftly within the grace period. The techniques I've described here have proven successful in my past experiences. Always start with the project recovery and then move onto verifying the domain configurations. Remember to use the gcloud cli to automate the domain recovery and mapping, making the process less tedious and more efficient. Finally, keep your eyes on the time if you are operating within a deleted project. The window does close after 30 days.
