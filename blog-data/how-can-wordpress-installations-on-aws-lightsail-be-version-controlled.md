---
title: "How can WordPress installations on AWS Lightsail be version controlled?"
date: "2024-12-23"
id: "how-can-wordpress-installations-on-aws-lightsail-be-version-controlled"
---

Alright, let's tackle this. Version controlling WordPress installations, especially on something like AWS Lightsail, isn’t always straightforward, but it's absolutely crucial for sane development and maintenance. I’ve definitely been in the trenches with this, more times than I care to recall, often learning the hard way what *not* to do. Early in my career, I tried manually backing up and FTP-ing changes – a process that quickly turned into a chaotic nightmare of lost work and corrupted databases. Needless to say, I adopted version control quickly.

The core issue with WordPress lies in its multifaceted nature. It's not just code; it’s also a database and a media library. Simply throwing the `/wp-content` directory into a git repository isn't the full solution. You'll encounter challenges with database changes, uploaded images, theme customizations, and plugin updates. To truly version control a WordPress installation effectively on Lightsail, we need to manage these components separately and cohesively.

First, let's address the code itself, meaning your themes, plugins, and any custom code you’ve created. The obvious solution is git. You should create a git repository for these components. I typically structure my repo with separate folders for `themes`, `plugins`, and any bespoke code (often under a `custom` folder).

Here's a practical example of how you might structure your `.gitignore` file in the root of your WordPress installation after you've initialized your git repository within the `/wp-content` directory:

```
# Ignore everything
*
# Except these specific directories, recursively
!themes/
!plugins/
!custom/
!uploads/

# within those directories, you might want to exclude specific files

themes/*/node_modules
themes/*/package-lock.json
plugins/*/node_modules
plugins/*/package-lock.json
```

This setup allows you to track the code changes in your theme and plugins, but it avoids tracking node_modules, .lock files, and the massive `/uploads` directory by default. We *will* manage the uploads directory, but not directly within the git repository.

Now, let’s move on to handling database changes. Simply dumping a database into version control is problematic. Database schema changes need to be managed carefully, preferably with migrations. For this, I've found a tool called "wp-migrate-db" invaluable (available as a WordPress plugin, although there are other options). It allows you to create SQL migration files that reflect changes to the database schema. This ensures that database changes are reproducible, versioned, and can be applied on different environments easily. Remember to treat your database schema like code – changes should be incremental, tested, and versioned.

Instead of versioning the entire database dump, we will version migration scripts generated by this tool, or similar, in our git repository. These migrations can be placed within a `migrations` directory in the root of your repository. An example of how these migrations might look is as follows:

```sql
-- Migration 1: Add new custom table
CREATE TABLE IF NOT EXISTS my_custom_table (
   id INT AUTO_INCREMENT PRIMARY KEY,
   data VARCHAR(255)
);

-- Migration 2: Add a new column to existing table
ALTER TABLE wp_users
ADD COLUMN user_location VARCHAR(255);
```

These scripts are idempotent. It’s the most robust way to ensure the same database structure across different development and production environments. The migration history will now be part of your version control. To run these migrations, you would have a separate script or a process integrated into your deployment pipeline.

Finally, let’s address the elephant in the room: media uploads. Trying to track your `/uploads` directory directly via Git is a bad idea, as it typically becomes large, slow and impractical, especially as your website grows. Instead, use a cloud storage solution like AWS S3, and then utilize a plugin that offloads your media. The plugin then acts as a bridge, ensuring that WordPress treats media stored on S3 as if it were local. This not only provides version control for your code but scales better.

Furthermore, you can utilize the aws cli to sync the uploads folder from your server to an S3 bucket, and vice versa. Here is an example of bash script to do this.

```bash
#!/bin/bash
#This assumes you have configured the AWS CLI on the server
#Define source directory and destination s3 bucket
source_dir="/var/www/html/wp-content/uploads/"
dest_s3="s3://my-bucket-name/wp-uploads/"
sync_uploads(){
    # Sync to S3
    aws s3 sync "$source_dir" "$dest_s3" --delete
    echo "Uploads synced to S3"
}
sync_downloads(){
    # sync from s3
     aws s3 sync  "$dest_s3" "$source_dir" --delete
    echo "Uploads synced from S3"
}

case "$1" in
    "upload")
        sync_uploads
        ;;
    "download")
       sync_downloads
       ;;
     *)
        echo "Usage: $0 (upload|download)"
        exit 1
        ;;
esac
exit 0
```
This script, when executed with `upload` as its argument, would synchronize your `/wp-content/uploads` directory to your specified S3 bucket; the `download` argument would bring down from your bucket to your server. You could set up a cron job to execute the upload script periodically. And the `download` script would be executed during your deployment pipeline.

In summary, to effectively version control a WordPress installation on AWS Lightsail, focus on these three key areas:

1.  **Code Versioning**: Use git to track themes, plugins, and custom code.
2.  **Database Versioning**: Utilize migration tools to track and manage database schema changes.
3.  **Media Storage**: Offload media files to a service like S3, and use plugins for integration with WordPress.

For deeper dives into best practices, I highly recommend looking into "Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation" by Jez Humble and David Farley for a comprehensive understanding of CI/CD pipelines, which are crucial for the complete process. For WordPress specifically, consider the "WordPress Plugin Development" book by Brad Williams and Justin Tadlock, which provides helpful insights on structuring WordPress projects for version control. Also, study the documentation on AWS CLI and S3 for better understanding of your cloud media storage. These resources will provide you with a more structured and technical understanding, building upon the points I've outlined here. You'll find that by using this methodology, you'll not only achieve reliable version control, but also improve your workflow and ensure consistent deployments across your WordPress environments.