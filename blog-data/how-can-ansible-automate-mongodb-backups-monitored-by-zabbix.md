---
title: "How can Ansible automate MongoDB backups monitored by Zabbix?"
date: "2024-12-23"
id: "how-can-ansible-automate-mongodb-backups-monitored-by-zabbix"
---

 I remember a particularly harrowing experience a few years back when a database crash coincided with a failing backup process—it underscored the absolute necessity of robust automated backups and monitoring. Setting up MongoDB backups orchestrated by Ansible, with Zabbix keeping a watchful eye, is a powerful combination for disaster recovery and operational awareness. I'll walk you through the process, focusing on the practicalities I've encountered over time.

The core idea is to use Ansible to handle the backup mechanism, primarily through `mongodump`, and then to use Zabbix to monitor the success or failure of these backups, along with other important MongoDB metrics. This setup allows for scheduled, consistent backups and alerts if things go south. It's not a fire-and-forget system, though; it requires careful configuration and ongoing maintenance.

First, let's discuss the Ansible side. We'll craft a playbook that executes `mongodump` on the target MongoDB server and stores the backups in a designated location. The playbook should also be idempotent, so it can be run repeatedly without creating duplicates or causing unintended effects. Here’s a basic example of the task:

```yaml
---
- hosts: mongodb_servers
  become: true
  vars:
    backup_dir: "/var/backups/mongodb"
    timestamp: "{{ ansible_date_time.iso8601_basic }}"
    dump_filename: "mongodb_backup_{{ timestamp }}.gz"
    full_dump_path: "{{ backup_dir }}/{{ dump_filename }}"

  tasks:
    - name: Ensure backup directory exists
      file:
        path: "{{ backup_dir }}"
        state: directory
        mode: 0755
    - name: Create MongoDB dump
      command: "mongodump --gzip --archive={{ full_dump_path }}"
      register: dump_result

    - name: Log dump result
      debug:
         var: dump_result

    - name: Clean up old backups
      find:
        paths: "{{ backup_dir }}"
        patterns: "mongodb_backup_*.gz"
        age: "7d"
        state: file
      register: old_backups

    - name: Delete old backups
      file:
        path: "{{ item.path }}"
        state: absent
      with_items: "{{ old_backups.files }}"
      when: old_backups.files is defined and old_backups.files | length > 0
```

This playbook, executed against a group of servers labeled `mongodb_servers`, does several things. It creates a timestamped filename for each backup, creates the backup directory if it doesn't exist, executes `mongodump` into an archive, logs the result, and cleans up backups older than 7 days. The `register` variable is critical; it captures the output of the `mongodump` command, which can then be logged for audit purposes or used for more advanced processing.

Now, moving on to Zabbix integration. The crucial point here isn't just about whether the `mongodump` command executes successfully, but also if the backup files are actually being generated and if they are consistent. We accomplish this by setting up custom Zabbix checks using User Parameters. The agent on the MongoDB server would then run these checks and send data back to the Zabbix server.

Here’s how we'd define a user parameter in the `/etc/zabbix/zabbix_agentd.conf.d/mongodb_backup.conf` file on the MongoDB server:

```
UserParameter=mongodb.backup.last_success,/bin/sh -c 'ls -t /var/backups/mongodb/mongodb_backup_*.gz 2>/dev/null | head -n 1 | xargs -I {}  stat -c %Y {} | awk \'{print $1}\' || echo 0'
UserParameter=mongodb.backup.file_count,/bin/sh -c 'ls /var/backups/mongodb/mongodb_backup_*.gz 2>/dev/null | wc -l'
```

The first user parameter, `mongodb.backup.last_success`, checks the modification time of the latest backup file. It returns the timestamp or '0' if no backup files are present. The second, `mongodb.backup.file_count`, counts the number of backup files in the backup directory. On the Zabbix server, these two items can be polled, and triggers configured around them for alerting. A trigger could be set to fire if the last successful backup timestamp is older than the expected time frame or if the number of backup files is less than or greater than the expected amount.

Finally, to truly leverage Zabbix, we must monitor MongoDB itself beyond just backups. Here's where the built-in MongoDB monitoring template of Zabbix really shines. It needs to be configured, usually via the Zabbix UI. However, if you need to have a custom solution or enhance the standard template, you could implement custom checks that pull specific data using the mongo shell. I would not recommend parsing text outputs with tools like `grep` within Zabbix; Instead, always use JSON output where feasible.

Here's an example of how you could create a user parameter to monitor the number of connections:

```
UserParameter=mongodb.connections.current,/usr/bin/mongo --quiet --eval 'printjson(db.serverStatus().connections.current)'
```

The user parameter leverages the mongo shell to pull the current connection count. In Zabbix, you would configure a new item with type “Zabbix agent” and key ‘mongodb.connections.current’. Then, you would configure alerts based on thresholds. You could extend this further by monitoring metrics like opcounters, asserts, or replica set status. For more comprehensive monitoring, consider using the Zabbix agent's `system.run` feature to execute custom scripts that pull even more granular data from mongo.

For learning more, I recommend consulting the official MongoDB documentation, which is incredibly detailed and well-maintained. Also, the book "MongoDB: The Definitive Guide" by Kristina Chodorow is an excellent resource. For the Zabbix side, the official Zabbix documentation is your best bet. I also suggest examining practical examples and community-provided templates from resources like the Zabbix forums and GitHub, though scrutinize any third-party resource carefully.

In practice, this approach needs to be adapted to your specific circumstances. For instance, you might want to encrypt your backups before storage, or integrate with an offsite storage solution using Ansible. Remember, automation is a tool, and like any tool, it needs to be used with care and attention to detail. Regularly review the backups, monitor Zabbix alerts, and refine your system as needed based on what the monitoring data reveals. This layered approach offers a resilient and well-monitored system.
