---
title: "How can I save SFTP settings for reusable use across projects?"
date: "2024-12-23"
id: "how-can-i-save-sftp-settings-for-reusable-use-across-projects"
---

Alright, let's talk about persisting sftp settings for streamlined project workflows. I've been down that rabbit hole many times, and trust me, juggling sftp credentials across different projects can quickly become a logistical nightmare. It's not just about remembering hostnames, ports, and usernames; it's also about managing private keys, dealing with different connection options, and ensuring security. Forget copy-pasting configurations; we need a more robust solution.

Over the years, I've found a multi-pronged approach works best. Firstly, centralize your configuration using dedicated configuration management tools. Second, embrace automation. Third, utilize your operating system's built-in tools effectively. We'll unpack these individually, and I'll show you some code that I've actually used and refined in past projects.

**Centralized Configuration Management:**

The problem with manually managing sftp settings is that they tend to become fragmented and inconsistently applied. Configuration drift, where different projects use slightly modified configurations, inevitably leads to confusion and errors. To avoid this, I strongly suggest adopting configuration management principles. That could mean using tools like Ansible, Chef, or Puppet, depending on the size and scale of your operations.

For smaller projects, however, a simpler approach may be sufficient. I've frequently used a custom configuration file, often in json format, combined with a simple python script, and this has proven quite effective. The idea is to have a central repository where all sftp configurations are stored, version-controlled, and accessible from any project.

For instance, imagine a `sftp_configs.json` file structured like this:

```json
{
  "production": {
    "host": "prod.example.com",
    "port": 22,
    "username": "deployuser",
    "private_key_path": "/home/user/.ssh/prod_key",
    "remote_base_path": "/var/www/html/app"
  },
   "staging": {
    "host": "staging.example.com",
    "port": 22,
    "username": "stage_deployer",
    "private_key_path": "/home/user/.ssh/stage_key",
    "remote_base_path": "/var/www/staging"
  },
   "development": {
    "host": "dev.example.local",
    "port": 22,
    "username": "dev_user",
    "private_key_path": "/home/user/.ssh/dev_key",
     "remote_base_path": "/var/www/dev"
  }
}
```

This json format is both machine-readable and human-editable, which is essential for maintaining clarity. Now, with the configuration in json, let’s explore how we could access it through python.

**Automation and Scripting:**

Once you've centralized your settings, the next step involves automating the sftp process. Relying on manual `sftp` commands is just as prone to errors as managing settings manually. I've found that using scripting languages like python, specifically the `paramiko` library, is a game-changer. It allows programmatic access to sftp, enabling consistent and repeatable operations.

Here's a basic python script that shows how to use the configuration file we created above:

```python
import json
import paramiko
import os


def upload_files(config_name, local_files, json_config_path='sftp_configs.json'):
    """Upload files to a server using configuration from a json file."""
    try:
        with open(json_config_path, 'r') as f:
            configs = json.load(f)
            if config_name not in configs:
                raise ValueError(f"Configuration '{config_name}' not found.")
            config = configs[config_name]

            hostname = config.get('host')
            port = config.get('port', 22) # Defaulting to port 22 if not found.
            username = config.get('username')
            private_key = config.get('private_key_path')
            remote_base = config.get('remote_base_path')


        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=hostname, port=port, username=username, key_filename=private_key)
        sftp = ssh_client.open_sftp()
        for local_file in local_files:
            remote_file = os.path.join(remote_base, os.path.basename(local_file))
            sftp.put(local_file, remote_file)
            print(f"Uploaded '{local_file}' to '{remote_file}'")
        sftp.close()
        ssh_client.close()
    except FileNotFoundError:
        print("Error: Could not locate json config file")
    except ValueError as e:
      print(f"Error: {e}")
    except paramiko.SSHException as e:
        print(f"Error connecting: {e}")
    except Exception as e:
      print(f"An unexpected error occurred: {e}")



if __name__ == '__main__':
    files_to_upload = ['local_file1.txt', 'local_file2.txt', 'local_file3.txt']
    # Ensure you have these files, or replace with paths to existing files
    for file in files_to_upload:
        if not os.path.exists(file):
            with open(file, 'w') as f:
                f.write("This is a dummy file")
    upload_files('production', files_to_upload)
    # upload_files('staging', files_to_upload)
```

The script above loads sftp configurations from our `sftp_configs.json`, establishes an ssh connection using `paramiko`, opens an sftp session, and uploads a list of files to a remote destination. It includes basic error handling, such as handling the case of a missing json file, which is essential for a production-grade script. You can extend it further to accommodate tasks like downloading files, creating directories, or deleting files.

Let's add one more example, this time focusing on downloading files using our existing structure:

```python
import json
import paramiko
import os


def download_files(config_name, remote_files, local_dir, json_config_path='sftp_configs.json'):
    """Download files from a remote server."""
    try:
        with open(json_config_path, 'r') as f:
            configs = json.load(f)
            if config_name not in configs:
                raise ValueError(f"Configuration '{config_name}' not found.")
            config = configs[config_name]

            hostname = config.get('host')
            port = config.get('port', 22)
            username = config.get('username')
            private_key = config.get('private_key_path')
            remote_base = config.get('remote_base_path')


        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=hostname, port=port, username=username, key_filename=private_key)
        sftp = ssh_client.open_sftp()
        if not os.path.exists(local_dir):
          os.makedirs(local_dir)
        for remote_file in remote_files:
            full_remote_path = os.path.join(remote_base, remote_file)
            local_file = os.path.join(local_dir, os.path.basename(remote_file))
            sftp.get(full_remote_path, local_file)
            print(f"Downloaded '{full_remote_path}' to '{local_file}'")
        sftp.close()
        ssh_client.close()
    except FileNotFoundError:
        print("Error: Could not locate json config file")
    except ValueError as e:
      print(f"Error: {e}")
    except paramiko.SSHException as e:
        print(f"Error connecting: {e}")
    except Exception as e:
      print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
  files_to_download = ["remote_file1.txt", "remote_file2.txt"] #Ensure these exist on your remote server
  local_directory = "downloads"
  download_files('production', files_to_download, local_directory)
```

This snippet extends the previous functionality to download remote files to a specified local directory. It emphasizes the reusability of our configuration data. This is the strength of this method. You are not limited to upload actions. You can easily create helper functions that fit specific project needs using that single configuration file, such as a `backup_remote_directory` action.

**Operating System Tools:**

In addition to these tools, it's essential not to overlook the features provided by your operating system. For example, `ssh-agent` can cache decrypted private keys in memory so that they don’t need to be re-entered every time you connect. This not only simplifies connections but is also more secure than having keys saved in plain text. Similarly, you can configure ssh using a `~/.ssh/config` file to add specific host definitions.

For instance, you could add a stanza similar to:
```
Host prod.example.com
    User deployuser
    IdentityFile ~/.ssh/prod_key
    Port 22
```
to your `~/.ssh/config` file.

These settings can then be accessed simply by using `sftp prod.example.com` in your terminal. These operating system level settings are incredibly powerful because they are used by all tools that interact with ssh.

**Resource Recommendations:**

For a deeper dive, I’d recommend several resources:

1.  **"High Performance Python" by Micha Gorelick and Ian Ozsvald:** This book is invaluable for understanding how to write performant python code, especially when it comes to tasks like file transfers and network communication.
2. **The Paramiko documentation:** The official paramiko website provides comprehensive documentation and examples on how to use the library effectively.
3. **"Infrastructure as Code" by Kief Morris:** This book will help you understand the overall principles of configuration management and automation, extending the concepts beyond just sftp.

In conclusion, saving sftp settings for reusable use requires a structured and automated approach. Relying on configuration management, implementing automation with scripting languages, and utilising your operating system tools are fundamental to building consistent, efficient and secure deployment pipelines. It’s a continuous improvement process, and you'll constantly refine the steps that work best for your projects. The key takeaway is to move away from manual management and towards programmatic control of your sftp tasks, and never copy and paste connection information. Trust me, your future self will thank you for it.
