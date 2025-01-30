---
title: "Why does Ansible hang during ZOS Core job submission?"
date: "2025-01-30"
id: "why-does-ansible-hang-during-zos-core-job"
---
The primary reason for Ansible hangs during z/OS core job submission frequently stems from insufficient handling of the return codes and asynchronous nature of the JCL submission process.  My experience troubleshooting this on several large mainframe migrations highlighted the critical need for robust error handling and timeout mechanisms within the Ansible playbook.  Ignoring the asynchronous nature of the z/OS job submission leads to Ansible prematurely assuming completion, resulting in hangs or inaccurate reporting.

**1. Understanding Asynchronous Job Submission on z/OS:**

Unlike many other operating systems, z/OS job submission is inherently asynchronous. When you submit a JCL job using tools like `tso`, `omvs`, or even an Ansible module, the operating system acknowledges the submission but doesn't immediately provide a definitive completion status.  The job proceeds independently, potentially taking considerable time depending on its complexity and system load.  Ansible, by default, waits for a command to complete before proceeding to the next task. This model clashes directly with the z/OS job submission's asynchronous behavior. If Ansible doesn't account for this asynchronous nature, it essentially hangs indefinitely, waiting for a response that may never come within the default timeout period.  The critical oversight is the expectation of immediate feedback; the submitted job requires a separate mechanism for status checking.


**2. Implementing Robust Error Handling and Asynchronous Monitoring:**

To rectify this, we must adopt a different strategy:  submit the job and subsequently poll for its completion status.  This necessitates a change from Ansible's default synchronous execution model. We can achieve this using the `wait_for_connection` module in conjunction with a custom script or a more sophisticated approach involving the `uri` module to interact with z/OS system services that provide job status information.

**3. Code Examples:**

**Example 1: Using `wait_for_connection` with a simple status check (less robust):**

This example demonstrates a basic approach, utilizing `wait_for_connection` module to wait for a file indicating job completion.  This is less robust than using direct system calls, but illustrates the principle of asynchronous polling.

```yaml
- name: Submit z/OS job
  shell: "tso SUBMIT {{ jcl_file }}"
  register: job_submission_result

- name: Wait for job completion (simplified)
  wait_for_connection:
    host: localhost
    port: 22  # SSH port (replace with appropriate method for status check)
    delay: 10
    timeout: 300 # 5 minutes timeout
  until: test -f /path/to/job_completion_indicator

- name: Process job output
  #Further tasks to retrieve job output after completion
  # ...
```

**Commentary:** This approach assumes a job completion indicator file is created upon successful job termination. The `wait_for_connection`  module is used here as a placeholder for a more targeted method of checking for job completion. The `until` clause checks for the file's existence periodically.  This method relies heavily on the external file, which could introduce problems if its creation fails.


**Example 2: Utilizing `uri` module with a z/OS system service:**

This example provides a more sophisticated and reliable method.  It leverages the `uri` module to query a z/OS system service (e.g., a REST API, if available, or a custom-written service).  This requires a z/OS service capable of providing job status information.

```yaml
- name: Submit z/OS job
  shell: "tso SUBMIT {{ jcl_file }}"
  register: job_submission_result

- name: Poll for job status
  uri:
    url: "http://zosexec.example.com/jobstatus/{{ job_submission_result.job_id }}"
    method: GET
    validate_certs: no #Adjust as necessary for your certificate setup.
  register: job_status
  until: job_status.json.status == "COMPLETED"
  retries: 30
  delay: 10

- name: Handle job completion (or failure)
  debug:
    msg: "Job {{ job_submission_result.job_id }} completed with status: {{ job_status.json.status }}"
  when: job_status.json.status == "COMPLETED"

- name: Handle job failure
  fail:
    msg: "Job {{ job_submission_result.job_id }} failed with status: {{ job_status.json.status }}"
  when: job_status.json.status != "COMPLETED"
```

**Commentary:** This is a more robust approach.  It directly queries the z/OS system for the job's status, avoiding the limitations of relying on a simple file indicator. Error handling is improved by explicitly checking the status and taking appropriate actions.  Remember to replace placeholder values (URL, job ID extraction) with your actual implementation.


**Example 3: Implementing custom script for more complex scenarios:**

For exceptionally complex scenarios or when interacting with less standardized z/OS services, a custom script (e.g., in REXX or Python running in an OMVS environment) can provide greater flexibility and control.  Ansible can then execute this script and handle the output.

```yaml
- name: Submit z/OS job
  shell: "tso SUBMIT {{ jcl_file }}"
  register: job_submission_result

- name: Run custom script to monitor job status
  shell: "/path/to/my_job_monitor.sh {{ job_submission_result.job_id }}"
  register: job_monitor_result

- name: Process job status from script output
  set_fact:
    job_status: "{{ job_monitor_result.stdout | b64decode | from_yaml }}"  # Assuming YAML output from script

- name: Handle job completion
  debug:
    msg: "Job status: {{ job_status.status }}"
  when: job_status.status == "COMPLETED"

- name: Handle job failure
  fail:
    msg: "Job failed: {{ job_status.error_message }}"
  when: job_status.status != "COMPLETED"

```

**Commentary:** This example showcases the use of a custom script for granular control over the job monitoring process. The script handles the specifics of interacting with z/OS APIs or utilities.  Error handling is essential in this approach, ensuring failures in the script are properly reported to Ansible.  The output from the custom script needs careful formatting (YAML is a suitable option) for easy parsing by Ansible.


**4. Resource Recommendations:**

IBM z/OS documentation on JCL and job submission.  Ansible documentation on the `shell`, `uri`, and `wait_for_connection` modules.  Understanding REXX or Python scripting for z/OS environments.  Consult IBM's knowledge base and support resources for z/OS-specific troubleshooting.  Thorough understanding of JCL error codes and how to interpret them programmatically.


In summary, overcoming Ansible hangs during z/OS job submission requires a fundamental shift from synchronous to asynchronous processing. Implementing robust polling mechanisms, handling asynchronous responses appropriately, and using effective error handling within your Ansible playbook are crucial steps towards reliable automation of z/OS core jobs. The choice between using the `wait_for_connection`, `uri` module or custom scripting depends on the complexity of the job, the availability of appropriate z/OS services, and the level of control you require.  Remember that thorough testing and error handling are indispensable to building robust and reliable Ansible playbooks for z/OS.
