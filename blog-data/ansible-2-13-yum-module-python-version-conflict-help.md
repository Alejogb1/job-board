---
title: "Ansible 2.13 Yum Module:  Python Version Conflict - Help!"
date: '2024-11-08'
id: 'ansible-2-13-yum-module-python-version-conflict-help'
---

```yaml
---
- name: Deploy Services
  hosts: centos-6-vm
  gather_facts: true

  tasks:
    - name: Patch
      become: true
      when: ansible_distribution_major_version | int == 6
      command: yum update -y
    - name: Patch
      become: true
      when: ansible_distribution_major_version | int >= 7
      yum:
        name: "*"
        security: true
        state: latest
        update_cache: true
```

