---
title: "How can Apache Airflow authenticate users via LDAP securely?"
date: "2024-12-23"
id: "how-can-apache-airflow-authenticate-users-via-ldap-securely"
---

Okay, let's tackle user authentication in Apache Airflow with LDAP. I've spent my fair share of time wrestling with this particular setup, and it’s definitely crucial to get it locked down properly for any production environment. It's less about a simple on/off switch and more about carefully configuring several pieces to play nicely together.

The core issue here is bridging the gap between Airflow's internal user management system and your organization's LDAP directory. Airflow doesn't inherently understand LDAP; it needs an adapter or a bridge. What we’re going to do is configure Airflow to leverage `flask-ldap3-login`, a Flask extension that does the heavy lifting for us, handling the LDAP query and authentication process. It’s important to recognize that we aren't actually *importing* user data into Airflow. We are verifying credentials against the LDAP server on each login attempt.

From my experience, this involves several steps, starting with getting the core LDAP settings right in the `airflow.cfg` file. We're not just slapping in any old values here; accuracy is key. The configuration file needs specifics such as the LDAP server address (including the port), the search base, the user DN template, and potentially a bind DN and password for querying. You typically use a service account dedicated to this if you can avoid using a general-purpose user, which enhances security and reduces impact if one account becomes compromised.

So, let’s walk through the configuration options and then I'll illustrate it with code snippets.

First, within the `[webserver]` section of your `airflow.cfg` file, you’ll need to enable LDAP authentication by setting:

```
auth_backend = airflow.providers.ldap.auth_manager.LDAPAuthManager
```
This tells Airflow's webserver to look at our LDAP module for authentication.

Next, the configuration settings specific to the ldap extension. Here’s a basic setup, and it's crucial to understand that the specifics will change depending on your LDAP schema and network configuration.

```
[ldap]
ldap_uri = ldaps://your_ldap_server.com:636
ldap_bind_user = "cn=airflow_svc,ou=applications,dc=example,dc=com"
ldap_bind_password = "your_secure_password"
ldap_search_base = "ou=users,dc=example,dc=com"
ldap_user_name_attr = uid
ldap_user_filter = (objectClass=person)
ldap_user_dn_template = "uid={0},ou=users,dc=example,dc=com"
ldap_user_email_attr = mail
ldap_group_base = "ou=groups,dc=example,dc=com"
ldap_group_name_attr = cn
ldap_group_filter = (objectClass=groupOfNames)
ldap_group_member_attr = member
```

Let’s break down these configuration parameters:

*   `ldap_uri`: The URI for your LDAP server. I recommend using `ldaps://` to encrypt the connection, usually on port 636 (rather than `ldap://` which isn't encrypted). Security here is non-negotiable.
*   `ldap_bind_user`: This is the username used to initially connect and query the LDAP directory. It should be a service account with read access only, never a real user account.
*   `ldap_bind_password`: The password for the bind user. Secure this, maybe using Airflow's secret backend.
*   `ldap_search_base`: The base DN to search for users. This limits where the LDAP server searches and keeps the query efficient.
*   `ldap_user_name_attr`: The LDAP attribute that uniquely identifies a user, in our example, "uid". This will usually be something like `uid` or `samaccountname`.
*   `ldap_user_filter`: This is a search filter to select the entries that represent users. `(objectClass=person)` is common, but might need adjustment for your LDAP schema.
*   `ldap_user_dn_template`: This is the string format used to build a user’s Distinguished Name (DN) to login. The "{0}" represents where the entered username will go.
*    `ldap_user_email_attr`: The LDAP attribute for email. Used when creating a user in airflow.
*   `ldap_group_base`: The base DN for searching groups.
*   `ldap_group_name_attr`: The attribute representing the group name, such as `cn`.
*   `ldap_group_filter`: Similar to user filter, but for groups, `(objectClass=groupOfNames)` is a typical value.
*  `ldap_group_member_attr`: The attribute used to list members within groups (such as "member").

Now for the code snippets. These aren't code to execute directly, but rather represent simplified snippets illustrating key parts of how Airflow interacts with LDAP:

**Snippet 1: Authentication Process**

```python
from flask_ldap3_login import LDAP3LoginManager, AuthenticationResponse
import ldap3
from airflow.configuration import conf

def authenticate_user(username, password):
  ldap_uri = conf.get("ldap", "ldap_uri")
  ldap_bind_user = conf.get("ldap", "ldap_bind_user")
  ldap_bind_password = conf.get("ldap", "ldap_bind_password")
  ldap_search_base = conf.get("ldap", "ldap_search_base")
  ldap_user_dn_template = conf.get("ldap", "ldap_user_dn_template")
  
  ldap_manager = LDAP3LoginManager()

  user_dn = ldap_user_dn_template.format(username)

  response = ldap_manager.authenticate(
      ldap_uri,
      user_dn,
      password,
      bind_dn = ldap_bind_user,
      bind_pw = ldap_bind_password,
      connect_timeout=5,
      authentication_type = ldap3.SIMPLE
  )

  if isinstance(response, AuthenticationResponse) and response.is_valid():
    return True # Authentication Success
  else:
    return False # Authentication Failure

```
This snippet demonstrates the core login workflow. Flask-ldap3-login handles the connection, binding with the service account, and the subsequent attempt to login with the user's credentials.

**Snippet 2: User Lookup and User Creation**

```python
from flask_ldap3_login import LDAP3LoginManager
from airflow.configuration import conf
import ldap3

def get_user_details_from_ldap(username):
    ldap_uri = conf.get("ldap", "ldap_uri")
    ldap_bind_user = conf.get("ldap", "ldap_bind_user")
    ldap_bind_password = conf.get("ldap", "ldap_bind_password")
    ldap_search_base = conf.get("ldap", "ldap_search_base")
    ldap_user_name_attr = conf.get("ldap", "ldap_user_name_attr")
    ldap_user_filter = conf.get("ldap", "ldap_user_filter")
    ldap_user_email_attr = conf.get("ldap", "ldap_user_email_attr")

    ldap_manager = LDAP3LoginManager()
    ldap_server = ldap3.Server(ldap_uri, get_info=ldap3.ALL)
    conn = ldap3.Connection(ldap_server, user=ldap_bind_user, password=ldap_bind_password)
    conn.bind()

    search_filter = f"(&({ldap_user_name_attr}={username}){ldap_user_filter})"
    conn.search(search_base=ldap_search_base, search_filter=search_filter, attributes=[ldap_user_email_attr])

    if conn.entries:
      email = str(conn.entries[0][ldap_user_email_attr])
      conn.unbind()
      return email
    else:
       conn.unbind()
       return None
```
This snippet illustrates how to pull additional attributes, in this case, the user’s email from LDAP and it's also called when a new user tries to log in the first time and there is no user in airflow database with such username.

**Snippet 3: Group Membership Handling**

```python
from flask_ldap3_login import LDAP3LoginManager
from airflow.configuration import conf
import ldap3

def get_user_groups(username):
    ldap_uri = conf.get("ldap", "ldap_uri")
    ldap_bind_user = conf.get("ldap", "ldap_bind_user")
    ldap_bind_password = conf.get("ldap", "ldap_bind_password")
    ldap_search_base = conf.get("ldap", "ldap_search_base")
    ldap_user_dn_template = conf.get("ldap", "ldap_user_dn_template")
    ldap_group_base = conf.get("ldap", "ldap_group_base")
    ldap_group_name_attr = conf.get("ldap", "ldap_group_name_attr")
    ldap_group_filter = conf.get("ldap", "ldap_group_filter")
    ldap_group_member_attr = conf.get("ldap", "ldap_group_member_attr")

    ldap_manager = LDAP3LoginManager()
    ldap_server = ldap3.Server(ldap_uri, get_info=ldap3.ALL)
    conn = ldap3.Connection(ldap_server, user=ldap_bind_user, password=ldap_bind_password)
    conn.bind()

    user_dn = ldap_user_dn_template.format(username)
    groups = []
    search_filter = f"(&{ldap_group_filter}({ldap_group_member_attr}={user_dn}))"
    conn.search(search_base=ldap_group_base, search_filter=search_filter, attributes=[ldap_group_name_attr])
    if conn.entries:
        for entry in conn.entries:
            groups.append(str(entry[ldap_group_name_attr]))
    conn.unbind()
    return groups
```
This final snippet shows how you'd pull group information for authorization purposes. This typically happens in conjunction with Airflow's role management, which can be configured to assign user roles based on their group membership. This requires further coding in the `auth_manager.py` file within the airflow provider, but the principle of using the data from LDAP is here.

For deeper understanding, I suggest exploring the official documentation for Apache Airflow and `flask-ldap3-login`. In addition, "Understanding LDAP" by Timothy A. Howes et al., is a great resource to get deep into the LDAP protocols themselves. Another highly useful book is "LDAP System Administration" by Gerald Carter which dives into the practical aspects of LDAP administration.

Remember, security is a journey, not a destination. Always test your configuration thoroughly, and make sure your service account has only the necessary permissions. This careful approach will prevent a whole heap of problems in the long run.
