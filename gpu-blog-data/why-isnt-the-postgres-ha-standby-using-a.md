---
title: "Why isn't the Postgres HA standby using a certificate?"
date: "2025-01-30"
id: "why-isnt-the-postgres-ha-standby-using-a"
---
The failure of a PostgreSQL High Availability (HA) standby server to utilize a certificate for secure communication is almost invariably rooted in a misconfiguration within the `pg_hba.conf` file, specifically concerning the authentication method employed for connections from the standby server to the primary.  My experience troubleshooting this issue across numerous production deployments consistently points to this single, often overlooked, detail.  Improperly configured authentication bypasses the certificate verification mechanism, rendering the certificate itself functionally useless.

**1. Clear Explanation:**

PostgreSQL's HA setup, typically using streaming replication, relies on secure communication between the primary and standby servers.  While a certificate might be installed and configured on both servers, the connection process is governed by the authentication method specified in `pg_hba.conf`.  This file dictates which authentication mechanisms are accepted for connections from specific IP addresses or networks.  If a method that doesn't inherently involve certificate verification – like `trust`, `password`, or `peer` – is specified, the server will accept the connection regardless of the certificate’s presence or validity.  The certificate remains unused, leaving the communication channel vulnerable.  The expectation that merely installing the certificate on both nodes guarantees secure communication is inaccurate; it's the authentication method that determines whether the certificate is actually employed.

To successfully use a certificate, the `pg_hba.conf` file must explicitly direct the connection to use `cert` authentication.  This forces the server to verify the client certificate presented during the connection attempt.  Failure to configure this appropriately leaves the replication process open to various security risks, negating the protective benefit of the deployed certificates.  Further, the certificate itself must be correctly configured on both the primary and standby servers, including the necessary certificate authority (CA) chain for verification.  Missing or improperly configured CA certificates can also cause failures even when `cert` authentication is specified.  Finally, the correct `sslmode` setting (typically `verify-ca` or `verify-full`) must be present in the connection string used by the standby.

**2. Code Examples with Commentary:**

**Example 1:  Incorrect `pg_hba.conf` configuration (trust-based authentication):**

```
# TYPE  DATABASE        USER            ADDRESS                 METHOD
host    all             all             127.0.0.1/32            trust
host    all             all             0.0.0.0/0               trust
host    replication     replicationuser 192.168.1.100/32       trust
```

This configuration uses `trust` authentication, meaning any connection from the specified IP addresses is accepted without verification.  Even if certificates are installed, they are ignored.  The standby server will connect, but the connection will be insecure.


**Example 2: Correct `pg_hba.conf` configuration (certificate-based authentication):**

```
# TYPE  DATABASE        USER            ADDRESS                 METHOD
host    all             all             127.0.0.1/32            peer
host    all             all             0.0.0.0/0               reject
host    replication     replicationuser 192.168.1.100/32       cert
```

This configuration explicitly uses `cert` authentication for the replication user (`replicationuser`) connecting from the standby server’s IP address (192.168.1.100).  All other connections are either allowed via `peer` authentication (for local connections) or rejected.  This setup correctly leverages the installed certificate for secure communication.

**Example 3:  Connection string with appropriate `sslmode`:**

This snippet demonstrates the client-side (standby) configuration required to interact with the server utilizing certificate-based authentication.  This needs to be configured within the `postgresql.conf` file on the standby server or within the connection string used by the replication process.

```sql
-- Assuming replicationuser connects to the primary at 192.168.1.101
--  with the certificate located at /path/to/replicationuser.crt
--  and connecting to the database named 'mydb'
--  The crucial aspect is sslmode=verify-ca or verify-full
'host=192.168.1.101 port=5432 dbname=mydb user=replicationuser password=replicationpassword sslmode=verify-ca sslcert=/path/to/replicationuser.crt sslrootcert=/path/to/ca.crt'
```

This example highlights the importance of `sslmode=verify-ca` (or `verify-full` for even stricter verification).  This ensures that the client-side actively attempts to verify the server's certificate using the provided CA certificate (`sslrootcert`).


**3. Resource Recommendations:**

The PostgreSQL documentation is indispensable for understanding authentication mechanisms and SSL configuration. The official documentation provides detailed explanations of `pg_hba.conf`,  SSL configuration options, and the intricacies of streaming replication.  Refer to the PostgreSQL manual for comprehensive guides on setting up and troubleshooting SSL and HA configurations.  Additionally, dedicated PostgreSQL administration guides offer practical advice and best practices for securing your database deployments.  Consulting these resources is vital for mastering the nuances of PostgreSQL security and ensuring robust HA setups.  Finally, a strong understanding of network security concepts, specifically concerning certificate management and PKI, is also essential for successful implementation.
