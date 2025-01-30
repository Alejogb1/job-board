---
title: "Why does a Jenkins Docker container behind a proxy fail to download plugins due to a certificate error?"
date: "2025-01-30"
id: "why-does-a-jenkins-docker-container-behind-a"
---
The crux of Jenkins Docker container plugin download failures behind a proxy, exhibiting certificate errors, often stems from a mismatch between the Java Virtual Machine's (JVM) trusted certificate store within the container and the proxy's certificate. Specifically, the JVM, responsible for network communications within the Jenkins Docker image, relies on a pre-configured set of trusted certificate authorities (CAs). When a proxy employs a self-signed or internally generated certificate – a common practice for enterprise environments – the JVM, lacking the corresponding root CA certificate, flags the connection as untrusted and refuses to proceed, resulting in plugin download failures.

I've personally encountered this exact scenario multiple times during the deployment of Jenkins-based CI/CD pipelines within corporate networks where security policies mandate proxy usage and often internal certificate authorities. My approach to addressing this problem consistently involves manipulating the JVM's certificate store within the Docker container.

The primary mechanism for verifying secure connections (HTTPS) is through a trust chain involving certificates. When a client, in this case, the Jenkins JVM, initiates a connection to a server (e.g., a Jenkins plugin repository), the server presents its certificate. The client verifies the certificate's validity by tracing it back to a trusted root CA certificate. If the server's certificate is signed by an unrecognized or absent CA, the verification fails, resulting in a certificate error. When Jenkins attempts to download plugins via a proxy that employs a non-public CA, the JVM rejects the connection.

To circumvent this, we need to import the proxy’s root CA certificate into the Jenkins container's JVM trust store. The trust store is managed through a utility called `keytool`, provided with the Java installation. The general process is to first obtain the root CA certificate of the proxy, typically as a `.crt` or `.pem` file. This certificate needs to be copied into the Docker image and then imported into the JVM's trust store.

**Code Example 1: Dockerfile Modification**

The initial step involves modifying the Dockerfile of your Jenkins image. If you are using the official `jenkins/jenkins:lts` image, you will need to create a derived image incorporating the necessary changes:

```dockerfile
FROM jenkins/jenkins:lts

# Copy the proxy CA certificate into the image
COPY proxy-ca.crt /tmp/proxy-ca.crt

# Update cacerts with custom certificates
RUN keytool -import -trustcacerts -keystore "${JAVA_HOME}/jre/lib/security/cacerts" -storepass changeit -noprompt -alias proxy-ca -file /tmp/proxy-ca.crt

# Optionally, update the JVM options for using the system proxy settings
ENV JAVA_OPTS="-Dhttp.proxyHost=<your_proxy_host> -Dhttp.proxyPort=<your_proxy_port> -Dhttps.proxyHost=<your_proxy_host> -Dhttps.proxyPort=<your_proxy_port>"

# Remove the temporary certificate file
RUN rm /tmp/proxy-ca.crt
```

*   `FROM jenkins/jenkins:lts`: This line specifies the base Jenkins image.
*   `COPY proxy-ca.crt /tmp/proxy-ca.crt`: Copies your proxy’s root CA certificate to `/tmp` within the container. This file needs to be present in the same directory as your Dockerfile, named `proxy-ca.crt`, or modified accordingly in the copy instruction.
*   `RUN keytool ...`:  This is where the core modification occurs. `keytool` imports the certificate, trusting it for future connections. `-keystore "${JAVA_HOME}/jre/lib/security/cacerts"` specifies the location of the JVM trust store. `-storepass changeit` is the default password for the trust store. `-alias proxy-ca` sets a unique alias for the imported certificate. `-noprompt` avoids interactive input.
*  `ENV JAVA_OPTS ...`: This line configures JVM to use HTTP/HTTPS proxy settings for outbound network communication. Replace `<your_proxy_host>` and `<your_proxy_port>` with the actual proxy settings, if applicable.
*   `RUN rm /tmp/proxy-ca.crt`: Cleans up the temporary certificate file.

After making these changes, you’ll need to build the new image using `docker build -t <your_image_name> .`.  You'll then use the new image to start the Jenkins container.

**Code Example 2: Alternative Script-Based Import**

An alternative approach, useful for dynamic environments where rebuilding images is undesirable, involves running a script during the Jenkins container’s startup. This requires that the certificate is somehow made available at startup (e.g., mounted as a volume).

```bash
#!/bin/bash

# Wait for Jenkins to be ready before running the configuration
while ! wget -q --spider http://localhost:8080/login; do
    echo "Waiting for Jenkins to be ready..."
    sleep 5
done

echo "Jenkins is ready, proceeding to import certificate."

# Check if cacerts are already configured
if ! keytool -list -keystore "${JAVA_HOME}/jre/lib/security/cacerts" -storepass changeit -alias proxy-ca > /dev/null 2>&1; then

    # Import the proxy CA certificate into the keystore
    keytool -import -trustcacerts -keystore "${JAVA_HOME}/jre/lib/security/cacerts" -storepass changeit -noprompt -alias proxy-ca -file /var/jenkins_home/proxy-ca.crt

     echo "Certificate imported successfully."
 else
    echo "Certificate already imported."
fi


# Set Java Proxy Environment Variables if not already set.
if [ -z "$JAVA_OPTS" ]; then
    export JAVA_OPTS="-Dhttp.proxyHost=<your_proxy_host> -Dhttp.proxyPort=<your_proxy_port> -Dhttps.proxyHost=<your_proxy_host> -Dhttps.proxyPort=<your_proxy_port>"
    echo "Proxy Environment variables set."

elif ! echo "$JAVA_OPTS" | grep -q "http.proxyHost"
then
    export JAVA_OPTS="$JAVA_OPTS -Dhttp.proxyHost=<your_proxy_host> -Dhttp.proxyPort=<your_proxy_port> -Dhttps.proxyHost=<your_proxy_host> -Dhttps.proxyPort=<your_proxy_port>"
     echo "Proxy Environment variables appended."
else
    echo "Proxy Environment variables already present."
fi

exec "$@"

```

*   This script first waits until Jenkins is accessible. Then it checks if the certificate has been previously imported, avoiding redundant import. It then uses `keytool` to import the certificate located at `/var/jenkins_home/proxy-ca.crt` (assuming the file is mounted as a volume). Subsequently, it ensures the JAVA_OPTS environment variable is set.
*   To implement this, save the script as `setup.sh`, make it executable via `chmod +x setup.sh` and place it in the docker container’s entrypoint. The following should be appended to your `Dockerfile` to change the entrypoint: `COPY setup.sh /usr/local/bin/setup.sh` and `ENTRYPOINT ["/usr/local/bin/setup.sh", "/bin/tini", "--", "/usr/local/bin/jenkins.sh"]`. Ensure that `/var/jenkins_home/proxy-ca.crt` is mounted as a volume to ensure the certificate is available to this script. The execution of `jenkins.sh` is passed as arguments to the script and executed only after the script runs.
*   The proxy settings inside `JAVA_OPTS` are also applied during the startup.

**Code Example 3: Passing Certificate and Proxy via Environment Variables**

For maximum flexibility, the root CA certificate can be passed as an environment variable:

```dockerfile
FROM jenkins/jenkins:lts

# Install required base64 command line tool if not already present
RUN apt-get update && apt-get install -y base64

# Entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
```

and the entrypoint script `entrypoint.sh`:

```bash
#!/bin/bash

# Ensure JAVA_HOME is defined
if [ -z "$JAVA_HOME" ]; then
   export JAVA_HOME=/usr/java/openjdk-11
fi

# Wait for Jenkins to be ready
while ! wget -q --spider http://localhost:8080/login; do
    echo "Waiting for Jenkins to be ready..."
    sleep 5
done

echo "Jenkins is ready, proceeding to import certificate."

# Check if certificate is provided via environment variable
if [ -n "$PROXY_CA_CERT_BASE64" ]; then
  # Decode base64 encoded certificate
  echo "$PROXY_CA_CERT_BASE64" | base64 -d > /tmp/proxy-ca.crt
  # Import certificate, ensuring no duplications
  if ! keytool -list -keystore "${JAVA_HOME}/jre/lib/security/cacerts" -storepass changeit -alias proxy-ca > /dev/null 2>&1; then
        keytool -import -trustcacerts -keystore "${JAVA_HOME}/jre/lib/security/cacerts" -storepass changeit -noprompt -alias proxy-ca -file /tmp/proxy-ca.crt
        echo "Certificate imported successfully."
  else
    echo "Certificate already imported."
  fi
   rm /tmp/proxy-ca.crt
else
  echo "No certificate found in PROXY_CA_CERT_BASE64 variable. Skipping certificate import."
fi

# Set proxy parameters
if [ -z "$JAVA_OPTS" ]; then
    export JAVA_OPTS="-Dhttp.proxyHost=$PROXY_HOST -Dhttp.proxyPort=$PROXY_PORT -Dhttps.proxyHost=$PROXY_HOST -Dhttps.proxyPort=$PROXY_PORT"
    echo "Proxy settings added."
 elif ! echo "$JAVA_OPTS" | grep -q "http.proxyHost"
 then
     export JAVA_OPTS="$JAVA_OPTS -Dhttp.proxyHost=$PROXY_HOST -Dhttp.proxyPort=$PROXY_PORT -Dhttps.proxyHost=$PROXY_HOST -Dhttps.proxyPort=$PROXY_PORT"
     echo "Proxy settings appended."
 else
     echo "Proxy settings already present."
 fi

exec "$@"
```

*   This approach checks for a base64 encoded certificate in `PROXY_CA_CERT_BASE64`.  The proxy hostname and port are read from environment variables `PROXY_HOST` and `PROXY_PORT`.
*   This method enhances flexibility by avoiding changes to Dockerfile when a certificate needs changing. This also avoids the need to mount a volume for the certificate. The certificate itself is provided as a runtime environment variable. The execution of `jenkins.sh` is passed as arguments to the script and executed only after the script runs.

In situations involving proxy certificate errors, a deep dive into the JVM’s trust store is often necessary. While these examples address the core issue, the exact method might vary slightly based on your specific Jenkins and proxy configurations. Always refer to the documentation of your specific proxy server for detailed certificate retrieval methods.

For further information regarding certificates, keytool, and JVM options, consult the official Oracle Java documentation. Likewise, the official Jenkins documentation and Docker documentation provide information on Jenkins setup and Docker fundamentals, respectively. These references will enhance understanding and troubleshooting skills.
