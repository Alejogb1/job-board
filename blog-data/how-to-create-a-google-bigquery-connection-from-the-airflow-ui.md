---
title: "How to create a Google BigQuery connection from the Airflow UI?"
date: "2024-12-23"
id: "how-to-create-a-google-bigquery-connection-from-the-airflow-ui"
---

Alright, let's dive into establishing a Google BigQuery connection via the Airflow UI. I've tackled this exact challenge a few times in past projects, notably when we were migrating legacy data pipelines from on-premises infrastructure to a cloud-based setup. The intricacies can sometimes feel a bit layered, but it's manageable with a solid understanding of the underlying components.

Initially, you might think it's just about dropping in credentials, but there’s more to it than that, especially when considering best practices for security and maintainability. Essentially, the process boils down to configuring an Airflow connection that specifies how to authenticate with Google Cloud Platform (GCP) and access BigQuery. The approach hinges on leveraging Airflow’s connection management and typically relies on service account keys for authentication, though other methods are possible, like workload identity federation, which is worth exploring in more secure environments.

First off, before even touching the Airflow UI, you'll need a service account in GCP with the correct permissions to interact with BigQuery. This is crucial. Avoid using personal accounts or giving the service account overly broad permissions; stick to the principle of least privilege. Specifically, the service account should have roles such as `roles/bigquery.dataEditor` or `roles/bigquery.jobUser`, depending on the specific tasks your workflows will perform in BigQuery. I usually prefer `roles/bigquery.jobUser` for most job submissions and then grant more specific roles on an as-needed basis. This separation enhances security and troubleshooting down the line.

Once the service account is set up and has the requisite permissions, you need to generate a JSON key file for it. This is the critical piece for authentication. *Do not* commit this key file directly to your version control system. Instead, store it securely, for instance, using Airflow’s secrets management backend (like HashiCorp Vault or GCP Secret Manager), and then configure the Airflow connection to access it from there.

Now, regarding creating the connection via the Airflow UI, the workflow goes something like this: navigate to the ‘Admin’ section, then ‘Connections’. Click on ‘Create’ and select ‘Google Cloud Platform’ as the connection type. Here is where the specifics come in. You’ll need to provide the following:

1.  **Conn Id:** A descriptive identifier for your connection, like `gcp_bigquery_conn` or something project specific like `data_ingestion_bigquery_conn`.
2.  **Conn Type:** Select ‘Google Cloud Platform’.
3.  **Project Id:** The id of the Google Cloud Project where BigQuery resides.
4.  **Keyfile JSON:** This is the section where things can vary. If you opt to securely store your key in a secrets backend, you will not directly paste the json contents here. Instead, you’d leave it empty and provide the necessary configuration in the "extra" section.

The "extra" section is a crucial aspect that often gets overlooked. It allows you to configure various authentication parameters. When using a secrets manager, you usually specify the path to the key file. For example, if you are using the secrets backend available on GCP, it might look something like:

```json
{
  "key_path": "projects/<your_project_id>/secrets/<secret_name>/versions/latest",
  "use_application_default_credentials": false
}
```
Note, you should set `use_application_default_credentials` to false. Also, notice the key is stored as a secret on gcp secret manager using it's path instead of storing the raw json. Alternatively, if you opt to store the entire key directly in the Airflow connections you might do this.

```json
{
  "keyfile_dict": {
  	  "type": "service_account",
  	  "project_id": "your-project-id",
  	  "private_key_id": "your-private-key-id",
  	  "private_key": "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY\n-----END PRIVATE KEY-----\n",
  	  "client_email": "your-service-account-email",
  	  "client_id": "your-service-account-client-id",
  	  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  	  "token_uri": "https://oauth2.googleapis.com/token",
  	  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  	  "client_x509_cert_url": "your-service-account-certificate-url"
    }
}
```

Remember, the `keyfile_dict` should be a properly escaped JSON string representing your service account’s private key. While the direct storage works, be extremely cautious with it. I strongly advise using an external secrets manager. It's far more secure and aligns better with industry standards.

Thirdly, if you're implementing workload identity federation (WIF), the configuration will be substantially different and you must set `use_application_default_credentials: true` and you will specify the properties needed to create the WIF token on the "extra" parameter. This is a more advanced configuration, and details can vary significantly depending on your environment.

```json
{
    "use_application_default_credentials": true,
    "impersonation_chain": [
      "projects/your-project/serviceAccounts/your-service-account-email@your-project.iam.gserviceaccount.com"
    ],
      "credentials": {
        "type": "external_account",
        "audience": "//iam.googleapis.com/projects/your-project/locations/global/workloadIdentityPools/your-pool-id/providers/your-provider-id",
        "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
        "token_url": "https://sts.googleapis.com/v1/token",
        "service_account_impersonation": {
        	"target_principal": "projects/your-project/serviceAccounts/your-service-account-email@your-project.iam.gserviceaccount.com"
       },
       "workload_identity_pool_id": "your-pool-id",
       "provider_id": "your-provider-id",
       "credential_source": {
          "file": "/var/run/secrets/kubernetes.io/serviceaccount/token"
       }
   }
}
```
This will allow your Airflow pod to make use of the workload identity federation token on the pod.

After setting up the connection, it's crucial to test it. You can do this within the Airflow UI by selecting the connection and pressing "Test". This verifies that Airflow can successfully authenticate with GCP using the provided details. This step can save you hours of debugging later down the road. If the connection test fails, carefully double-check the service account permissions, the key file, and the 'extra' configurations. Errors in any of these will typically cause authentication failures.

To gain a deeper understanding, I strongly suggest exploring Google's official documentation on service accounts and IAM roles. Specifically, review the documentation on how to create, manage, and secure service accounts. For detailed insights on workload identity federation, look at the guides provided by Google Cloud. Additionally, consider reviewing 'Designing Data-Intensive Applications' by Martin Kleppmann, which, although not specific to Airflow or GCP, provides valuable context on data architecture patterns and security implications which apply to these situations. Also, 'Effective DevOps' by Jennifer Davis and Ryn Daniels is a worthwhile read, as it discusses infrastructure as code and secrets management concepts, which are very relevant.

Finally, remember, a well-configured Airflow connection is not only about functionality; it's also about following sound security practices and promoting maintainability. These efforts pay off significantly as your data pipelines grow and scale. By using a secrets backend, regularly rotating keys, and following the principle of least privilege, you'll create a robust system that stands the test of time. This also significantly improves your team's productivity in the long run because your infrastructure is more maintainable and less error-prone.
