---
title: "How to Authenticate an Azure ContainerRegistryClient with a scope mapped token username/password?"
date: "2024-12-15"
id: "how-to-authenticate-an-azure-containerregistryclient-with-a-scope-mapped-token-usernamepassword"
---

alright, so you're hitting the classic problem of wanting to access an azure container registry (acr) using a service principal and a token that's scoped down to just what it needs, instead of using, say, the admin user or a full access key. been there, done that, got the t-shirt (and several all-nighters). let's unpack this.

the crux of it is that the azure container registry client library, specifically the `containerregistryclient` class, needs some form of authentication to know who's calling and whether they're allowed to do what they're asking. using a username and password combo with a scoped token is a pretty secure way to go, and it's absolutely what you should be doing in any kind of production environment. hardcoding keys is just asking for trouble (lessons learned the hard way, i assure you).

i remember once, back in my early days, i thought i could just drop a full access key into the ci/cd pipeline. well, let's just say that key didn't stay private for long. a script i had running in a development environment got a bit too curious and decided to expose everything i was using in the console, and it ended up on github. the sheer panic i felt that day... i don't want that for anyone. since then, i've become a strong advocate for minimal access tokens.

so, how do we do this with the `containerregistryclient`? it's all about crafting the correct credentials and passing them in at the client's initialization stage. the key isn't just the token, it's the fact that we need to understand how the client is expecting the username and password to be structured. the username must be the "token" string, and the password must be the token itself. that's all. i mean, it's straightforward but definitely not obvious if you haven't done it before. this is the classic case of needing to read the documentation or the source code. or stackoverflow, of course.

let me give you a python example:

```python
from azure.containerregistry import ContainerRegistryClient
from azure.identity import ClientSecretCredential

def get_acr_client_with_scoped_token(
    acr_name,
    tenant_id,
    client_id,
    client_secret,
    scope
):

    #first we create the credentials for the service principal
    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret
    )

    #now we get the token from the credential using the scope
    token_result = credential.get_token(scope)

    #acr username and password creation
    token = token_result.token
    username = "token"
    password = token
    
    #finally we create the client with the username and password
    registry_uri = f"https://{acr_name}.azurecr.io"
    client = ContainerRegistryClient(
            registry_uri,
            username=username,
            password=password
        )

    return client
    

# how to use it
if __name__ == "__main__":
   # i'm using environment variables here for security
    import os
    acr_name = os.environ.get("ACR_NAME")
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    client_id = os.environ.get("AZURE_CLIENT_ID")
    client_secret = os.environ.get("AZURE_CLIENT_SECRET")
    scope = f"https://{acr_name}.azurecr.io/.default"
    client = get_acr_client_with_scoped_token(
        acr_name,
        tenant_id,
        client_id,
        client_secret,
        scope
    )

    # example operation, fetch the repository name
    repository_names = client.list_repository_names()
    for name in repository_names:
        print(name)

```

in this example, you will see i’m using `azure.identity` to handle the credential obtaining, it's a far superior way to manage credentials than hardcoding them anywhere. the `clientsecretcredential` class takes in your service principal's tenant id, client id, and secret. then, with the scope, i obtain a token that i can use to create an authentication tuple that is then used in the initialization of `containerregistryclient`. the scope is crucial. it specifies what resources this token has access to. for an acr, a good approach is to use the ".default" scope that targets the acr itself and grants access based on assigned roles. you then set the username to the string "token" and the password is the token you obtained from azure active directory (aad).

now, if you're more a javascript person, the process is very similar.  here's a quick typescript example using the `azure-container-registry` library and the `@azure/identity` library:

```typescript
import { ContainerRegistryClient } from "@azure/container-registry";
import { ClientSecretCredential } from "@azure/identity";

async function getAcrClientWithScopedToken(
  acrName: string,
  tenantId: string,
  clientId: string,
  clientSecret: string,
  scope: string
): Promise<ContainerRegistryClient> {
  const credential = new ClientSecretCredential(
    tenantId,
    clientId,
    clientSecret
  );
  const tokenResult = await credential.getToken(scope);
  const token = tokenResult.token;
  const username = "token";
  const password = token;

  const registryUri = `https://${acrName}.azurecr.io`;
  const client = new ContainerRegistryClient(
    registryUri,
    username,
    password
  );

  return client;
}


async function main() {
    // i'm using environment variables here for security
    const acrName = process.env.ACR_NAME as string;
    const tenantId = process.env.AZURE_TENANT_ID as string;
    const clientId = process.env.AZURE_CLIENT_ID as string;
    const clientSecret = process.env.AZURE_CLIENT_SECRET as string;
    const scope = `https://${acrName}.azurecr.io/.default`

    const client = await getAcrClientWithScopedToken(
        acrName,
        tenantId,
        clientId,
        clientSecret,
        scope
    );
    const repositoryNames = client.listRepositoryNames();

    for await (const name of repositoryNames){
        console.log(name);
    }

}
  
main().catch((err) => {
    console.error("An error occurred:", err);
    process.exit(1);
});

```

this typescript code mirrors the python example. it uses `clientsecretcredential` to authenticate the service principal, fetches a scoped token, and then initializes the `containerregistryclient` using the token as the password with username set to "token". nothing too shocking, although the async/await is more prominent as we're dealing with promises and i'm a firm believer in the benefits of promises.

and finally, here is one java example, using the azure sdk for java:

```java
import com.azure.identity.ClientSecretCredentialBuilder;
import com.azure.identity.ClientSecretCredential;
import com.azure.containers.containerregistry.ContainerRegistryClient;
import com.azure.containers.containerregistry.ContainerRegistryClientBuilder;

public class AcrClientExample {

    public static ContainerRegistryClient getAcrClientWithScopedToken(
            String acrName,
            String tenantId,
            String clientId,
            String clientSecret,
            String scope) {


        ClientSecretCredential credential = new ClientSecretCredentialBuilder()
            .tenantId(tenantId)
            .clientId(clientId)
            .clientSecret(clientSecret)
            .build();

        String token = credential.getToken(scope).getToken();
        String username = "token";
        String password = token;
        
       String registryUri = String.format("https://%s.azurecr.io", acrName);

        ContainerRegistryClient client = new ContainerRegistryClientBuilder()
           .endpoint(registryUri)
           .username(username)
           .password(password)
           .buildClient();

        return client;
    }

    public static void main(String[] args) {
        // i'm using environment variables here for security
        String acrName = System.getenv("ACR_NAME");
        String tenantId = System.getenv("AZURE_TENANT_ID");
        String clientId = System.getenv("AZURE_CLIENT_ID");
        String clientSecret = System.getenv("AZURE_CLIENT_SECRET");
        String scope = String.format("https://%s.azurecr.io/.default", acrName);

        ContainerRegistryClient client = getAcrClientWithScopedToken(
                acrName,
                tenantId,
                clientId,
                clientSecret,
                scope);
        
        client.listRepositoryNames().forEach(System.out::println);
    }
}
```

again, very similar, we're using `clientsecretcredential` and creating the `containerregistryclient` the same way we did before. notice that these examples are very similar across languages. the core pattern is the same, even if the syntax changes a bit. this similarity is something you'll notice across many of the azure sdks. they tend to follow the same general design patterns which i think is great.

so, to recap, the trick is to understand that the `containerregistryclient` accepts a "username" and "password", but the username should be the string "token", and the password is the aad token itself you got using a service principal with the correct scope. it’s not rocket science once you know the secret (i.e., read the documentation). i really do recommend reading the documentation for this particular library. its really well documented.

as for further reading, i would suggest the official azure documentation, it's improved a lot. look for the "authentication" section for `azure-sdk-for-python` `azure-sdk-for-js` and `azure-sdk-for-java` since this is where they really dive into the details on how to handle credentials. the "azure active directory documentation" can also be really useful to understand how oauth 2.0 works and the logic of scopes, but this might be a rabbit hole if you just want to solve the initial problem. additionally, the book "oauth 2.0 simplified" by aaron parecki is a really good resource if you want to go more in deep about the subject.

and finally, one more thing, i once spent a whole afternoon trying to figure out why my token wasn't working. turns out, the service principal's role assignment on the acr wasn't set to pull images. so always, always, double-check your role assignments. there’s nothing worse than a permissions issue disguised as a syntax error, trust me.
