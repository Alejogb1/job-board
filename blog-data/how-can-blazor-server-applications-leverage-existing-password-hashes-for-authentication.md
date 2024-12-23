---
title: "How can Blazor Server applications leverage existing password hashes for authentication?"
date: "2024-12-23"
id: "how-can-blazor-server-applications-leverage-existing-password-hashes-for-authentication"
---

Alright, let's tackle this one. Instead of launching straight into definitions, perhaps it's more useful to recount a scenario where this became critical. Years ago, during a migration of a legacy asp.net webforms application to a blazor server implementation, we faced exactly this challenge: a mountain of users with pre-existing password hashes. Simply forcing a password reset on everyone was, shall we say, not a politically viable solution, and ethically, it's a poor experience for users. We had to devise a method to use those legacy hashes seamlessly in our shiny new blazor app.

The crux of the matter is that blazor server, like most modern authentication systems, typically relies on identity frameworks (such as asp.net core identity) which often store *their own* hash formats. We can't directly plug in an arbitrarily formatted hash. The key here is understanding that authentication is essentially two-staged: first, the user presents a password, and second, *that* password is used to generate a hash, which is compared with the stored hash. We're not altering this core process, but rather focusing on how we can accommodate those pre-existing hashes *during the comparison* stage.

Essentially, what we need to do is create a custom password hasher. Asp.net core provides an abstraction for this, `ipasswordhasher<tuser>`. This interface allows us to override the hash creation and verification logic. In our situation, it meant implementing a custom hasher that understood the legacy hashing algorithm. To do this effectively, we first had to know exactly what hashing algorithm the legacy system was using. That could have been anything from md5, to sha1, to something proprietary. In my case, it was an earlier version of sha256 with a specific salt generation strategy. Once that is known, implementation can begin.

Here's a simplified code example showing how you might approach this. Assume for the moment, the legacy system used sha1:

```csharp
using Microsoft.AspNetCore.Identity;
using System.Security.Cryptography;
using System.Text;

public class LegacySha1PasswordHasher : IPasswordHasher<IdentityUser>
{
    public string HashPassword(IdentityUser user, string password)
    {
        // This would be for new passwords; in our case, this won't be used
        // if you want users to be able to change their password and use a new system hash
        // you would implement this as well.
        throw new NotImplementedException("Not used for legacy hashing");
    }


    public PasswordVerificationResult VerifyHashedPassword(IdentityUser user, string hashedPassword, string providedPassword)
    {
         using (SHA1 sha1 = SHA1.Create())
         {
            var salt = GetSalt(user.Id);
            var combined =  Encoding.UTF8.GetBytes(salt + providedPassword);
            byte[] hashedBytes = sha1.ComputeHash(combined);
            string providedHash = Convert.ToBase64String(hashedBytes);

            if (hashedPassword == providedHash)
            {
                return PasswordVerificationResult.Success;
            }
            else
            {
                return PasswordVerificationResult.Failed;
            }
        }
    }
        private string GetSalt(string userId)
    {
      // This is an example of how to retrieve salt, modify as needed
      // In reality it may be stored in database, or derived from userId
        return userId.Substring(0, 8);
    }
}
```

Notice the `HashPassword` method is essentially a stub because we are not hashing passwords using the legacy method. We want to retain asp.net identity's hashing strategy for new passwords. The crux of the work is in `verifyhashedpassword`. The method retrieves the salt based on `userid` - this is for demonstration, in a real system, this would involve retrieving it from wherever the original application was getting it (database, some calculation, etc). It then takes the provided password, combines it with the salt and calculates the hash using sha1, and checks if it matches the stored hash. If it matches, we return `success`. Otherwise, `failed`.

To make this active, you need to configure it within the services configuration in `startup.cs` (or `program.cs` in .net 6+):

```csharp
// In ConfigureServices(...)
services.AddIdentity<IdentityUser, IdentityRole>()
    .AddEntityFrameworkStores<ApplicationDbContext>()
    .AddDefaultTokenProviders()
    .AddPasswordHasher<IdentityUser, LegacySha1PasswordHasher>();
```

Here's another example, assuming your legacy system used bcrypt:

```csharp
using Microsoft.AspNetCore.Identity;
using BCrypt.Net;

public class LegacyBcryptPasswordHasher : IPasswordHasher<IdentityUser>
{
    public string HashPassword(IdentityUser user, string password)
    {
         throw new NotImplementedException("Not used for legacy hashing");
    }

    public PasswordVerificationResult VerifyHashedPassword(IdentityUser user, string hashedPassword, string providedPassword)
    {
        try {
          if (BCrypt.Net.BCrypt.Verify(providedPassword, hashedPassword))
          {
               return PasswordVerificationResult.Success;
          }
         else{
               return PasswordVerificationResult.Failed;
           }
        }
        catch{
             return PasswordVerificationResult.Failed;
       }
    }
}
```

And the same change to the startup file:

```csharp
 services.AddIdentity<IdentityUser, IdentityRole>()
    .AddEntityFrameworkStores<ApplicationDbContext>()
    .AddDefaultTokenProviders()
    .AddPasswordHasher<IdentityUser, LegacyBcryptPasswordHasher>();
```

In this bcrypt example, we're using the `BCrypt.Net` nuget package, which handles salt management internally. The beauty of bcrypt is that you often don't have to retrieve a salt - it's incorporated into the hash itself. Notice the `try...catch` as the bcrypt verification can throw exceptions when given improperly formatted data, so it's important to handle this.

One more illustration; if the legacy system used some kind of *custom* salt and hashing algorithm:

```csharp
using Microsoft.AspNetCore.Identity;
using System.Security.Cryptography;
using System.Text;
using System.Linq;

public class LegacyCustomPasswordHasher : IPasswordHasher<IdentityUser>
{

    public string HashPassword(IdentityUser user, string password)
    {
         throw new NotImplementedException("Not used for legacy hashing");
    }


    public PasswordVerificationResult VerifyHashedPassword(IdentityUser user, string hashedPassword, string providedPassword)
    {
         var salt = GetCustomSalt(user); //Retrieves salt
        var combinedString =  salt + providedPassword;
         var bytes = Encoding.UTF8.GetBytes(combinedString);
         var hash = customHash(bytes);
       var providedHash = Convert.ToBase64String(hash);

        if (hashedPassword == providedHash)
           {
             return PasswordVerificationResult.Success;
          }
        else
            {
             return PasswordVerificationResult.Failed;
          }

    }

    private byte[] customHash(byte[] data){
        // this is a *very* simplified and non-secure hashing algorithm, replace with actual code
        return data.Reverse().ToArray();
    }
     private string GetCustomSalt(IdentityUser user)
    {
        // Replace with the actual way the salt is retrieved for the specific legacy system
          return user.Id.Reverse().ToString();
    }

}
```
and again, the change in startup:

```csharp
 services.AddIdentity<IdentityUser, IdentityRole>()
    .AddEntityFrameworkStores<ApplicationDbContext>()
    .AddDefaultTokenProviders()
    .AddPasswordHasher<IdentityUser, LegacyCustomPasswordHasher>();
```

Here you see the most flexible example where I've placed a placeholder method `customHash`, this highlights that you could do any kind of complex password verification, as long as you provide the implementation for that. It's *crucial* that this code precisely replicates the original hashing implementation, down to the salt strategy, string encoding, and every step of the hashing calculation itself.

This approach allows you to seamlessly bridge the old authentication system into the new one without forcing users to reset their passwords. It's not ideal, since you're still relying on potentially older and less secure hashing algorithms for existing passwords, so as part of the migration it's best to inform the users they should update their password and transition them to asp.net's standard password hashes over time. This is accomplished by using the default identity hasher when a user sets a new password.

For further reading on cryptographic best practices, I highly recommend "Cryptography Engineering" by Niels Ferguson, Bruce Schneier, and Tadayoshi Kohno. It provides a solid foundation for understanding how cryptographic primitives are used in real-world applications. Additionally, the official Microsoft documentation on asp.net core identity provides detailed insights into password hashing and the `ipasswordhasher` interface, as well as best practices for handling sensitive data. Also reviewing the documentation for any of the third party hash packages you use, such as BCrypt, is essential to make sure you understand their recommendations. Remember, when dealing with sensitive data like passwords, precision and a strong understanding of the underlying cryptographic mechanisms are absolutely paramount.
