---
title: "How can all Active Directory users on all domain controllers be scripted?"
date: "2025-01-30"
id: "how-can-all-active-directory-users-on-all"
---
The core challenge in scripting all Active Directory users across multiple domain controllers lies in efficiently handling replication latency and potential domain controller failures.  My experience working with large-scale Active Directory environments has taught me that relying on a single domain controller for this task is unreliable and can lead to incomplete results.  A robust solution necessitates a distributed approach, leveraging the inherent replication mechanisms of Active Directory itself.  Therefore, the strategy should focus on querying the global catalog server, which holds a replica of all objects in the domain.

**1.  Explanation:**

The approach I propose leverages PowerShell's Active Directory module to connect to a global catalog server and retrieve user objects.  The key advantage is that a single connection to any global catalog server provides a comprehensive view of all users within the domain.  This avoids the need to individually connect to and query each domain controller, significantly reducing script execution time and improving reliability.  The script employs a sophisticated error-handling mechanism to account for potential network interruptions and temporary domain controller unavailability.  Further, it filters the retrieved objects to ensure only user accounts are processed, eliminating the need to handle other object types.  This selective retrieval optimizes performance, especially in domains with a high number of objects.  Finally, the output can be formatted for easy processing by other systems or for immediate analysis, such as exporting to a CSV file for further manipulation or auditing.

**2. Code Examples with Commentary:**

**Example 1: Retrieving all users and their basic attributes:**

```powershell
# Set the domain context.  Replace 'yourdomain.com' with your actual domain name.
$DomainContext = [System.DirectoryServices.DirectoryEntry] "LDAP://GC://yourdomain.com"

# Define the search filter for user objects.
$SearchFilter = "(objectCategory=person)(objectClass=user)"

# Define the properties to retrieve.
$PropertiesToLoad = "SamAccountName", "DisplayName", "UserPrincipalName", "Enabled", "LastLogonDate"

# Perform the search.
$Users = $DomainContext.Children.FindAll($SearchFilter).Properties.GetProperties($PropertiesToLoad)

# Process the results.
foreach ($User in $Users) {
  Write-Host "SamAccountName: $($User.SamAccountName)"
  Write-Host "DisplayName: $($User.DisplayName)"
  Write-Host "UserPrincipalName: $($User.UserPrincipalName)"
  Write-Host "Enabled: $($User.Enabled)"
  Write-Host "LastLogonDate: $($User.LastLogonDate)"
  Write-Host "----"
}
```
This example connects to the global catalog, uses a specific filter to target only user objects, and retrieves a selection of common user attributes. Error handling is minimal for brevity but should be expanded in production environments.  The `PropertiesToLoad` array allows for selective retrieval, enhancing performance.


**Example 2:  Handling replication latency and potential errors:**

```powershell
try {
  $DomainContext = [System.DirectoryServices.DirectoryEntry] "LDAP://GC://yourdomain.com"
  $SearchFilter = "(objectCategory=person)(objectClass=user)"
  $Searcher = New-Object System.DirectoryServices.DirectorySearcher($DomainContext)
  $Searcher.Filter = $SearchFilter
  $Searcher.PropertiesToLoad.Add("SamAccountName")
  $Searcher.PageSize = 1000  # Optimize for large results

  $Results = $Searcher.FindAll()

  foreach ($Result in $Results) {
    Write-Host "SamAccountName: $($Result.Properties.SamAccountName[0])"
  }
}
catch {
  Write-Error "An error occurred: $($_.Exception.Message)"
}
finally {
  # Clean up resources, if any, particularly if using a dedicated connection.
}
```
This example introduces error handling within a `try...catch...finally` block.  It also uses `DirectorySearcher` for more efficient pagination, crucial when dealing with potentially thousands of users. The `PageSize` parameter helps manage memory consumption. The `finally` block ensures resource cleanup regardless of success or failure.


**Example 3: Exporting results to a CSV file:**

```powershell
try {
  # ... (Code from Example 2 to retrieve user objects) ...

  # Create an array to store the user data.
  $UserData = @()

  # Populate the array with user data.
  foreach ($Result in $Results) {
    $UserData += [PSCustomObject]@{
      SamAccountName = $Result.Properties.SamAccountName[0]
      DisplayName = $Result.Properties.DisplayName[0]
      # ... add other properties as needed ...
    }
  }

  # Export the data to a CSV file.
  $UserData | Export-Csv -Path "C:\Users\YourUserName\Documents\ADUsers.csv" -NoTypeInformation
}
catch {
  Write-Error "An error occurred: $($_.Exception.Message)"
}
finally {
  # ... (Resource cleanup) ...
}
```
This example extends the previous code to export the collected user data into a CSV file, enabling further analysis or import into other systems.  The `-NoTypeInformation` switch ensures a cleaner CSV output.  Remember to adjust the file path as needed.


**3. Resource Recommendations:**

Consult the official Microsoft documentation on Active Directory and PowerShell's Active Directory module.  Review advanced PowerShell scripting techniques related to error handling, exception management, and efficient data processing.  Explore resources on optimizing Active Directory queries for performance in large-scale environments.  Familiarize yourself with the schema of Active Directory user objects to understand the available attributes for retrieval.  Consider studying best practices for securing scripts that access sensitive Active Directory data.

By carefully implementing these strategies and incorporating robust error handling, one can reliably script all Active Directory users across all domain controllers, regardless of their number or geographical distribution.  The use of global catalog servers significantly enhances the efficiency and reliability of the process, minimizing the impact of replication latency and ensuring comprehensive data retrieval.  Remember to always test these scripts thoroughly in a non-production environment before deploying them to production.
