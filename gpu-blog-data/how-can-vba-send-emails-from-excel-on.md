---
title: "How can VBA send emails from Excel on less secure devices?"
date: "2025-01-30"
id: "how-can-vba-send-emails-from-excel-on"
---
The challenge of sending emails via VBA from Excel on less secure devices, particularly those lacking modern authentication protocols, stems from the evolution of email security standards. Many legacy systems or personal setups don't support OAuth 2.0 or similar modern authentication mechanisms. Instead, they often rely on older protocols like SMTP with basic authentication, which contemporary email providers increasingly deprecate or outright block due to security risks. My work supporting legacy office systems has frequently encountered this issue.

The core of the problem lies in the limitations imposed by the email server’s security configuration and the authentication capabilities available within VBA. Modern email providers, such as Gmail or Outlook.com, require OAuth 2.0 authentication for programmatic access, which involves token-based authorization instead of direct username and password transmission. VBA, particularly in older Excel versions, lacks built-in support for this modern authentication process. Consequently, directly attempting to use SMTP with basic authentication (username and password) often fails, resulting in connection errors or emails being blocked by the provider. The challenge, then, becomes how to circumvent these modern security requirements while still enabling VBA to send emails, specifically on older or less secure devices.

The most straightforward workaround involves using an SMTP server that still supports basic authentication or relaying through an email service that allows application-specific passwords. The former is becoming less common, so the focus must shift to the latter, which often requires a dedicated setup step from the user. This process typically involves logging into the email account's security settings online, generating a unique application password, and using that password within the VBA code instead of the user’s primary password. This approach adds a layer of separation and control, making it somewhat less risky than directly storing the primary account password in a VBA script. However, It does not represent ideal security practice but it allows to achieve the task in less secure scenarios.

The underlying logic in VBA code involves creating an `Outlook.Application` object, a `MailItem` object, defining the recipient(s), subject, and body, and then using the `.Send` or `.Display` method to either send the email immediately or display it to the user for review. However, when the email server requires more than basic authentication, VBA needs to use a different approach. The first example demonstrates the basic, potentially problematic, SMTP method. The second and third code examples show how to modify this to leverage a specific email address and application password.

**Code Example 1: The Basic SMTP Approach (Generally Not Recommended)**

```vba
Sub SendEmailBasic()
    Dim objOutlook As Object
    Dim objMail As Object

    Set objOutlook = CreateObject("Outlook.Application")
    Set objMail = objOutlook.CreateItem(0)

    With objMail
        .To = "recipient@example.com"
        .Subject = "Test Email from VBA"
        .Body = "This is a test email sent using basic SMTP configuration."
        .Send ' or .Display to show before sending
    End With

    Set objMail = Nothing
    Set objOutlook = Nothing

End Sub
```

This basic code utilizes the user's default Outlook configuration and attempts to send a message, assuming SMTP with basic authentication. This will fail with modern email providers. In my experience, trying this with Gmail will throw an error relating to the authentication mechanism. It will only work in setups where your email client and your server allow username and password authentication without modern security practices.

**Code Example 2: Using Specific SMTP Server and Credentials**

```vba
Sub SendEmailWithSMTP()
    Dim objOutlook As Object
    Dim objMail As Object
    Dim objConf As Object

    Set objOutlook = CreateObject("Outlook.Application")
    Set objMail = objOutlook.CreateItem(0)
    Set objConf = objOutlook.Session.Accounts.Item(1) ' Assuming default account is required, user might need to adjust

    With objMail
        .To = "recipient@example.com"
        .Subject = "Email using app password"
        .Body = "This is a test email sent with specific SMTP settings and app password."

        .Configuration.Item("http://schemas.microsoft.com/mapi/id/{00020301-0000-0000-C000-000000000046}/") = "smtp.example.com"  ' SMTP Server Address
        .Configuration.Item("http://schemas.microsoft.com/mapi/id/{00020303-0000-0000-C000-000000000046}/") = 587  ' SMTP Port
        .Configuration.Item("http://schemas.microsoft.com/mapi/id/{0002030a-0000-0000-C000-000000000046}/") = "sender@example.com"  ' Sender Email Address (same as account)
        .Configuration.Item("http://schemas.microsoft.com/mapi/id/{00020306-0000-0000-C000-000000000046}/") = "app_password" ' Application password
        .Configuration.Item("http://schemas.microsoft.com/mapi/id/{0002030b-0000-0000-C000-000000000046}/") = 2  ' Use SSL/TLS
        .Send

    End With


    Set objMail = Nothing
    Set objOutlook = Nothing
    Set objConf = Nothing
End Sub
```

In this example, I am directly setting specific email server settings using an application-specific password. Crucially, the `app_password` placeholder would need to be replaced by the password generated in the email account's settings. The server address (`smtp.example.com`), port (587), and sender's email address (`sender@example.com`) must also be configured correctly. Moreover the index 1 in `objOutlook.Session.Accounts.Item(1)` should correspond with the selected default account in outlook, otherwise will not work properly. Also, note that the `Configuration.Item` keys are GUIDs referencing to the needed parameters, these must not be altered.  The final `2` value, indicates SSL/TLS usage.

**Code Example 3: Using Outlook Account and App Password**

```vba
Sub SendEmailAppPass()
    Dim objOutlook As Object
    Dim objMail As Object
    Dim objAcc As Object

    Set objOutlook = CreateObject("Outlook.Application")
    Set objMail = objOutlook.CreateItem(0)


    ' Specify the account to use (replace with correct account name if necessary)
    Set objAcc = GetAccountByDisplayName("sender@example.com", objOutlook)
     If objAcc Is Nothing Then
            MsgBox "Account not found!"
            Exit Sub
    End If

     With objMail
        .To = "recipient@example.com"
        .Subject = "Email using app password from specific account"
        .Body = "This is a test email sent with app password using an Outlook account."
        .SentOnBehalfOfName = "sender@example.com"
        .SaveSentMessageFolder = objAcc.SentMailFolder 'Store in sent items in sender mailbox
         
         .Send
    End With

    Set objMail = Nothing
    Set objOutlook = Nothing
    Set objAcc = Nothing
End Sub

Function GetAccountByDisplayName(accountName As String, outlookApp As Object) As Object
    Dim account As Object
    For Each account In outlookApp.Session.Accounts
        If account.DisplayName = accountName Then
            Set GetAccountByDisplayName = account
            Exit Function
        End If
    Next account
    Set GetAccountByDisplayName = Nothing
End Function

```

This last example improves on the previous one, as it selects the account dynamically based on its display name. The `GetAccountByDisplayName` function loops through all accounts and returns the correct one based on the name passed in the function, in this case `sender@example.com`. The email is sent on behalf of the provided account using `.SentOnBehalfOfName = "sender@example.com"`, and will also be stored inside the selected user's sent items folder with the line `.SaveSentMessageFolder = objAcc.SentMailFolder`. This last example is more user friendly, and better to use in multi-account scenarios. As in example 2, the application password must be set in the corresponding email provider settings.

These examples highlight various approaches, but they all require careful consideration of the email server configuration, the availability of an application password, and the understanding of Outlook's object model for VBA. It's imperative to avoid hardcoding sensitive information such as passwords directly in the VBA code. Storing these externally or using a password vault may be useful, but that is outside the scope of the original question.

For further study and understanding of the concepts discussed, exploring online documentation related to the Outlook object model in VBA, SMTP protocol settings, and email security practices is strongly recommended. Specifically, researching Microsoft's documentation regarding the Outlook object model will be invaluable for understanding object properties and methods related to email sending. Additionally, understanding modern email authentication processes through educational resources covering OAuth 2.0 will offer additional perspective. Furthermore, investigating the security guidelines and recommendations from your email provider regarding the use of application passwords will provide the specific steps needed to perform the authentication. I have found both the official Microsoft documentation and other online resources, especially related to SMTP, highly useful.
