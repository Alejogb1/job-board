---
title: "How can SQL mail be formatted using HTML?"
date: "2025-01-30"
id: "how-can-sql-mail-be-formatted-using-html"
---
SQL Server’s `sp_send_dbmail` procedure, while primarily designed for text-based notifications, offers robust support for HTML formatting within the email body. This capability significantly enhances the readability and usability of database alerts and reports sent via email. My experience supporting large-scale database systems has shown me that properly formatted HTML emails are crucial for quickly conveying information to various stakeholders, ranging from developers needing immediate error notifications to business users requiring regular performance reports. Achieving this, however, requires understanding how to construct the HTML content and the specific parameters of `sp_send_dbmail`.

The key to using HTML in SQL mail lies in the `@body` parameter of the `sp_send_dbmail` stored procedure and the `@body_format` parameter, which must be set to ‘HTML’. The `@body` parameter accepts a string, and this string can contain valid HTML markup. SQL Server itself doesn't interpret HTML in any special way; it simply passes the string containing HTML tags directly to the email subsystem. It's the email client of the recipient that interprets and renders the HTML. Therefore, valid HTML is crucial for predictable formatting. Moreover, the email client can introduce inconsistencies due to varying levels of HTML support across applications.

The following three code examples demonstrate practical ways to incorporate HTML formatting within database email notifications. They showcase how to utilize basic HTML structures, embedded CSS for styling, and a more sophisticated approach that might be used in data-driven reports.

**Example 1: Simple HTML Formatting for Error Notification**

This first example illustrates a simple error notification utilizing basic HTML elements to structure the message for improved clarity. It provides a clean and easily readable notification without excessive complexity.

```sql
DECLARE @Subject VARCHAR(255);
DECLARE @Body NVARCHAR(MAX);

SET @Subject = 'Database Error Notification';

SET @Body = N'<p><b>Error Occurred:</b></p>
               <p>An error was detected in the database.</p>
               <p><b>Error Details:</b></p>
               <ul>
                 <li><b>Date:</b> ' + CONVERT(VARCHAR(20), GETDATE(), 120) + '</li>
                 <li><b>Error Code:</b> 1234</li>
                 <li><b>Message:</b> A generic error message occurred. Please investigate.</li>
               </ul>
               <p><i>This is an automated notification.</i></p>';

EXEC msdb.dbo.sp_send_dbmail
	@profile_name = 'YourDatabaseMailProfile',  -- Replace with your profile name
    @recipients = 'recipient@example.com',     -- Replace with recipient's email address
    @subject = @Subject,
    @body = @Body,
    @body_format = 'HTML';
```

**Commentary on Example 1:**

*   The `@body` variable is declared as `NVARCHAR(MAX)` to accommodate potentially lengthy HTML strings.
*   Basic HTML tags like `<p>`, `<b>`, `<ul>`, and `<li>` are used to format the email body content.
*   The `CONVERT` function is used to insert the current date and time, demonstrating dynamic content.
*   The `sp_send_dbmail` procedure is called with `@body_format = 'HTML'`, which is critical for proper formatting.
*   The profile name and recipient email addresses should be replaced with environment specific values.

**Example 2: Embedding Inline CSS for Styling**

This example builds upon the first by incorporating inline CSS to add styling to the email, enhancing its visual appearance. This approach is generally more compatible with older email clients than linked stylesheets.

```sql
DECLARE @Subject VARCHAR(255);
DECLARE @Body NVARCHAR(MAX);

SET @Subject = 'Critical Database Alert';

SET @Body = N'<div style="font-family: Arial, sans-serif; color: #333;">
               <h2 style="color: #FF0000; border-bottom: 1px solid #ddd; padding-bottom: 5px;">Critical Database Alert</h2>
               <p style="margin-bottom: 10px;">A critical event has been detected in the database.</p>
               <table style="width: 100%; border-collapse: collapse;">
                   <tr style="background-color: #f0f0f0;">
                       <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Field</th>
                       <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Value</th>
                   </tr>
                   <tr>
                       <td style="padding: 8px; border: 1px solid #ddd;">Event Time</td>
                       <td style="padding: 8px; border: 1px solid #ddd;">' + CONVERT(VARCHAR(20), GETDATE(), 120) + '</td>
                   </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">Event Type</td>
                        <td style="padding: 8px; border: 1px solid #ddd;">High Severity</td>
                   </tr>
               </table>
               <p style="font-style: italic; margin-top: 15px;">Please take immediate action.</p>
             </div>';

EXEC msdb.dbo.sp_send_dbmail
	@profile_name = 'YourDatabaseMailProfile', -- Replace with your profile name
    @recipients = 'recipient@example.com',     -- Replace with recipient's email address
    @subject = @Subject,
    @body = @Body,
    @body_format = 'HTML';
```

**Commentary on Example 2:**

*   Inline CSS is used via the `style` attribute in HTML elements for styling, which works well with most email clients.
*   The example introduces more complex HTML elements, such as `<div>`, `<h2>`, and `<table>`, to present the data in a structured manner.
*   Table elements are used to present information in a clear and organized way.
*   The email now incorporates a heading and styled paragraph for emphasis.
*   Background colors and border styles are used within the table to create a more visually appealing presentation.

**Example 3: HTML Table Generation from Query Results**

This example dynamically generates an HTML table based on a database query, which is common in automated reporting scenarios. It uses dynamic SQL to construct the HTML string from the dataset.

```sql
DECLARE @Subject VARCHAR(255);
DECLARE @Body NVARCHAR(MAX);
DECLARE @HTMLTable NVARCHAR(MAX);

SET @Subject = 'Daily Server Statistics';

SET @HTMLTable = N'<table style="width:100%; border-collapse: collapse;">
                    <tr style="background-color:#f0f0f0;">
                        <th style="padding:8px; border: 1px solid #ddd; text-align: left;">Server Name</th>
                        <th style="padding:8px; border: 1px solid #ddd; text-align: left;">CPU Usage (%)</th>
                        <th style="padding:8px; border: 1px solid #ddd; text-align: left;">Memory Usage (MB)</th>
                   </tr>';

SELECT @HTMLTable = @HTMLTable + N'<tr>
                                    <td style="padding:8px; border: 1px solid #ddd;">' + Server_Name + N'</td>
                                    <td style="padding:8px; border: 1px solid #ddd;">' + CONVERT(NVARCHAR(10),CPU_Usage) + N'</td>
                                    <td style="padding:8px; border: 1px solid #ddd;">' + CONVERT(NVARCHAR(10),Memory_Usage) + N'</td>
                                </tr>'
FROM (SELECT 'Server1' as Server_Name, 75 AS CPU_Usage, 16000 as Memory_Usage
	UNION ALL SELECT 'Server2', 30, 32000
	UNION ALL SELECT 'Server3', 55, 8000) AS ServerStats

SET @HTMLTable = @HTMLTable + N'</table>';

SET @Body = N'<div style="font-family: Arial, sans-serif;">
               <h2>Daily Server Statistics</h2>
                ' + @HTMLTable + N'
                <p style="font-style: italic; margin-top: 15px;">Data generated at: ' + CONVERT(VARCHAR(20), GETDATE(), 120) + '</p>
              </div>';

EXEC msdb.dbo.sp_send_dbmail
    @profile_name = 'YourDatabaseMailProfile', -- Replace with your profile name
    @recipients = 'recipient@example.com',     -- Replace with recipient's email address
    @subject = @Subject,
    @body = @Body,
    @body_format = 'HTML';
```

**Commentary on Example 3:**

*   This example constructs the HTML table dynamically by iterating over a result set.
*   The HTML table structure is constructed incrementally using string concatenation within the query.
*   A subquery replaces a real data source for demonstration.
*   The dynamically generated table is integrated into the final email body.
*   This method is useful for sending regular reports containing data directly from SQL Server.

**Resource Recommendations:**

For furthering understanding of HTML, numerous online resources are available, including websites covering basic HTML syntax, CSS styling, and general web development principles. Books on web technologies also contain relevant information. A strong grasp of HTML and CSS is crucial for effectively designing formatted emails via database mail. Specific knowledge of email client compatibility can also be helpful. Many online communities discuss email formatting quirks and best practices, offering solutions to common problems that arise with HTML emails. I have found consulting these resources invaluable in ensuring email displays correctly across diverse email clients.
