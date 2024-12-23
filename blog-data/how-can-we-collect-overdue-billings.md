---
title: "How can we collect overdue billings?"
date: "2024-12-23"
id: "how-can-we-collect-overdue-billings"
---

, let's talk overdue billings. It’s a problem I've faced multiple times, often finding myself in the weeds after a project launch. The key, I’ve learned, isn't about chasing payments after they're already late; it's about structuring systems that minimize late payments in the first place and, when they do happen, handle them effectively and systematically. Let's dive into some strategies and code examples that have proven valuable in my experience.

The initial approach always starts with prevention. Clear, concise billing processes are paramount. This includes automated invoicing systems that trigger at predefined intervals, offering multiple payment options, and sending reminder notifications before the actual due date. I've seen situations where simply clarifying payment terms upfront in the contract, and then reinforcing them via automated notifications, significantly reduces the likelihood of late payments. Lack of clarity breeds ambiguity, which directly translates to payment delays.

When prevention fails, a structured escalation process is necessary. This isn't about being aggressive; it's about being systematic. I generally recommend a three-stage process: friendly reminders, followed by a formal overdue notice, and then, if necessary, engagement with a collections agency or legal action. Each stage has its own cadence and level of formality. Let's look at a code example that automates the first two stages.

```python
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(recipient_email, subject, body, sender_email, sender_password):
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP_SSL('smtp.example.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(message)
        print(f"Email sent successfully to {recipient_email}")
    except Exception as e:
         print(f"Error sending email to {recipient_email}: {e}")


def check_overdue_invoices(invoice_data, sender_email, sender_password):
    today = datetime.date.today()
    for invoice_id, data in invoice_data.items():
        due_date = data['due_date']
        recipient_email = data['email']
        amount = data['amount']
        days_overdue = (today - due_date).days

        if days_overdue > 0:
            if days_overdue <= 7:
                subject = "Friendly Reminder: Invoice Due"
                body = f"This is a friendly reminder that invoice {invoice_id} for the amount of ${amount} was due on {due_date}. Please submit payment as soon as possible."
                send_email(recipient_email, subject, body, sender_email, sender_password)

            elif days_overdue <= 30:
                subject = "Overdue Payment Notice: Invoice " + str(invoice_id)
                body = f"This is a formal notification that invoice {invoice_id} for the amount of ${amount} is now overdue. The due date was {due_date}. Please remit payment immediately."
                send_email(recipient_email, subject, body, sender_email, sender_password)

            else:
                 print(f"Invoice {invoice_id} is significantly overdue. Consider next steps.")

invoice_data = {
    "INV001": {
        "due_date": datetime.date(2024, 5, 1),
        "email": "client1@example.com",
        "amount": 1500
    },
     "INV002": {
        "due_date": datetime.date(2024, 5, 15),
        "email": "client2@example.com",
        "amount": 2000
    },
    "INV003": {
         "due_date": datetime.date(2024, 4, 1),
        "email": "client3@example.com",
        "amount": 500
    }
}

sender_email = "your_email@example.com" #replace this with your email
sender_password = "your_password" #replace this with your password
check_overdue_invoices(invoice_data, sender_email, sender_password)
```
This python script leverages the `datetime` and `smtplib` modules. It reads invoice data, calculates the overdue time, and sends email reminders if the payment is overdue, scaling in severity depending on the number of days overdue. Note that for production environments, it is crucial to utilize a robust email service and manage credentials securely (consider using environment variables instead of hardcoding). Also, proper error handling and logging are necessary to monitor the process.

Beyond automated reminders, it's crucial to track overdue payments and identify patterns. For this, I often build or use a database system that stores invoice information, due dates, payments made, and flags for overdue status. This data-driven approach helps me understand which clients frequently have issues and allows for proactive intervention. A simple database structure might look something like this, focusing on the most relevant fields for this scenario:

```sql
CREATE TABLE invoices (
    invoice_id VARCHAR(255) PRIMARY KEY,
    client_id INT,
    due_date DATE,
    amount DECIMAL(10, 2),
    payment_date DATE,
    status VARCHAR(20) DEFAULT 'pending'
);

CREATE TABLE clients (
    client_id INT PRIMARY KEY,
    email VARCHAR(255),
    name VARCHAR(255)
    -- other client info
);
--Example
INSERT into clients (client_id, email, name)
VALUES
(1, 'client1@example.com', 'Test Client 1'),
(2, 'client2@example.com', 'Test Client 2'),
(3, 'client3@example.com', 'Test Client 3');

INSERT into invoices (invoice_id, client_id, due_date, amount)
VALUES
('INV001', 1, '2024-05-01', 1500),
('INV002', 2, '2024-05-15', 2000),
('INV003', 3, '2024-04-01', 500);
```
This SQL snippet creates a simple database schema with `invoices` and `clients` tables. The `invoices` table tracks invoice details and the `clients` table keeps client information. In a real application, additional columns would be included, such as invoice creation date, the product or service involved, etc. However, this example is streamlined for demonstration purposes. Using SQL, you can easily perform queries to identify overdue bills, track payment history, and gain valuable insights into trends. For example, a query to show all invoices that are overdue would be:

```sql
SELECT inv.invoice_id, cli.name, inv.due_date, inv.amount, inv.status
FROM invoices inv
INNER JOIN clients cli ON inv.client_id = cli.client_id
WHERE inv.status = 'pending' and inv.due_date < CURRENT_DATE;
```
This SQL query will return a list of invoices that are overdue, including details on the client and the due date. This allows for tracking down who owes you money and can be used in conjunction with the python email code shown earlier.

Finally, I’ve found that offering flexible payment arrangements, within reason, can significantly improve recovery rates. Sometimes a client is genuinely struggling, and offering a payment plan or a slight adjustment to the invoice can be a more practical approach than immediately resorting to collections. This doesn't mean accepting every request, but having a defined process for evaluating requests for payment adjustments can be beneficial. For instance, it could be automated via an API that allows clients to request extensions, which are then reviewed before approval. Let's illustrate this with a basic json data response example

```json
{
    "invoiceId": "INV003",
    "paymentPlanRequested": true,
    "reason": "Temporary cash flow issues due to unexpected expenditures.",
    "requestedNewDueDate": "2024-06-15",
    "status": "pending",
    "approval": null
}
```

This json structure represents the request a client might make for a payment plan. We can use this information to decide if we want to create a custom payment schedule for the client.

To further your understanding of these topics, I’d highly recommend looking into *Database Design for Mere Mortals* by Michael J. Hernandez and John L. Viescas for a detailed perspective on database design. *Understanding Email Infrastructure: A Practical Guide to Setting Up and Managing Email Servers* by Chris Ballew can be quite helpful for understanding the intricacies of email systems. Finally, *The Elements of Style* by William Strunk Jr. and E.B. White is an excellent resource for ensuring clear and concise communication, which is vital in this process. These resources can provide a more thorough understanding of the concepts and techniques mentioned here.

In summary, recovering overdue billings involves a proactive and structured approach. It isn’t just about chasing late payments, but about setting up the right systems to prevent them in the first place and having a reliable, scalable process to manage them when they do arise.
