---
title: "How to AWS notify after 3 consecutive fail sign in IAM user attempts?"
date: "2024-12-14"
id: "how-to-aws-notify-after-3-consecutive-fail-sign-in-iam-user-attempts"
---

ah, i see the problem. user’s trying to lock down iam access after too many failed login attempts, and they need an automated way to get notified. i've been down this road before, let me tell you. it's a surprisingly common security thing when you need to keep the bad actors out.

way back when, i was managing a small cluster of ec2 instances. we had this one user, bless their heart, who kept mistyping their password. not just once or twice, but consistently. at first, it was kinda funny. then it got annoying because we were constantly unlocking their account. eventually it was obvious there was a bigger problem, either they were getting attacked or there was some sort of phishing. we needed a way to be alerted when these login failures piled up. this was when i learned the hard way why you need to automate everything and not check logs manually for errors... 

so, how to tackle this in aws? well, there isn't a single, out-of-the-box setting in iam that says "notify me after three failed attempts". it's not that easy unfortunately. we're going to need to piece together a few services to make this happen. cloudtrail will capture the events, cloudwatch will monitor the logs, and sns will handle the notification part of the plan. it's more work to set up, but it gives you more control, and also it's a really useful thing to learn the tools of the aws ecosystem.

first things first, let's make sure cloudtrail is logging the right events. you want to verify that cloudtrail is configured and logging iam related events, especially the 'consolelogin' event. usually, this is enabled by default, but always double check. if it isn't logging these events then we have no way to know when login failures happen.

once cloudtrail is set up, we need to create a cloudwatch log metric filter that detects the failure pattern. we'll look for specific error messages related to failed login attempts. this is the part that needs a little config. here's how to define the metric filter using the aws cli:

```bash
aws logs put-metric-filter \
    --log-group-name "your_cloudtrail_log_group_name" \
    --filter-name "failed_login_attempts" \
    --filter-pattern '{ $.eventName = "ConsoleLogin" && $.errorMessage ="Failed authentication" }' \
    --metric-transformations metricName=failed_login_attempt_count,metricNamespace=security_alerts,metricValue=1
```

replace `"your_cloudtrail_log_group_name"` with your actual cloudtrail log group name. this filter looks for ‘consolelogin’ events with "failed authentication" as error message. whenever this event occurs, it increments the `failed_login_attempt_count` metric in the `security_alerts` namespace.

next, we need to create a cloudwatch alarm based on this metric. this alarm will trigger when the metric goes above a certain threshold within a period of time. we'll set the threshold to 3 and the period to something reasonable, like 5 minutes. here's the aws cli command for it:

```bash
aws cloudwatch put-metric-alarm \
    --alarm-name "failed_login_attempts_alarm" \
    --metric-name "failed_login_attempt_count" \
    --namespace "security_alerts" \
    --statistic "Sum" \
    --period 300 \
    --threshold 3 \
    --comparison-operator "GreaterThanOrEqualToThreshold" \
    --evaluation-periods 1 \
    --unit "Count" \
    --alarm-actions "arn:aws:sns:your_aws_region:your_account_id:your_sns_topic_name"
```
again, you'll need to replace `"your_aws_region:your_account_id:your_sns_topic_name"` with your actual sns topic arn. the period is in seconds (300 is 5 minutes), the threshold is 3, and we're summing up the number of failures. after 3 failed attempts, this alarm will be triggered. i actually almost forgot about these numbers, but thankfully i remember i have them on my notebook.

lastly, we need an sns topic to receive the notification when the alarm is triggered. if you don't have one yet, you can create one with the following aws cli command:

```bash
aws sns create-topic --name "failed_login_alerts"
```

then, you'll need to subscribe an email or other notification endpoint to the topic. to subscribe an email address, use this:

```bash
aws sns subscribe \
  --topic-arn "arn:aws:sns:your_aws_region:your_account_id:your_sns_topic_name" \
  --protocol email \
  --notification-endpoint "your_email@example.com"
```
aws will then send a confirmation email to the address you provide and you'll need to confirm it. now every time the alarm goes off, you'll get notified.

and voila! you've got a notification system for failed sign-in attempts. it took me a little while the first time i did it, but now it's muscle memory.

a few things i learned the hard way:

*   always double check your log group name in cloudtrail settings and double check the metric filters, as they can be a pain to debug after the fact. if the filter isn’t right, you simply won’t get the alarms. also be careful with copy and pasting names. small typos can take a lot of time to debug.
*   test this on a non-production environment first and make sure everything works as expected before you apply it to a live production environment. nobody likes alarm fatigue.
*   use descriptive names for all your resources. this makes debugging a lot easier later on. i've spent way too long trying to figure out what `alarm-1234` actually did.
*   be mindful of the period you set. too short, and you might get a ton of false positives. too long, and you might miss attacks early on. it’s a matter of tweaking.
*   remember that the logs may have a delay to be ingested into cloudwatch, so it won't be instant notification.

if you're looking to understand this more deeply, i'd recommend checking out the official aws documentation for cloudtrail, cloudwatch, and sns. also, the book "aws certified security specialty study guide" by ben piper and david clark is a great reference if you really want to dive into aws security topics. another good read is "aws cookbook" by matthew reiche. these books helped me quite a bit when i was starting.

this setup, while a little involved, provides a much more robust solution than manually tracking failed login attempts and it is a good exercise to improve your understanding of these core aws services. and by the way, why did the developer quit their job? because they didn't get array raise! (i'm sorry).
