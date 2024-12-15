---
title: "How to Azure - Get Billing cost for a specified date using Ruby?"
date: "2024-12-15"
id: "how-to-azure---get-billing-cost-for-a-specified-date-using-ruby"
---

alright, so you're after pulling azure billing costs for a particular date using ruby, eh? been there, done that. it's one of those things that sounds straightforward, but can have a few gotchas if you're not careful. i've banged my head against the wall on similar issues before, so let me give you the lowdown based on my experience.

first off, you'll need the azure sdk for ruby. if you haven't installed it already, you're gonna want to add these gems to your `gemfile`

```ruby
gem 'azure-sdk-billing', '~> 0.18'
gem 'ms_rest_azure', '~> 0.12'
gem 'ms_rest', '~> 0.8'
```

then run `bundle install` of course.

once you've got the gems sorted, the next crucial part is authentication. azure uses azure active directory (aad) for auth, and that's where things can get a little messy if you haven't done it before. i remember spending a whole evening trying to get the correct service principal set up. good times, haha.

i strongly advise against using your personal azure credentials directly in your code – that's a big no-no. instead, you should create an aad application and a service principal, granting it the minimum necessary permissions to access billing data. there are a ton of tutorials out there, but azure's own documentation on service principals is a good place to begin, usually.

for the ruby code, you’re going to need a few things:

*   your tenant id
*   your client id (the application id of your aad app)
*   your client secret (a secret key for your aad app)
*   your subscription id

you can grab all of these from the azure portal under active directory and subscriptions.

with all that in place, here's a ruby code snippet that should do the trick for getting the costs for one specific date. i've added a bit of error handling, because let’s face it, stuff happens.

```ruby
require 'azure_mgmt_billing'
require 'date'

def get_billing_cost_for_date(tenant_id, client_id, client_secret, subscription_id, date)

  credentials =  MsRestAzure::ApplicationTokenProvider.new(tenant_id, client_id, client_secret)
    .get_authentication_header
  billing_client = Azure::Billing::Profiles::ProfilesService.new(credentials)

  begin

    start_date = date.strftime('%Y-%m-%d')
    end_date = (date + 1).strftime('%Y-%m-%d') # date is inclusive so you need to +1 day

    filter = "properties/usageStart ge '#{start_date}' and properties/usageEnd le '#{end_date}'"
    result = billing_client.usage.list(subscription_id, {filter: filter})
    total_cost = 0
    result.value.each do |usage|
      total_cost += usage.properties.pretax_cost
    end
    return total_cost
  rescue  MsRest::HttpOperationError => e
        puts "Error fetching billing data: #{e.response.body}"
        return nil
  rescue => e
        puts "An unexpected error ocurred: #{e.message}"
        return nil
  end
end

# example usage
tenant_id = "your_tenant_id"
client_id = "your_client_id"
client_secret = "your_client_secret"
subscription_id = "your_subscription_id"
date = Date.new(2024, 03, 10)

total_cost = get_billing_cost_for_date(tenant_id, client_id, client_secret, subscription_id, date)

if total_cost
    puts "Total cost for #{date}: #{total_cost}"
else
   puts "Failed to get the cost for #{date}"
end

```

this code initializes the billing client using the authentication information you provide, then makes a call to the `list` method, filtering by the date you specified. notice how we’re converting ruby's `date` object into an `iso` string, that’s crucial for the api. and also notice how the `properties/usageStart` is `ge` and the `properties/usageEnd` is `le`, this makes the query inclusive based on the start and end date. i've found that date boundaries can be very tricky with these apis, and it's always better to be very specific. you could improve this part by making `ge` and `le` `gt` and `lt` and subtract/add an instant of time to be extra sure. that's a thing you can enhance.

now, if you want to get the costs for a date range, that's a slightly different beast. you'll need to loop through each day and add those individual costs or create a bigger query using date ranges if that makes sense in your application. here's a slightly modified version that loops, to handle a range:

```ruby
require 'azure_mgmt_billing'
require 'date'

def get_billing_cost_for_date_range(tenant_id, client_id, client_secret, subscription_id, start_date, end_date)
  credentials =  MsRestAzure::ApplicationTokenProvider.new(tenant_id, client_id, client_secret)
      .get_authentication_header
    billing_client = Azure::Billing::Profiles::ProfilesService.new(credentials)

    total_cost = 0

  begin
      (start_date..end_date).each do |date|
          date_start_str = date.strftime('%Y-%m-%d')
          date_end_str = (date + 1).strftime('%Y-%m-%d') # date is inclusive so you need to +1 day
          filter = "properties/usageStart ge '#{date_start_str}' and properties/usageEnd le '#{date_end_str}'"
          result = billing_client.usage.list(subscription_id, {filter: filter})
          result.value.each do |usage|
            total_cost += usage.properties.pretax_cost
          end
      end

    return total_cost
  rescue  MsRest::HttpOperationError => e
        puts "Error fetching billing data: #{e.response.body}"
        return nil
  rescue => e
        puts "An unexpected error ocurred: #{e.message}"
        return nil
  end
end

# example usage
tenant_id = "your_tenant_id"
client_id = "your_client_id"
client_secret = "your_client_secret"
subscription_id = "your_subscription_id"
start_date = Date.new(2024, 03, 01)
end_date = Date.new(2024, 03, 10)

total_cost = get_billing_cost_for_date_range(tenant_id, client_id, client_secret, subscription_id, start_date, end_date)

if total_cost
    puts "Total cost from #{start_date} to #{end_date}: #{total_cost}"
else
    puts "Failed to get the cost for range from #{start_date} to #{end_date}"
end
```
in this example the key thing is to use ruby ranges and iterate over each day getting the total cost, summing it in each step of the loop. this is not performant and you could make a query with a time range, but i'm providing a simple solution with basic knowledge of programming for a clearer understanding of what is happening step by step.

one more thing that i found out the hard way is that the billing data isn't always available immediately. there's usually some latency before it appears in the api. so, if you're querying for a date, make sure to allow some time for the data to be processed. depending on the resources you're using the latency could be as low as few minutes and as big as a couple of hours.

another thing i’ve learned is to be mindful of the api rate limits, although they are not that low, making requests too frequently will lead to your code not working correctly because of throttling. it’s always good to implement a backoff strategy in the case of api errors, but that's another topic for another time.

finally, you can filter by other properties such as resource types and meter categories, that's up to your specific needs. the azure billing api is pretty flexible and that will be beneficial depending on the reports you want to generate. take a look into the official documentation of the rest api, it's usually pretty well done, and that way you can customize your requests.

also, i highly recommend reading some books on system design, sometimes that's a better use of your time than looking at many different tutorials that could just provide limited knowledge of the subject.

you can learn a lot from [“designing data-intensive applications” by martin kleppmann] for the core concepts and [“release it!: design and deploy production-ready software” by michael t. nygard] to get an idea of how to handle code in production and be more professional in your craft. they are both good books and will help you in other areas of your development process as well.

```ruby
require 'azure_mgmt_billing'
require 'date'

def get_billing_data_detailed(tenant_id, client_id, client_secret, subscription_id, date)
  credentials =  MsRestAzure::ApplicationTokenProvider.new(tenant_id, client_id, client_secret)
      .get_authentication_header
    billing_client = Azure::Billing::Profiles::ProfilesService.new(credentials)

  begin
    start_date = date.strftime('%Y-%m-%d')
    end_date = (date + 1).strftime('%Y-%m-%d')
    filter = "properties/usageStart ge '#{start_date}' and properties/usageEnd le '#{end_date}'"

    result = billing_client.usage.list(subscription_id, {filter: filter})
    detailed_usage = []
      result.value.each do |usage|
          detailed_usage << {
              meter_category: usage.properties.meter_category,
              resource_type: usage.properties.resource_type,
              pretax_cost: usage.properties.pretax_cost,
              usage_quantity: usage.properties.usage_quantity,
              unit: usage.properties.unit
          }
      end
    return detailed_usage
    rescue  MsRest::HttpOperationError => e
          puts "Error fetching billing data: #{e.response.body}"
          return nil
    rescue => e
            puts "An unexpected error ocurred: #{e.message}"
            return nil
    end
end


# example usage
tenant_id = "your_tenant_id"
client_id = "your_client_id"
client_secret = "your_client_secret"
subscription_id = "your_subscription_id"
date = Date.new(2024, 03, 10)

detailed_data = get_billing_data_detailed(tenant_id, client_id, client_secret, subscription_id, date)

if detailed_data
    detailed_data.each do |usage|
      puts "Meter Category: #{usage[:meter_category]}"
        puts "Resource Type: #{usage[:resource_type]}"
        puts "Cost: #{usage[:pretax_cost]}"
        puts "Usage quantity: #{usage[:usage_quantity]}"
        puts "Unit: #{usage[:unit]}"
      puts "---"
    end

else
    puts "Failed to get detailed data"
end


```
this final example code, demonstrates how to access the more detailed usage data, this will be useful to investigate costs of each category and resources you might be using in the cloud.

that should cover pretty much everything you need to get going. let me know if you're hitting any snags, and good luck!
