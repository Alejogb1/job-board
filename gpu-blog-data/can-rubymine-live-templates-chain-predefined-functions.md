---
title: "Can RubyMine live templates chain predefined functions?"
date: "2025-01-30"
id: "can-rubymine-live-templates-chain-predefined-functions"
---
RubyMine's live template system, while powerful, does not directly support the chaining of predefined functions in the way one might expect from a functional programming paradigm. This means you cannot write a template that takes a variable, passes it through function `A`, then takes the result of `A` and passes it to function `B`, all within the template definition. However, the system provides a mechanism to achieve similar results using a combination of variables and built-in functions, which I've leveraged extensively in my workflow over the last few years, often to significantly reduce boilerplate. The limitation stems from the fact that live templates primarily function as text expansion engines with rudimentary variable processing, not as a full-fledged functional interpreter.

Instead of chaining, we define multiple template variables, each with a specific function applied to it or another variable. The key is to understand the order of expansion. RubyMine expands the template according to the order of variables defined in the template settings. So, if we create a variable that modifies another variable, we can achieve a pseudo-chaining effect. For instance, suppose we need to generate a database migration that uses a timestamp in various formats for indexing and column naming. We can define separate variables for date, timestamp, and different representations, referencing the preceding results. We don't chain functions in a functional sense, but we do transform values step by step by using other evaluated variables.

The crucial component for this technique lies in RubyMine’s built-in functions available in live template variable expressions. These include functions for manipulating strings, numbers, dates, and even generating UUIDs. While these functions are not user-defined, the existing collection is robust enough to handle a wide range of formatting and manipulation tasks. To create a chain, I typically define each intermediate value as its own variable, allowing me to inspect its value and correct errors during development. Then I use those variables when constructing the final template expansion. Let's look at some concrete examples.

**Example 1: Generating a Migration Timestamp**

This example demonstrates extracting date and time components from the current time and formatting them. The live template is triggered using the keyword `mig`.
```
# ${DATE}
# ${DATE_UNDER}
# ${DATE_TIME}
class Create${CLASS_NAME} < ActiveRecord::Migration[${VERSION}]
  def change
    create_table :${TABLE_NAME} do |t|
      t.timestamps
    end
    add_index :${TABLE_NAME}, :created_at
    add_index :${TABLE_NAME}, :updated_at
  end
end
```
Here are the variable definitions:
```
CLASS_NAME:  camelCase(fileNameWithoutExtension())
TABLE_NAME:  underscore(fileNameWithoutExtension()).pluralize()
VERSION: "7.0" # hardcoded version
DATE: date() # current date yyyy-MM-dd
DATE_UNDER:  replace(DATE,"-","_") # date with underscores e.g. 2023_10_27
DATE_TIME:  date("yyyyMMdd_HHmmss") # timestamp yyyyMMdd_HHmmss
```

**Commentary:**

In this setup, the `DATE` variable is assigned the current date using the `date()` function. The `DATE_UNDER` variable uses the `replace` function to replace hyphens with underscores. Then `DATE_TIME` extracts a timestamp using the format specifier "yyyyMMdd_HHmmss".  These variables are used in comments within the generated code so that debugging and troubleshooting is easier.
While there's no explicit function chaining, we obtain derived values through multiple variable transformations. The expansion results will look something like:

```ruby
# 2023-10-27
# 2023_10_27
# 20231027_143542
class CreateUserProfiles < ActiveRecord::Migration[7.0]
  def change
    create_table :user_profiles do |t|
      t.timestamps
    end
    add_index :user_profiles, :created_at
    add_index :user_profiles, :updated_at
  end
end

```

**Example 2: Creating an API endpoint path**

This template generates a URL endpoint given a resource name. The trigger is `api`.
```
#${URL_PATH}
#${UPPER_URL_PATH}
get '/${API_VERSION}/:${RESOURCE_PARAM_NAME}', to: '${CONTROLLER}#index'
get '/${API_VERSION}/:${RESOURCE_PARAM_NAME}/:${RESOURCE_PARAM_NAME}/:id', to: '${CONTROLLER}#show'
post '/${API_VERSION}/:${RESOURCE_PARAM_NAME}', to: '${CONTROLLER}#create'
put '/${API_VERSION}/:${RESOURCE_PARAM_NAME}/:id', to: '${CONTROLLER}#update'
delete '/${API_VERSION}/:${RESOURCE_PARAM_NAME}/:id', to: '${CONTROLLER}#destroy'
```
Here are the variable definitions:
```
RESOURCE_NAME: complete()
URL_PATH: underscore(RESOURCE_NAME).pluralize()
UPPER_URL_PATH:  toUpperCase(URL_PATH)
RESOURCE_PARAM_NAME: singularize(URL_PATH)
CONTROLLER: camelCase(RESOURCE_NAME).pluralize() + 'Controller'
API_VERSION: "v1"
```

**Commentary:**

This example uses a series of string manipulation functions. The `RESOURCE_NAME` is provided upon template expansion. The `URL_PATH` is then created by first taking the `RESOURCE_NAME` and making it `underscore`d and then pluralizing it. The `UPPER_URL_PATH` is an upper-case version of that path. The `RESOURCE_PARAM_NAME` is created using the singularized form of `URL_PATH`, which means that URL path variable will be used in routing parameter definitions. Finally, a `CONTROLLER` name is assembled.
An expansion might look like this:

```ruby
#user_profiles
#USER_PROFILES
get '/v1/:user_profile', to: 'UserProfilesController#index'
get '/v1/:user_profile/:user_profile/:id', to: 'UserProfilesController#show'
post '/v1/:user_profile', to: 'UserProfilesController#create'
put '/v1/:user_profile/:id', to: 'UserProfilesController#update'
delete '/v1/:user_profile/:id', to: 'UserProfilesController#destroy'
```

**Example 3: Generating a test with a timestamp**

This template generates a test method name with a time component. Trigger is `test`.
```
# ${METHOD_NAME}
test "${TEST_NAME}" do
  skip "Not implemented"
end
```
Variable definitions are as follows:
```
TEST_NAME:  complete()
METHOD_NAME:  concat("test_",replace(date("yyyyMMdd_HHmmss"),"_",""))
```

**Commentary:**

Here, we demonstrate generating a method name based on the current timestamp. The `date` function generates the timestamp in a specific format which is then processed to remove any underscores. Then the prefix `test_` is prepended.

An example expansion might be:

```ruby
# test_20231027144830
test "My Awesome Test" do
  skip "Not implemented"
end
```

To improve the development experience, I use descriptive variable names, so that during template configuration, it is easier to see the transformation steps. This makes the template behavior transparent and easy to modify or troubleshoot.

While we cannot chain functions directly as one might in a functional language, this system allows for a significant degree of flexibility. For the given functionality, RubyMine live templates are not designed for full-fledged functional processing. These templates are built for quick text manipulations, and the available functions are very good for that task. This approach has saved me a significant amount of time over the years, especially when adhering to code consistency standards. By carefully planning the variable relationships, one can effectively emulate a "chaining" approach and greatly enhance productivity.

For further exploration of live templates, consult the "Live Templates" section in the official JetBrains RubyMine documentation. Additionally, exploring the built-in function list available in the template variable settings dialog is essential. It’s also worthwhile examining the pre-existing templates provided by JetBrains within RubyMine; they often showcase valuable techniques. There are also many blogs and tutorials that provide examples of using live templates with real-world use cases.
