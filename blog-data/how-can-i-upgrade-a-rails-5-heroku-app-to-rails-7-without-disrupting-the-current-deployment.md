---
title: "How can I upgrade a Rails 5 Heroku app to Rails 7 without disrupting the current deployment?"
date: "2024-12-23"
id: "how-can-i-upgrade-a-rails-5-heroku-app-to-rails-7-without-disrupting-the-current-deployment"
---

, let’s tackle this. Upgrading a Rails application, especially one deployed to Heroku, is a task I’ve navigated a few times now, and each time it’s a bit of a delicate dance. The jump from Rails 5 to Rails 7 is significant, so we'll need to approach this methodically to ensure zero downtime. It’s not something you want to just push and pray on a live system. In my past experience, neglecting proper preparation has always led to some fairly stressful debugging sessions, usually at the least convenient time.

The core of this process isn't just about changing versions in a `Gemfile`; it’s about systematically addressing the breaking changes, deprecations, and underlying architectural shifts that occur between these major versions. Here's my strategy, refined through those experiences.

**Phase 1: Preparation and Compatibility Assessment**

Before even touching the codebase, let's solidify our battle plan. The first step is to *carefully* review the Rails release notes from Rails 5.1 all the way to 7.0. This is crucial. Don't just skim; actually go through them. Pay particular attention to deprecations in Rails 5, the removals and breaking changes in Rails 6 and 7, and changes to default configurations. The Rails Guides offer excellent documentation on the changes made between versions. The "Upgrading Ruby on Rails" section within the guides should become your go-to resource during this transition. Understanding what's coming at you is half the battle.

Next, we need to scrutinize our `Gemfile`. Upgrade all gems to their latest versions compatible with Rails 5. This minimizes potential conflicts during the Rails upgrade process. Then we want to scan for gems that might have known incompatibility issues with later Rails versions. A good starting point is reviewing the gem's README or repository's issue tracker. For example, outdated authentication gems or gems that hook deeply into ActiveRecord might cause headaches. Be prepared to potentially migrate away from problematic libraries. If possible, have a staging environment identical to your production environment to test on. You will need this.

**Phase 2: Incremental Upgrade (Local Development First)**

Now we move to the actual code changes. We start locally, in a branch distinct from the main development branch, say something like `rails7-upgrade`. This is where we introduce changes one piece at a time, moving from Rails 5.1 to 5.2, then Rails 6.0, then 6.1, and finally to Rails 7.0.

Here’s how a typical iteration might look using our `Gemfile`:

**Example 1: Initial Gemfile Change (Rails 5.1 to 5.2)**

```ruby
# Current Gemfile (Rails 5.0.x)

source 'https://rubygems.org'
git_source(:github) { |repo| "https://github.com/#{repo}.git" }

ruby '2.5.3'
gem 'rails', '~> 5.0.0'
# ... other gems

# Updated Gemfile for Rails 5.1
source 'https://rubygems.org'
git_source(:github) { |repo| "https://github.com/#{repo}.git" }

ruby '2.5.3' # Or an appropriate compatible ruby version for the target Rails version
gem 'rails', '~> 5.1.0'
# ... other gems
```

After making this update, run `bundle update` and then address any issues that arise during your test suites. You will almost certainly need to tweak your code to handle deprecations from this upgrade. Repeat this incremental upgrade process all the way up to Rails 7.0. Be sure to read the release notes for each iteration and address each breaking change in sequence. Remember, doing it piece by piece makes it more manageable.

**Example 2: Updating for Rails 6 Breaking Changes**

```ruby
# Gemfile after upgrading to Rails 5.2
# Update to a suitable Ruby Version for Rails 6
ruby '2.7.0'
gem 'rails', '~> 6.0.0'
# ... potentially adjust other gems here too.
```
After updating to Rails 6, certain default behaviours may have changed. For instance, Rails 6 by default disables autoloading in production, which can lead to hard to trace issues if not addressed. You might need to change your `config/environments/production.rb` file to enable eager loading.

```ruby
# config/environments/production.rb
Rails.application.configure do
  # ... other settings
  config.eager_load = true # Ensure all classes are loaded on boot.
end
```
Test thoroughly after this and ensure everything is still working as expected.

**Example 3: Final Rails 7 Gemfile Update**

```ruby
# Updated Gemfile for Rails 7
ruby '3.0.0' # or a higher Ruby version
gem 'rails', '~> 7.0.0'
# ... updated gems
```

Rails 7 introduces some more significant changes, such as import maps by default. We'd need to ensure we either migrate or disable the import maps if we choose not to adopt them yet. For example, you might opt to keep using Webpacker initially and delay your migration to Import Maps, which requires a specific set up. Make sure you are prepared for changes to how Javascript is bundled and included on your pages.

Throughout this process, pay very close attention to deprecation warnings. Rails will offer helpful warnings when you are using code that will be removed in future versions. Address these warnings explicitly; do not ignore them. Ignoring these warnings simply kicks the can down the road, making the transition more difficult.

**Phase 3: Testing, Testing, and More Testing**

This phase is the most crucial. It’s time to employ a robust suite of automated tests: unit, integration, and system tests. Cover as much of the functionality as you can, and remember that even if tests pass, visual review can also reveal issues that are not always caught by automated tests, particularly with front-end changes.

Furthermore, consider using tools like `brakeman` for static analysis to identify security vulnerabilities that might have inadvertently been introduced during the upgrade. Static analyzers can also help highlight potential problem areas that could cause runtime errors. It is also a great time to update your CI pipeline to include these checks.

**Phase 4: Staged Deployment Strategy on Heroku**

Here's where we deviate from a single-push strategy, ensuring zero downtime:

1. **Fork the Production Database:** To avoid potentially destructive updates directly on production, create a fork of your production database that is only being used in your staging environment. We want a carbon copy of the production data set before applying any migration.
2. **Deploy to a Staging Environment:** Deploy the upgraded branch to your staging environment on Heroku, configured to point to this database fork. Ensure your staging environment closely mirrors production in terms of configurations.
3. **Extensive Staging Testing:** Use this environment to thoroughly test the application, and involve your QA team if you have one. This includes load testing, checking all critical paths, and making sure that any performance issues have been addressed.
4. **Blue-Green Deployment:** Set up a blue-green deployment process. Essentially, you’ll deploy your updated application (the "green" environment) alongside your current production deployment (the "blue" environment), both configured to point to your main production database. Both versions would still be served by different Dynos.
5. **Traffic Shifting:** Gradually shift a portion of your production traffic to the green deployment. Monitor this traffic shift closely and rollback quickly if there are any issues. You could accomplish this through Heroku's traffic-shifting feature or using other techniques involving your load balancer or CDN.
6. **Full Rollout:** Once you are fully confident in the new deployment, switch all traffic to the green environment. Once the old version has zero traffic, you can remove the old application and its resources.
7. **Database Migrations:** Only after you've migrated 100% of the traffic to the green environment, apply your database migrations on production, making sure that no requests are running on the old application that expects the old database schema. This can be challenging as migrations in Rails can sometimes be destructive, if not coded properly.

**Technical Resources:**

For further depth, I recommend consulting these resources:

*   **"Upgrading Ruby on Rails" section of the Rails Guides:** This will be your constant companion throughout the entire process. It outlines every significant change between each version.
*   **"Confident Ruby" by Avdi Grimm:** While not directly about Rails upgrades, it’s an excellent resource for building well-structured and tested applications, helping ensure stability when making major version upgrades.
*   **"Working Effectively with Legacy Code" by Michael Feathers:** If you are working on an application that hasn’t seen many updates for a long time, this is a must-read. It focuses on the techniques for gradually improving code quality in older applications.

This process isn’t a one-size-fits-all template. It's adapted from many attempts, often involving late nights fixing unexpected issues. However, by being meticulous, testing thoroughly, and deploying strategically, you can make this transition as smooth as possible without disrupting your live application. Always, always be prepared to rollback if something goes wrong. Good luck!
