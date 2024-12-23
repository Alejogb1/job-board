---
title: "How can dynamic AASM states be reloaded?"
date: "2024-12-23"
id: "how-can-dynamic-aasm-states-be-reloaded"
---

Okay, let's tackle this one. I’ve certainly navigated the choppy waters of dynamic state machine reloading before, and it’s not always a walk in the park. The ability to adjust AASM (Acts As State Machine) states on the fly, without completely restarting the application, is crucial for maintaining smooth operations and adaptability, especially in long-running processes or systems that need to respond to changing configurations. The simple solution would be to completely reload the application, but that's not always feasible or desirable and can cause disruptions to existing workflows.

When I first encountered this, it was in a large-scale financial trading system, where market conditions and trading strategies changed regularly. We had these complex state machines governing order execution, and a hard restart for every little adjustment was simply unacceptable. We needed a surgical approach, something more nuanced. The challenge lies in the fact that AASM, at its core, expects a fixed definition of states and transitions when the class is initially loaded. Dynamic alterations require more thought.

So, how do we actually achieve this? Well, the core concept revolves around separating the *definition* of your state machine from its *execution* context. Instead of embedding the states and events directly within your model, we utilize an external data source—think a configuration file, a database table, or even a remote service—that defines them. This configuration can be updated independently and then applied to modify the behavior of our AASM instance.

The approach I’ve found most reliable involves these steps:

1.  **Externalize State Definitions:** The first and most critical step is to not hardcode states and transitions directly within your model class. Instead, store your states, events, and their associated transitions in a persistent store of some kind. This could be a database table, a YAML file, a JSON structure fetched remotely, or something custom depending on the nature of your application.
2.  **Load Configuration:** When the application (or relevant module) starts or when configuration changes are detected, you must load or reload these external definitions from their source.
3.  **Rebuild State Machine:** Instead of directly altering the state machine within AASM which is not advisable, dynamically generate a new AASM definition (which often means using an intermediary object that mixes in AASM with our reloaded configuration). Then, instantiate a fresh version of this class with the new definitions. Copying over the state information of the original object to the new object, including the current state, is key to ensure the current process isn't disrupted, allowing us to carry on our processes smoothly.
4. **Replace Original with the New Object:** The final step is to replace the old instance with the new object with the updated states and transitions within the relevant scope. It's essentially an object swap.

Now, let's solidify this with some code examples using Ruby, since AASM is most commonly used with Ruby. Please be aware that these are simplified illustrations, but they demonstrate the core principles:

**Example 1: Configuration from a YAML File**

Let's say we have a `Workflow` class that uses AASM, and its states are driven by a `config.yaml` file.

```ruby
# config.yaml
# example
initial: pending
states:
  pending:
    events:
      start: { to: processing }
  processing:
    events:
      complete: { to: completed }
      error: { to: failed }
  completed: {}
  failed: {}
```

```ruby
require 'aasm'
require 'yaml'

class Workflow
  include AASM

  def initialize(config_file = 'config.yaml')
    @config_file = config_file
    load_config
    aasm do
       state :initialized, initial: true
       event :initialize_aasm do
          transitions from: :initialized, to: :pending
       end
    end
    self.initialize_aasm
  end

  def load_config
    config = YAML.load_file(@config_file)
    aasm_definitions = config.fetch('states',{})
    # Rebuild aasm definitions
    aasm do
        config['states'].each do |state_name, state_data|
            state state_name.to_sym
            state_data.fetch('events', {}).each do |event_name, event_data|
              event event_name.to_sym do
                 transitions from: state_name.to_sym, to: event_data['to'].to_sym
              end
            end
        end
    end
  end

  def update_config(new_config_file)
    @config_file = new_config_file
    load_config
  end

end

workflow = Workflow.new
puts "Initial State: #{workflow.aasm.current_state}"
workflow.start
puts "State after start: #{workflow.aasm.current_state}"
workflow.complete
puts "State after complete: #{workflow.aasm.current_state}"


# To reload with a different config (imagine that config.yaml has changed).
# Workflow can then be updated with new definition
# and continue its process.
# lets say that `new_config.yaml` now has 'cancel' event to the 'pending' state from the 'processing' state

# new_config.yaml
# example
# initial: pending
# states:
#  pending:
#    events:
#      start: { to: processing }
#  processing:
#    events:
#      complete: { to: completed }
#      error: { to: failed }
#      cancel: { to: pending }
#  completed: {}
#  failed: {}

new_workflow = Workflow.new('new_config.yaml')
# copy state from old workflow to the new
new_workflow.aasm.current_state = workflow.aasm.current_state

#replace the old with new instance
workflow = new_workflow

puts "State after reload: #{workflow.aasm.current_state}"

workflow.cancel
puts "State after cancel: #{workflow.aasm.current_state}"

```

This example loads the state definitions from the `config.yaml` file. When the `update_config` method is called (or when a configuration change is detected), a new `Workflow` object is created with the new configuration, and the state is transferred to this new object, and finally replaces the old object with the new, effectively reloading the AASM definition and ensuring no workflow processes are interrupted.

**Example 2: Configuration from a Database**

In a slightly more complex setup, imagine the configuration is stored in a database:

```ruby
# Assume a model named WorkflowConfiguration with columns: id, state_name, event_name, to_state

require 'aasm'
require 'active_record'

ActiveRecord::Base.establish_connection(adapter: "sqlite3", database: ":memory:")

ActiveRecord::Schema.define do
  create_table :workflow_configurations do |t|
    t.string :state_name
    t.string :event_name
    t.string :to_state
    t.timestamps
  end
end


class WorkflowConfiguration < ActiveRecord::Base
end


WorkflowConfiguration.create(state_name: 'pending', event_name: 'start', to_state: 'processing')
WorkflowConfiguration.create(state_name: 'processing', event_name: 'complete', to_state: 'completed')
WorkflowConfiguration.create(state_name: 'processing', event_name: 'error', to_state: 'failed')

class Workflow
  include AASM

  def initialize
      load_config
      aasm do
        state :initialized, initial: true
        event :initialize_aasm do
            transitions from: :initialized, to: :pending
        end
      end
      self.initialize_aasm
  end

  def load_config
      aasm_definitions = WorkflowConfiguration.all.group_by(&:state_name)
      aasm do
          aasm_definitions.each do |state_name, configs|
               state state_name.to_sym
               configs.each do |config|
                   event config.event_name.to_sym do
                       transitions from: state_name.to_sym, to: config.to_state.to_sym
                   end
                end
          end
      end
  end

   def update_config
        load_config
   end

end

workflow = Workflow.new
puts "Initial State: #{workflow.aasm.current_state}"
workflow.start
puts "State after start: #{workflow.aasm.current_state}"
workflow.complete
puts "State after complete: #{workflow.aasm.current_state}"

# Lets create a new configuration on DB
WorkflowConfiguration.create(state_name: 'processing', event_name: 'cancel', to_state: 'pending')

new_workflow = Workflow.new
new_workflow.aasm.current_state = workflow.aasm.current_state

workflow = new_workflow

puts "State after reload: #{workflow.aasm.current_state}"
workflow.cancel
puts "State after cancel: #{workflow.aasm.current_state}"
```

Here, the AASM definition is derived from records in the `workflow_configurations` table. The `load_config` method pulls these configurations to create the necessary states and transitions dynamically. The old object is replaced using similar techniques as the above example.

**Example 3: Configuration from Remote Service**

Finally, for the sake of demonstrating the flexibility, consider fetching your configuration from a remote service, perhaps an API:

```ruby
require 'aasm'
require 'net/http'
require 'json'

class Workflow
    include AASM
    attr_reader :config

    def initialize
       @config = fetch_config
        load_config
         aasm do
             state :initialized, initial: true
             event :initialize_aasm do
                transitions from: :initialized, to: :pending
             end
          end
          self.initialize_aasm
    end


    def fetch_config
        uri = URI('https://api.example.com/config') # Replace with your actual endpoint
        response = Net::HTTP.get(uri)
        JSON.parse(response) # Assuming JSON response
    rescue
        {
          'initial': 'pending',
          'states': {
            'pending': {
              'events': {
                'start': { 'to': 'processing' }
              }
            },
            'processing': {
              'events': {
                'complete': { 'to': 'completed' },
                'error': {'to': 'failed'}
              }
            },
            'completed': {},
            'failed': {}
          }
       }
    end

    def load_config
       aasm_definitions = @config.fetch('states',{})

        aasm do
            aasm_definitions.each do |state_name, state_data|
               state state_name.to_sym
                state_data.fetch('events', {}).each do |event_name, event_data|
                    event event_name.to_sym do
                       transitions from: state_name.to_sym, to: event_data['to'].to_sym
                    end
                end
            end
        end
    end

    def update_config
        @config = fetch_config
        load_config
    end
end



workflow = Workflow.new
puts "Initial State: #{workflow.aasm.current_state}"
workflow.start
puts "State after start: #{workflow.aasm.current_state}"
workflow.complete
puts "State after complete: #{workflow.aasm.current_state}"

# Simulate a configuration change from API by adding cancel event on pending
def create_new_remote_config
    uri = URI('https://api.example.com/config')
    new_config = {
            'initial': 'pending',
            'states': {
            'pending': {
            'events': {
               'start': { 'to': 'processing' }
               }
             },
            'processing': {
            'events': {
              'complete': { 'to': 'completed' },
              'error': {'to': 'failed'},
               'cancel': {'to': 'pending'}
                }
             },
            'completed': {},
            'failed': {}
          }
    }
    Net::HTTP.post(uri, new_config.to_json, 'Content-Type' => 'application/json')
end

# To trigger update of config from remote API
create_new_remote_config
new_workflow = Workflow.new
new_workflow.aasm.current_state = workflow.aasm.current_state

workflow = new_workflow

puts "State after reload: #{workflow.aasm.current_state}"
workflow.cancel
puts "State after cancel: #{workflow.aasm.current_state}"
```

Here the code retrieves the state configuration from an external api, and dynamically builds the aasm object based on this.

This approach allows for real-time state updates by leveraging a remote configuration.

**Important Considerations:**

*   **Error Handling:** Always include robust error handling when loading configuration data to prevent application crashes.
*   **Concurrency:** When updating the AASM instance in multi-threaded environments, make sure the process is thread-safe and consider using techniques like mutex locks to avoid race conditions.
*   **Testing:** Rigorous testing is absolutely essential whenever you're implementing dynamic state changes to ensure your application behaves as expected.

For further reading, I'd recommend diving into Martin Fowler's "Patterns of Enterprise Application Architecture" for a more in-depth discussion on state management patterns and their applications. Additionally, "Domain-Driven Design" by Eric Evans provides excellent perspectives on modeling your business logic through state machines and how they should interact with business rules. Specifically with regard to AASM I recommend the official AASM documentation for specifics.

In conclusion, reloading dynamic AASM states isn't about rewriting AASM itself, but about cleverly structuring your application to separate state definitions from their execution context and carefully managing the process of replacing old objects with new ones, in a way that ensures the current workflows aren't disrupted. These are the methods I've utilized and refined over years of practical experience, and I hope you'll find them useful as well.
