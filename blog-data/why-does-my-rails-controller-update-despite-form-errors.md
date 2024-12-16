---
title: "Why does my Rails controller update despite form errors?"
date: "2024-12-16"
id: "why-does-my-rails-controller-update-despite-form-errors"
---

Okay, let's tackle this. I've seen this particular head-scratcher more times than I care to remember, and it always boils down to a few specific nuances in how Rails handles form submissions and data persistence. It’s a classic symptom of how the framework’s lifecycle interacts with your code, and it’s something we’ve all had to debug at some point.

The scenario, as I understand it, is that you’re submitting a form, your model's validations are kicking in (good!), errors are being generated (also good, in a way, it means validations are working!), but despite those errors, the record is still being updated in the database, which, of course, is not what you want. It feels counterintuitive, because logically, one might expect the update to be entirely blocked on the presence of validation failures.

The core issue revolves around the way Rails interprets the result of a `save` operation, and importantly, the specific method you're using for that. Let's get into the weeds. Generally speaking, in a typical rails controller's `update` action, you'd have something that looks like this, a structure that's served us well across many projects, and which I've personally tweaked in many iterations:

```ruby
def update
  @record = Record.find(params[:id])
  if @record.update(record_params)
    redirect_to @record, notice: 'Record was successfully updated.'
  else
    render :edit, status: :unprocessable_entity
  end
end
```

The crucial part, and the source of our problem, is `record.update(record_params)`. This method attempts to update the attributes of the record and then tries to save it to the database. The key here is that `update` *first* updates the attributes on the object *in memory*, and *then* it triggers the save process. If the save fails due to validations, that's fine - Rails has mechanisms for that, and you are checking them. But, crucially, the attributes have already been updated on `@record`.

The crucial, hidden detail in `update` is that it doesn't return `true` or `false` strictly based on whether the save *succeeded*. Instead, it returns `true` if the record was valid *at some point*, meaning it passed validations, and it returned `false` if *it did not*. However, due to the two step process, the update to the object's attributes happens *before* validation, meaning even if the save fails, your `@record` has already been updated with the new form values. The problem is not that the database record is updated when validations fail (it is not, because save failed), but that your `@record` *object* in memory is updated. When you render the `:edit` view, that object, already carrying the new (and invalid) attributes, is what gets rendered. This creates the illusion of database changes.

This behavior is not a bug; it's designed this way to facilitate showing validation errors within the form. Without this behavior, your form would display the *previous* valid data, which doesn't provide clear feedback to the user.

Okay, so what’s the fix? Well, there are a couple of things we could try. Let's examine a revised approach. Instead of relying solely on `update`, we can perform the attribute updates and the save process explicitly in two steps. Here's the amended code:

```ruby
def update
  @record = Record.find(params[:id])
  @record.assign_attributes(record_params) # Update attributes on the object
  if @record.valid? # Explicitly check if the record *with the new attributes* is valid
    if @record.save # Only save if the validations pass
      redirect_to @record, notice: 'Record was successfully updated.'
    else
      # This branch is unlikely to be hit, but included for thoroughness
      render :edit, status: :unprocessable_entity
    end
  else
    render :edit, status: :unprocessable_entity
  end
end
```

This version of the `update` action first explicitly sets the attributes with `assign_attributes` which performs the in-memory update. Then, it checks if the *modified* record is valid before attempting to persist to the database. If it’s not valid, we immediately short-circuit the save, preventing the confusion, and render the edit form, with errors. This ensures that the database is only touched when all validations pass. Crucially, even if the validations fail, the original data for that model record is still available, because we have not performed the save.

Now, there’s another approach that can be useful, particularly in scenarios where the model includes custom save behavior or you want finer-grained control. Instead of `update`, we can explicitly use `assign_attributes` followed by a conditional save. Here’s an example:

```ruby
def update
  @record = Record.find(params[:id])
  @record.assign_attributes(record_params)
  if @record.save
    redirect_to @record, notice: 'Record was successfully updated.'
  else
    render :edit, status: :unprocessable_entity
  end
end
```
In this case, if the save fails, rendering the edit form will show the invalid changes in the form, exactly as intended, but without persisting them to the database if validations fail. The difference, in practice, between this and the previous example is that this approach doesn't explicitly check for validations before saving, whereas the previous one does. This approach still separates attribute assignment from saving, providing better control.

The key thing to remember is the two-step process of attribute assignment and persistence. `update` does both of those in sequence, whereas `assign_attributes` only updates the object in memory. Therefore, checking `valid?` immediately after attribute assignment allows you to make an informed decision about persistence.

A further tip is to be vigilant about callback methods in your models. Before or after save callbacks could be subtly altering values, sometimes in unexpected ways. I recall a time when I inherited a project where a `before_save` callback was auto-generating a slug on the record every time it was saved, which led to a similar "updated despite errors" confusion. Scrutinize these callbacks during your debugging process.

For further learning, I'd highly recommend examining the source code of Active Record, specifically the `update`, `assign_attributes`, and `save` methods. It’s a very enlightening exercise. I'd also point to the Rails documentation itself on ActiveRecord validations and callbacks. They are clear, and quite helpful if you examine the examples carefully. In terms of books, "Agile Web Development with Rails 7" provides a detailed explanation of these core concepts. In particular, pay close attention to the lifecycle of active record objects, which the book explains nicely, giving context to the sequence of methods like assign_attributes, validations, and save. Also, the "Rails 7 API Documentation" is a treasure trove of specific details. Finally, for a deeper understanding of the patterns at play here, I would look at some of the more advanced sections of "Patterns of Enterprise Application Architecture" by Martin Fowler, as these issues often mirror problems with data mapping and domain layers in general.

Hopefully, this clarifies why your controller might appear to be updating records despite validation errors. The devil is often in the details, and in Rails, that often involves the subtle differences in methods that modify and save data.
