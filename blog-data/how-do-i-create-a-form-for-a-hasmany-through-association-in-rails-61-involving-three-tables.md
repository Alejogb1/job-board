---
title: "How do I create a form for a has_many through association in Rails 6.1 involving three tables?"
date: "2024-12-23"
id: "how-do-i-create-a-form-for-a-hasmany-through-association-in-rails-61-involving-three-tables"
---

Alright, let’s tackle this. I’ve certainly been down this road before, and it’s a common scenario when you’re dealing with relational databases in Rails, especially when introducing that `has_many :through` association. The trick is understanding how Rails handles nested attributes and then translating that into a workable form.

Let's assume for a moment you’re building a system for a university and you have three tables. First, `students`. Second, `courses`. And third, an intermediary `enrollments` table that links students to courses, indicating which student is enrolled in which course. Here, `enrollments` is your joining table, crucial for the `has_many :through` relationship.

**The Model Setup**

We have the basic models laid out like this:

```ruby
class Student < ApplicationRecord
  has_many :enrollments
  has_many :courses, through: :enrollments
end

class Course < ApplicationRecord
  has_many :enrollments
  has_many :students, through: :enrollments
end

class Enrollment < ApplicationRecord
  belongs_to :student
  belongs_to :course
end
```

This structure clearly defines how the tables relate: a student can have many courses through enrollments, and a course can have many students also through enrollments. Now, the form… that’s where things get interesting.

**The Form Challenge**

The aim here isn’t just to edit a student or course directly, but to manage the *association* between them. We want a form, typically on the student’s editing interface, where we can select which courses a particular student is enrolled in. This means we need to construct a form capable of handling nested attributes, specifically nested attributes for our `enrollments` table.

**Nested Attributes & `accepts_nested_attributes_for`**

Rails provides `accepts_nested_attributes_for`, which is precisely what we need here. This method, placed in the model, lets us update attributes of a related model in the parent's form. Let's modify our `Student` model:

```ruby
class Student < ApplicationRecord
  has_many :enrollments
  has_many :courses, through: :enrollments

  accepts_nested_attributes_for :enrollments, allow_destroy: true
end
```

Here, `accepts_nested_attributes_for :enrollments` tells Rails that we'll send data for creating, updating, or destroying `enrollment` records through our student form. The `allow_destroy: true` option is critical because it permits removing enrollments by setting a `_destroy` key on the nested attributes hash, a feature that’ll be important in our form.

**The Form Construction**

Now, let's consider the form structure. Since we’re generally working with an existing student, we typically are in an `edit` view, so the form would look something like this (using `form_with`):

```erb
<%= form_with model: @student, local: true do |form| %>

  <!-- Existing student fields -->
  <%= form.label :name %>
  <%= form.text_field :name %>

  <!-- Enrollment fields (this is the crux of it) -->
  <h3>Enrolled Courses</h3>
  <%= form.fields_for :enrollments do |enrollment_form| %>
    <% course = enrollment_form.object.course %>
      <div>
        <%= hidden_field_tag "student[enrollments_attributes][#{enrollment_form.index}][id]", enrollment_form.object.id %>
        <%= hidden_field_tag "student[enrollments_attributes][#{enrollment_form.index}][_destroy]", 0 %>
        <%= check_box_tag "student[enrollments_attributes][#{enrollment_form.index}][id]", course.id, !course.nil?, { onclick: "toggleEnrollment(this, #{enrollment_form.index})" } %>
        <%= course.name if course %>

      </div>
  <% end %>
    <br>
    <div>
      <% Course.all.each do |course| %>
        <div>
          <%= check_box_tag "student[enrollments_attributes][#{SecureRandom.uuid}][course_id]", course.id, false, { onclick: "toggleNewEnrollment(this)"} %>
          <%= course.name %>
        </div>
      <% end %>
    </div>
  <%= form.submit %>
<% end %>

<script>

    function toggleNewEnrollment(checkbox) {
        if (!checkbox.checked) {
            checkbox.closest('div').remove();
        }
    }

    function toggleEnrollment(checkbox, index) {
    const hiddenDestroyField = document.querySelector(`input[name='student[enrollments_attributes][${index}][_destroy]']`);
    hiddenDestroyField.value = checkbox.checked ? 0 : 1;
    }

</script>
```

Here’s a detailed breakdown:

1.  **`form_with model: @student`**: This is a standard Rails form for our `Student` object.
2.  **`form.fields_for :enrollments`**: This creates a set of nested fields for our `enrollments`. Notice that each `enrollment_form` iterates through existing enrollments. This is key because we need to handle the existing ones.
3.  **Checkboxes:** In both the existing and new `enrollment` fields, we provide a checkbox for each course.  The logic for whether a student is enrolled in a course is handled by the checkbox’s `checked` state, which corresponds to existing enrollments or creating new enrollments. We must send all enrollment details every time we update a student so Rails can create, modify, or destroy as needed.
4.  **Hidden Fields:** Crucially, we’re including `hidden_field_tag`s for the `id` and the `_destroy` parameters to both update existing enrollments or destroy them. The JavaScript function `toggleEnrollment` toggles the `_destroy` value, from `0` (keep) to `1` (destroy). Additionally, the `toggleNewEnrollment` removes the new enrollment rows when unchecked.
5.  **New Enrollment Options:** Here we have an additional set of checkboxes that loops through all available courses. These allow us to create new enrollments when the form is submitted. Since we are creating a new enrollment rather than modifying an existing, we do not need the `id` or `_destroy` hidden fields.
6. **SecurityRandom UUID:** Since there is no `index` to use when creating new enrollments, we generate a random uuid when creating new checkboxes. This is so that Rails can discern which records are new versus existing.

**The Controller**

The controller logic needs to permit the nested attributes:

```ruby
class StudentsController < ApplicationController

  def edit
    @student = Student.find(params[:id])
  end

  def update
    @student = Student.find(params[:id])
    if @student.update(student_params)
      redirect_to @student, notice: 'Student was successfully updated.'
    else
      render :edit
    end
  end

  private

  def student_params
    params.require(:student).permit(:name, enrollments_attributes: [:id, :course_id, :_destroy])
  end
end
```

In the `student_params` method, the `permit` call specifically authorizes the `enrollments_attributes`, which will include their id, the course_id, and `_destroy`.

**Resource Recommendations**

For further exploration, I strongly suggest diving into the official Rails documentation on Active Record Associations and Nested Attributes. Additionally, "Agile Web Development with Rails" by Sam Ruby et al. provides extensive coverage of model associations and form construction. For a more in-depth understanding of relational database concepts, C.J. Date's "An Introduction to Database Systems" remains a classic.

**Important Caveats**

One thing to note: The approach above relies on sending all enrollment details on every submission. It is important to be aware of this as application complexity increases, and could result in performance issues with very large datasets. You may want to consider alternative approaches with AJAX and partial updates if this becomes an issue. Another caveat is, that when creating new enrollment records, we have to generate a new uuid to use as an index for the nested attributes. This is why we use `SecureRandom.uuid` when generating the new checkboxes in the form.

This setup, although it may appear complex on the surface, is actually quite elegant in its simplicity, and can be generalized to handle any `has_many :through` association with nested attributes in Rails. It's an area that can seem confusing initially, but once you’ve spent some time building out these types of forms, it becomes second nature. Remember, understanding how Rails maps form data to model attributes is key to getting it right. And while this is a good solution, always think through the requirements of your application, as alternative approaches may be more appropriate in certain circumstances.
