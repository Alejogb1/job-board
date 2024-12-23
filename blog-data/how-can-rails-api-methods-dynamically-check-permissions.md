---
title: "How can Rails API methods dynamically check permissions?"
date: "2024-12-23"
id: "how-can-rails-api-methods-dynamically-check-permissions"
---

Alright, let's unpack this. Permissions in Rails APIs—a cornerstone for any robust application. I’ve spent a good chunk of my career ensuring APIs are not just functional, but also secure, and dealing with dynamic permissions has been a recurring theme. Rather than a static, “user is admin” approach, we often need something finer-grained, adaptable to various user roles and resource access levels. This isn't a trivial task, but certainly manageable with the right strategies.

Dynamic permission checking in a Rails API essentially means validating, often at the controller level, whether the current user is authorized to perform a specific action on a particular resource, where the permissions themselves might be determined by a multitude of factors. These factors could include the user’s role, the current state of the resource, or even external service responses. Hardcoding these rules is a recipe for disaster—it leads to brittle code, difficult maintenance, and a serious security risk.

My go-to approach invariably involves a combination of a well-defined permission system and an easily extensible authorization framework, usually built on top of something like cancancan or pundit (although I've moved away from those lately, as the requirements have become much more bespoke). I prefer a more tailored solution now, and let me elaborate on why. The pre-packaged solutions are useful but can sometimes be too rigid for the kind of granular control required in complex systems.

When I build permissions systems, I favor a model where each action has a specific “permission” associated with it. This permission isn't a string like “edit_post” but rather a checkable unit that encapsulates logic. These permission units can be as simple as checking a user's role or as complex as evaluating a set of conditions derived from both the user and the target resource. This approach gives far greater flexibility than using static role-based permissions. It also allows us to implement data-dependent authorization, which is crucial.

Let's start with a basic example to illustrate this concept. Imagine you're building an API for a document management system. Users can create documents, but they should only be able to edit or delete documents they created, or where an admin or manager grants those specific privileges. First, we define our permission class:

```ruby
# app/services/permissions/document_permission.rb
class Permissions::DocumentPermission
  def initialize(user, document)
    @user = user
    @document = document
  end

  def can_edit?
    @user.admin? || @document.user_id == @user.id || is_collaborator_with_edit_access?
  end

  def can_delete?
      @user.admin? || @document.user_id == @user.id || is_collaborator_with_delete_access?
  end

  private

  def is_collaborator_with_edit_access?
    @document.collaborators.exists?(user_id: @user.id, access_level: 'edit')
  end

  def is_collaborator_with_delete_access?
    @document.collaborators.exists?(user_id: @user.id, access_level: 'delete')
  end
end
```

Here, `DocumentPermission` encapsulates the permission checks specific to documents, checking not just user roles, but also resource ownership and external collaborations. The crucial point here is that the logic is not contained within the controller, but is instead managed as a distinct unit. Now, in the controller action, this would be used as follows:

```ruby
# app/controllers/api/v1/documents_controller.rb
class Api::V1::DocumentsController < ApplicationController
  before_action :authenticate_user!
  before_action :set_document, only: [:update, :destroy]

  def update
    permission = Permissions::DocumentPermission.new(current_user, @document)
    unless permission.can_edit?
       return render json: { error: 'Not authorized to edit this document' }, status: :forbidden
    end

    if @document.update(document_params)
      render json: @document
    else
      render json: { errors: @document.errors }, status: :unprocessable_entity
    end
  end

  def destroy
    permission = Permissions::DocumentPermission.new(current_user, @document)
      unless permission.can_delete?
        return render json: { error: 'Not authorized to delete this document' }, status: :forbidden
      end

    @document.destroy
    head :no_content
  end

  private

  def set_document
      @document = Document.find(params[:id])
  rescue ActiveRecord::RecordNotFound
      render json: {error: 'Document not found'}, status: :not_found
  end

  def document_params
    params.require(:document).permit(:title, :content)
  end
end

```

This snippet showcases how you could use the previously defined permission check in the `update` and `destroy` actions of a `DocumentsController`. Notice how we're not hardcoding logic within the controller. If the criteria for editing a document changes, you simply modify the `DocumentPermission` class, maintaining a separation of concerns.

Let's consider a slightly more advanced scenario. Imagine we need to fetch user data, but certain attributes are only visible to the user themselves or to a specific role like “manager”. This involves not just action-level permissions, but also data-level permissions. This typically involves overriding the serialization process.

```ruby
# app/serializers/user_serializer.rb
class UserSerializer < ActiveModel::Serializer
  attributes :id, :username, :email, :created_at

  def attributes(*args)
    hash = super
    if  object == current_user || current_user.manager?
      hash[:phone_number] = object.phone_number
      hash[:address] = object.address
    end
    hash
  end
  
  def current_user
      scope
  end
end
```

Here, in `UserSerializer`, certain attributes are conditionally included based on the current user context. This approach makes sure that even when the data is retrieved, unauthorized users don't get sensitive information. Note that the `scope` used is typically provided by a gem like `active_model_serializers` that provides the context from the request. I have not added the full setup for this gem for brevity.

Finally, consider a case where permissions depend on external factors. Imagine an API for booking events where permissions depend on whether an event seat is available. This involves interacting with a service to check seats:

```ruby
# app/services/permissions/event_permission.rb
class Permissions::EventPermission
  def initialize(user, event, booking_service)
     @user = user
    @event = event
    @booking_service = booking_service
  end

  def can_book?
    return false unless @user.is_registered
    @booking_service.available_seats(@event) > 0
  end
end

#app/services/booking_service.rb
class BookingService
  def available_seats(event)
      event.total_seats - event.bookings.count
  end
end

#app/controllers/api/v1/events_controller.rb
class Api::V1::EventsController < ApplicationController
    def book
     event = Event.find(params[:id])
     booking_service = BookingService.new
    permission = Permissions::EventPermission.new(current_user, event, booking_service)
    unless permission.can_book?
        return render json: {error: 'Booking not available'}, status: :forbidden
    end
      #Logic to book event goes here
  end
end
```

In this example, `EventPermission` uses the `BookingService` to check for seat availability, which affects the user's booking permissions, illustrating how external factors can also dynamically influence permissions.

For resources, I'd strongly advise looking at the "Domain-Driven Design: Tackling Complexity in the Heart of Software" by Eric Evans, especially for the concepts around the domain model and service layers. For a solid understanding of REST APIs in general, I’d suggest reading “RESTful Web Services” by Leonard Richardson and Sam Ruby. Furthermore, a solid resource for understanding authorization best practices is "Web Security: A White Hat Perspective" by Michael Howard. These books offer a more rigorous background on many principles discussed, and I reference them often.

In practice, achieving a robust and dynamically permissioned API is an ongoing effort. It requires constant refinement, particularly as business logic becomes more complex, however by focusing on distinct, testable, and reusable permission logic, you can ensure a flexible and secure system. In my experience, the initial overhead in building something tailored pays off significantly in long-term maintainability and scalability. The approach I’ve outlined provides a framework which can be extended and adapted to an enormous variety of use cases. This is not a one-size-fits-all situation; it’s about understanding the fundamentals of building a reliable, permission-checking framework.
