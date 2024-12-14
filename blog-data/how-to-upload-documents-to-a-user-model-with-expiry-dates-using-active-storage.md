---
title: "How to Upload documents to a User Model with expiry dates using Active Storage?"
date: "2024-12-14"
id: "how-to-upload-documents-to-a-user-model-with-expiry-dates-using-active-storage"
---

alright, let's talk about uploading documents to a user model with expiry dates using active storage. i've been down this road a few times, and it always seems a bit trickier than it first appears. it's one of those things that looks straightforward on the surface, but the devil, as they say, is in the details.

so, first off, you're dealing with a few different things here: active storage, which handles the file uploads, and then the expiry logic, which we'll need to implement ourselves. active storage is great for handling the actual storage of the files on different backends (like aws s3, google cloud storage, or even just locally) but it doesn't have any built-in functionality for setting expiration dates or automatically cleaning up expired documents. that's what we'll be tackling.

here's a breakdown of how i've handled this in the past, and it's a method i think works pretty well:

**the model**

let's start with your user model. we'll assume you've already got active storage set up and configured correctly. it's not that hard: `rails active_storage:install` does the trick and setting up the storage services in `config/storage.yml`. you should probably go over that documentation to get it right if you didn't do it before.

now, let's add the attachment and expiry logic. we'll use a `has_one_attached` association for a single document, but it's easily adaptable to `has_many_attached` if you need multiple files per user. we'll also need to store the expiration date itself, and for that i would use a database column named `document_expires_at` which should be of type `datetime`.

```ruby
class User < ApplicationRecord
  has_one_attached :document

  # document expiration
  attribute :document_expires_in, :integer
  attribute :document_expires_at, :datetime

  before_save :set_document_expires_at, if: :document_expires_in_changed?

  def document_expired?
    return false unless document_expires_at.present?
    document_expires_at < time.current
  end

  def document_with_expiry_upload(file, expiry_in_days)
    self.document.attach(file)
    self.document_expires_in = expiry_in_days.days
    self.save!
  end

  private

  def set_document_expires_at
    self.document_expires_at = Time.current + document_expires_in
  end

end

```

in this snippet:

*   `has_one_attached :document` sets up the active storage attachment.
*   `document_expires_in` is a virtual attribute which holds the duration for document validity, it does not save to the db.
*   `document_expires_at` will be the timestamp when the document expires, this one saves to the db.
*   `before_save :set_document_expires_at, if: :document_expires_in_changed?` makes sure that the expiration timestamp is calculated only if the expiry duration is changed.
*   `document_expired?` checks if the document is expired.
*   `document_with_expiry_upload` is our custom method which is more like a helper method that receives the file to upload and how long will it be valid, it will save the duration and then do the necessary steps.

**the controller**

next up, the controller action that handles the document uploads, the code will be something like this:

```ruby
class UsersController < ApplicationController
  def update
    @user = User.find(params[:id])

    if params[:user][:document].present? && params[:user][:document_expires_in].present?
      expiry_in_days = params[:user][:document_expires_in].to_i
      @user.document_with_expiry_upload(params[:user][:document], expiry_in_days)
      redirect_to @user, notice: 'document uploaded successfully.'
    else
      @user.update(user_params)
      render :edit, status: :unprocessable_entity
    end
  end

  private

  def user_params
    params.require(:user).permit(:name)
  end
end
```

key aspects here:

*   we are checking for the presence of document and expiry duration params before using our custom method.
*   if the document is present we calculate the expiry time in days to an integer and send it together with the document to our custom method.

**the cleanup job**

now, for the most crucial part: automatically cleaning up expired documents. we'll use a background job for this, which is the best method to do this, instead of using a rake task or another method that could be problematic.

```ruby
class CleanupExpiredDocumentsJob < ApplicationJob
  queue_as :default

  def perform
    User.find_each do |user|
      if user.document.attached? && user.document_expired?
        user.document.purge
        user.update(document_expires_at: nil) # remove the expiration time too.
      end
    end
  end
end
```

this job:

*   finds each user, the `find_each` method is preferred here due to performance reasons over `User.all`.
*   checks if the user has a document attached and is expired.
*   if so, it purges the attachment from active storage and removes the expiry timestamp too, so it doesn't try to delete it again.

**scheduling the job**

you'll need to schedule this job to run periodically. i usually use a gem like 'whenever' to make that easier. i generally schedule this job to run once a day in production, but you can adjust the schedule depending on how often you expect documents to expire. in your `config/schedule.rb` file you would add something like this:

```ruby
every 1.day, at: '4:00 am' do
  runner "CleanupExpiredDocumentsJob.perform_later"
end
```

this code would schedule the job to run every day at 4 am.

**things to consider**

*   **error handling:** you'll probably want to add some error handling in your background job, just in case something fails. catching exceptions would be better to know if a process failed.
*   **file sizes:** active storage handles file sizes pretty well, but you might want to put some limitations to your uploads.
*   **storage backend:** choosing the proper storage backend is crucial to avoid problems, using local storage is generally not a good idea for production environments.
*   **testing:** make sure to test everything thoroughly specially the cleanup job. a failing job can lead to problems.

**other options (but not so great)**

there are other ways to handle this. for example, you could try to delete the documents when the user tries to download them if they are expired. but that is not good since it would require to check the expiry date every time a user wants to get the file. it would increase processing and is not ideal. using the background job is the correct way. i've seen others use database triggers, but i'm not a fan, since that adds extra complexity to the database logic.

**resources**

to further your understanding of this you can check this out:

*   the official rails guide on active storage: i know it's not a book but if you didn't go over this you should definitely do it, it provides good coverage about the basics of active storage and the different features it provides.
*   the 'working with active storage in rails' talk from 'gorails'. it provides some examples and explanations that you might find useful.
*   the 'designing data-intensive applications' book by martin kleppmann: it's not about rails or active storage, but it's a good read about the general problems about data storage, and how to design applications to handle data storage problems better, it gives you a wider view of the problems we have to solve.
*   the book 'patterns of enterprise application architecture' by martin fowler: you may find that some patterns of this book are useful, this is a classic and may help to design applications better.

oh, and one more thing. a sql query walks into a bar, joins two tables, and orders a drink. but when the bartender asks why it's there, it simply replies: "i don't know, i was just following instructions!"

anyways, i think this gives you a solid foundation for handling document uploads with expiry dates using active storage in your rails app. good luck with it. i'm sure you will get it working soon enough. let me know if you have any more specific questions.
