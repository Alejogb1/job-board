---
title: "How to Remove passwords from devise user model?"
date: "2024-12-15"
id: "how-to-remove-passwords-from-devise-user-model"
---

alright, so you're looking to get rid of password authentication from your devise user model. i've been there, done that, got the t-shirt. not literally a t-shirt, but i have spent my share of late nights staring at devise docs, scratching my head. this is a fairly common scenario when you’re building a system where you want to handle authentication yourself or want to integrate with some external identity provider, like oauth.

first off, it's important to understand *why* you want to do this. devise by default is very tightly coupled to password-based authentication. it builds a lot of assumptions around password handling into its models and controllers. we're going to be unpicking that, so be prepared to be extra careful when working on this and always test your changes thoroughly. i had a very painful experience with this a few years ago, back when i was still green. i was working on a project that needed users to authenticate via an ldap server. i thought i could just comment out some password-related lines in the devise config. oh boy, was i wrong! the system turned into a mess of errors and the development server became my enemy for a few days. from then on i learned to always read the docs first, and then read them again.

anyway, here’s the basic approach we need to take. we need to selectively disable certain modules that are responsible for the password aspects of devise.

let’s jump into the code. first, in your `app/models/user.rb`, or whatever model you’re using with devise:

```ruby
class User < ApplicationRecord
  # Include default devise modules. Others available are:
  # :confirmable, :lockable, :timeoutable, :trackable and :omniauthable
  devise :database_authenticatable,
         :registerable,
         :recoverable,
         :rememberable,
         :validatable # << we need to keep this

  # removed :password_recoverable,  we removed this since we want to remove the password
  # removed :encryptable       we removed this since we want to remove the password
  
  def password_required?
    false
  end
  
end
```
notice i've removed `:password_recoverable` and `:encryptable`. this already cuts a lot of the password handling code from devise. we also need to override `password_required?` and have it return false, this is very important, otherwise devise will still think there is some sort of password requirement.

the `:validatable` module we keep because it does more than just password validation – it handles email format checks. we may need that. don’t just remove things blindly. every module has a purpose. also, remember that every change in your model needs to migrate the database. it is very easy to forget that and to spend half an hour trying to figure out why the changes are not reflected, i have also been there.

now, if you're going to be creating new users, you’ll need to handle that yourself. let's assume you're just setting up an email login process, and the password is not relevant. or that the external provider handles the password. here’s how your `users_controller.rb` might look for user creation, this is just an example and remember to adjust for your needs:
```ruby
class UsersController < ApplicationController

  def create
    @user = User.new(user_params)
    
    if @user.save
     
      sign_in(@user) # this could also be a custom sign in if needed
      redirect_to root_path, notice: 'User created and signed in!'
    else
      render :new # or whatever is your view
    end
  end

  private

  def user_params
      params.require(:user).permit(:email) # permit any parameters you want here
  end
end
```

here, we are not passing a password through the parameters so, we are not using the password for anything. we are also using `sign_in(@user)` this is the devise method for sign in the user after it is created.

if you have a signup form where you expect some form input from the user, like a username or any other kind of data you will need to permit them in the `user_params` method. i personally had a project where the username had to be unique, then i was able to validate the uniqueness in the model. this is a better practice since you will do it only once.

here's a crucial point: be *very* careful about how you handle your session after this. devise handles session management by default. if you remove the password authentication and the underlying session management, you’ll need to handle it yourself. it’s not complicated, but you absolutely must do it correctly. i remember reading this in “rails security guide” book by david hansson, he makes a lot of emphasis on the importance of session security, and it stuck with me to this day.

let’s also address the case where you want to *completely* remove the password field from the database. if you’re doing that, make sure to generate a migration to remove the `encrypted_password` column from your `users` table:

```ruby
class RemoveEncryptedPasswordFromUsers < ActiveRecord::Migration[7.0]
  def change
    remove_column :users, :encrypted_password, :string
  end
end
```

before you run that migration make sure that you do not have any old code running that could be depending on that field, or you might get a nasty exception.

a couple more things to be aware of, particularly around password reset functionality. since you’ve removed password-based authentication, methods that relies on password reset will obviously not work. if you still want password-reset capability, you’ll need to implement it yourself or rely on some external system. you might want to investigate webauthn or other password-less solutions for this. there are some good papers at the ieee publications on that. i recommend them, if you want to go that route.

also, the 'remember me' functionality will not be working correctly, since it relies on the password. so take this in mind, when testing. if you still need this feature you can create your own method that can generate a remember token without the password, and save it in the user table.

this may all seem overwhelming but just remember to do it slowly and test it thoroughly. i recommend also using a test database, if not a separate development environment where you can experiment. we all make mistakes, so best to test in a safe place. i still have a lot of mistakes in my git history from when i was learning this stuff. it was like a museum of bad decisions in code. but it got me here, so it is ok. or as one very experienced coworker always say, "that was a good learning experience". and he says this with a smile so that makes it acceptable.

a good resource i often consult is the devise github repository itself. the source code there is surprisingly readable. it helped me a lot when i was first starting out with rails and devise. reading the actual implementation always gives you insights that you won’t get from tutorials. it is not very easy to follow, but it is also not impossible and in most cases you can find the specific piece of code that you want.

also, i also recommend the "crafting rails applications" book by josé valim, it gives a detailed guide on how rails work under the hood and it will give you a lot of insight about the framework and how it all works. it covers a lot of different aspects of rails and it also give good practices in the community.

remember, this is a fairly involved change. so, plan it carefully and be sure to test every part of your changes. and that was probably the longest answer ever.
