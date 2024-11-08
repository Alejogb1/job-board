---
title: "QR Code to Database: Updating Records the Easy Way"
date: '2024-11-08'
id: 'qr-code-to-database-updating-records-the-easy-way'
---

```ruby
# routes.rb
get '/update_quantity/:item_id', to: 'items#update_quantity'

# items_controller.rb
class ItemsController < ApplicationController
  def update_quantity
    item = Item.find(params[:item_id])
    item.quantity -= 1
    item.save
    # Optional: redirect to a success page or show a confirmation message
    redirect_to root_path
  end
end
```
