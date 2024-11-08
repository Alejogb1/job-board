---
title: "Quick Tip: Sending Flash Messages in JSON Responses with Rails (DRY Approach)"
date: '2024-11-08'
id: 'quick-tip-sending-flash-messages-in-json-responses-with-rails-dry-approach'
---

```ruby
def do_stuff
  if params['stuff']
    begin
      Helper.do_stuff params['stuff']
      flash[:success] = 'Stuff done'
    rescue Exception => ex
      flash[:error] = ex.message
    end
  else
    flash[:error] = 'No stuff provided'
  end

  respond_to do |format|
    format.html { redirect_to :action => 'index' }
    format.json { render json: flash }
  end
end
```
