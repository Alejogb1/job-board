---
title: "how to call a predicate from another predicate in prolog?"
date: "2024-12-13"
id: "how-to-call-a-predicate-from-another-predicate-in-prolog"
---

Okay so you've got a situation where you need to trigger one predicate from inside another in Prolog yeah I get that it's like trying to get your function to call another function but Prolog's a bit different isn't it

Been there done that seen the t-shirt probably even designed the t-shirt myself at this point Let me tell you a story back in my early Prolog days I was working on this project it was a rule-based expert system for diagnosing network issues not gonna lie a bit ambitious for someone who barely knew what a cut was but hey we all start somewhere right So I had this predicate `check_connectivity(Device)` that was supposed to do a bunch of stuff like ping the device check its interfaces you know the drill And then I had this other predicate `analyze_logs(Device)` that was obviously meant to sift through the device logs for error messages

The genius me thought ok well if connectivity is down we definitely need to check the logs Makes sense in my head right WRONG I wrote something like this

```prolog
check_connectivity(Device) :-
  ping(Device, Status),
  (Status == ok -> true ;
    analyze_logs(Device)).

analyze_logs(Device) :-
  read_log_file(Device, LogContent),
  parse_log(LogContent, Errors),
  report_errors(Errors).
```

Yeah you can see the issue can't you It was a complete mess `analyze_logs` was getting called regardless of whether the ping was successful or not It was supposed to be an OR condition but noob-me made it a weird IF-ELSE it worked kinda but not in a good way The logic was just off and I was getting log analyses on healthy devices like what? Debugging that was a nightmare I spent hours staring at the tracer trying to figure out where my brain went wrong

The thing you gotta remember is that Prolog is all about logical flow not imperative control It's not like Python or Java where you're explicitly telling the program to do step by step in that order Prolog tries to satisfy goals it finds what works and doesn't care about what you might think is the "flow" so my mistake was assuming that the logic was implied and it wasn't

Okay lets get to the point the "how" part

The most basic way to do this is just to literally write it in your predicate's body you just put the other predicate name where you want it like this

```prolog
process_item(Item) :-
    validate_item(Item),
    normalize_item(Item, NormalizedItem),
    store_item(NormalizedItem).

validate_item(Item) :-
    % Some validation logic
    is_valid_type(Item).

normalize_item(Item, NormalizedItem) :-
    % Some normalization logic
    to_lower_case(Item, NormalizedItem).

store_item(Item) :-
  %Store the item
  write_to_db(Item).

is_valid_type(Item) :-
    % Check if Item is of correct type
    Item \= invalid.

to_lower_case(Item, LowerCaseItem):-
    downcase_atom(Item,LowerCaseItem).

write_to_db(Item) :-
    %Writing to db placeholder function
    format('~w~n', [Item]).
```

This means that to satisfy the goal `process_item(X)` Prolog first needs to satisfy `validate_item(X)` then `normalize_item(X)` and finally `store_item(X)` Each predicate is called in sequence and the variable bindings are passed along the way

Now sometimes you want more control like for example you might want to call a predicate only if a condition is met like my failed logging idea from way back then the `->` operator or cut (`!`) is your best friend but be careful with cut that thing is dangerous and can bite you in the a** so best if you avoid it if you can

```prolog
process_data(Data) :-
    check_precondition(Data, Status),
    (Status == ok -> process_valid_data(Data) ;
    process_invalid_data(Data)).

check_precondition(Data, Status) :-
    % Check preconditions
    is_data_complete(Data, IsComplete),
    (IsComplete == true -> Status = ok ; Status = invalid).

process_valid_data(Data) :-
    % Process if valid
    format('~w is valid data~n', [Data]).

process_invalid_data(Data) :-
    % Process if invalid
    format('~w is invalid data~n', [Data]).

is_data_complete(Data, IsComplete):-
    %Check if data is ok
    Data \= incomplete,
    IsComplete = true.

is_data_complete(_, false).
```

Here `check_precondition` sets a status based on some condition and that status drives a conditional call to either `process_valid_data` or `process_invalid_data` You can create more complex logical structures using these if-else conditions

Sometimes when you get into more complicated programs you'll have multiple options each requiring different sub-predicates here's an example of that:

```prolog
handle_request(Request) :-
    process_request(Request, Response),
    send_response(Response).

process_request(Request, Response) :-
    request_type(Request, Type),
    process_request_type(Type, Request, Response).


process_request_type(Type, Request, Response) :-
  Type == query, !, handle_query(Request, Response).
process_request_type(Type, Request, Response) :-
  Type == update, !, handle_update(Request, Response).
process_request_type(Type, Request, Response) :-
    Type == create, !, handle_create(Request, Response).
process_request_type(_, _, 'Error: unknown request type').

handle_query(Request, Response) :-
    % Logic for handling a query
    format('~w: Query processed~n', [Request]),
    Response = ok_query.


handle_update(Request, Response) :-
  % Logic for handling an update
  format('~w: Update processed~n', [Request]),
  Response = ok_update.

handle_create(Request, Response) :-
  % Logic for handling an create request
  format('~w: Create processed~n', [Request]),
  Response = ok_create.

request_type(query_1, query).
request_type(update_1, update).
request_type(create_1, create).
```

The main `handle_request` predicate first gets the request and it's type based on `request_type` then `process_request` figures out what specific process to trigger based on that type using `process_request_type`. The cut in there (`!`) tells prolog that if a type is matched then do not check the following cases that one might create problems so that's a quick and dirty way to handle multiple types of logic but do not use them if you are not experienced

Now you're thinking "Okay this is good stuff but where do I learn more?" Well I'm not gonna give you links go read some books man or papers there's good stuff there you can find "The Craft of Prolog" by Richard O'Keefe you get a full deep-dive into Prolog best practices or "Programming in Prolog" by Clocksin and Mellish is a classic intro it gives you the fundamental knowledge you need but if you want to get more specific papers on logic programming there's tons of those on ACM or IEEE explorer do a little digging you might get some surprises

I hope this clarifies it for you and you can now finally finish your expert system network troubleshooting like I should have done back then I remember that night i had to give up and get some sleep I just hope you don't give up on programming when things go south that's the worse you can do you gotta keep pushing no matter what
