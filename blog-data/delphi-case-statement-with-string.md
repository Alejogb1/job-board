---
title: "delphi case statement with string?"
date: "2024-12-13"
id: "delphi-case-statement-with-string"
---

 so delphi case statement with strings right been there done that got the t-shirt Let me tell you this ain't exactly a walk in the park like some other languages I've seen Yeah some folks might say use a bunch of `if then else` but trust me that road leads to spaghetti code madness faster than you can say "compiler error" and we all know that’s not a good look

So here’s the deal Delphi’s `case` statement is primarily designed for ordinal types integers characters enums that sort of jazz Strings however they are like that one cousin you have who doesn’t quite fit in at family gatherings Not directly compatible with `case` So you can't just drop in a string and expect it to work you need to get a little creative You have options though not as many as i would like but they get the job done

The simplest but most restrictive way is using an integer hash So here is how it went for me first time it happened at a place i worked at back in 2012 I had to process different text commands like "start" "stop" "restart" coming in from a socket connection It was a simple automation app controlling some testing equipment I was like ok case statement here i come But oh boy was i wrong Delphi was like NOPE and i was like ok i see you So i decided to hash the strings into integers and then i compared the integers That’s what i did and below is how i did it

```delphi
function StringHash(const Str: string): Integer;
var
  i: Integer;
  hash: Integer;
begin
  hash := 0;
  for i := 1 to Length(Str) do
  begin
    hash := hash * 31 + Ord(Str[i]);
  end;
  Result := hash;
end;

procedure ProcessCommand(const Command: string);
var
  hash: Integer;
begin
  hash := StringHash(Command);
  case hash of
    StringHash('start'):
      begin
        // Start stuff here
        ShowMessage('Starting');
      end;
    StringHash('stop'):
      begin
        // Stop stuff here
        ShowMessage('Stopping');
      end;
    StringHash('restart'):
      begin
         //Restart stuff here
         ShowMessage('Restarting');
      end;
   else
      ShowMessage('Unknown Command');
  end;
end;
```

Now see that’s some code that works and it’s simple enough right You create your function hash the string to an integer and use the hash function inside the case statement for each branch Pretty standard really There are other more elaborate and better hashing algorithms you can check Knuth's "The Art of Computer Programming" if you are really keen but for simple scenarios that’s good enough. This method gets the job done no doubt about that but it comes with a caveat Potential hash collisions that means different strings might end up with the same hash which is basically very bad so you always need to keep that in mind That's the issue i had some weird cases would just not work due to collisions and the app was misbehaving i ended up debugging it for almost a whole day until i realised the issue was not with the business logic but with my hash function! So you need to be very careful using this method You'd think that after that mishap i would learn but nope i tried the following method next time i had to do something similar and i had to fix that too!

Another option which i found out while browsing a few forums back in 2015 is using a lookup table or a dictionary type data structure Delphi doesn’t have a built-in dictionary in the standard library pre 2009 although that’s not an issue since you can easily implement one with generics Nowadays of course it is not a problem since we have the `TDictionary` but back in the day we had to write it from scratch using `TList` or some such data structure and frankly speaking i was too lazy to write a dictionary from scratch every single time I just wanted a simple lookup mechanism so I did this with a class declaration. So basically you store the string and a associated handler to execute when a specific string is found. This method avoids the collision issues from the previous example That approach worked really well for the project i was working on at that time and I felt really proud of myself

```delphi
type
  TCommandHandler = procedure of object;

  TCommandDispatcher = class
  private
    FCommands: TList<TCommandHandler>;
    FCommandNames: TList<string>;
    function GetIndex(const Command: string): Integer;
  public
     constructor Create;
    destructor Destroy; override;
    procedure AddCommand(const CommandName: string; const Handler: TCommandHandler);
    procedure DispatchCommand(const Command: string);
  end;

implementation
constructor TCommandDispatcher.Create;
begin
  FCommands := TList<TCommandHandler>.Create;
  FCommandNames := TList<string>.Create;
end;

destructor TCommandDispatcher.Destroy;
begin
  FCommands.Free;
  FCommandNames.Free;
  inherited;
end;

function TCommandDispatcher.GetIndex(const Command: string): Integer;
var
  i: Integer;
begin
  Result := -1;
  for i := 0 to FCommandNames.Count -1 do
  begin
    if FCommandNames[i] = Command then
    begin
      Result := i;
      Exit;
    end;
  end;
end;


procedure TCommandDispatcher.AddCommand(const CommandName: string; const Handler: TCommandHandler);
begin
    FCommandNames.Add(CommandName);
    FCommands.Add(Handler);
end;

procedure TCommandDispatcher.DispatchCommand(const Command: string);
var
  Index : Integer;
begin
  Index := GetIndex(Command);
  if Index <> -1 then
  begin
     FCommands[Index]();
  end
  else
  begin
    ShowMessage('Unknown command: ' + Command);
  end;
end;

// Example usage
procedure StartHandler;
begin
  ShowMessage('Start command executed');
end;

procedure StopHandler;
begin
  ShowMessage('Stop command executed');
end;

procedure RestartHandler;
begin
   ShowMessage('Restart command executed');
end;


var
  Dispatcher: TCommandDispatcher;

begin
  Dispatcher := TCommandDispatcher.Create;
  Dispatcher.AddCommand('start', StartHandler);
  Dispatcher.AddCommand('stop', StopHandler);
  Dispatcher.AddCommand('restart', RestartHandler);

  Dispatcher.DispatchCommand('start');
  Dispatcher.DispatchCommand('stop');
  Dispatcher.DispatchCommand('restart');
  Dispatcher.DispatchCommand('invalid');

  Dispatcher.Free;
end.
```

The cool thing is this isn’t just limited to string command processing it’s a nice dispatcher pattern you can use almost anywhere which is why it’s so useful. You need to make sure you handle the case where there is an unknown command of course otherwise your app will be very unfriendly. This approach is better in terms of collision avoidance because it uses direct string comparison instead of hashes but it has a little performance penalty because it needs to do a linear search which means the more commands you have the slower it gets Now in my current role i am in a team where we are handling tons of text coming in from users and you would not believe this but we actually use the dictionary method nowadays with `TDictionary` which is much much simpler and better than what I used to use and the performance is great

Another approach i saw somewhere in the Borland newsgroups from back in 2003 (it is amazing how much information is there in those archives) is to use a `TStringList` So basically you can use a `TStringList` to store strings and then using the `IndexOf` function of this class to determine the index of a string it is another way to create a sort of lookup table This method uses a list which has some interesting properties but is pretty similar to the previous one. Here is how this method looks

```delphi
procedure ProcessCommand(const Command: string);
var
  CommandList: TStringList;
  Index: Integer;
begin
  CommandList := TStringList.Create;
  try
    CommandList.Add('start');
    CommandList.Add('stop');
    CommandList.Add('restart');

    Index := CommandList.IndexOf(Command);
    case Index of
      0:
        ShowMessage('Starting');
      1:
        ShowMessage('Stopping');
      2:
        ShowMessage('Restarting');
    else
      ShowMessage('Unknown Command');
    end;
  finally
    CommandList.Free;
  end;
end;

```

It works but be careful because `TStringList` is really powerful and has a lot of stuff inside that you might not need. So for a simple use case like this it’s probably an overkill but for complex cases that's a really nice option to have.

So there you have it three approaches to tackle the Delphi `case` statement and string problem. The hash approach is quick and easy but can be a headache The dictionary/lookup method is more robust and less prone to collisions The string list method works as a good middle ground.

If you want to deep dive into Delphi performance optimization then i would recommend you to check "Delphi in Depth" by Cary Jensen and Loy Anderson is a classic in my opinion. Also check out "Code Complete" by Steve McConnell it is more of a general programming book but the principles and techniques there apply perfectly to Delphi. There is also "Effective Delphi" by Alister Christie It has a bunch of very useful things that you should learn about Delphi.

And one thing I learnt the hard way is that premature optimization is the root of all evil especially when you're doing string processing so choose the best method based on your specific case Don't just copy paste what someone in the internet tells you that they think it is the best and I think that is a good rule to live by overall.
