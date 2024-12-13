---
title: "solverok function excel vba?"
date: "2024-12-13"
id: "solverok-function-excel-vba"
---

Okay so someone's asking about `solverok` in excel vba yeah I've been there dude many times like way way too many times alright let me just dump what I know about it its honestly not that complicated once you've wrestled it a bit

So `SolverOK` in VBA is basically your gateway to making Excel's Solver tool dance to your programmatic tune its part of the Solver add-in which you gotta enable first by the way if you havent done that already VBA cant see it If youre here and dont know that go to Excel Options add-ins and select the Solver Addin at the bottom Excel's settings can be surprisingly simple sometimes its like a car you have a steering wheel but that doesn't mean the engine is running you get me?

I remember I was building this massive optimization model back in the day maybe 2012ish something like that I was fresh out of uni and thought I was hot stuff you know the type I had spreadsheets longer than my attention span I was trying to find the optimal mix of ingredients for this theoretical cake company I had in my head. Yeah cake not exactly high tech but you have to start somewhere I was using Solver to find the minimum cost to hit some nutrition targets I was writing VBA macros to automate everything and thats when I first met my nemesis `SolverOK`

See the main issue I ran into is that `SolverOK` isn't just a magic wand that makes all your optimization problems go away you have to set it up properly its like telling a robot where to walk but not giving it a map it just spins in place so you have to define a bunch of parameters first you have to specify the objective function the changing variable cells the constraints etc etc and if you mess up even one tiny little thing `SolverOK` will throw an error or worse just give you a nonsensical result and you will have a hell of a time debuggign this its not that the debugger is bad but the process is so complex that the debugging process makes you feel like you are solving the problem again which well is not what you need since you already are solving it this was the most annoying thing I had to deal with I mean why would I solve the problem on debug? it does not even make sense

So let's break this down into the parts that matter a `SolverOK` call in VBA typically looks something like this:

```vba
Sub RunSolver()
    SolverReset
    SolverOk SetCell:="$B$10", MaxMinVal:=1, ValueOf:=0, ByChange:="$B$2:$B$6", Engine:=1, EngineDesc:="GRG Nonlinear"
    SolverAdd CellRef:="$B$2:$B$6", Relation:=1, FormulaText:="1"
    SolverAdd CellRef:="$B$2:$B$6", Relation:=3, FormulaText:="0"
    SolverSolve UserFinish:=True
End Sub
```

Alright so let me explain this like we are on the same page here first `SolverReset` clears out any previous Solver settings you dont want old ghosts messing with your new solve it's like clearing your browser history before you search for something embarrasing so first things first clear the mess

Then we have the main event `SolverOk` the set cell argument `SetCell:="$B$10"` that specifies where your objective function is located in this case cell B10 so B10 should have the formula that calculates the thing you are trying to maximize or minimize it's a reference to the cell containing your ultimate goal in my cake example it was a formula that added all the cost per gram of ingredients and multiplied by how much you were using

`MaxMinVal:=1` determines if you are maximizing which is a 1 or minimizing which would be a 2 in this particular case I am setting it to 1 means its maximizing and well 2 means its minimizing simple stuff right

Then `ValueOf:=0` is if you are trying to set the SetCell to a specific value. If you are only maximizing or minimizing this can be a 0 this is set to 0 in this example cause were maximizing if you were setting the objective to a given value it would be that value

`ByChange:="$B$2:$B$6"` are the cells which the solver changes to find the optimal solution these are your adjustable variables like how much of each ingredient you use in the cake example this range is B2 to B6

`Engine:=1` chooses the engine type there are 3 the 1 being GRG Non-Linear which works better for non-linear problems then we have 2 the Simplex LP which is for linear problems and then there is 3 the Evolutionary which is for non-smooth problems use this last one if the first two are not working for you and if you want to do complex multi-variate optimization. `EngineDesc:="GRG Nonlinear"` just tells it to use that engine type which if you know the engine id you dont need but it is good for self-documentation of your code.

Now the part that most new users get wrong and its very hard to debug well actually its hard to debug almost everything with this function is the `SolverAdd` stuff here we are setting the constraints which is very important this solver will use these contraints to navigate the search space. `CellRef:="$B$2:$B$6"` the same variables from `ByChange` these are the cell variables we are contraining `Relation:=1` means this value has to be greater than or equal to the number specified in the `FormulaText` which will be 1 in this case `Relation:=3` means the value must be less than or equal to the number specified in the `FormulaText` which will be 0

So this code forces every variable from B2 to B6 to be between 0 and 1 which is a very commom situation when doing optimization so you can not be using less than zero and more than 1 for these variables that you defined in the `ByChange` this is simple but you would not imagine the amount of time I wasted in silly errors like this

Finally `SolverSolve UserFinish:=True` it tells Excel to actually solve the model. The `UserFinish:=True` means you want to get control back after the solver finishes its work otherwise you might have to manually get the data out from excel.

The `SolverOK` function itself returns a boolean value true or false based on whether or not the solver parameters were correctly set this is useful for error handling and you should always use it to see if something went wrong if it returns false it means something in your parameter setup was invalid and you have to correct it otherwise the `SolverSolve` might or might not return an answer and then you would have a hell of a time debugging

Another example I want to show you is adding multiple constraints at once:

```vba
Sub RunSolverWithMultipleConstraints()
    SolverReset
    SolverOk SetCell:="$C$15", MaxMinVal:=2, ValueOf:=0, ByChange:="$C$2:$C$10"
    SolverAdd CellRef:="$C$2:$C$10", Relation:=4, FormulaText:="integer"
    SolverAdd CellRef:="$C$12", Relation:=2, FormulaText:="100"
    SolverAdd CellRef:="$C$13", Relation:=3, FormulaText:="50"
    SolverSolve UserFinish:=True
End Sub
```

In this example `SolverAdd CellRef:="$C$2:$C$10", Relation:=4, FormulaText:="integer"` specifies that the changing variables must be whole numbers this is a common constraint when you are for example trying to optimize production of things that you can not produce as fractions. And `SolverAdd CellRef:="$C$12", Relation:=2, FormulaText:="100"` means that cell C12 which probably has some other formula calculating something else has to be less than or equal to 100 and `SolverAdd CellRef:="$C$13", Relation:=3, FormulaText:="50"` means that cell C13 has to be greater than or equal to 50 so you are adding multiple constratins and even constraint types at once all the relations are explained here 1 is greater or equals 2 is equal to 3 is less or equal to and 4 is integer

This example shows you that you can add constraints to more than just the changing cells.

One more example because honestly this is important and it involves one of the most frustrating errors when using the solver the "solver could not find a feasible solution" error which is often solved by simply changing the starting values in the changing variable range

```vba
Sub RunSolverWithStartingValues()
    SolverReset
    SolverOk SetCell:="$D$12", MaxMinVal:=2, ValueOf:=0, ByChange:="$D$2:$D$6"
    SolverAdd CellRef:="$D$2:$D$6", Relation:=3, FormulaText:="10"
    SolverAdd CellRef:="$D$2:$D$6", Relation:=1, FormulaText:="0"
    Range("D2").Value = 2
    Range("D3").Value = 3
    Range("D4").Value = 5
    Range("D5").Value = 7
    Range("D6").Value = 1
    SolverSolve UserFinish:=True
End Sub
```

This example is similar to the first with a simple change we manually set the starting values on the changing variable range D2 to D6 because if Excel starts at say 0 on every value and your optimization problem requires something greater than zero you will be given the dreaded "solver could not find a feasible solution" and you will get stuck for hours with a simple variable initializaiton.

Look `SolverOK` isn't perfect sometimes its like trying to teach a cat to code it requires patience persistence and a lot of trial and error It once took me two whole days to find a simple error where I had flipped the relation and it was all because I was tired of looking at spreadsheets but at the end I am glad I had gone through it I am still glad I go through it I like it so I would also recommend if you want to start using VBA like this go through with it because this is one of the most common types of optimization that people will need to do.

If you want to get deeper into optimization techniques I'd suggest checking out something like "Numerical Optimization" by Jorge Nocedal and Stephen J. Wright it's a hefty book but it covers a lot of the theory behind algorithms used by solver or if you want a simpler approach to optimization maybe check out "Introduction to Linear Optimization" by Dimitris Bertsimas and John Tsitsiklis if your optimization is linear in nature. Those resources will give you a solid grasp of what's actually going on behind the scenes.

Hope this helps and good luck with your optimization problems.
