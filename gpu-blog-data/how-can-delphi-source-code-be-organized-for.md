---
title: "How can Delphi source code be organized for optimal compiler performance?"
date: "2025-01-30"
id: "how-can-delphi-source-code-be-organized-for"
---
Delphi's compiler, while generally efficient, can still benefit substantially from a strategically organized codebase. Specifically, the dependency graph formed by unit uses clauses and interface definitions significantly impacts compile times. A sprawling, tangled dependency web forces the compiler to repeatedly process units, negating potential optimizations like cached intermediate code representations. Optimizing source code organization centers around minimizing these unnecessary recompilations.

My experience over several large Delphi projects, ranging from financial applications to real-time control systems, has consistently shown that even modest attention to unit organization yields considerable reductions in build times. The problem often arises subtly; an initially well-structured application can gradually degrade as new features are added without strict adherence to architectural principles. The primary focus should always be on managing unit dependencies, aiming for shallow, unidirectional relationships rather than deep, cyclic ones.

A key technique is the separation of concerns, both at an architectural and at a unit level. A poorly factored unit, doing too much, will inevitably be a dependency of many other units, causing a ripple effect whenever changes are made. Similarly, circular dependencies, where unit A uses unit B, and unit B uses unit A (either directly or indirectly), are particularly pernicious. The compiler cannot process these units effectively in isolation, often requiring it to recompile both, and sometimes several other affected units, after even minor modifications. The goal, therefore, is to design units that represent cohesive responsibilities and to manage dependencies strictly through interfaces and minimal implementation knowledge. This means striving for low coupling and high cohesion.

Here's a breakdown of specific strategies and code examples:

**1. Interface Separation & Abstraction:**

Direct unit dependencies should be primarily through interfaces, not concrete implementations. Define interfaces in dedicated units and then implement them in separate units. This allows for flexibility and reduced dependencies. The use of interfaces is a cornerstone to reduce the impact of implementation changes on other units, as only the interface contracts are relevant in the usage.

```delphi
// IDataService.pas (Interface Definition)
unit IDataService;

interface

type
  IDataService = interface
    ['{E4997873-170F-4A7E-BA66-0E683BC2C138}']
    function GetData: string;
  end;

implementation

end.

// DataServiceImpl.pas (Implementation)
unit DataServiceImpl;

interface

uses
  IDataService;

type
  TDataServiceImpl = class(TInterfacedObject, IDataService)
  public
    function GetData: string;
  end;

implementation

function TDataServiceImpl.GetData: string;
begin
  Result := 'Data from Implementation';
end;

end.

// MainUnit.pas (Usage)
unit MainUnit;

interface

uses
  IDataService;

type
  TMainForm = class(TForm)
  private
    FDataService : IDataService;
  public
    constructor Create(AOwner: TComponent); override;
    procedure ShowData;
  end;

implementation

uses
  DataServiceImpl; // Direct dependency on implementation only here

constructor TMainForm.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FDataService := TDataServiceImpl.Create; //Concrete instance created
end;

procedure TMainForm.ShowData;
begin
  ShowMessage(FDataService.GetData);
end;

end.
```

In this example, `MainUnit` uses the `IDataService` interface, not the concrete `TDataServiceImpl`. This means if the `TDataServiceImpl` is modified, `MainUnit` does not need to be recompiled, unless the interface `IDataService` changes. The direct dependency from `MainUnit` to the implementation (`DataServiceImpl`) is isolated to the point of the instantiation of `FDataService`. This limits the impact of implementation changes on the consuming modules.

**2. Unit Granularity:**

A unit should contain a logically related set of declarations and implementations. Avoid “catch-all” units that contain unrelated code. Such units tend to create unnecessary dependencies, as unrelated modules may end up requiring such a monolithic unit, increasing their dependency graph. Large units, too, also increase the compiler load as it must spend more resources processing the code itself, especially during subsequent compiles.

```delphi
// BadExample.pas (Large, unrelated code)
unit BadExample;

interface

type
  TDataProcessor = class
    function ProcessData(Data: string): string;
  end;

  TReportGenerator = class
    procedure GenerateReport(Data: string);
  end;

  TLogger = class
    procedure Log(Message: string);
  end;


implementation

function TDataProcessor.ProcessData(Data: string): string;
begin
  Result := 'Processed ' + Data;
end;

procedure TReportGenerator.GenerateReport(Data: string);
begin
  ShowMessage('Report: ' + Data);
end;

procedure TLogger.Log(Message: string);
begin
  OutputDebugString(PChar(Message));
end;

end.

//ImprovedExample.pas (Separated Concerns)

unit DataProcessorUnit;

interface

type
  TDataProcessor = class
    function ProcessData(Data: string): string;
  end;

implementation

function TDataProcessor.ProcessData(Data: string): string;
begin
  Result := 'Processed ' + Data;
end;

end.

unit ReportGeneratorUnit;

interface

type
  TReportGenerator = class
    procedure GenerateReport(Data: string);
  end;

implementation

procedure TReportGenerator.GenerateReport(Data: string);
begin
  ShowMessage('Report: ' + Data);
end;

end.

unit LoggerUnit;

interface

type
  TLogger = class
    procedure Log(Message: string);
  end;

implementation

procedure TLogger.Log(Message: string);
begin
  OutputDebugString(PChar(Message));
end;

end.
```

Here, splitting the single `BadExample` unit into three distinct units, `DataProcessorUnit`, `ReportGeneratorUnit`, and `LoggerUnit`, improves modularity. Units using only the `TDataProcessor` no longer depend on the `ReportGeneratorUnit`, thus limiting dependencies. This also leads to cleaner, more understandable modules, as each unit has a singular responsibility.

**3. Dependency Inversion and Abstract Factories:**

Dependency inversion principle encourages using abstractions (interfaces or abstract classes) as dependencies rather than concrete implementations. Abstract factories can centralize the creation of these concrete implementation instances, hiding these dependencies behind an abstraction, further reducing coupling. This principle complements the separation of interface from implementation, enabling a more flexible and maintainable design.

```delphi
// ILoggerFactory.pas (Abstract Factory)
unit ILoggerFactory;

interface

uses
  LoggerUnit; // Only direct reference to a concrete implementation is in factory creation

type
  ILoggerFactory = interface
    ['{847170A2-F002-4C87-84A8-6F1C6A1C966C}']
    function CreateLogger: TLogger;
  end;


implementation

end.

// LoggerFactoryImpl.pas (Concrete Factory)
unit LoggerFactoryImpl;

interface

uses
  ILoggerFactory, LoggerUnit;

type
  TLoggerFactoryImpl = class(TInterfacedObject, ILoggerFactory)
  public
    function CreateLogger: TLogger;
  end;

implementation

function TLoggerFactoryImpl.CreateLogger: TLogger;
begin
  Result := TLogger.Create;
end;

end.

// BusinessLogicUnit.pas (Usage)
unit BusinessLogicUnit;

interface

uses
  ILoggerFactory; //Dependency on ILogger, NOT concrete TLogger

type
  TBusinessLogic = class
  private
   FLogger: TLogger;
  public
    constructor Create(LoggerFactory: ILoggerFactory);
    procedure ProcessData(Data: string);
  end;

implementation
uses
  LoggerFactoryImpl; //Concrete factory creation

constructor TBusinessLogic.Create(LoggerFactory: ILoggerFactory);
begin
    FLogger := LoggerFactory.CreateLogger;
end;

procedure TBusinessLogic.ProcessData(Data: string);
begin
  FLogger.Log('Processing Data: ' + Data);
  // ... business logic
end;
end.


// ApplicationStartup.pas
unit ApplicationStartup;

interface
uses BusinessLogicUnit, LoggerFactoryImpl;
procedure Main;

implementation

procedure Main;
var
  LoggerFactory : ILoggerFactory;
  BusinessLogic : TBusinessLogic;
begin
  LoggerFactory := TLoggerFactoryImpl.Create; //Concrete factory instance
  BusinessLogic := TBusinessLogic.Create(LoggerFactory);
  BusinessLogic.ProcessData('My Data');
end;
end.
```

In this example, `BusinessLogicUnit` only depends on the `ILoggerFactory` interface. The specific `TLogger` instance is created by the `LoggerFactoryImpl`, hiding the concrete `TLogger` dependency. This adds another layer of indirection and reduces compilation dependencies. The `ApplicationStartup` unit is responsible for resolving the concrete dependencies. Changes to `LoggerUnit` will not affect `BusinessLogicUnit` provided the interface `TLogger` remains consistent.

**Resource Recommendations:**

For further study, I recommend researching principles of software architecture and design, specifically focusing on SOLID principles. Texts on object-oriented design patterns, and specifically on dependency injection and inversion, offer valuable insights. Studying advanced Delphi programming techniques, including the use of interfaces, generics, and anonymous methods, can also aid in developing more decoupled and efficient code. Practical experience, alongside disciplined refactoring, is the ultimate teacher in mastering these techniques. It’s crucial to monitor and profile your applications build time to identify and address dependencies or architectural problems early. Examining the compiled units (.dcu files) and the usage section of the Delphi documentation provides insight on the compiler behavior, and hence provides hints on how to further optimize the build process. Finally, studying large Delphi projects (open source or closed source) provides inspiration for organization and best practices.
