---
title: "How can constructor dependency injection be tested using Unity?"
date: "2024-12-23"
id: "how-can-constructor-dependency-injection-be-tested-using-unity"
---

Okay, let's tackle this. I've spent quite a bit of time in the trenches working with Unity, and testing dependency injection, particularly constructor injection, always requires a bit of finesse. I remember one particularly frustrating project where untestable components were causing cascade failures throughout our CI pipeline – a real headache. So, let me walk you through how I typically approach testing constructor dependency injection within Unity, along with some practical code examples and recommendations.

The fundamental issue here is that constructor injection forces a dependency onto a class when it's created. This is great for decoupling, but less convenient for direct unit testing, since we can't just new-up an object without also satisfying all of its dependency requirements. The goal is to isolate the class under test, focusing solely on its logic and avoiding testing the behavior of its dependencies. This is where mocking and, to some degree, stubbing becomes essential.

The basic principle is to create mock or stub implementations of your dependencies, which can then be injected during the constructor call of your test object. Let's assume a simple scenario. Imagine you have a `GameController` class that depends on an `IPlayerDataService` to load and save player data. This is very common. Here's a basic C# code structure that mirrors something I have encountered:

```csharp
public interface IPlayerDataService
{
    PlayerData LoadPlayerData(string playerId);
    void SavePlayerData(string playerId, PlayerData data);
}

public class PlayerData
{
    public int Level { get; set; }
    public int Score { get; set; }
}

public class GameController : MonoBehaviour
{
    private readonly IPlayerDataService _playerDataService;

    public GameController(IPlayerDataService playerDataService)
    {
       _playerDataService = playerDataService;
    }

    public void LoadGame(string playerId)
    {
        PlayerData data = _playerDataService.LoadPlayerData(playerId);
        //Further game logic here based on loaded data
        Debug.Log($"Game loaded for player: {playerId}, Level:{data.Level}");
    }
     public void SaveGame(string playerId, PlayerData currentData)
     {
        _playerDataService.SavePlayerData(playerId,currentData);
         Debug.Log($"Game data saved for player: {playerId}");
    }

     public void IncrementLevel(string playerId) {
          PlayerData currentData = _playerDataService.LoadPlayerData(playerId);
          currentData.Level++;
          _playerDataService.SavePlayerData(playerId,currentData);

     }
}

```

Now, how do we test the `GameController` without also testing (or depending on) an actual database or file system that `PlayerDataService` might interact with? This is where testing frameworks and mocking come in. Using a framework like NSubstitute, a highly versatile mocking library that I prefer, or any mocking framework that integrates well with Unity, you can create mock implementations. We'll use NSubstitute for this example.

Here's our first test case using NSubstitute and assuming your project is set up to use test runners:

```csharp
using NSubstitute;
using NUnit.Framework;
using UnityEngine;

public class GameControllerTests
{
    [Test]
    public void LoadGame_CallsPlayerDataServiceCorrectly()
    {
        // Arrange - create a mock of IPlayerDataService
        var mockPlayerDataService = Substitute.For<IPlayerDataService>();
        var testData = new PlayerData { Level = 5, Score = 100 };
        mockPlayerDataService.LoadPlayerData("test_player").Returns(testData);

        // Act - create GameController with the mocked dependency and call the LoadGame method
        var gameController = new GameController(mockPlayerDataService);
        gameController.LoadGame("test_player");

        // Assert - verify the calls to the mock
        mockPlayerDataService.Received(1).LoadPlayerData("test_player");
    }
}

```

In this first test, I set up a mock instance of `IPlayerDataService` using NSubstitute. I define what the `LoadPlayerData` method should return when called with the specific id. Then, I create a `GameController` instance, providing this mock during its construction, and subsequently call the `LoadGame` method using the same id. Finally, I verify that the `LoadPlayerData` method was indeed called. This tests the specific interaction between the `GameController` and its dependency, without needing a real implementation for `PlayerDataService`. This isolation is crucial.

Here's a second example that test the `IncrementLevel` functionality:

```csharp
using NSubstitute;
using NUnit.Framework;
using UnityEngine;

public class GameControllerTests
{
    [Test]
    public void IncrementLevel_IncrementsLevelAndSavesCorrectly()
    {
        // Arrange
        var mockPlayerDataService = Substitute.For<IPlayerDataService>();
        var initialData = new PlayerData { Level = 3, Score = 150 };
        var expectedData = new PlayerData { Level = 4, Score = 150 };

        mockPlayerDataService.LoadPlayerData("test_player").Returns(initialData);

         // Act
        var gameController = new GameController(mockPlayerDataService);
        gameController.IncrementLevel("test_player");


        // Assert
        mockPlayerDataService.Received(1).LoadPlayerData("test_player");
        mockPlayerDataService.Received(1).SavePlayerData("test_player", Arg.Is<PlayerData>(data => data.Level == expectedData.Level && data.Score == expectedData.Score));
    }
}

```

In this case, I'm simulating the interaction. I set initial data, define how it should change within `GameController`, and verify that the save function receives the correctly modified data. This gives me confidence that `GameController` not only loads the data but modifies it correctly before saving.

One more snippet to showcase further the ability to check the parameters used with our mocked service:
```csharp
using NSubstitute;
using NUnit.Framework;
using UnityEngine;

public class GameControllerTests
{
      [Test]
    public void SaveGame_CallsPlayerDataServiceCorrectlyWithData()
    {
       // Arrange
        var mockPlayerDataService = Substitute.For<IPlayerDataService>();
        var testData = new PlayerData { Level = 7, Score = 200 };

        // Act
        var gameController = new GameController(mockPlayerDataService);
        gameController.SaveGame("test_player_save", testData);

        // Assert
        mockPlayerDataService.Received(1).SavePlayerData("test_player_save", Arg.Is<PlayerData>(data => data.Level == testData.Level && data.Score == testData.Score));
    }
}
```

This last example shows how you can check that the correct instance of `PlayerData` is passed to the `SavePlayerData` method, with the exact values you provided. It makes sure that data was not modified in an unexpected way before being sent to the dependency.

This approach covers the main aspects of testing constructor dependency injection, however here's a few extra considerations. For complex dependencies, you might need a mix of mocks and stubs. A stub is a more simplified mock that returns pre-defined values and does not verify interaction, this can be helpful where your test only requires return value without tracking the interaction. Consider also using a testing library that suits your needs. Some developers find Moq to be a good alternative to NSubstitute. You could even develop hand-rolled mocks for particularly simple scenarios, but I usually find this not to be worth the time or effort. Lastly, be sure to explore concepts like the "Arrange-Act-Assert" pattern, which is a great strategy for structuring tests effectively.

For further study on dependency injection and testing, I highly recommend looking into *Working Effectively with Legacy Code* by Michael Feathers, which is invaluable for understanding how to test objects that weren't designed with testing in mind, and *Refactoring: Improving the Design of Existing Code* by Martin Fowler. Additionally, the resources and documentation provided for mocking frameworks like NSubstitute or Moq are always beneficial. Reading about design principles like SOLID (especially the 'D' for dependency inversion) will definitely deepen your understanding.

Testing constructor injection in Unity can seem initially tricky, but by embracing the power of mocking and focusing on isolating your class's logic, you can achieve highly testable and maintainable code. This also sets the foundation for more complex systems that rely on robust component interaction testing. Good luck in your journey and please feel free to share if you face any more challenges; I'm always keen to learn from others' experiences as well.
