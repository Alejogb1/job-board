---
title: "simulating sports matches in online game?"
date: "2024-12-13"
id: "simulating-sports-matches-in-online-game"
---

Okay so simulating sports matches online right I've been down this rabbit hole more times than I care to admit Let me tell you it's not as straightforward as throwing some numbers at a wall and hoping for the best especially when you’re trying to keep it feeling somewhat realistic and responsive in an online game environment

Alright so let’s unpack this thing My first real foray into this was back when I was doing a small indie project a 2D pixel art football game Yeah I know I'm not proud of the art but the simulation was the real meat of it at least for me I was young and naive I thought a simple random number generator would cut it Oh boy was I wrong The games were just chaos teams scoring at ridiculous rates and no real sense of strategy it was like watching toddlers play a sport that they barely understand it was bad really bad I quickly realized that I needed something more structured

The first thing I tried that kinda helped but not really was a probability-based approach each action in the game like a pass or a shot was governed by a probability value these probabilities would then be modified by player stats and other factors This worked okay-ish in practice but it felt very predictable and boring It wasn’t hard to exploit this system once you got the hang of it and it certainly didn't feel very dynamic It was just a bunch of random numbers deciding the outcome of a match and that's no fun

Here's what I was using back then using Python just to give you a quick example cause you never know when python might be useful

```python
import random

class Player:
    def __init__(self, name, passing_skill, shooting_skill):
        self.name = name
        self.passing_skill = passing_skill
        self.shooting_skill = shooting_skill

def simulate_pass(player1, player2):
    pass_success_prob = player1.passing_skill / (player1.passing_skill + player2.passing_skill)
    if random.random() < pass_success_prob:
        return True
    else:
        return False

def simulate_shot(player, defense_skill):
    shot_success_prob = player.shooting_skill / (player.shooting_skill + defense_skill)
    if random.random() < shot_success_prob:
        return True
    else:
        return False
```

You can see it's super basic It’s just generating random numbers that are then used to compare against our probabilities it's simple but it quickly falls apart when you're talking about a complex game

So what did I learn The problem wasn’t that probability was useless but that I was just applying it too simply I was thinking of individual actions as isolated events instead of thinking of them within a larger context of the game state A game is much more than just a series of passes and shots it's a dynamic evolving system

The next big leap was to transition to a state-based simulation I needed to track the overall status of the game like the position of the ball the location of players and so on Based on this the actions could then be evaluated with a more dynamic view of the game instead of just numbers. This is when things started to get really interesting I used a grid-based system for simplicity's sake even though I knew I should probably have gone for a more complex representation.

I had to start defining game states which is a big pain. This meant encoding things like player positions ball position and even team formations into discrete chunks of data. Then actions would move things from one game state to another and use probability only to determine how exactly the transition from state A to state B is done. That gave me a far more consistent and predictable result which made it feel more like an actual game.

To give you an idea here's a simplified version of how a move can be represented. Bear in mind this was a very early prototype and I was still learning so don’t judge too harshly.

```python
class GameState:
    def __init__(self, ball_position, player_positions):
        self.ball_position = ball_position
        self.player_positions = player_positions

def apply_action(current_state, action):
    new_ball_position = action.calculate_new_ball_position(current_state)
    new_player_positions = action.calculate_new_player_positions(current_state)
    return GameState(new_ball_position, new_player_positions)
```

`apply_action` takes the current state of the game and a given action and then it returns the new state after applying the action. Each specific action will be responsible for defining the specific ways it modifies the game's state.

For example a player making a pass could involve calculating the new position of the ball based on the passing skills of the player and the position of the target receiver. I didn't just randomly move the ball from point A to point B. I also introduced slight randomness to simulate the imperfectness of player actions even the most skilled players will not land a perfect pass every time. A slightly off angle or a weaker pass it all depends on the probabilities.

Another problem I had was synchronization I had multiple clients trying to play the game simultaneously. So when simulating I had to be sure the game is in sync for all the players. I remember a very frustrating night trying to fix a bug where one player would see the ball in one position and the other in another it was a terrible experience. I was chasing my own tail for hours it was a real mess

To tackle this we used deterministic simulation with a server-side authority. The server performs the game logic and sends the game state information to the clients. All game actions initiated by the client are just that actions and all the resulting game-state changes are decided and broadcasted by the server. This makes sure that everyone gets the same information at the same time.

I was using an approach similar to this where every client's game runs on a tick and only the server is authoritative over the state of the game.

```python
def process_client_action(server_state, client_action):
    if validate_action(server_state, client_action):
        new_server_state = apply_action(server_state, client_action)
        return new_server_state
    else:
        return server_state

def simulate_game_tick(server_state):
    for queued_action in get_all_queued_actions():
        new_server_state = process_client_action(server_state, queued_action)
        server_state = new_server_state
    # Other game logic like AI logic
    return server_state
```

The server validates and applies every action coming from the client and it is the source of truth. This makes things predictable and helps with dealing with game-state inconsistencies

This is a big area and you need to read up on it as much as possible. I'd recommend reading “Game Programming Gems” for more detail into game-state management. There’s a lot of good stuff in there covering various simulation architectures. You can also look into the book “Real-Time Rendering” there's a section on simulation that helped me understand how to calculate movements and collisions more realistically. And for more theory I'd check some of the papers on the Discrete Event Simulation field it is not specifically about game programming but it does help to understand the underlying concepts.

Also one thing I learned and this was a hard one is that perfect simulations are overrated sometimes you want the game to have a tiny bit of uncertainty to make it more engaging. Adding some small inconsistencies can actually improve the player experience because let's be honest who wants to play a game where the outcome is always 100% predictable? I mean at that point it is like watching a simulation run not playing a game. But then again that's what simulations do or is it? I am getting confused and so are you probably.

Anyway simulating sports games is a complex problem and there are a lot of different ways to tackle it You need to think about things like player stats game state management synchronization and even adding some uncertainty. It's not a simple matter of applying basic probability you really need to understand the underlying mechanics and find the right balance between simulation and fun. Good luck on your journey cause you will probably need it.
