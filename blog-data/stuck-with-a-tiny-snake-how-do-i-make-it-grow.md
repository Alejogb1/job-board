---
title: "Stuck with a Tiny Snake?  How Do I Make It Grow?"
date: '2024-11-08'
id: 'stuck-with-a-tiny-snake-how-do-i-make-it-grow'
---

```python
import pygame
import time
import sys
import random


a = print("1) Easy")
b = print("2) Medium")
c = print("3) Hard")

while True:
    difficulty = input("Enter Difficulty Level: ")
    if difficulty == "1":
        speed = 5
        break
    elif difficulty == "2":
        speed = 6
        break
    elif difficulty == "3":
        speed = 8
    else:
        print("Choose from Above Options Only!")


# Initialise Game
pygame.init()
clock = pygame.time.Clock()


# Screen and Window Size:
screen_width = 800
screen_height = 700
screen = pygame.display.set_mode((screen_width, screen_height))
caption = pygame.display.set_caption("Snake Game")
icon = pygame.image.load("snake.png")
pygame.display.set_icon(icon)

# Colors
red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
white = (255, 255, 255)

# Snake Editing
x1 = 350
y1 = 300
snake = pygame.Rect([x1, y1, 20, 20])

x1_change = 0       
y1_change = 0

snake_size = 15

snk_list = []
snk_length = 1

# Snake Food
food_x = random.randint(30, screen_width - 40)
food_y = random.randint(30, screen_height - 40)
food_height = 15
food_width = 15

# Game State
game_over = True

# Game over
font = pygame.font.SysFont("freelansbold.tff", 64)

# Score Counter
score = 0
score_font = pygame.font.SysFont("chiller", 50)


# TO INCREASE SNAKE LENGTH LOGIC:
def plot_snake(gameWindow, color, snk_list, snake_size):
    for x, y in snk_list:
        pygame.draw.rect(gameWindow, color, [x, y, snake_size, snake_size])


def game_over_text(text, color):
    x = font.render(text, True, (240, 0, 0))
    screen.blit(x, [screen_width//2 - 135, screen_height//2 - 25])


def score_show():
    text = score_font.render("Score: " + str(score), True, (255, 255, 255))
    screen.blit(text, (20, 10))


def main_loop():
    global x1, y1, x1_change, y1_change, game_over, food_x, food_y, score, speed, snk_list, snake_size
    global snk_length
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # User Input
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                x1_change = -snake_size
                y1_change = 0
            elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                x1_change = snake_size
                y1_change = 0
            elif event.key == pygame.K_UP or event.key == pygame.K_w:
                y1_change = -snake_size
                x1_change = 0
            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                y1_change = snake_size
                x1_change = 0

    # Game Over Checking
    if x1 >= screen_width or x1 < 0 or y1 >= screen_height or y1 < 0:
        game_over = False

    x1 += x1_change
    y1 += y1_change

    snake_rect = pygame.Rect(x1, y1, snake_size, snake_size)
    food_rect = pygame.Rect(food_x, food_y, snake_size, snake_size)
    if snake_rect.colliderect(food_rect):
        snk_length += 1
        score += 1
        food_x = random.randint(30, screen_width - 40)
        food_y = random.randint(30, screen_height - 40)
        speed += 1

    # Drawing On Screen
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, red, [x1, y1, 15, 15])
    pygame.draw.rect(screen, green, [food_x, food_y, food_width, food_height])
    score_show()
    pygame.display.flip()

    #SNAKE LENGTH LOGIC
    snk_list.insert(0, [x1, y1])
    if len(snk_list) > snk_length:
        del snk_list[-1]

    plot_snake(screen, red, snk_list, snake_size)
    # Final Initialisation
    pygame.display.flip()
    clock.tick(speed)


# Main Game Loop
while game_over:
    main_loop()

# Game_Over
screen.fill((0, 0, 0))
game_over_text("Game Over!!!", (255, 0, 0))
pygame.display.flip()
time.sleep(2)
pygame.quit()
quit()

```
