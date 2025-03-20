from collections import deque
import random
import sys

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from config import params
from utils.structs import Direction


# DEBUG OPTS
np.set_printoptions(threshold=sys.maxsize)


class SnakeEnv(gym.Env):
    def __init__(self, grid_size, renderer, agent):
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(3)  # 0=No-op, 1=Left, 2=Right
        self.observation_space = spaces.Box(low=0, high=3, shape=(self.grid_size, self.grid_size), dtype=np.uint8)  # 0=space, 1=snake head, 2=snake body, 3=food
        self.snake = deque()            # list of body segments to update position incrementally
        self.snake_segments = set()     # set of body segments for fast lookups during collision checks
        self.direction = None
        self.food = None
        self.done = False
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size    # max steps per episode

        self.renderer = renderer
        self.agent = agent

    def reset(self, *, seed=None, options=None):
        start_x, start_y = self.grid_size // 2, self.grid_size // 2
        self.snake = deque([(start_x, start_y),
                            (start_x - 1, start_y),
                            (start_x - 2, start_y)])
        self.snake_segments = set(self.snake)
        self.direction = Direction.RIGHT
        self.food = self.spawn_food()
        self.done = False
        self.steps = 0      # keeps track of how much the snake has traveled after the last food gobble

        return self.get_state()

    def step(self, action):     # return new_state, reward, terminated, truncated, info
        self.steps += 1

        # change dir
        if action == 1:
            self.direction = Direction((self.direction.value - 1) % 4)    # left turn: R -> U -> L -> D
        elif action == 2:
            self.direction = Direction((self.direction.value + 1) % 4)    # right turn: R -> D -> L -> U

        # advance head
        head_x, head_y = self.snake[0]
        new_head = (head_x, head_y - 1)  # default direction = up

        if self.direction == Direction.RIGHT:
            new_head = (head_x + 1, head_y)
        elif self.direction == Direction.DOWN:
            new_head = (head_x, head_y + 1)
        elif self.direction == Direction.LEFT:
            new_head = (head_x - 1, head_y)

        # remove tail first to prevent snake from eating its tail (maybe a less known variant but i like this version)
        tail = self.snake.pop()
        self.snake_segments.remove(tail)

        terminated = new_head in self.snake_segments or self.is_wall_collision(new_head)

        if terminated:
            return self.get_state(), -10, True, False, {}       # strong punishment for death

        # add new head
        self.snake.appendleft(new_head)
        self.snake_segments.add(new_head)

        # if food eaten, add back tail
        if new_head == self.food:
            reward = 1      # reward for eating food
            self.snake.append(tail)
            self.snake_segments.add(tail)
            self.food = self.spawn_food()
            self.steps = 0
        else:
            reward = -0.01      # keep the snake honest, no loafing around!

        truncated = self.steps >= self.max_steps

        return self.get_state(), reward, False, truncated, {}

    def close(self):
        self.renderer.close()

    def get_state(self):
        state = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        for seg_x, seg_y in self.snake:
            state[seg_y, seg_x] = 2     # snake body

        head_x, head_y = self.snake[0]
        state[head_y, head_x] = 1       # snake head

        state[self.food[1], self.food[0]] = 3   # food

        return state

    def spawn_food(self):
        while True:
            food_pos = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if food_pos not in self.snake_segments:
                return food_pos

    def is_wall_collision(self, pos):
        x, y = pos
        return x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size

    # UI delegation
    def render(self, mode="human"):
        if mode == "human" and self.renderer:
            self.renderer.render(self.get_state(), self.agent.score)
        else:
            raise NotImplementedError(f"No renderer set or render mode {mode} is not supported.")

    def run(self):
        # game loop
        if not self.agent:
            raise ValueError("No agent in env")

        state = self.reset()

        running = True

        while running:
            self.render()  # Render the game

            if pygame.event.peek(pygame.QUIT):
                pygame.quit()
                return

            action = self.agent.get_action(state)

            if action == -1:
                running = False

            # Step the environment
            state, reward, terminated, truncated, _ = self.step(action)

            if reward == 1:
                self.agent.score += 1

            # Check if game over
            if terminated or truncated:
                print(f"Game Over! Score = {self.agent.score}")
                running = False

        self.close()
