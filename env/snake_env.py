from collections import deque
import random
import sys

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from config import FOOD_PICKUP_REWARD, STEP_COST, CRASH_COST, TRUNCATION_COST, PROGRESS_REWARD, DIGRESS_COST, \
    IN_PLACE_CIRCLING_COST, MAX_WANDERING_STEPS_MULTIPLIER, SELF_LOOP_CRASH_PREVENTION_MULTIPLER, TURN_COST
from utils import manhattan_distance
from utils.structs import Direction


# DEBUG OPTS
np.set_printoptions(threshold=sys.maxsize)


class SnakeEnv(gym.Env):
    def __init__(self, grid_size, renderer=None, agent=None):
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(3)  # 0=No-op, 1=Left, 2=Right
        # self.observation_space = spaces.Box(low=0, high=3, shape=(self.grid_size, self.grid_size), dtype=np.uint8)  # 0=space, 1=snake head, 2=snake body, 3=food
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(13,), dtype=np.float16)
        self.snake = deque()            # list of body segments to update position incrementally
        self.snake_segments = set()     # set of body segments for fast lookups during collision checks
        self.direction = None
        self.food = None
        self.done = False
        self.steps = 0
        self.max_steps = MAX_WANDERING_STEPS_MULTIPLIER * self.grid_size * self.grid_size    # max steps without picking up food

        self.renderer = renderer
        self.render_mode = "human"
        self.agent = agent

        # experimental: to boost initial exploration instead of spot circling
        self.recently_visited = deque(maxlen=6)

    def reset(self, *, seed=None, options=None):
        start_x, start_y = self.grid_size // 2, self.grid_size // 2
        self.snake = deque([(start_x, start_y),
                            (start_x - 1, start_y),
                            (start_x - 2, start_y)])
        self.snake_segments = set(self.snake)
        self.direction = Direction.RIGHT
        self.food = self.spawn_food(reset=True)
        self.done = False
        self.steps = 0      # keeps track of how much the snake has traveled after the last food gobble

        # self.recently_visited = deque(reversed(self.snake), maxlen=6)

        # return self.get_state(), {}
        return self.get_state_experimental(), {}

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
            # return self.get_state(), CRASH_COST, True, False, {}       # strong punishment for death
            return self.get_state_experimental(), CRASH_COST, True, False, {}

        # add new head
        self.snake.appendleft(new_head)
        self.snake_segments.add(new_head)

        # if food eaten, add back tail
        if new_head == self.food:
            reward = FOOD_PICKUP_REWARD      # reward for eating food
            self.snake.append(tail)
            self.snake_segments.add(tail)
            self.food = self.spawn_food()
            self.steps = 0
        else:
            reward = STEP_COST      # keep the snake honest, no loafing around!

            old_head = head_x, head_y
            old_dist = manhattan_distance(old_head, self.food)
            new_dist = manhattan_distance(new_head, self.food)
            if new_dist < old_dist:
                reward += PROGRESS_REWARD
            else:
                reward += DIGRESS_COST

        # if new_head in self.recently_visited:
        #     reward += IN_PLACE_CIRCLING_COST
        # self.recently_visited.append(new_head)

        if action in (1, 2):
            reward += TURN_COST     # to smoothen path

        truncated = self.steps >= self.max_steps
        if truncated:
            reward = TRUNCATION_COST

        # return self.get_state(), reward, False, truncated, {}
        return self.get_state_experimental(), reward, False, truncated, {}

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

    def spawn_food(self, reset=False):
        # handling sparse reward issue
        # spawn food more in the center at the beginning and slowly move out as the snake grows
        # radius = (self.grid_size // 8) + ((len(self.snake) - 3) // 5)       # every 5 food eaten increases radius by 1
        # cx = cy = self.grid_size // 2
        # x_min = max(cx - radius, 0)
        # x_max = min(cx + radius, self.grid_size - 1)
        # y_min = max(cy - radius, 0)
        # y_max = min(cy + radius, self.grid_size - 1)
        # food_pos_choices = [(cx + 4, cy), (cx + 4, cy + 4), (cx, cy + 4), (cx - 4, cy + 4), (cx - 4, cy), (cx, cy)]
        # ptr = 0
        # attempts = 0
        while True:
            # if reset:
            #     ptr = 0
            # food_pos = food_pos_choices[ptr]
            # ptr = (ptr + 1) % len(food_pos_choices)
            # attempts += 1
            # if attempts > 20:
            #     food_pos = (cx + 6, cy)
            # food_pos = random.randint(x_min, x_max), random.randint(y_min, y_max)
            food_pos = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if food_pos not in self.snake_segments:
                return food_pos
                # yield food_pos

    def is_wall_collision(self, pos):
        x, y = pos
        return x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size

    # UI delegation
    def render(self, mode="human"):
        if mode == "human" and self.renderer:
            self.renderer.render(self.get_state(), self.agent.score)
        elif mode == "ansi":
            grid = np.full((self.grid_size, self.grid_size), " ")
            for x, y in self.snake:
                grid[y, x] = "o"
            grid[self.snake[0][1], self.snake[0][0]] = "O"
            grid[self.food[1], self.food[0]] = "*"
            print("#  " * self.grid_size)
            print("\n".join(["  ".join(row) for row in grid]))
            print("#  " * self.grid_size)
        else:
            raise NotImplementedError(f"No renderer set or render mode {mode} is not supported.")

    # game loop to invoke in main
    def run(self, close_renderer_after_run=True):
        # game loop
        if not self.agent:
            raise ValueError("No agent in env")

        if hasattr(self.agent, "model") and not self.agent.model:    # ai agents, model not loaded
            self.agent.init_model(self)

        state, _ = self.reset()

        running = True

        while running:
            self.render(mode="human")  # Render the game

            if pygame.event.peek(pygame.QUIT):
                pygame.quit()
                return

            action = self.agent.get_action(state)

            if action == -1:
                running = False

            # Step the environment
            state, reward, terminated, truncated, _ = self.step(action)

            if reward == FOOD_PICKUP_REWARD:
                self.agent.score += 1

            # Check if game over
            if terminated or truncated:
                print(f"Game Over! Score = {self.agent.score}")
                running = False

        close_renderer_after_run and self.close()

    # train loop for ai agents
    def train_agent(self, timesteps):
        self.agent.init_model(self)
        self.agent.train(timesteps)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.snake[0]
        # hits boundary
        if pt[0] >= self.grid_size or pt[0] < 0 or pt[1] >= self.grid_size or pt[1] < 0:
            return True
        # hits itself
        if pt in self.snake_segments:
            return True

        return False

    def get_state_experimental(self):
        head_x, head_y = self.snake[0]
        point_l = (head_x - 1, head_y)
        point_r = (head_x + 1, head_y)
        point_u = (head_x, head_y - 1)
        point_d = (head_x, head_y + 1)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # # Danger straight
            # (dir_r and self.is_collision(point_r)) or
            # (dir_l and self.is_collision(point_l)) or
            # (dir_u and self.is_collision(point_u)) or
            # (dir_d and self.is_collision(point_d)),
            #
            # # Danger right
            # (dir_u and self.is_collision(point_r)) or
            # (dir_d and self.is_collision(point_l)) or
            # (dir_l and self.is_collision(point_u)) or
            # (dir_r and self.is_collision(point_d)),
            #
            # # Danger left
            # (dir_d and self.is_collision(point_r)) or
            # (dir_u and self.is_collision(point_l)) or
            # (dir_r and self.is_collision(point_u)) or
            # (dir_l and self.is_collision(point_d)),

            # distances to possible collisions from turning from current position, clamped to min val of 0
            *map(lambda x: max(0, x), self.get_normalized_collision_distances()),

            # Move direction
            dir_r,
            dir_d,
            dir_l,
            dir_u,

            # Food location
            self.food[0] < head_x,  # food left
            self.food[0] > head_x,  # food right
            self.food[1] < head_y,  # food up
            self.food[1] > head_y,  # food down
        ]

        return np.array(state, dtype=np.float16)

    '''
    Collision distances while going straight after the following turns:
    left u-turn (turn left twice), left turn, straight (no turn), right turn, right u-turn (turn right twice)
    '''
    def get_normalized_collision_distances(self):
        head = self.snake[0]

        def move(pos, direction):
            x, y = pos
            if direction == Direction.UP:
                return x, y - 1
            elif direction == Direction.DOWN:
                return x, y + 1
            elif direction == Direction.LEFT:
                return x - 1, y
            else:
                return x + 1, y

        def distance_to_collision(direction, pos=None):
            if not pos:
                pos = head
            x, y = pos
            dist = 0
            while True:
                x, y = move((x, y), direction)
                dist += 1
                if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size or (x, y) in self.snake_segments:
                    if (x, y) in self.snake_segments:
                        dist -= SELF_LOOP_CRASH_PREVENTION_MULTIPLER * self.grid_size       # potential crash into self is more severe as there may not be a way out
                    break
            return dist / self.grid_size

        left_dir = Direction((self.direction.value - 1) % 4)
        right_dir = Direction((self.direction.value + 1) % 4)
        u_turn_dir = Direction((self.direction.value + 2) % 4)

        d_straight = distance_to_collision(self.direction)
        d_left = distance_to_collision(left_dir)
        d_right = distance_to_collision(right_dir)
        d_left_u_turn = 1 / self.grid_size + distance_to_collision(u_turn_dir, move(head, left_dir))     # move to left of head and then go backwards
        d_right_u_turn = 1 / self.grid_size + distance_to_collision(u_turn_dir, move(head, right_dir))   # move to right of head and then go backwards

        return d_left_u_turn, d_left, d_straight, d_right, d_right_u_turn

