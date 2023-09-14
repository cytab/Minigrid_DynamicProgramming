#!/usr/bin/env python3

from __future__ import annotations

import gymnasium as gym
import pygame
from gymnasium import Env
import numpy as np
from minigrid.core.actions import ActionsReduced
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

ALL_POSSIBLE_ACTIONS = (ActionsReduced.right, ActionsReduced.left, ActionsReduced.forward, ActionsReduced.backward, )

class ManualControl:
    def __init__(
        self,
        env: Env,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False
        self.gamma = 0.99
        self.threshold = 1e-6

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)
        current_agent_pose = (self.env.agent_pos[0],  self.env.agent_pos[1])
        J = self.calculate_values()
        policy = self.calculate_greedy_policy(J)
        while True:
            action = policy[current_agent_pose]
            self.step(action)
            current_agent_pose = (self.env.agent_pos[0], self.env.agent_pos[1])

    def step(self, action: ActionsReduced):
        _ , reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": ActionsReduced.left,
            "right": ActionsReduced.right,
            "up": ActionsReduced.forward,
            "down": ActionsReduced.backward,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
            action_possible = self.env.get_possible_move()
            print(action_possible)
        else:
            print(key)
    
    def best_action_value(self, J, s):
        best_a = None
        best_value = float('-inf')
        self.env.set_state(s)
        for a in ALL_POSSIBLE_ACTIONS:
            transitions = self.env.get_transition_probs(a, cost_value=0.5)
            expected_v =  0
            expected_r = 0
            for (prob, r, state_prime) in transitions:
                expected_r += prob*r
                expected_v = prob*J[state_prime]
            v = expected_r + self.gamma*expected_v
            if v > best_value:
                best_value = v
                best_a = a
        return best_a, best_value
    
    def calculate_values(self):
        J = {}
        states = self.env.get_all_states()
        for s in states:
            J[s] = 0
            
        while True:
            big_change = 0
            for s in self.env.get_states_non_terminated():
                old_v = J[s]
                _, new_v = self.best_action_value(J, s)
                J[s] = new_v
                big_change = max(big_change, np.abs(old_v-new_v))

            if big_change < self.threshold :
                break
        return J
    
    def initialize_random_policy(self):
        policy = {}
        for s in self.env.get_states_non_terminated():
            policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
        return policy
    
    def calculate_greedy_policy(self, J):
        policy = self.initialize_random_policy()
        for s in policy.keys():
            self.env.set_state(s)
            best_a , _ = self.best_action_value(J, s)
            policy[s] = best_a
        return policy



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        default="MiniGrid-Empty-Reduced-16x16-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="draw the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default="640",
        help="set the resolution for pygame rendering (width and height)",
    )

    args = parser.parse_args()

    env: MiniGridEnv = gym.make(
        args.env_id,
        tile_size=args.tile_size,
        render_mode="human",
        agent_pov=args.agent_view,
        agent_view_size=args.agent_view_size,
        screen_size=args.screen_size,
    )

    # TODO: check if this can be removed
    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, args.tile_size)
        env = ImgObsWrapper(env)

    manual_control = ManualControl(env, seed=args.seed)
    manual_control.start()
