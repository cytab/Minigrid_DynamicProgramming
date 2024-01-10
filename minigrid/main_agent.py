#!/usr/bin/env python3

from __future__ import annotations

import gymnasium as gym
import pygame
from gymnasium import Env
from minigrid.core.grid import Grid
import numpy as np
from minigrid.core.actions import ActionsReduced
from minigrid.envs.empty_reduced import EmptyReducedEnv
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from minigrid.Agent_2 import AssistiveAgent
ALL_POSSIBLE_ACTIONS = (ActionsReduced.right, ActionsReduced.left, ActionsReduced.forward, ActionsReduced.backward, ActionsReduced.stay)
        
class MainAgent:
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

    def start(self, agent: AssistiveAgent):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)
        current_agent_pose = (self.env.agent_pos[0],  self.env.agent_pos[1])
        J, Q = self.value_iteration()
        #policy_l = self.deduce_policy(J)

        dist = self.boltzmann_policy(Q=Q, eta=0.95)

        J2 = agent_2.calculate_Values(dist)
        Q2 = agent_2.calculate_Values(dist,Q=True)
        J2_l, Q2_l = agent_2.value_iteration(dist)
  
        policy_agent2 = agent_2.calculate_greedy_policy(J2, dist)
        policy_agent2_l = agent_2.deduce_policy(J2, dist)
        print(policy_agent2)
        print("-------------------------")
        print(policy_agent2_l)
        print("-------------------------")
        policy_agent2_n = agent_2.calculate_greedy_policy(J2_l, dist)
        policy_agent2_n_l = agent_2.deduce_policy(J2_l, dist)
        print(policy_agent2_n)
        print("-------------------------")
        print(policy_agent2_n_l)
        print("-------------------------")
        """while True:
            #resolve dynamic programming of agent 2
            J2 = agent_2.calculate_Values(dist)
            Q2 = agent_2.calculate_Values(p_action=dist, Q=True)
            print(Q2)
            # deduce the actual optimal policy
            policy_agent2 = agent_2.calculate_greedy_policy(J2, dist)
        
            #take agent 2 action in the world
            agent_2.step(policy_agent2[current_agent_pose])
        
            #recalculate Q function of agent 1
            Q = self.calculate_values(Q=True)
        
            #new distribution of action of agent 1 
            dist = self.boltzmann_policy(Q=Q, eta=5)
            #print(dist)
            #action = policy[current_agent_pose] # uncomment for deterministic action
        
            # generate an action from distribution
            action = ActionsReduced(self.generate_action(current_agent_pose, dist=dist))
        
            # take agent 1 action in the world
            self.step(action)
        
            # update agent pose
            current_agent_pose = (self.env.agent_pos[0], self.env.agent_pos[1])"""

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
    
    def best_action_value(self, J, s, Q=False):
        best_a = None
        best_value = float('-inf')
        self.env.set_state(s)
        q = {a:0 for a in ALL_POSSIBLE_ACTIONS}
        for a in self.env.get_possible_move(s):
            transitions = self.env.get_transition_probs(a, cost_value=1)
            expected_v =  0
            expected_r = 0
            for (prob, r, state_prime) in transitions:
                expected_r += prob*r
                expected_v += prob*J[state_prime]
            v = expected_r + self.gamma*expected_v
            q[a] = v
            if v > best_value:
                best_value = v
                best_a = a
        if not Q:
            return best_a, best_value
        else:
            return q, best_a, best_value
        
    def value_iteration(self, Q=False):
        states = self.env.get_all_states()
        Q= {}
        J = {}
        for s in states:
            Q[s] = {}
            J[s] = 0
            for a in ALL_POSSIBLE_ACTIONS:
                    Q[s][a] = 0
        while True:
            big_change = 0
            old_J = J.copy()
            for s in self.env.get_states_non_terminated(): 
                self.env.set_state(s)
                for a in self.env.get_possible_move(s):
                    next_state_reward = []
                    transitions = self.env.get_transition_probs(a, cost_value=1)
                    for (prob, r, state_prime) in transitions:
                        reward = prob*(r + self.gamma*old_J[state_prime])
                        next_state_reward.append(reward)
                        
                    Q[s][a]=((np.sum(next_state_reward)))
                J[s] = max(Q[s].values())
                big_change = max(big_change, np.abs(old_J[s]-J[s]))
            if big_change <= self.threshold :
                break
        return J, Q
    
    def calculate_values(self, Q=False):
        states = self.env.get_all_states()
        if not Q:
            J = {}
            for s in states:
                J[s] = 0
            while True:
                big_change = 0
                old_v = J.copy()       
                for s in self.env.get_states_non_terminated(): 
                    _, new_v = self.best_action_value(old_v, s)
                    J[s] = new_v
                    big_change = max(big_change, np.abs(old_v[s]-J[s]))
                    
                if big_change < self.threshold :
                    break
            return J
        else:
            Q= {}
            J = {}
            for s in states:
                Q[s] = {}
                J[s] = 0
                for a in ALL_POSSIBLE_ACTIONS:
                    Q[s][a] = 0
            while True:
                big_change = 0
                old_v = J.copy()       
                for s in self.env.get_states_non_terminated(): 
                    Q[s], _, new_v = self.best_action_value(old_v, s, Q)
                    J[s] = new_v
                    big_change = max(big_change, np.abs(old_v[s]-new_v))  
                if big_change < self.threshold :
                    break
            #while True:
            #    print('faIT')
            #    big_change = 0
            #    temp = Q.copy()        
            #    for s in states:
            #        for a in self.env.get_possible_move(s):
            #            Q[s][a] = self.value_action(temp, s, a)
            #            big_change = max(big_change, np.abs(temp[s][a]-Q[s][a]))
            #    if big_change < self.threshold:
            #        break;
            return Q    
    
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
    
    def deduce_policy(self, J):
        policy = {}
        for s in self.env.get_states_non_terminated():
            policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
        for s in self.env.get_states_non_terminated():
            self.env.set_state(s) 
            Q_table = np.zeros(len(ALL_POSSIBLE_ACTIONS))
            for action in ALL_POSSIBLE_ACTIONS :
                transitions = self.env.get_transition_probs(action, cost_value=1)
                for (prob, r, state_prime) in transitions:
                    Q_table[int(action)] += prob*(r + self.gamma*J[state_prime])
            policy[s] = ActionsReduced(np.argmax(Q_table))
        return policy                   
                
        
    #output the distribution over action in all state of agent 1
    def boltzmann_policy(self, Q, eta):
        dist = {}
        total_prob = {}
        states = self.env.get_all_states()
        for s in states:
            dist[s] = {}
            total_prob[s] = 0
            for a in ALL_POSSIBLE_ACTIONS:
                dist[s][a] = (np.exp(eta*Q[s][a]))
                total_prob[s] += dist[s][a]
            for a in ALL_POSSIBLE_ACTIONS:
                dist[s][a] = (dist[s][a])/(total_prob[s])
        return dist
    
    def generate_action(self, state, dist):
        prob = [dist[state][a] for a in ALL_POSSIBLE_ACTIONS]
        generated_action = np.random.choice(ALL_POSSIBLE_ACTIONS, p=prob)
        return generated_action


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        default="MiniGrid-Empty-Reduced-8x8-v0",
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
    
    #env = EmptyReducedEnv(render_mode="human", size =16)
    #print(env.reduced)
    # TODO: check if this can be removed
    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, args.tile_size)
        env = ImgObsWrapper(env)

    agent_1 = MainAgent(env, seed=args.seed)
    agent_2 = AssistiveAgent(env=env, seed=args.seed)
    agent_1.start(agent_2)
