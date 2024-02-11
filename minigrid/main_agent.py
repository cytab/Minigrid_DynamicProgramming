#!/usr/bin/env python3

from __future__ import annotations
import pickle, time
import gymnasium as gym
import pygame
import pprint
from gymnasium import Env
from minigrid.core.grid import Grid
import numpy as np
from minigrid.core.actions import ActionsReduced, ActionsAgent2, WorldSate, GoalState
from minigrid.envs.empty_reduced import EmptyReducedEnv
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from minigrid.Agent_2 import AssistiveAgent

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

ALL_POSSIBLE_ACTIONS = (ActionsReduced.right, ActionsReduced.left, ActionsReduced.forward, ActionsReduced.backward, ActionsReduced.stay)
#ALL_POSSIBLE_WOLRD = (WorldSate.open_door, WorldSate.closed_door)
ALL_POSSIBLE_WOLRD = ((WorldSate.open_door1,WorldSate.open_door2), (WorldSate.open_door1,WorldSate.closed_door2), (WorldSate.closed_door1, WorldSate.open_door2), (WorldSate.closed_door1, WorldSate.closed_door2))
ALL_POSSIBLE_GOAL = (GoalState.green_goal,GoalState.red_goal)



def belief_state(env, previous_dist_g, dist_boltzmann, w, s):
        # be carful of dynamic of w that needs the action of agent 2
        current_dist = previous_dist_g
        normalizing_factor = 0
        print(current_dist[ALL_POSSIBLE_GOAL[0]])
        for i in range(len(ALL_POSSIBLE_GOAL)):
            conditional_state_world = 0
            for a in env.get_possible_move(s):
                conditional_state_world += dist_boltzmann[w][s][ALL_POSSIBLE_GOAL[i]][a]
            current_dist[ALL_POSSIBLE_GOAL[i]] = conditional_state_world*previous_dist_g[ALL_POSSIBLE_GOAL[i]]
            normalizing_factor += conditional_state_world*previous_dist_g[ALL_POSSIBLE_GOAL[i]]
        
        current_dist = {ALL_POSSIBLE_GOAL[i]: current_dist[ALL_POSSIBLE_GOAL[i]]/normalizing_factor for i in range(len(ALL_POSSIBLE_GOAL))}
        
        return current_dist

plt.style.use('fivethirtyeight')
step = []
belief_State_Tracker = {ALL_POSSIBLE_GOAL[i]: [] for i in range(len(ALL_POSSIBLE_GOAL))}
def animate(i):
    plt.cla()
    plt.plot(step, belief_State_Tracker[ALL_POSSIBLE_GOAL[0]], label='Green goal')
    plt.plot(step, belief_State_Tracker[ALL_POSSIBLE_GOAL[1]], label='Red goal')
    
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.draw()
    plt.pause(0.05)
    

# plt.tight_layout()
# plt.show()
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
        #N = 1000
        #miinimum_step = 50
        #count_sucess = 0 
        #for i in range(N):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)
        current_agent_pose = (self.env.agent_pos[0],  self.env.agent_pos[1])
        
        initial_time = time.time()
        J, Q = self.value_iteration_multiple_goal()
        value_iteration_elapsed_time = initial_time - time.time()
        print('Elpased time for value iteration with multiple goal:')
        print(value_iteration_elapsed_time)
        ###   --- T = 246s = 4 min
        #file = open("J.pkl", "wb")
        #pickle.dump(J, file)
        #file.close()
        #file = open("Q.pkl", "wb")
        #pickle.dump(Q, file)
        #file.close()
        #file = open("dictionary_data.pkl", "rb")
        #output = pickle.load(file)
        #print(output)
        #file.close()
        #policy = self.deduce_policy(J)
        
        ##J, Q = self.value_iteration()
        #pprint.PrettyPrinter(width=20).pprint(Q)
        
        
        # Determine initial policy
        dist = self.boltzmann_policy_multiple_goal(Q,eta=6)
        ###   --- T = 
        #pprint.PrettyPrinter(width=20).pprint(dist)
        #print("-------------------------")
        #J2, Q2 = agent_2.value_iteration(dist)
        #pprint.PrettyPrinter(width=20).pprint(Q2)
            # deduce the actual optimal policy
        #policy_agent2 = agent_2.deduce_policy(J2, dist)
        #pprint.PrettyPrinter(width=20).pprint(policy_agent2)
        prior = {ALL_POSSIBLE_GOAL[0]: 0.5, ALL_POSSIBLE_GOAL[1]: 0.5}
        belief_State_Tracker[ALL_POSSIBLE_GOAL[0]].append(prior[ALL_POSSIBLE_GOAL[0]])
        belief_State_Tracker[ALL_POSSIBLE_GOAL[1]].append(prior[ALL_POSSIBLE_GOAL[1]])
        count = 0
        step.append(count)
        #ani = FuncAnimation(plt.gcf(), animate, interval=60)
        #ani.save('growingCoil.mp4', writer = 'ffmpeg', fps = 30)
        self.env.grid.set(self.env.rooms[0].doorPos[0], self.env.rooms[0].doorPos[1], None)
        self.env.grid.set(self.env.rooms[1].doorPos[0], self.env.rooms[1].doorPos[1], None) 
        while True:
            plt.ion()
            current_world = self.env.get_world_state()
            print(current_world)
            g = GoalState.red_goal
            belief = belief_state(env=self.env, previous_dist_g=prior, dist_boltzmann=dist, w=current_world, s=current_agent_pose)
            
            action = ActionsReduced(self.generate_action(state=current_agent_pose, worldState=current_world, goal=g,dist=dist))
            terminated = self.step(action)
            if terminated or step[-1] == 500:
                break
            count += 1
            step.append(count)
            belief_State_Tracker[ALL_POSSIBLE_GOAL[0]].append(belief[ALL_POSSIBLE_GOAL[0]])
            belief_State_Tracker[ALL_POSSIBLE_GOAL[1]].append(belief[ALL_POSSIBLE_GOAL[1]])
            animate(i=1)
            # update agent pose
            current_agent_pose = (self.env.agent_pos[0], self.env.agent_pos[1])
            prior = belief

        plt.ioff()
        #plt.plot(step, belief_State_Tracker[ALL_POSSIBLE_GOAL[0]], label='Green goal')
        #plt.plot(step, belief_State_Tracker[ALL_POSSIBLE_GOAL[1]], label='Red goal')
        plt.show()
        """while True:
            # get world state
            current_world = self.env.get_world_state()
            #print("State of current world: ")
            #print(current_world)
            
            #current agent goal 
            g = GoalState.green_goal
            #resolve dynamic programming of agent 2
            J2, Q2 = agent_2.value_iteration(dist)
            #pprint.PrettyPrinter(width=20).pprint(Q2)
            # deduce the actual optimal policy
            policy_agent2 = agent_2.deduce_policy(J2, dist)
            
            #take agent 2 action in the world
            #print("Action taken by agent 2 : ")
            #print(policy_agent2[current_world][current_agent_pose][g])
            agent_2.step(policy_agent2[current_world][current_agent_pose][g])
        
            #recalculate Q function of agent 1
            J, Q = self.value_iteration()
        
            #new distribution of action of agent 1 
            dist = self.boltzmann_policy(Q=Q, eta=3)
            
            # generate an action from distribution
            action = ActionsReduced(self.generate_action(state=current_agent_pose, worldState=current_world, goal=GoalState.green_goal,dist=dist))
        
            # take agent 1 action in the world
            terminated = self.step(action)
        
            # update agent pose
            current_agent_pose = (self.env.agent_pos[0], self.env.agent_pos[1])
            if terminated:
                count_sucess = count_sucess + 1
                break
            step = step + 1
             
        print("---------------------------------------- Sucess rate -------------------------------------")
        print(count_sucess/N)
        """

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
        return terminated

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()
    """
    def best_action_value(self, J, s, Q=False): # NOT UPDATED WITH WORLD AND GOAL STATE
        best_a = None
        best_value = float('-inf')
        self.env.set_state(s)
        q = {a:0 for a in ALL_POSSIBLE_ACTIONS}
        for a in ALL_POSSIBLE_ACTIONS:
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
        """
    
    def initializeJ_Q(self, g=GoalState.green_goal):
        states = self.env.get_all_states()
        Q= {}
        J = {}
        big_change ={}
        for w in ALL_POSSIBLE_WOLRD:
            Q[w] = {}
            J[w] = {}
            big_change[w] = {}
            for s in states:
                self.env.set_state(s)
                J[w][s]= {}
                Q[w][s] = {}
                for i in range(len(ALL_POSSIBLE_GOAL)):
                    big_change[w][ALL_POSSIBLE_GOAL[i]] = 0
                    J[w][s][ALL_POSSIBLE_GOAL[i]] = 0
                    Q[w][s][ALL_POSSIBLE_GOAL[i]] = {}
                for a in self.env.get_possible_move(s):
                    Q[w][s][ALL_POSSIBLE_GOAL[i]][a] = 0
        return J, Q, states, big_change 
    
    def bellman_equation(self,J, g, w, a, s):
        next_state_reward = []
        transitions = self.env.get_transition_probs(a, cost_value=1)
        for (prob, r, state_prime) in transitions:
            #print(s)
            #print(state_prime)
            reward = prob*(r + self.gamma*J[w][state_prime][g])
            next_state_reward.append(reward)
        return next_state_reward
    
    def initialize_variation(self):
        big_change = {}
        for w in ALL_POSSIBLE_WOLRD:
                big_change[w] = {}
                for i in range(len(ALL_POSSIBLE_GOAL)):
                    big_change[w][ALL_POSSIBLE_GOAL[i]] = 0
        return big_change 
    
    def variation_superiorTothreshold(self, variation):
        breaking_flag = True
        
        for i in range(len(ALL_POSSIBLE_GOAL)):
            for w in ALL_POSSIBLE_WOLRD:
                if variation[w][ALL_POSSIBLE_GOAL[i]] <= self.threshold:
                    breaking_flag = True * breaking_flag
                else:
                    breaking_flag = False * breaking_flag
        return breaking_flag
                    
    def value_iteration(self, g=GoalState.green_goal):
        J, Q, states, big_change = self.initializeJ_Q()
        g =g
        while True:
            big_change = self.initialize_variation()
            old_J = J
            for w in ALL_POSSIBLE_WOLRD:
                self.env.open_door_manually(w)
                for s in self.env.get_states_non_terminated(): 
                    self.env.set_state(s)
                    # open the door in Value iteration
                    temp = J[w][s][g]
                    #do things to set goals
                    for a in self.env.get_possible_move(s):
                        next_state_reward = self.bellman_equation(J, g, w, a, s) 
                        Q[w][s][g][a]=((np.sum(next_state_reward)))
                                              
                    J[w][s][g] = max(Q[w][s][g].values())
                
                    big_change[w][g] = max(big_change[w][g], np.abs(temp-J[w][s][g]))
                    # close the door
                self.env.open_door_manually(w)    
            if self.variation_superiorTothreshold(big_change):
                break
        return J, Q
       
    def value_iteration_multiple_goal(self):
        # set by default
        self.env.set_env_to_goal(GoalState.green_goal)
        J, Q, states, big_change = self.initializeJ_Q()
        while True:
            big_change = self.initialize_variation()
            for g in ALL_POSSIBLE_GOAL:
                self.env.set_env_to_goal(g)
                for w in ALL_POSSIBLE_WOLRD:
                    self.env.open_door_manually(w)
                    for s in self.env.get_states_non_terminated():
                        self.env.set_state(s)
                        # open the door in Value iteration
                        temp = J[w][s][g]
                        #do things to set goals
                        for a in self.env.get_possible_move(s):
                            next_state_reward = self.bellman_equation(J, g, w, a, s) 
                            Q[w][s][g][a]=((np.sum(next_state_reward)))
                        
                        J[w][s][g] = max(Q[w][s][g].values())
                
                        big_change[w][g] = max(big_change[w][g], np.abs(temp-J[w][s][g]))
                        #close the door
                    self.env.open_door_manually(w)
            if self.variation_superiorTothreshold(big_change):
                break
        return J,Q      
       
    """
    def calculate_values(self, Q=False):# NOT UPDATED WITH WORLD AND GOAL STATE
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
    
    def initialize_random_policy(self): # NOT UPDATED WITH WORLD AND GOAL STATE
        policy = {}
        for s in self.env.get_states_non_terminated():
            policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
        return policy
    
    def calculate_greedy_policy(self, J): # NOT UPDATED WITH WORLD AND GOAL STATE
        policy = self.initialize_random_policy()
        for s in policy.keys():
            self.env.set_state(s)
            best_a , _ = self.best_action_value(J, s)
            policy[s] = best_a
        return policy
    """
    
    def deduce_policy(self, J):
        policy = {}
        self.env.set_env_to_goal(GoalState.green_goal)
        for i in range(len(ALL_POSSIBLE_GOAL)):
            for w in ALL_POSSIBLE_WOLRD:
                policy[w] = {}
                for s in self.env.get_states_non_terminated():
                    policy[w][s] = {}
                    policy[w][s][ALL_POSSIBLE_GOAL[i]] = np.random.choice(ALL_POSSIBLE_ACTIONS)

        for i in range(len(ALL_POSSIBLE_GOAL)):
            self.env.set_env_to_goal(ALL_POSSIBLE_GOAL[i])
            for w in ALL_POSSIBLE_WOLRD:
                # open the door in Value iteration
                self.env.open_door_manually(w)
                for s in self.env.get_states_non_terminated():
                    self.env.set_state(s)
                    Q_table = np.zeros(len(ALL_POSSIBLE_ACTIONS))
                    for action in self.env.get_possible_move(s) :
                        next_state_reward = self.bellman_equation(J, ALL_POSSIBLE_GOAL[i], w, action, s) 
                        Q_table[int(action)] = np.sum(next_state_reward) # the problem is here
                    policy[w][s][ALL_POSSIBLE_GOAL[i]] = ActionsReduced(np.argmax(Q_table))
                self.env.open_door_manually(w)
        return policy                   
                  
    #output the distribution over action in all state of agent 1
    def boltzmann_policy(self, Q, eta):
        #  IMPROVE INITIALIZATION OF DIC 
        dist = {}
        total_prob = {}
        
        states = self.env.get_states_non_terminated()
        for w in ALL_POSSIBLE_WOLRD:
            self.env.open_door_manually(w)
            dist[w] = {}
            total_prob[w] = {}
            #print(w)
            g = GoalState.green_goal
            for s in states:
                self.env.set_state(s)
                dist[w][s] = {}
                total_prob[w][s] = {}
                
                dist[w][s][g] = {}
                total_prob[w][s][g] = 0
                for a in self.env.get_possible_move(s): # still debugging this part but works fine
                    dist[w][s][g][a] = 0
                #for a in ALL_POSSIBLE_ACTIONS :
                print(dist[w][s][g][a] )
                for a in self.env.get_possible_move(s):
                    #print(a)
                    # use max normalization method where we use exp(array - max(array))
                    # instead of exp(arr) which can cause infinite value
                    # we can improve this part of the code
                    dist[w][s][g][a] = (np.exp(eta*(Q[w][s][g][a] - max(Q[w][s][g].values()))))
                    total_prob[w][s][g] += dist[w][s][g][a]
                for a in self.env.get_possible_move(s):
                    dist[w][s][g][a] = (dist[w][s][g][a])/(total_prob[w][s][g])
            # CLOSE the door in Value iteration
            self.env.open_door_manually(w)
        return dist
    
    def boltzmann_policy_multiple_goal(self, Q, eta):
        #  IMPROVE INITIALIZATION OF DIC 
        dist = {}
        total_prob = {}
        
        states = self.env.get_states_non_terminated()
        
        for w in ALL_POSSIBLE_WOLRD:
            self.env.open_door_manually(w)
            dist[w] = {}
            total_prob[w] = {}
            #print(w)
            for s in states:
                self.env.set_state(s)
                dist[w][s] = {}
                total_prob[w][s] = {}
                for i in range(len(ALL_POSSIBLE_GOAL)):
                    dist[w][s][ALL_POSSIBLE_GOAL[i]] = {}
                    total_prob[w][s][ALL_POSSIBLE_GOAL[i]] = 0
                    for a in self.env.get_possible_move(s): # still debugging this part but works fine
                        dist[w][s][ALL_POSSIBLE_GOAL[i]][a] = 0
                    #for a in ALL_POSSIBLE_ACTIONS :
                    for a in self.env.get_possible_move(s):
                        #print(a)
                        # use max normalization method where we use exp(array - max(array))
                        # instead of exp(arr) which can cause infinite value
                        # we can improve this part of the code
                        dist[w][s][ALL_POSSIBLE_GOAL[i]][a] = (np.exp(eta*(Q[w][s][ALL_POSSIBLE_GOAL[i]][a] - max(Q[w][s][ALL_POSSIBLE_GOAL[i]].values()))))
                        total_prob[w][s][ALL_POSSIBLE_GOAL[i]] += dist[w][s][ALL_POSSIBLE_GOAL[i]][a]
                    for a in self.env.get_possible_move(s):
                        dist[w][s][ALL_POSSIBLE_GOAL[i]][a] = (dist[w][s][ALL_POSSIBLE_GOAL[i]][a])/(total_prob[w][s][ALL_POSSIBLE_GOAL[i]])
            # CLOSE the door in Value iteration
            self.env.open_door_manually(w)
        return dist
       
    def generate_action(self, state, worldState, goal, dist):
        possible_action = [a for a in dist[worldState][state][goal].keys()]
        prob = [dist[worldState][state][goal][a] for a in dist[worldState][state][goal].keys()]
        generated_action = np.random.choice(possible_action, p=prob)
        print(generated_action)
        return generated_action


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
