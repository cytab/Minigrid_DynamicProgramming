#!/usr/bin/env python3

from __future__ import annotations
import pickle, time
import gymnasium as gym
import pygame, copy
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
#ALL_POSSIBLE_GOAL = (GoalState.green_goal)


def belief_state(env, previous_dist_g, dist_boltzmann, w, s, previous_state, action_2=None):
        # be carful of dynamic of w that needs the action of agent 2
        #PROCESS ENVIRONEMENT IF POSSIBLE 
        current_dist = previous_dist_g
        normalizing_factor = 0
        for i in range(len(ALL_POSSIBLE_GOAL)):
            conditional_state_world = 0
            env.set_state(previous_state)
            for a in env.get_possible_move(previous_state):
                transition = env.get_transition_probs(a, cost_value=1)
                for (_,_,state_prime) in transition:
                    if state_prime == s:
                        conditional_state_world += dist_boltzmann[ALL_POSSIBLE_GOAL[i]][w][previous_state][a]
            current_dist[ALL_POSSIBLE_GOAL[i]] = conditional_state_world*previous_dist_g[ALL_POSSIBLE_GOAL[i]]
            normalizing_factor += current_dist[ALL_POSSIBLE_GOAL[i]]
            
        current_dist = {ALL_POSSIBLE_GOAL[i]: current_dist[ALL_POSSIBLE_GOAL[i]]/normalizing_factor for i in range(len(ALL_POSSIBLE_GOAL))}
        return current_dist

plt.style.use('fivethirtyeight')
step = []
belief_State_Tracker = {ALL_POSSIBLE_GOAL[i]: [] for i in range(len(ALL_POSSIBLE_GOAL))}
def animate(i):
    plt.cla()
    plt.plot(step, belief_State_Tracker[ALL_POSSIBLE_GOAL[0]], label='Green goal', color='green')
    plt.plot(step, belief_State_Tracker[ALL_POSSIBLE_GOAL[1]], label='Red goal', color='red')
    
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
        self.threshold = 1e-1   
        self.status = dict()
        
        for i in range(len(ALL_POSSIBLE_GOAL)):
            self.status[ALL_POSSIBLE_GOAL[i]] = {}
            for w in ALL_POSSIBLE_WOLRD:
                self.status[ALL_POSSIBLE_GOAL[i]][w] = False
        
    def start(self, agent: AssistiveAgent):
        #N = 1000
        #miinimum_step = 50
        #count_sucess = 0 
        #for i in range(N):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)
        current_agent_pose = (self.env.agent_pos[0],  self.env.agent_pos[1])
        initial_time = time.time()
        # lorsqu'il n'y a pas l'operateur max on a :
            # une boucle de calcul complet dure maximum 0.41 s 0.12s (home pc)
            # nombre iteraation 1439
        # lorsqu'il y a  l'operateur max on a :
            # une boucle de calcul complet dure maximum 0.42 s 0.132 (home pc)
            # nombre iteration 1444
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
        
        #J, Q = self.value_iteration()
        #pprint.PrettyPrinter(width=20).pprint(Q)
        
        
        # Determine initial policy
        dist = self.boltzmann_policy_multiple_goal(Q,eta=9)
        ###   --- T = 
        #pprint.PrettyPrinter(width=20).pprint(dist)
        #print("-------------------------")
        J2, Q2 = agent_2.value_iteration_baseline(dist)
        #pprint.PrettyPrinter(width=20).pprint(Q2)
            # deduce the actual optimal policy
        #policy_agent2 = agent_2.deduce_policy(J2, dist)
        #pprint.PrettyPrinter(width=20).pprint(policy_agent2)
        previous_State = None
        prior = {ALL_POSSIBLE_GOAL[0]: 0.5, ALL_POSSIBLE_GOAL[1]: 0.5}
        belief_State_Tracker[ALL_POSSIBLE_GOAL[0]].append(prior[ALL_POSSIBLE_GOAL[0]])
        belief_State_Tracker[ALL_POSSIBLE_GOAL[1]].append(prior[ALL_POSSIBLE_GOAL[1]])
        count = 0
        step.append(count)
        #ani = FuncAnimation(plt.gcf(), animate, interval=60)
        #ani.save('growingCoil.mp4', writer = 'ffmpeg', fps = 30)
        #self.env.grid.set(self.env.rooms[0].doorPos[0], self.env.rooms[0].doorPos[1], None)
        #self.env.grid.set(self.env.rooms[1].doorPos[0], self.env.rooms[1].doorPos[1], None)
        ''' 
        while True:
            plt.ion()
            previous_State = (self.env.agent_pos[0], self.env.agent_pos[1])
            current_world = self.env.get_world_state()
            g = GoalState.red_goal
            
            action = ActionsReduced(self.generate_action(state=current_agent_pose, worldState=current_world, goal=g,dist=dist))
            terminated = self.step(action)
            
            current_agent_pose = (self.env.agent_pos[0], self.env.agent_pos[1])
            if terminated or step[-1] == 500:
                break
            count += 1
            
            step.append(count)
            belief = belief_state(env=self.env, previous_dist_g=prior, dist_boltzmann=dist, w=current_world, s=current_agent_pose, previous_state=previous_State)

            belief_State_Tracker[ALL_POSSIBLE_GOAL[0]].append(belief[ALL_POSSIBLE_GOAL[0]])
            belief_State_Tracker[ALL_POSSIBLE_GOAL[1]].append(belief[ALL_POSSIBLE_GOAL[1]])
            animate(i=1)
            # update agent pose
            
            prior = belief
        plt.ioff()
        #plt.plot(step, belief_State_Tracker[ALL_POSSIBLE_GOAL[0]], label='Green goal')
        #plt.plot(step, belief_State_Tracker[ALL_POSSIBLE_GOAL[1]], label='Red goal')
        plt.show()
        '''
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
        for i in range(len(ALL_POSSIBLE_GOAL)):
            Q[ALL_POSSIBLE_GOAL[i]] = {}
            J[ALL_POSSIBLE_GOAL[i]] = {}
            big_change[ALL_POSSIBLE_GOAL[i]] = {}
            for w in ALL_POSSIBLE_WOLRD:
                Q[ALL_POSSIBLE_GOAL[i]][w] = {}
                J[ALL_POSSIBLE_GOAL[i]][w] = {}
                big_change[ALL_POSSIBLE_GOAL[i]][w] = 0
                for s in states:
                    self.env.set_state(s)
                    J[ALL_POSSIBLE_GOAL[i]][w][s]= 0
                    Q[ALL_POSSIBLE_GOAL[i]][w][s] = {}
                    for a in self.env.get_possible_move(s):
                        Q[ALL_POSSIBLE_GOAL[i]][w][s][a] = 0
        return J, Q, states, big_change 
    
    def bellman_equation(self,J, g, w, a, s):
        next_state_reward = []
        transitions = self.env.get_transition_probs(a, cost_value=1)
        for (prob, r, state_prime) in transitions:
            #print(s)
            #print(state_prime)
            reward = prob*(r + self.gamma*J[g][w][state_prime])
            next_state_reward.append(reward)
        return next_state_reward
    
    def initialize_variation(self):
        big_change = {}
        for i in range(len(ALL_POSSIBLE_GOAL)):
            big_change[ALL_POSSIBLE_GOAL[i]] = {}
            for w in ALL_POSSIBLE_WOLRD:
                big_change[ALL_POSSIBLE_GOAL[i]][w] = 0
        return big_change 
    
    def variation_superiorTothreshold(self, variation):
        breaking_flag = True
        
        for i in range(len(ALL_POSSIBLE_GOAL)):
            for w in ALL_POSSIBLE_WOLRD:
                if variation[ALL_POSSIBLE_GOAL[i]][w] <= self.threshold:
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
                    temp = J[g][w][s]
                    #do things to set goals
                    for a in self.env.get_possible_move(s):
                        next_state_reward = self.bellman_equation(J, g, w, a, s) 
                        Q[g][w][s][a]=((np.sum(next_state_reward)))
                                              
                    J[g][w][s] = max(Q[g][w][s].values())
                
                    big_change[g][w] = np.abs(temp-J[g][w][s])
                    # close the door
                self.env.open_door_manually(w)    
            if self.variation_superiorTothreshold(big_change):
                break
        return J, Q
       
    def value_iteration_multiple_goal(self):
        # set by default
        self.env.set_env_to_goal(GoalState.green_goal)
        J, Q, states, big_change = self.initializeJ_Q()
        number_iter = 0
        
        while True:
            big_change = self.initialize_variation()
            for g in ALL_POSSIBLE_GOAL:
                self.env.set_env_to_goal(g)
                for w in ALL_POSSIBLE_WOLRD:
                    self.env.open_door_manually(w)
                    #if self.status[w][g] is False: # we only update the value iteration for value function that didn't converge yet
                    for s in self.env.get_states_non_terminated():
                        self.env.set_state(s)
                        # open the door in Value iteration
                        temp = J[g][w][s]
                        #do things to set goals
                        for a in self.env.get_possible_move(s):
                            next_state_reward = self.bellman_equation(J, g, w, a, s) 
                            Q[g][w][s][a]=((np.sum(next_state_reward)))
                        
                        J[g][w][s] = max(Q[g][w][s].values())
                
                        big_change[g][w] = max(big_change[g][w], np.abs(temp-J[g][w][s]))
                        #close the door
                    self.env.open_door_manually(w)
            if self.variation_superiorTothreshold(big_change):
                break
            number_iter += 1
        print(number_iter)
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
        for i in range(len(ALL_POSSIBLE_GOAL)):
            dist[ALL_POSSIBLE_GOAL[i]] = {}
            total_prob[ALL_POSSIBLE_GOAL[i]] = {}
            for w in ALL_POSSIBLE_WOLRD:
                self.env.open_door_manually(w)
                dist[ALL_POSSIBLE_GOAL[i]][w] = {}
                total_prob[ALL_POSSIBLE_GOAL[i]][w] = {}
                #print(w)
                for s in states:
                    self.env.set_state(s)
                    dist[ALL_POSSIBLE_GOAL[i]][w][s] = {}
                    total_prob[ALL_POSSIBLE_GOAL[i]][w][s] = 0
                    for a in self.env.get_possible_move(s): # still debugging this part but works fine
                        dist[ALL_POSSIBLE_GOAL[i]][w][s][a] = 0
                        #for a in ALL_POSSIBLE_ACTIONS :
                    for a in self.env.get_possible_move(s):
                        #print(a)
                        # use max normalization method where we use exp(array - max(array))
                        # instead of exp(arr) which can cause infinite value
                        # we can improve this part of the code
                        dist[ALL_POSSIBLE_GOAL[i]][w][s][a] = (np.exp(eta*(Q[ALL_POSSIBLE_GOAL[i]][w][s][a] - max(Q[ALL_POSSIBLE_GOAL[i]][w][s].values()))))
                        total_prob[ALL_POSSIBLE_GOAL[i]][w][s] += dist[ALL_POSSIBLE_GOAL[i]][w][s][a]
                    for a in self.env.get_possible_move(s):
                        dist[ALL_POSSIBLE_GOAL[i]][w][s][a] = (dist[ALL_POSSIBLE_GOAL[i]][w][s][a])/(total_prob[ALL_POSSIBLE_GOAL[i]][w][s])
                # CLOSE the door in Value iteration
                self.env.open_door_manually(w)
        return dist
       
    def generate_action(self, state, worldState, goal, dist):
        possible_action = [a for a in dist[goal][worldState][state].keys()]
        prob = [dist[goal][worldState][state][a] for a in dist[goal][worldState][state].keys()]
        generated_action = np.random.choice(possible_action, p=prob)
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
