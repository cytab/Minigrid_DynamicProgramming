#!/usr/bin/env python3

from __future__ import annotations
import pickle, time
import pomdp_py
import gymnasium as gym
import cvxpy as cp
import torch
import pygame, copy
from gymnasium import Env
from minigrid.core.grid import Grid
import numpy as np
from minigrid.core.actions import ActionsReduced, ActionsAgent2, WorldSate, GoalState
from minigrid.core.actions import *
from minigrid.envs.empty_reduced import EmptyReducedEnv
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from minigrid.Agent_2 import AssistiveAgent
from pomdp_py import to_pomdp_file
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
import sys
import ast
from scipy.optimize import fsolve, root
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QComboBox,
                             QListWidget, QTextEdit, QPushButton, QGridLayout)
from PyQt5.QtCore import Qt
import json


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, WorldSate):
            return {"__enum__": str(obj)}
        elif isinstance(obj, tuple):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

# Function to save dictionary
def save_dict_to_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, cls=CustomEncoder)

def custom_decoder(dct):
    if "__enum__" in dct:
        name, member = dct["__enum__"].split(".")
        return getattr(WorldSate, member)
    return dct

# Function to load the dictionary
def load_dict_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f, object_hook=custom_decoder)
                    
def convert_keys_to_str(d):
    if isinstance(d, dict):
        return {str(k): convert_keys_to_str(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_keys_to_str(i) for i in d]
    else:
        return d



VIEW_DICTIONNARY = False
USE_LIBRARY = True
ALL_POSSIBLE_ACTIONS = (ActionsReduced.right, ActionsReduced.left, ActionsReduced.forward, ActionsReduced.backward, ActionsReduced.stay)
#ALL_POSSIBLE_WOLRD = (WorldSate.open_door, WorldSate.closed_door)

ALL_POSSIBLE_WOLRD = ((WorldSate.open_door1,WorldSate.open_door2), (WorldSate.open_door1,WorldSate.closed_door2), (WorldSate.closed_door1, WorldSate.open_door2), (WorldSate.closed_door1, WorldSate.closed_door2))
ALL_POSSIBLE_GOAL = (GoalState.green_goal,GoalState.red_goal)
#ALL_POSSIBLE_GOAL = GoalState.green_goal


def belief_state(env, previous_dist_g, dist_boltzmann, w, s, previous_state, action_2=None):
    # be carful of dynamic of w that needs the action of agent 2
    #PROCESS ENVIRONEMENT IF POSSIBLE 
    current_dist = previous_dist_g
    normalizing_factor = 0
    for i in range(len(ALL_POSSIBLE_GOAL)):
        conditional_state_world = 0.000001
        env.set_state(previous_state)
        for a in env.get_possible_move(previous_state):
            transition = env.get_transition_probs(a, cost_value=1)
            for (_,_,state_prime) in transition:
                if state_prime == s:
                    conditional_state_world += dist_boltzmann[ALL_POSSIBLE_GOAL[i]][w][previous_state][a]
        current_dist[ALL_POSSIBLE_GOAL[i]] = conditional_state_world*previous_dist_g[ALL_POSSIBLE_GOAL[i]]
        normalizing_factor += current_dist[ALL_POSSIBLE_GOAL[i]]
    if normalizing_factor > 0:
        current_dist = {ALL_POSSIBLE_GOAL[i]: current_dist[ALL_POSSIBLE_GOAL[i]]/normalizing_factor for i in range(len(ALL_POSSIBLE_GOAL))}
    return current_dist

belief_State_Tracker = {ALL_POSSIBLE_GOAL[i]: [] for i in range(len(ALL_POSSIBLE_GOAL))}
step = []
'''
plt.style.use('fivethirtyeight')
def run_conv(args):
    agent, filename, discount = args
    to_pomdp_file(agent, filename, discount)
'''
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
        '''
        for i in range(len(ALL_POSSIBLE_GOAL)):
            self.status[ALL_POSSIBLE_GOAL[i]] = {}
            for w in ALL_POSSIBLE_WOLRD:
                self.status[ALL_POSSIBLE_GOAL[i]][w] = False
        '''
    def run_simulation(self, agent, discrete_num, dist, g=GoalState.green_goal,  num_simulations=2, max_steps=300):
        count = 0
        agent.set_discretize_num(discrete_num=discrete_num)
        print(f'@@@@@@@@@@@@ Robot policy with discrete_num={discrete_num} @@@@@@@@@@@@')

        start = time.time()
        J2_temp, Q2_temp = agent.value_iteration_baseline(dist)
        end = time.time() - start
        print('Duration for value iteration:', end)

        policy_agent2 = agent.deduce_policy_multiple_goal(J2_temp, dist)

        all_rewards_human_robot = []
        action_counts = {ActionsAgent2.take_key1: [], ActionsAgent2.take_key2: []}

        for sim in range(num_simulations):
            print(f"Simulation {sim + 1}/{num_simulations} for discrete_num={discrete_num}")
            print("------------------------------------------------------------- Start Simulation -------------------------------------------------------------")

            self.reset(self.seed)
            step.append(count)
            prior = {ALL_POSSIBLE_GOAL[0]: PROB_SIM_GREEN_GOAL, ALL_POSSIBLE_GOAL[1]: 1 - PROB_SIM_GREEN_GOAL}
            belief = prior
            belief_State_Tracker[ALL_POSSIBLE_GOAL[0]].append(belief[ALL_POSSIBLE_GOAL[0]])
            belief_State_Tracker[ALL_POSSIBLE_GOAL[1]].append(belief[ALL_POSSIBLE_GOAL[1]])
            current_agent_pose = (self.env.agent_pos[0], self.env.agent_pos[1])
            count = 0
            human_robot_cumulative_reward = 0
            self.env.set_env_to_goal(g)

            # Collect rewards
            collected_reward_vector_human_robot = []
            terminated = False

            while not terminated and count < max_steps:
                previous_State = (self.env.agent_pos[0], self.env.agent_pos[1])
                current_world = self.env.get_world_state()
                print(f"Step {count}")
                print("-------------------------------------------------------------")
                print(f"Green goal: {g}, Previous state: {previous_State}, Current world state: {current_world}")
                print(f"Belief at this time step: {belief}")

                approx_belief = agent.approx_prob_to_belief(belief[ALL_POSSIBLE_GOAL[0]])
                action_robot = policy_agent2[approx_belief][current_world][current_agent_pose]
                reward_robot = agent_2.step(action_robot)
                print("Action taken by Agent 2:", action_robot, "Robot reward:", reward_robot)

                # Track occurrences of GoalState.take1 and GoalState.take2
                if action_robot == ActionsAgent2.take_key1:
                    action_counts[ActionsAgent2.take_key1].append(count)
                elif action_robot == ActionsAgent2.take_key2:
                    action_counts[ActionsAgent2.take_key2].append(count)

                current_world = self.env.get_world_state()
                action_human = ActionsReduced(self.generate_action(state=current_agent_pose, worldState=current_world, goal=g, dist=dist))
                terminated, reward_human = self.step(action_human)
                print("Action taken by Human:", action_human, 'Human reward:', reward_human)

                current_agent_pose = (self.env.agent_pos[0], self.env.agent_pos[1])
                human_robot_cumulative_reward += reward_robot + reward_human
                

                print(f"Current Agent Pose: {current_agent_pose}")
                print(f"Cumulative Rewards (Human + Robot): {human_robot_cumulative_reward}")
                print("-------------------------------------------------------------")

                count += 1
                belief = belief_state(env=self.env, previous_dist_g=prior, dist_boltzmann=dist, w=current_world, s=current_agent_pose, previous_state=previous_State)
                prior = belief

                # Collect rewards
                collected_reward_vector_human_robot.append(human_robot_cumulative_reward)

            # Pad rewards if simulation ends early
            while len(collected_reward_vector_human_robot) < max_steps:
                collected_reward_vector_human_robot.append(human_robot_cumulative_reward)

            all_rewards_human_robot.append(collected_reward_vector_human_robot)

        # Convert rewards to numpy array
        all_rewards_human_robot = np.array(all_rewards_human_robot, dtype=np.float64)

        # Calculate mean and standard deviation
        mean_rewards = np.nanmean(all_rewards_human_robot, axis=0)
        std_rewards = np.nanstd(all_rewards_human_robot, axis=0)

        mean_action_counts = {
        ActionsAgent2.take_key1: np.mean(action_counts[ActionsAgent2.take_key1]) if action_counts[ActionsAgent2.take_key1] else 0,
        ActionsAgent2.take_key2: np.mean(action_counts[ActionsAgent2.take_key2]) if action_counts[ActionsAgent2.take_key2] else 0
        }

        print("Mean Action Counts:")
        print(f"GoalState.take1 mean count: {mean_action_counts[ActionsAgent2.take_key1]}")
        print(f"GoalState.take2 mean count: {mean_action_counts[ActionsAgent2.take_key2]}")

        return mean_rewards, std_rewards, mean_action_counts
    
    def compute_dist(self, eta, g):
        print(f"Computing dist for eta={eta}")
        self.env.set_env_to_goal(g)
        J, Q = self.value_iteration_multiple_goal()
        dist = self.boltzmann_policy_multiple_goal(Q, eta=eta)
        return dist
    
    def start(self, agent: AssistiveAgent):
        #N = 1000
        #miinimum_step = 50
        #count_sucess = 0 
        #for i in range(N):
        """Start the window display with blocking event loop"""
        # Data for the histgram
        '''
        actions = ['Haut', 'Bas', 'Gauche', 'Droite', 'Attendre']
        probabilities = [0.2, 0.35, 0.15, 0.1, 0.2]  # Example probabilities

        prob_1_1_open_doors_green = [0.01, 0.4994381549665872, 0.01, 0.4994381549665872, 0.01223412786521707]
        prob_7_4_open_doors_green = [0.01, 0.9894615931350329, 0.01, 0.01, 0.05384068649515891]
        prob_4_4_open_doors_green = [0.01, 0.49978007905860766,0.01, 0.49978007905860766, 0.04398418827708831]
        prob_10_8_open_doors_green = [0.01, 0.9898384891317205, 0.01, 0.01, 0.016149093871353055]
        prob_7_4_open_doors_green_rat_0_25 = [0.13305095468775188, 0.33171378008335184, 0.13305095468775188, 0.13305095468775188, 0.2691333558533926]
        prob_7_4_open_doors_green_rat_0_8 = [0.032121371683456974, 0.5975575544788935,0.032121371683456974,0.032121371683456974, 0.30607833047073546]

        prob_1_1_open_doors_red = [0.01, 0.49964592205498537, 0.01, 0.49964592205498537, 0.01223412786521707]
        prob_7_4_open_doors_red = [0.01, 0.01, 0.01, 0.9896752830127303, 0.03247169872635597]
        prob_4_4_open_doors_red = [0.01, 0.01, 0.01, 0.9594615931350331, 0.02384068649515893]
        prob_10_8_open_doors_red = [0.01, 0.4995176038517855, 0.4995176038517855, 0.01, 0.0646732462595567]

        prob_7_4_open_doors_red_rat_0_25 = [0.13132053739951044, 0.13132053739951044, 0.13132053739951044, 0.3366841023778008, 0.26935428542366785]
        prob_7_4_open_doors_red_rat_0_8 = [0.03002360791635148,0.03002360791635148, 0.03002360791635148,0.6108164836880408, 0.2991126925629049]

        prob_1_1_close_doors_red = [0.012334891422532415, 0.012334891422532415, 0.012334891422532415, 0.012334891422532415, 0.9895066043430986]
        #prob_7_4_close_doors_red = [0.00012334891422532415, 0.00012334891422532415, 0.00012334891422532415, 0.00012334891422532415, 0.9995066043430986]
        #prob_4_4_close_doors_red = []
        #prob_10_8_close_doors_red = []
        # Create the plot
        # Define a color palette suitable for research papers
        colors = ['#4C72B0', '#AAAAA8', '#C44E52', '#8172B2', '#CCB974']

        # Create the plot with different colors for each action
        # Create the plot with a white background
        fig, ax = plt.subplots(figsize=(8, 6))  # Figure background
        fig.patch.set_facecolor('white')  # Set the figure background to white
        ax.set_facecolor('white')  # Set the axes (grid) background to white

        # Create the bar chart
        bars = ax.bar(actions, prob_7_4_open_doors_green_rat_0_8, color=colors, edgecolor='black')

        # Add titles and labels
        ax.set_title("Actions at (7, 4), Green Objective", fontsize=14, weight='bold')
        ax.set_xlabel("Actions", fontsize=12)
        ax.set_ylabel("Probability", fontsize=12)
        ax.set_ylim(0, 1)  # Probabilities range from 0 to 1

        # Add gridlines with white background
        ax.grid(axis='y', linestyle='--', alpha=0.7, color='gray')  # Gridlines over white
        # Make the grid edges black
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)  # Optional: Set the thickness of the edge
        # Hide parts of the spines to emphasize a white background outside the grid
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add a legend
        ax.legend(bars, actions, title="Actions", fontsize=10, title_fontsize=11, loc="upper right")

        # Improve the layout for a research paper
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()

        # Show the plot
        plt.show()
        '''
        
        '''
        # N_values = [5, 15, 30]
        # taill_grille = ['12x12', '16x16', '32x32']
        # computation_times = [135.73, 414, 855.4]
        # computation_times_state_humain = [414, 981.11, 8977]
        # computation_time_sarsop_state_human = [20399, 244483+23703, z+z1] # [20399, 374757]

        # fig, ax = plt.subplots(figsize=(8, 6))
        # fig.patch.set_facecolor('white')
        # ax.set_facecolor('white')

        # ax.plot(taill_grille, computation_times_state_humain, marker='o', linestyle='-', color='b')
        # ax.set_title("Computation time of policy")
        # ax.set_xlabel("Grid Word size")
        # ax.set_ylabel("Time (seconds)")
        # ax.grid(axis='y', linestyle='--', alpha=0.7, color='gray')

        # for spine in ax.spines.values():
        #     spine.set_edgecolor('black')
        #     spine.set_linewidth(1.2)

        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        # # Set exact y-axis ticks
        # ax.set_yticks(computation_times_state_humain)
        # ax.set_yticklabels([f'{time:.2f}' for time in computation_times])

        # plt.tight_layout()
        # plt.show()
        '''
        
        self.reset(self.seed)
        #self.env.place_agent(top=(1,5))
        current_agent_pose = (self.env.agent_pos[0],  self.env.agent_pos[1])
        g = GoalState.green_goal
        
        if self.env.multiple_goal:
            self.env.set_env_to_goal(g)
            print('@@@@@@@@@@@@@@@@@@ Human policy Level 0 calculus @@@@@@@@@@@@@@@@@@')
            J, Q = self.value_iteration_multiple_goal()
            dist = self.boltzmann_policy_multiple_goal(Q,eta=0.8)
            f = open("dist.txt","w")
        
            # write file
            f.write( str(dist) )
            
            # close file
            f.close()
            
            # agent 2
            start = time.time()
            if USE_LIBRARY:
                print('@@@@@@@@@@@@@@@@@@ Robot policy Level 0 calculus @@@@@@@@@@@@@@@@@@')

                J2_temp, Q2_temp = agent_2.value_iteration_baseline(dist)
                f = open("J2.txt","w")
            
                # write file
                f.write( str(J2_temp) )
                
                
            
                # close file
                f.close()
                end = time.time() - start 
                print('Duree pour resolution : ')
                print(end)
                policy_agent2 = agent_2.deduce_policy_multiple_goal(J2_temp, dist)
            
            '''
            print('@@@@@@@@@@@@@@@@@@ Human policy ADAPTATIVE Level 1 calculus @@@@@@@@@@@@@@@@@@')

            J_iterative, Q_iterative = self.value_iteration_iterative_game(agent=agent_2, p_action=dist)
            dist_iterative = self.boltzmann_policy_iterative_game(Q=Q_iterative, agent=agent_2, eta=9)
            
            print('@@@@@@@@@@@@@@@@@@ Robot policy Level 1 calculus @@@@@@@@@@@@@@@@@@')

            J_iterative_1, Q_iterative_1 = agent_2.value_iteration_baseline_iterative_game(dist_iterative)
            policy_agent2_niveau_1 = agent_2.deduce_policy_iterative_game(J=J_iterative_1, p_action=dist_iterative)
            '''
            
            if VIEW_DICTIONNARY:
                converted_dict = convert_keys_to_str(dist)
                converted_Q = convert_keys_to_str(Q2_temp)
                converted_policy2 = convert_keys_to_str(policy_agent2)
                #converted_iterative_q = convert_keys_to_str(Q_iterative)
                #converted_dist_iterative = convert_keys_to_str(dist_iterative)
                #converted_iterative_q_2 = convert_keys_to_str(Q_iterative_1)
                #converted_iterative_policy_agent2_niveau_1 = convert_keys_to_str(policy_agent2_niveau_1)
                # Application setup
                app = QApplication(sys.argv)
                viewer = DynamicDualDictViewer(converted_dict, converted_policy2, dict1_name='Poliy Agent H niveau 0', dict2_name='PoliCY Agent 2 niveau 0')
                
                sys.exit(app.exec_())
        else:
            J, Q = self.value_iteration()
            dist = self.boltzmann_policy(Q, eta=5)
        
        epsilon = 1e-5

        robotproblem = Robotproblem(word1=ALL_POSSIBLE_WOLRD[3][0], world2=ALL_POSSIBLE_WOLRD[3][1], pose=current_agent_pose, goal=g, env=env, dim=(16,16), human_probability=dist, epsilon=epsilon, initial_prob=PROB_SIM_GREEN_GOAL)

        count = 0
        # Parameters for the Monte Carlo simulation
        NUM_SIMULATIONS = 50  # Number of simulations to run
        MAX_STEPS = 300  # Maximum number of steps per simulation

        # Main code to iterate over discrete_num values and plot results
        #etas = [0.01, 0.8, 2]
        etas = [0.75, 0.5, 2]
        discrete_nums = [5, 15, 30]
        colors = ["orange", "blue", "green"]
        MAX_STEPS_ETA = [140, 80, 150]
        steps = np.arange(1, MAX_STEPS + 1)
        
        for eta in etas:
            print(f"Running simulations for eta={eta}")
            dist = self.compute_dist(eta, g=g)

            plt.figure(figsize=(12, 8))

            for i, discrete_num in enumerate(discrete_nums):
                mean_rewards, std_rewards, mean_action_counts = self.run_simulation(agent_2, discrete_num, dist, max_steps=MAX_STEPS_ETA[i], g=g, num_simulations=NUM_SIMULATIONS)
                steps = np.arange(1, MAX_STEPS_ETA[i] + 1)

                plt.plot(steps[:60], mean_rewards[:60], label=f"discrete_num={discrete_num}", color=colors[i])
                plt.fill_between(steps, mean_rewards - std_rewards, mean_rewards + std_rewards, color=colors[i], alpha=0.2)
                
                # plt.figure(figsize=(10, 6))
                # plt.bar(mean_action_counts.keys(), mean_action_counts.values(), color=['blue', 'orange'], alpha=0.7)
                # plt.xlabel("Action Type")
                # plt.ylabel("Mean Number of Times Action Taken")
                # plt.title("Histogram of Mean Actions Taken by Agent 2")
                # plt.xticks(ticks=[0, 1], labels=["ActionsAgent2.take_key1", "ActionsAgent2.take_key2"])
                # plt.grid(True)
                # plt.savefig("mean_action_histogram.png")
                # plt.show()

                
            plt.title(f"Average Human-Robot Reward (eta={eta})")
            plt.xlabel("Steps")
            plt.ylabel("Average Reward")
            plt.legend()
            plt.grid(True)

            # Save the plot to a file or display it
            plt.savefig(f"reward_plot_eta_{eta}.png")  # Save to file for later use
            plt.show()  # Show the plot
            
            
        # step.append(count)
        # prior = {ALL_POSSIBLE_GOAL[0]: PROB_SIM_GREEN_GOAL, ALL_POSSIBLE_GOAL[1]: 1-PROB_SIM_GREEN_GOAL}
        # belief = prior
        # belief_State_Tracker[ALL_POSSIBLE_GOAL[0]].append(belief[ALL_POSSIBLE_GOAL[0]])
        # belief_State_Tracker[ALL_POSSIBLE_GOAL[1]].append(belief[ALL_POSSIBLE_GOAL[1]])
        # collected_reward_vector_human = []
        # collected_reward_vector_robot = []
        # human_cumulative_reward = 0
        # robot_cumulative_reward = 0
        # g = GoalState.red_goal
        # terminated = False
        # while True :
        #     plt.ion()
        #     if not terminated and count < 300:
        #         previous_State = (self.env.agent_pos[0], self.env.agent_pos[1])
        #         current_world = self.env.get_world_state()
                
        #         print("real goal", g, "Actual state :", previous_State, "Actual world state :", current_world, "belief at this time step", belief)
        #         #take agent 2 action in the world
        #         approx_belief = agent.approx_prob_to_belief(belief[ALL_POSSIBLE_GOAL[0]])
        #         print("Action taken by agent 2 : %s" % str(policy_agent2[approx_belief][current_world][current_agent_pose]))
        #         reward_robot = agent_2.step(policy_agent2[approx_belief][current_world][current_agent_pose])
                
        #         action = ActionsReduced(self.generate_action(state=current_agent_pose, worldState=current_world, goal=g,dist=dist))
        #         terminated, reward_human = self.step(action)
        #         print("Action taken by Human : %s" % str(action))
        #         current_agent_pose = (self.env.agent_pos[0], self.env.agent_pos[1])
        #         human_cumulative_reward += reward_human
        #         robot_cumulative_reward += reward_robot
        #         count += 1
        #         step.append(count)
                
        #         belief = belief_state(env=self.env, previous_dist_g=prior, dist_boltzmann=dist, w=current_world, s=current_agent_pose, previous_state=previous_State)
                
        #         belief_State_Tracker[ALL_POSSIBLE_GOAL[0]].append(belief[ALL_POSSIBLE_GOAL[0]])
        #         belief_State_Tracker[ALL_POSSIBLE_GOAL[1]].append(belief[ALL_POSSIBLE_GOAL[1]])
        #         # update agent pose
        #         animate(i=1)
                
        #         prior = belief
        #         collected_reward_vector_robot.append(robot_cumulative_reward)
        #         collected_reward_vector_human.append(human_cumulative_reward)
                
        #     elif terminated and count < 300:
        #         g = GoalState.green_goal
        #         collected_reward_vector_robot.append(reward_robot)
        #         collected_reward_vector_human.append(reward_human)
            
        #     elif count >= 300:
        #         break
        # plt.ioff()
        # plt.plot(step, belief_State_Tracker[ALL_POSSIBLE_GOAL[0]], label='Green goal')
        # plt.plot(step, belief_State_Tracker[ALL_POSSIBLE_GOAL[1]], label='Red goal')
        # plt.show()    
        
        '''
        solve(
            problem,
            max_depth=12000,
            discount_factor=0.99,
            planning_time=5.0,
            exploration_const=50000,
            visualize=True,
            max_time=120,
            max_steps=500,
            solver_type='sarsop')
            
        '''
        # print(agent_2.discretize_belief)
        # print(agent_2.approx_prob_to_belief(0.11))
        
        solve(
            robotproblem,
            max_depth=12000,
            discount_factor=0.99,
            planning_time=5.0,
            exploration_const=50000,
            visualize=True,
            max_time=120,
            max_steps=500,
            solver_type='sarsop',
            humanproblem=False,
            human_intent=g,
            dist=dist,
            computed_policy=policy_agent2,
            agent2=agent_2)
        

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
            
            prior = b beta = cp.Variable()elief
        plt.ioff()
        #plt.plot(step, belief_State_Tracker[ALL_POSSIBLE_GOAL[0]], label='Green goal')
        #plt.plot(step, belief_State_Tracker[ALL_POSSIBLE_GOAL[1]], label='Red goal')
        plt.show()
        '''

    def start_simBeta(self, agent: AssistiveAgent, initial_eta=9, initialProb=0.5):
        global step
        eta = 0.5
        tru_Eta = 0.12
        self.reset(self.seed)
        current_agent_pose = (self.env.agent_pos[0],  self.env.agent_pos[1])
        g = GoalState.red_goal
        
        if self.env.multiple_goal:
            self.env.set_env_to_goal(g)
            print('@@@@@@@@@@@@@@@@@@ Human policy Level 0 calculus @@@@@@@@@@@@@@@@@@')
            J, Q = self.value_iteration_multiple_goal()
            dist = self.boltzmann_policy_multiple_goal(Q,eta=eta)
            real_dts = self.boltzmann_policy_multiple_goal(Q,eta=tru_Eta)
            f = open("dist.txt","w")
        
            # write file
            f.write( str(dist) )
            
            # close file
            f.close()
            
            # agent 2
            '''
            start = time.time()
            print('@@@@@@@@@@@@@@@@@@ Robot policy Level 0 calculus @@@@@@@@@@@@@@@@@@')

            J2_temp, Q2_temp = agent_2.value_iteration_baseline(dist)
            f = open("J2.txt","w")
        
            # write file
            f.write( str(J2_temp) )
            
            
        
            # close file
            f.close()
            end = time.time() - start 
            print('Duree pour resolution : ')
            print(end)
            policy_agent2 = agent_2.deduce_policy_multiple_goal(J2_temp, dist)
            '''
    
        else:
            J, Q = self.value_iteration()
            dist = self.boltzmann_policy(Q, eta=5)
        
        prior = {ALL_POSSIBLE_GOAL[0]: initialProb, ALL_POSSIBLE_GOAL[1]: 1-initialProb}
        collect_data = []
        count = 0
        step.append(count)
        belief = prior
        # Initialize the history dictionary
        history = {1: [], 5: [], 45: []}
        time_history = {1:0, 5:0, 45:0}
        # Number of
        # Monte Carlo runs
        num_runs = 20

        # N values to iterate over
        N_values = [1, 5, 45]

        # Storage for final statistics
        statistics = {}
        
        for n_t in N_values:
            all_estimate_evolutions = []  # Store estimate evolutions for all runs
            print(f"Running Monte Carlo Simulation for N = {n_t}")
            duratin_Taken = []
            for run in range(num_runs):
                
                collect_data = []  # To collect data for each simulation run        
                estimate_evolution = []  # Track evolution of estimates for this run

                #belief_State_Tracker = {ALL_POSSIBLE_GOAL[0]: [], ALL_POSSIBLE_GOAL[1]: []}
                #prior = {ALL_POSSIBLE_GOAL[0]: initialProb, ALL_POSSIBLE_GOAL[1]: 1-initialProb}  # Initialize belief state prior
                
                for n in range(n_t):
                    #agent_2.step(ActionsAgent2.take_key1)
                    #agent_2.step(ActionsAgent2.take_key2)
                    print(f"  Data collection - Run: {run + 1}, Iteration: {n + 1}")
                    self.reset(self.seed)
                    current_agent_pose = (self.env.agent_pos[0], self.env.agent_pos[1])
                    belief_State_Tracker = {ALL_POSSIBLE_GOAL[0]: [], ALL_POSSIBLE_GOAL[1]: []}
                    prior = {ALL_POSSIBLE_GOAL[0]: initialProb, ALL_POSSIBLE_GOAL[1]: 1-initialProb}  # Initialize belief state prior
                    #agent_2.step(ActionsAgent2.take_key1)
                    #agent_2.step(ActionsAgent2.take_key2)
                    step = []
                    step.append(count)
                    count = 0
                    while True:
                        if n > n_t / 2:
                            g = GoalState.green_goal
                        else:
                            g = GoalState.red_goal

                        previous_State = (self.env.agent_pos[0], self.env.agent_pos[1])
                        current_world = self.env.get_world_state()
                        action = ActionsReduced(self.generate_action(
                            state=current_agent_pose, worldState=current_world, goal=g, dist=real_dts))

                        collect_data.append([current_agent_pose, current_world, action, g, belief.get(ALL_POSSIBLE_GOAL[0], 0)])
                        terminated, r = self.step(action)

                        current_agent_pose = (self.env.agent_pos[0], self.env.agent_pos[1])
                        if terminated or count == 200:
                            print('!!!!!!!!!!!!!!!!!! terminated !!!!!!!!!!!!!!!!!!!!!!!!')
                            print(terminated)
                            break
                        count += 1

                        # Update belief state
                        belief = belief_state(env=self.env, previous_dist_g=prior, dist_boltzmann=dist,
                                            w=current_world, s=current_agent_pose, previous_state=previous_State)
                        belief_State_Tracker[ALL_POSSIBLE_GOAL[0]].append(belief[ALL_POSSIBLE_GOAL[0]])
                        belief_State_Tracker[ALL_POSSIBLE_GOAL[1]].append(belief[ALL_POSSIBLE_GOAL[1]])
                        step.append(count)
                        prior = belief

                # Estimate parameters using the BoltzmanEstimator
                estimator = BoltzmanEstimator(data=collect_data, q_function=Q, boltzman_policy=dist, initial_beta=eta)
                temp_evolution = estimator.gradient_iteration_hidden_goal(datas=collect_data)
                duratin_Taken.append(temp_evolution[1])
                all_estimate_evolutions.append(temp_evolution[0])  # Store evolution of estimates
                
            # Compute statistics at each iteration
            estimate_evolution_length = len(all_estimate_evolutions[0])  # Assumes all evolutions have the same length
            mean_estimate = np.zeros(estimate_evolution_length)
            variance_estimate = np.zeros(estimate_evolution_length)
            for i in range(estimate_evolution_length):
                values_at_iteration = []
                for run_evolution in all_estimate_evolutions:
                    if i < len(run_evolution):  # Ensure index is within bounds
                        values_at_iteration.append(run_evolution[i])
                    else:
                        print(f"Skipping index {i} for this run because its length is {len(run_evolution)}")
                
                if values_at_iteration:  # Check if we collected any values before computing mean/variance
                    mean_estimate[i] = np.mean(values_at_iteration)
                    variance_estimate[i] = np.var(values_at_iteration)
                else:
                    print(f"Warning: No values collected for iteration {i}")
            
            # Store the best estimate evolution and variance
            history[n_t] = mean_estimate
            statistics[n_t] = variance_estimate
            time_history[n_t] = np.mean(duratin_Taken)
            
        estimator.stock_history_hidden_goal(history, statistics, time_history)
            
        estimator.plot_beta_Estimation_com(gradient=True, groundtruth=tru_Eta, em=False)  
        estimator.plot_time() 
        '''
        data = np.array([[0.1, 0.5, 0.3], [0.4, 0.2, 0.6], [0.7, 0.8, 0.9]])
        row_labels = ['State 1', 'State 2', 'State 3']
        col_labels = ['Action A', 'Action B', 'Action C']

        # Plotting the table
        fig, ax = plt.subplots()
        ax.axis('tight')
        ax.axis('off')

        # Create the table
        table = ax.table(cellText=data, rowLabels=row_labels, colLabels=col_labels, cellLoc='center', loc='center')

        # Styling (optional)
        table.scale(1, 1.5)  # Scale the table to make it more readable
        table.auto_set_font_size(False)
        table.set_fontsize(12)

        plt.show()
        '''
        
        
    def step(self, action: ActionsReduced):
        terminated , reward, _, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")
        #self.env.set_state(self.env.agent_pos)
        #pose, reward = self.env.check_move(action, self.env.get_world_state())
        reward = self.env.get_reward_1(self.env.agent_pos[0], self.env.agent_pos[1], action)
        terminated = True if (self.env.agent_pos[0],  self.env.agent_pos[1]) == (self.env.goal_pose[0][0], self.env.goal_pose[0][1]) else False
        print(terminated)
        if terminated:
            print("terminated!")
            #self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()
        return terminated, reward 

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
        if self.env.multiple_goal :
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
                        for a in ALL_POSSIBLE_ACTIONS:
                            Q[ALL_POSSIBLE_GOAL[i]][w][s][a] = 0
            
        else:
                Q[ALL_POSSIBLE_GOAL] = {}
                J[ALL_POSSIBLE_GOAL] = {}
                big_change[ALL_POSSIBLE_GOAL] = {}
                for w in ALL_POSSIBLE_WOLRD:
                    Q[ALL_POSSIBLE_GOAL][w] = {}
                    J[ALL_POSSIBLE_GOAL][w] = {}
                    big_change[ALL_POSSIBLE_GOAL][w] = 0
                    for s in states:
                        self.env.set_state(s)
                        J[ALL_POSSIBLE_GOAL][w][s]= 0
                        Q[ALL_POSSIBLE_GOAL][w][s] = {}
                        for a in ALL_POSSIBLE_ACTIONS:
                            Q[ALL_POSSIBLE_GOAL][w][s][a] = 0
        return J, Q, states, big_change 
    
    def initializeJ_Q_iterative_game(self, agent, g = GoalState.green_goal) :
        states = self.env.get_all_states()
        Q= {}
        J = {}
        big_change ={}
        for i in range(len(ALL_POSSIBLE_GOAL)):
            Q[ALL_POSSIBLE_GOAL[i]] = {}
            J[ALL_POSSIBLE_GOAL[i]] = {}
            big_change[ALL_POSSIBLE_GOAL[i]] = {}
            for belief in agent.discretize_belief:
                Q[ALL_POSSIBLE_GOAL[i]][belief] = {}
                J[ALL_POSSIBLE_GOAL[i]][belief] = {}
                big_change[ALL_POSSIBLE_GOAL[i]][belief] = {}
                for w in ALL_POSSIBLE_WOLRD:
                    Q[ALL_POSSIBLE_GOAL[i]][belief][w] = {}
                    J[ALL_POSSIBLE_GOAL[i]][belief][w] = {}
                    big_change[ALL_POSSIBLE_GOAL[i]][belief][w] = 0
                    for s in states:
                        self.env.set_state(s)
                        J[ALL_POSSIBLE_GOAL[i]][belief][w][s]= 0
                        Q[ALL_POSSIBLE_GOAL[i]][belief][w][s] = {}
                        for a in ALL_POSSIBLE_ACTIONS:
                            Q[ALL_POSSIBLE_GOAL[i]][belief][w][s][a] = 0
        return J, Q, states, big_change 
                            
    def bellman_equation(self,J, g, w, a, s):
        next_state_reward = []
        transitions = self.env.get_transition_probs(a, cost_value=1)
        for (prob, r, state_prime) in transitions:
            reward = prob*(r + self.gamma*J[g][w][state_prime])
            next_state_reward.append(reward)
        return next_state_reward
    
    def world_dynamic_update(self, action, current_world):
        world_prime = None
        if not self.env.multiple_goal:
            if action == ActionsAgent2.take_key and current_world == WorldSate.closed_door :
                world_prime = WorldSate.open_door
            if action == ActionsAgent2.take_key and current_world == WorldSate.open_door :
                world_prime = WorldSate.open_door
            if action == ActionsAgent2.nothing:
                world_prime = current_world
        else:
            if action == ActionsAgent2.take_key1 and current_world[0] == WorldSate.closed_door1:
                world_prime = (WorldSate.open_door1, current_world[1])
            elif action == ActionsAgent2.take_key2 and current_world[1] == WorldSate.closed_door2:
                world_prime = (current_world[0], WorldSate.open_door2)
            elif action == ActionsAgent2.nothing:
                world_prime = current_world
            else:
                world_prime = current_world
        return world_prime
    
    def belief_state(self, previous_dist_g, dist_boltzmann, w, s, previous_state, action_2=None):
        # be carful of dynamic of w that needs the action of agent 2
        #PROCESS ENVIRONEMENT IF POSSIBLE 
        current_dist = previous_dist_g
        normalizing_factor = 0
        for i in range(len(ALL_POSSIBLE_GOAL)):
            conditional_state_world = 0
            self.env.set_state(previous_state)
            for a in ALL_POSSIBLE_ACTIONS:
                transition = self.env.get_transition_probs(a, cost_value=1)
                for (_,_,state_prime) in transition:
                    if state_prime == s:
                        conditional_state_world += dist_boltzmann[ALL_POSSIBLE_GOAL[i]][w][previous_state][a]
            current_dist[ALL_POSSIBLE_GOAL[i]] = conditional_state_world*previous_dist_g[ALL_POSSIBLE_GOAL[i]]
            normalizing_factor += current_dist[ALL_POSSIBLE_GOAL[i]]
        if normalizing_factor > 0:
            current_dist = {ALL_POSSIBLE_GOAL[i]: current_dist[ALL_POSSIBLE_GOAL[i]]/normalizing_factor for i in range(len(ALL_POSSIBLE_GOAL))}
        return current_dist
    
    def belief_state_discretize(self, belief, dist_boltzmann, w, s, previous_state, action_2=None):
        # be carful of dynamic of w that needs the action of agent 2
        #PROCESS ENVIRONEMENT IF POSSIBLE
        current_dist = {ALL_POSSIBLE_GOAL[0]: belief, ALL_POSSIBLE_GOAL[1]: 1-belief}
        current_dist = self.belief_state(previous_dist_g=current_dist, dist_boltzmann=dist_boltzmann, w=w, s=s, previous_state=previous_state)
        return current_dist[ALL_POSSIBLE_GOAL[0]]
    
    def bellman_equation_iterative_game(self,agent, p_action, J, g, belief, w, a, s):
        next_state_reward = []
        optimal_action2 = agent.policy(belief, w, s)
        #print(optimal_action2)
        self.env.check_move(action=optimal_action2, w=w)
        transitions = self.env.get_transition_probs( action=a, cost_value=1)
        for (prob, r, state_prime) in transitions:    
            world_prime = self.world_dynamic_update(optimal_action2, w)
            next_belief = self.belief_state_discretize(belief=belief, dist_boltzmann=p_action, w=world_prime,s=state_prime, previous_state=s)
            next_belief = agent.approx_prob_to_belief(next_belief)
            r = self.env.check_move(action=a,w=world_prime,cost_value=1)
            reward = prob*(r[1] + self.gamma*J[g][next_belief][world_prime][state_prime])
            next_state_reward.append(reward)
        #print(next_state_reward)
        self.env.check_move(action=optimal_action2, w=w)
        return next_state_reward
    
    def initialize_variation(self):
        big_change = {}
        if self.env.multiple_goal:
            
            
            for i in range(len(ALL_POSSIBLE_GOAL)):
                big_change[ALL_POSSIBLE_GOAL[i]] = {}
                for w in ALL_POSSIBLE_WOLRD:
                    big_change[ALL_POSSIBLE_GOAL[i]][w] = 0
            
        else:
                big_change[ALL_POSSIBLE_GOAL] = {}
                for w in ALL_POSSIBLE_WOLRD:
                    big_change[ALL_POSSIBLE_GOAL][w] = 0
                    
        return big_change 

    def initialize_variation_iterative_game(self, agent):
            big_change = {}
            if self.env.multiple_goal:
                
                
                for i in range(len(ALL_POSSIBLE_GOAL)):
                    big_change[ALL_POSSIBLE_GOAL[i]] = {}
                    for belief in agent.discretize_belief:
                        big_change[ALL_POSSIBLE_GOAL[i]][belief] = {}
                        for w in ALL_POSSIBLE_WOLRD:
                            big_change[ALL_POSSIBLE_GOAL[i]][belief][w] = 0
                        
            return big_change 

    def variation_superiorTothreshold(self, variation):
        breaking_flag = True
        if self.env.multiple_goal:
            
            
            for i in range(len(ALL_POSSIBLE_GOAL)):
                for w in ALL_POSSIBLE_WOLRD:
                    if variation[ALL_POSSIBLE_GOAL[i]][w] <= self.threshold:
                        breaking_flag = True * breaking_flag
                    else:
                        breaking_flag = False * breaking_flag
            
        else:
                for w in ALL_POSSIBLE_WOLRD:
                    if variation[ALL_POSSIBLE_GOAL][w] <= self.threshold:
                        breaking_flag = True * breaking_flag
                    else:
                        breaking_flag = False * breaking_flag
        return breaking_flag
    
    def variation_superiorTothreshold_iterative_game(self, agent, variation):
        breaking_flag = True
        for i in range(len(ALL_POSSIBLE_GOAL)):
            for belief in agent.discretize_belief:
                for w in ALL_POSSIBLE_WOLRD:
                    #print(variation[ALL_POSSIBLE_GOAL[i]][belief][w])
                    if variation[ALL_POSSIBLE_GOAL[i]][belief][w] <= self.threshold:
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
                
                    big_change[g][w] = max(big_change[g][w], np.abs(temp-J[g][w][s]))
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
                        for a in ALL_POSSIBLE_ACTIONS:
                            next_state_reward = self.bellman_equation(J, g, w, a, s) 
                            Q[g][w][s][a]=((np.sum(next_state_reward)))
                        
                        J[g][w][s] = max(Q[g][w][s].values())
                
                        big_change[g][w] = max(big_change[g][w], np.abs(temp-J[g][w][s]))
                        #close the door
                    self.env.open_door_manually(w)
            if self.variation_superiorTothreshold(big_change):
                break
            number_iter += 1
        #print(number_iter)
        return J,Q      
    
    def value_iteration_iterative_game(self, agent, p_action):
        # set by default
        self.env.set_env_to_goal(GoalState.green_goal)
        J, Q, states, big_change = self.initializeJ_Q_iterative_game(agent=agent)
        number_iter = 0
        temp = {}
        while True:
            big_change = self.initialize_variation_iterative_game(agent=agent)
            initial_time = time.time()
            for g in ALL_POSSIBLE_GOAL:
                self.env.set_env_to_goal(g)
                for belief in agent.discretize_belief: 
                    for w in ALL_POSSIBLE_WOLRD:
                        self.env.open_door_manually(w)
                        #if self.status[w][g] is False: # we only update the value iteration for value function that didn't converge yet
                        for s in self.env.get_states_non_terminated():
                            self.env.set_state(s)
                            # open the door in Value iteration
                            temp = J[g][belief][w][s]
                            #do things to set goals
                            for a in ALL_POSSIBLE_ACTIONS:
                                next_state_reward = self.bellman_equation_iterative_game(agent=agent, p_action=p_action, J=J, g=g, belief=belief, w=w, a=a, s=s) 
                                #print(next_state_reward)
                                Q[g][belief][w][s][a]=((np.sum(next_state_reward)))
                        
                            J[g][belief][w][s] = max(Q[g][belief][w][s].values())
                
                            big_change[g][belief][w] = max(big_change[g][belief][w], np.abs(temp-J[g][belief][w][s]))
                        #close the door
                    self.env.open_door_manually(w)
            value_iteration_elapsed_time = initial_time - time.time()
            print('Elpased time for value iteration with multiple goal:')
            print(value_iteration_elapsed_time)
            print(number_iter)
            if self.variation_superiorTothreshold_iterative_game(agent=agent, variation=big_change):
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
    '''
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
    '''      
    #output the distribution over action in all state of agent 1

    def boltzmann_policy(self, Q, eta):
        #  IMPROVE INITIALIZATION OF DIC 
        dist = {}
        total_prob = {}
        
        states = self.env.get_all_states()
        for w in ALL_POSSIBLE_WOLRD:
            self.env.open_door_manually(w)
            dist[w] = {}
            total_prob[w] = {}
            g = GoalState.green_goal
            for s in states:
                self.env.set_state(s)
                dist[w][s] = {}
                total_prob[w][s] = {}
                
                dist[w][s][g] = {}
                total_prob[w][s][g] = 0
                for a in ALL_POSSIBLE_ACTIONS: # still debugging this part but works fine
                    dist[w][s][g][a] = 0
                #for a in ALL_POSSIBLE_ACTIONS :
                for a in self.env.get_possible_move(s):
                    # use max normalization method where we use exp(array - max(array))
                    # instead of exp(arr) which can cause infinite value
                    # we can improve this part of the code
                    dist[w][s][g][a] = (np.exp(eta*(Q[g][w][s][a] - max(Q[g][w][s].values()))))
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
        
        states = self.env.get_all_states()
        for i in range(len(ALL_POSSIBLE_GOAL)):
            dist[ALL_POSSIBLE_GOAL[i]] = {}
            total_prob[ALL_POSSIBLE_GOAL[i]] = {}
            for w in ALL_POSSIBLE_WOLRD:
                self.env.open_door_manually(w)
                dist[ALL_POSSIBLE_GOAL[i]][w] = {}
                total_prob[ALL_POSSIBLE_GOAL[i]][w] = {}
                for s in states:
                    self.env.set_state(s)
                    dist[ALL_POSSIBLE_GOAL[i]][w][s] = {}
                    total_prob[ALL_POSSIBLE_GOAL[i]][w][s] = 0
                    for a in ALL_POSSIBLE_ACTIONS: # still debugging this part but works fine
                        dist[ALL_POSSIBLE_GOAL[i]][w][s][a] = 0
                        #for a in ALL_POSSIBLE_ACTIONS :
                    for a in ALL_POSSIBLE_ACTIONS:
                        # use max normalization method where we use exp(array - max(array))
                        # instead of exp(arr) which can cause infinite value
                        # we can improve this part of the code
                        dist[ALL_POSSIBLE_GOAL[i]][w][s][a] = (np.exp(eta*(Q[ALL_POSSIBLE_GOAL[i]][w][s][a] - max(Q[ALL_POSSIBLE_GOAL[i]][w][s].values()))))
                        total_prob[ALL_POSSIBLE_GOAL[i]][w][s] += dist[ALL_POSSIBLE_GOAL[i]][w][s][a]
                    for a in ALL_POSSIBLE_ACTIONS:
                        dist[ALL_POSSIBLE_GOAL[i]][w][s][a] = (dist[ALL_POSSIBLE_GOAL[i]][w][s][a])/(total_prob[ALL_POSSIBLE_GOAL[i]][w][s])
                # CLOSE the door in Value iteration
                self.env.open_door_manually(w)
        return dist

    def boltzmann_policy_iterative_game(self, agent, Q, eta):
        #  IMPROVE INITIALIZATION OF DIC 
        dist = {}
        total_prob = {}
        
        states = self.env.get_all_states()
        for i in range(len(ALL_POSSIBLE_GOAL)):
            dist[ALL_POSSIBLE_GOAL[i]] = {}
            total_prob[ALL_POSSIBLE_GOAL[i]] = {}
            for belief in agent.discretize_belief:
                dist[ALL_POSSIBLE_GOAL[i]][belief] = {}
                total_prob[ALL_POSSIBLE_GOAL[i]][belief] = {}
                for w in ALL_POSSIBLE_WOLRD:
                    self.env.open_door_manually(w)
                    dist[ALL_POSSIBLE_GOAL[i]][belief][w] = {}
                    total_prob[ALL_POSSIBLE_GOAL[i]][belief][w] = {}
                    for s in states:
                        self.env.set_state(s)
                        dist[ALL_POSSIBLE_GOAL[i]][belief][w][s] = {}
                        total_prob[ALL_POSSIBLE_GOAL[i]][belief][w][s] = 0
                        for a in ALL_POSSIBLE_ACTIONS: # still debugging this part but works fine
                            dist[ALL_POSSIBLE_GOAL[i]][belief][w][s][a] = 0
                        for a in ALL_POSSIBLE_ACTIONS :
                            # use max normalization method where we use exp(array - max(array))
                            # instead of exp(arr) which can cause infinite value
                            # we can improve this part of the code
                            dist[ALL_POSSIBLE_GOAL[i]][belief][w][s][a] = (np.exp(eta*(Q[ALL_POSSIBLE_GOAL[i]][belief][w][s][a] - max(Q[ALL_POSSIBLE_GOAL[i]][belief][w][s].values()))))
                            total_prob[ALL_POSSIBLE_GOAL[i]][belief][w][s] += dist[ALL_POSSIBLE_GOAL[i]][belief][w][s][a]
                        for a in ALL_POSSIBLE_ACTIONS:
                            dist[ALL_POSSIBLE_GOAL[i]][belief][w][s][a] = (dist[ALL_POSSIBLE_GOAL[i]][belief][w][s][a])/(total_prob[ALL_POSSIBLE_GOAL[i]][belief][w][s])
                    # CLOSE the door in Value iteration
                    self.env.open_door_manually(w)
        return dist
    
    def generate_action(self, state, worldState, goal, dist):
        possible_action = [a for a in dist[goal][worldState][state].keys()]
        prob = [dist[goal][worldState][state][a] for a in dist[goal][worldState][state].keys()]
        generated_action = np.random.choice(possible_action, p=prob)
        return generated_action

    def generate_action_iterative_game(self, belief, state, worldState, goal, dist):
        possible_action = [a for a in dist[goal][belief][worldState][state].keys()]
        prob = [dist[goal][belief][worldState][state][a] for a in dist[goal][belief][worldState][state].keys()]
        generated_action = np.random.choice(possible_action, p=prob)
        return generated_action
  
class BoltzmanEstimator:
    "EM or gradient descent estimation for boltzmann policy temperature variable beta"
    def __init__(self, data, q_function, boltzman_policy, initial_beta=9):
        
        # The format of the data is ([x_t, w^{'}_t, a^H,g]; [x_t, w^{'}_t, a^H,g])
        self.all_data = data
        self.initial_beta = initial_beta
        self.optimized_beta = initial_beta
        self.history_beta_gradient = []
        self.history_beta_gradient_hidden_goal = []
        self.history_beta_em = []
        self.history_beta_em_hidden_goal = []
        self.q_function = q_function
        self.boltzman_policy = boltzman_policy
        self.changing_data = []
        self.N_history_hidden = {}
        self.N_time_computation = {}
        self.statistics = []
        self.N = 0
    
    def reset_histogram(self):
        self.history_beta_em = []
        self.history_beta_em_hidden_goal = []
        
    def stock_history_hidden_goal(self, history, statistics, time_taken=[]):
       # self.N_history_hidden[self.N] = []
        self.N_history_hidden = history
        self.statistics = statistics
        self.N_time_computation = time_taken
        
    def gradient_iteration(self, datas, n_iterations=1e6, learning_rate=1e-2, decreasing_step=0.99, epsilon=1e-12):
        gradient = 0
        iteration = 0
        beta_old = 1
        beta = self.initial_beta
        self.history_beta_gradient.append(beta)
        convergeance = True
        erreur = 1
        debut = time.time()
        while(erreur >= epsilon):
            gradient = 0
            for data in datas:
                goal = data[3]
                a_t = data[2]
                w_t = data[1]
                x_t = data[0]
                q = self.q_function[goal][w_t][x_t][a_t]
                max_q = max(self.q_function[goal][w_t][x_t].values())
                exp_sum = sum(np.exp(beta* (self.q_function[goal][w_t][x_t][a]-max_q)) for a in ALL_POSSIBLE_ACTIONS)
                numerateur = sum(self.q_function[goal][w_t][x_t][a]* np.exp(beta*(self.q_function[goal][w_t][x_t][a]-max_q)) for a in ALL_POSSIBLE_ACTIONS)
                gradient += q - (numerateur/exp_sum)

            #update old value
            beta_old = beta
            #update new beta 
            beta +=learning_rate*gradient
            #compute error
            erreur = abs(beta-beta_old)
            #print(erreur)
            if iteration % 2000 == 0:
                #update learnig rate
                learning_rate = learning_rate*decreasing_step
                print(f"Iteration {iteration}: beta = {beta}, gradient = {gradient}")
            
            if iteration == n_iterations:
                convergeance = False
                break
            iteration += 1
            self.history_beta_gradient.append(beta)
        duration = time.time() - debut
        print(f"Iteration {iteration}: beta = {beta}, gradient = {gradient}, erreur = {erreur}, beta old = {beta_old}, duration in second = {duration}")
        if convergeance is True:
            self.optimized_beta = beta
            print(f"Iteration {iteration}: beta = {beta}, gradient = {gradient}")

    def gradient_iteration_hidden_goal(self, datas, n_iterations=1e6, learning_rate=1e-3, decreasing_step=0.90, epsilon=1e-12):
        gradient = 0
        iteration = 0
        beta_old = 1
        beta = self.initial_beta
        self.history_beta_gradient_hidden_goal.append(beta)
        convergeance = True
        erreur = 1
        debut = time.time()
        while(erreur >= epsilon):
            gradient = 0
            for data in datas:
                #goal = data[3]
                for g in ALL_POSSIBLE_GOAL:
                    
                    belief_t = data[4] if g == ALL_POSSIBLE_GOAL[0] else 1 - data[4]
                    a_t = data[2]
                    w_t = data[1]
                    x_t = data[0]
                    q = (self.q_function[g][w_t][x_t][a_t])
                    max_q = max(self.q_function[g][w_t][x_t].values())
                    exp_sum = sum(np.exp(beta* (self.q_function[g][w_t][x_t][a]-max_q)) for a in ALL_POSSIBLE_ACTIONS)
                    numerateur = sum(self.q_function[g][w_t][x_t][a]* np.exp(beta*(self.q_function[g][w_t][x_t][a]-max_q)) for a in ALL_POSSIBLE_ACTIONS)
                    gradient += belief_t*(q - (numerateur/exp_sum))

            #update old value
            beta_old = beta
            #update new beta 
            beta +=learning_rate*gradient
            #compute error
            erreur = abs(beta-beta_old)
            #print(erreur)
            if iteration % 2000 == 0:
                #update learnig rate
                learning_rate = learning_rate*decreasing_step
                print(f"Iteration {iteration}: beta = {beta}, gradient = {gradient}")
            
            if iteration == n_iterations:
                convergeance = False
                break
            iteration += 1
            self.history_beta_gradient_hidden_goal.append(beta)
        duration = time.time() - debut
        print(f"Iteration {iteration}: beta = {beta}, gradient = {gradient}, erreur = {erreur}, beta old = {beta_old}, duration in second = {duration}")
        if convergeance is True:
            self.optimized_beta = beta
            print(f"Iteration {iteration}: beta = {beta}, gradient = {gradient}")
        return (self.history_beta_gradient_hidden_goal, duration) 
            
    def maximum_expectation_iteration(self, datas, epsilon=1e-12, hidden_goal= False):
        iteration = 0
        self.beta_old = np.inf
        #beta = cp.Variable()
        #beta.value = self.initial_beta
        beta = torch.tensor(self.initial_beta, requires_grad=True)  # Initial value of beta
        learning_rate = 1e-2
        if not hidden_goal:
            self.history_beta_em.append(beta)
        else:
            self.history_beta_em_hidden_goal.append(beta)
        self.all_data = datas
        erreur = 1
        debut = time.time()
        self.changing_data = datas
        #max_norm = 1.0 
        while(erreur >= epsilon):
            #print(beta)
            
            #update beta
            self.beta_old = beta.item()
            if not hidden_goal:
                beta = root(self.func_to_optimize, 12e-6).x[0]
                self.history_beta_em.append(beta)

            else:
                # beta = root(self.func_to_optimize_hidden_goal, 12e-6).x[0]
                # self.history_beta_em_hidden_goal.append(beta)
                '''
                # Define the objective
                objective = cp.Maximize(self.func_to_optimize_hidden_goal_cvx(beta))

                # Define constraints (if any)
                constraints = [beta >= 0, beta <= 10]  # Example: beta must be non-negative
                
                # Setup the problem
                problem = cp.Problem(objective, constraints)
                # Solve the problem
                problem.solve()

                # Extract the optimal value
                optimal_beta = beta.value
                optimal_value = problem.value

                print("Optimal beta:", optimal_beta)
                print("Optimal value:", optimal_value)
                
                '''
                
                for step in range(10000):  # Iterate for a fixed number of steps
                    big_tot = 0  # Reset the objective

                    for data in self.changing_data:
                        for g in ALL_POSSIBLE_GOAL:
                            # Extract data components
                            belief_t = data[4] if g == ALL_POSSIBLE_GOAL[0] else 1 - data[4]
                            a_t = data[2]
                            w_t = data[1]
                            x_t = data[0]

                            # Compute Q-values and max Q-value
                            q_values = torch.tensor([self.q_function[g][w_t][x_t][a] for a in ALL_POSSIBLE_ACTIONS])
                            max_q = torch.max(q_values)

                            # Compute softmax probabilities
                            exp_q = torch.exp(beta * (q_values - max_q))
                            action_probs = exp_q / torch.sum(exp_q)

                            # Compute the current action's Q-value
                            q = self.q_function[g][w_t][x_t][a_t]

                            # Log-sum-exp term for numerical stability
                            exp_q_values = beta * (q_values - max_q)
                            log_sum_exp = torch.logsumexp(exp_q_values, dim=0)

                            # Compute the contribution to the total
                            tot = torch.sum(action_probs * (beta * q_values - log_sum_exp))
                            big_tot += belief_t * tot

                    # Maximize the objective
                    objective = big_tot
                    objective.backward()  # Compute gradients

                    # Clip gradient to prevent instability
                    beta.grad = torch.clamp(beta.grad, -1.0, 1.0)

                    # Update beta using gradient ascent
                    with torch.no_grad():
                        beta += learning_rate * beta.grad

                    # Clear gradients before the next step
                    beta.grad.zero_()

                    # Optionally: Print progress
                    if step % 10 == 0:
                        print(f"Step {step}: Objective = {objective.item()}, Beta = {beta.item()}")
                
            #print(beta)
            print(f"Iteration {iteration+1}, Beta: {beta.item()}")
            print(f"Iteration {iteration}: beta = {beta}, beta old = {self.beta_old}")
            #compute error
            erreur = abs(beta.item()-self.beta_old)
            #print(erreur)
            iteration += 1
        duration = time.time() - debut
        print(f"Iteration {iteration}: beta = {beta.item()}, erreur = {erreur}, beta old = {self.beta_old}, duration in second = {duration}")
        return (self.history_beta_em_hidden_goal, duration) if hidden_goal else (self.history_beta_em, duration)

    
    def func_to_optimize(self, beta):
        big_tot = 0
        for data in self.changing_data:
            goal = data[3]
            a_t = data[2]
            w_t = data[1]
            x_t = data[0]
            max_q = max(self.q_function[data[3]][w_t][x_t].values())
            action_probs = np.array([np.exp(self.beta_old *(self.q_function[data[3]][w_t][x_t][a]-max_q)) for a in ALL_POSSIBLE_ACTIONS])
            action_probs /= action_probs.sum()  # Normalize to get probabilities
            q = self.q_function[data[3]][w_t][x_t][a_t]
            
            exp_sum = sum(np.exp(beta* (self.q_function[data[3]][w_t][x_t][a]-max_q)) for a in ALL_POSSIBLE_ACTIONS)
            numerateur = sum(self.q_function[data[3]][w_t][x_t][a]* np.exp(beta*(self.q_function[data[3]][w_t][x_t][a]-max_q)) for a in ALL_POSSIBLE_ACTIONS)
            tot=0
            for i, a in enumerate(ALL_POSSIBLE_ACTIONS):
                tot += action_probs[i] *(q - (numerateur/exp_sum))
            big_tot += tot
        return big_tot
    
    def func_to_optimize_hidden_goal(self, beta):
        big_tot = 0
        for data in self.changing_data:
            for g in ALL_POSSIBLE_GOAL:
                    
                belief_t = data[4] if g == ALL_POSSIBLE_GOAL[0] else 1 - data[4]
                goal = data[3]
                a_t = data[2]
                w_t = data[1]
                x_t = data[0]
                max_q = max(self.q_function[g][w_t][x_t].values())
                action_probs = np.array([np.exp(self.beta_old *(self.q_function[g][w_t][x_t][a]-max_q)) for a in ALL_POSSIBLE_ACTIONS])
                action_probs /= action_probs.sum()  # Normalize to get probabilities
                q = self.q_function[g][w_t][x_t][a_t]
                
                exp_sum = sum(np.exp(beta* (self.q_function[g][w_t][x_t][a]-max_q)) for a in ALL_POSSIBLE_ACTIONS)
                numerateur = sum(self.q_function[g][w_t][x_t][a]* np.exp(beta*(self.q_function[g][w_t][x_t][a]-max_q)) for a in ALL_POSSIBLE_ACTIONS)
                tot=0
                for i, a in enumerate(ALL_POSSIBLE_ACTIONS):
                    tot += action_probs[i] *(q - (numerateur/exp_sum))
                big_tot += belief_t*tot
        return big_tot
    
    def func_to_optimize_hidden_goal_cvx(self, beta):
        big_tot = 0

        for data in self.changing_data:
            for g in ALL_POSSIBLE_GOAL:
                belief_t = data[4] if g == ALL_POSSIBLE_GOAL[0] else 1 - data[4]
                goal = data[3]
                a_t = data[2]
                w_t = data[1]
                x_t = data[0]

                # Compute Q-values
                q_values = [self.q_function[g][w_t][x_t][a] for a in ALL_POSSIBLE_ACTIONS]
                max_q = max(q_values)

                # Exponentials of adjusted Q-values
                exp_q = cp.hstack([cp.exp(self.beta_old * (q - max_q)) for q in q_values])
                
                # Action probabilities
                action_probs = exp_q / cp.sum(exp_q)

                # Numerator and denominator for current action
                q = [self.q_function[g][w_t][x_t][a] for a in ALL_POSSIBLE_ACTIONS]
                exp_q_values = cp.hstack([ (q_i - max_q) for q_i in q_values])

                # Contribution to the total
                tot_terms = []
                for i, a in enumerate(ALL_POSSIBLE_ACTIONS):
                    term = action_probs[i] * ( q[i] - cp.log_sum_exp(beta * exp_q_values))
                    tot_terms.append(term)
                tot = cp.sum(cp.hstack(tot_terms))

                # Add belief-weighted contribution to the total
                big_tot += belief_t * tot

        return big_tot
    
    def func_to_optimize_torch(self, beta):
        big_tot = 0
        for data in self.changing_data:
            for g in ALL_POSSIBLE_GOAL:
                belief_t = data[4] if g == ALL_POSSIBLE_GOAL[0] else 1 - data[4]
                a_t = data[2]
                w_t = data[1]
                x_t = data[0]

                q_values = [self.q_function[g][w_t][x_t][a] for a in ALL_POSSIBLE_ACTIONS]
                max_q = max(q_values)

                exp_q = torch.exp(self.beta_old * (torch.tensor(q_values) - max_q))
                action_probs = exp_q / exp_q.sum()

                q = self.q_function[g][w_t][x_t][a_t]
                exp_q_values = beta * (torch.tensor(q_values) - max_q)

                tot = torch.sum(
                    action_probs * (beta * q_values - torch.logsumexp(exp_q_values, dim=0))
                )

                big_tot += belief_t * tot

        return big_tot
    
    def plot_beta_Estimation(self, gradient=True, em=False, hidden_goal_comp=False, groundtruth=0.01):
        if gradient is True:
            if hidden_goal_comp is False:
                plt.plot(self.history_beta_gradient, marker='o', linestyle='-')
                plt.xlabel('Iteration')
                plt.ylabel('value')
                plt.title('EStimation Beta with gradient')
                
                # Add a horizontal reference line 
                plt.axhline(y=groundtruth, color='black', linestyle='--', label='Vrai valeur')
                plt.legend()
            else:
                plt.plot(self.history_beta_gradient_hidden_goal, marker='o', linestyle='-')
                plt.xlabel('Iteration')
                plt.ylabel('value')
                plt.title('EStimation Beta with gradient')
                
                # Add a horizontal reference line 
                plt.axhline(y=groundtruth, color='black', linestyle='--', label='Vrai valeur')
                plt.legend()
            
        elif em:
            if hidden_goal_comp is False:
                plt.plot(self.history_beta_em, marker='o', linestyle='-')
                plt.xlabel('Iteration')
                plt.ylabel('value')
                plt.title('EStimation Beta with EM')
                
                # Add a horizontal reference line 
                plt.axhline(y=groundtruth, color='black', linestyle='--', label='Vrai valeur')
                plt.legend()
            else:
                plt.plot(self.history_beta_em_hidden_goal, marker='o', linestyle='-')
                plt.xlabel('Iteration')
                plt.ylabel('value')
                plt.title('EStimation Beta with gradient')
                
                # Add a horizontal reference line 
                plt.axhline(y=groundtruth, color='black', linestyle='--', label='Vrai valeur')
                plt.legend()
        plt.show()
        
    def plot_beta_Estimation_com(self, gradient=True, em=False, groundtruth=0.01):
        colors = ['blue', 'green', 'orange', 'purple']  # Define colors for different cases
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        if gradient:
            print("Plotting Gradient Ascent Results...")

            for i, n_t in enumerate(self.N_history_hidden.keys()):
                    best_estimate_evolution = self.N_history_hidden[n_t]
                    variance_evolution = self.statistics[n_t]  # Ensure statistics are computed

                    lower_bound = best_estimate_evolution - np.sqrt(variance_evolution)
                    upper_bound = best_estimate_evolution + np.sqrt(variance_evolution)

                    # Plot gradient ascent evolution
                    ax.plot(range(len(best_estimate_evolution)), best_estimate_evolution,
                            color=colors[i % len(colors)], linestyle='-', linewidth=2,
                            label=f'Gradient ascent avec objectif inconnu K = {n_t}')

                    # Shaded region for variance
                    ax.fill_between(range(len(best_estimate_evolution)), lower_bound, upper_bound,
                                    color=colors[i % len(colors)], alpha=0.2,
                                    label=f'Variance K = {n_t}')
                

            ax.set_title("Évolution de l'estimation avec Gradient Ascent", fontsize=14, weight='bold')

        elif em:
            print("Plotting EM Results...")

            for i, n_t in enumerate(self.N_history_hidden.keys()):
                best_estimate_evolution = self.N_history_hidden[n_t]
                variance_evolution = self.statistics[n_t]

                lower_bound = best_estimate_evolution - np.sqrt(variance_evolution)
                upper_bound = best_estimate_evolution + np.sqrt(variance_evolution)

                # Plot EM estimation evolution
                ax.plot(range(len(best_estimate_evolution)), best_estimate_evolution,
                        color=colors[i % len(colors)], linestyle='-', linewidth=2,
                        label=f'EM avec objectif inconnu K = {n_t}')

                # Shaded region for variance
                ax.fill_between(range(len(best_estimate_evolution)), lower_bound, upper_bound,
                                color=colors[i % len(colors)], alpha=0.2,
                                label=f'Variance K = {n_t}')

            ax.set_title("Évolution de l'estimation avec EM", fontsize=14, weight='bold')

        # Common styling
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel("Itération", fontsize=12)
        ax.set_ylabel("Valeur de Beta", fontsize=12)
        plt.axhline(y=groundtruth, color='black', linestyle='--', label='Vraie valeur')

        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_time(self):
        # Define colors dynamically if K values grow
        colors = ['blue', 'green', 'orange', 'purple', 'red', 'cyan', 'magenta']
        K = ["1", "5", "45"]
        computation_times = list(self.N_time_computation.values())

        fig, ax = plt.subplots(figsize=(8, 6))  # Figure background
        fig.patch.set_facecolor('white')  # Set the figure background to white
        ax.set_facecolor('white')  # Set the axes (grid) background to white

        # Create the bar chart
        bars = ax.bar(K, computation_times, color=colors[:len(K)], edgecolor='black')

        # Add labels to each bar
        for bar, time in zip(bars, computation_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f'{time:.2f} s',
                    ha='center', va='bottom', fontsize=10, weight='bold')

        # Add titles and labels
        ax.set_title("Temps de calcul", fontsize=14, weight='bold')
        ax.set_xlabel("K", fontsize=12)
        ax.set_ylabel("Temps (s)", fontsize=12)

        # Set y-ticks to match computation times
        ax.set_yticks(computation_times)
        ax.set_yticklabels([f'{time:.2f}' for time in computation_times], fontsize=10)

        # Add gridlines with white background
        ax.grid(axis='y', linestyle='--', alpha=0.7, color='gray')

        # Make the grid edges black
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)

        # Hide unnecessary spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add legend
        ax.legend(bars, K, title="Valeur de K", fontsize=10, title_fontsize=11, loc="upper right")

        # Improve the layout for a research paper
        plt.xticks(fontsize=10)
        plt.tight_layout()

        # Show the plot
        plt.show()
    
    def belief_state(self, env, previous_dist_g, dist_boltzmann, w, s, previous_state, action_2=None):
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
            
        if normalizing_factor > 0:
            current_dist = {ALL_POSSIBLE_GOAL[i]: current_dist[ALL_POSSIBLE_GOAL[i]]/normalizing_factor for i in range(len(ALL_POSSIBLE_GOAL))}
        return current_dist
  
  
class DynamicDualDictViewer(QWidget):
    def __init__(self, dict1, dict2, dict1_name="Dictionary 1", dict2_name="Dictionary 2", parent=None):
        super().__init__(parent)
        self.dict1 = dict1
        self.dict2 = dict2
        self.current_dict1 = dict1
        self.current_dict2 = dict2
        self.dict1_name = dict1_name
        self.dict2_name = dict2_name

        self.active_history1 = []  # Active history of navigation for dict1
        self.active_history2 = []  # Active history of navigation for dict2

        self.passive_history1 = [(dict1, 'Root')]  # Passive history for dict1 (stores all visited states)
        self.passive_history2 = [(dict2, 'Root')]  # Passive history for dict2

        self.breadcrumb1 = []  # Track selected keys in dict1
        self.breadcrumb2 = []  # Track selected keys in dict2

        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        # Create a row layout for the dictionary comparison area
        self.comparison_layout = QGridLayout()

        # Breadcrumbs for tracking key paths
        self.breadcrumb1_label = QLabel('Path: /')
        self.breadcrumb2_label = QLabel('Path: /')
        self.comparison_layout.addWidget(self.breadcrumb1_label, 0, 0)
        self.comparison_layout.addWidget(self.breadcrumb2_label, 0, 1)

        # Dictionary names displayed at the top
        self.dict1_label = QLabel(f'{self.dict1_name}')
        self.dict2_label = QLabel(f'{self.dict2_name}')
        self.comparison_layout.addWidget(self.dict1_label, 1, 0)
        self.comparison_layout.addWidget(self.dict2_label, 1, 1)

        # Dropdowns for selecting keys from both dictionaries
        self.left_list = QListWidget()
        self.left_list.addItems(self.current_dict1.keys())
        self.left_list.setSelectionMode(QListWidget.MultiSelection)
        self.left_list.itemSelectionChanged.connect(self.on_select_left)
        self.comparison_layout.addWidget(self.left_list, 2, 0)

        self.right_list = QListWidget()
        self.right_list.addItems(self.current_dict2.keys())
        self.right_list.setSelectionMode(QListWidget.MultiSelection)
        self.right_list.itemSelectionChanged.connect(self.on_select_right)
        self.comparison_layout.addWidget(self.right_list, 2, 1)

        # Layout for multiple text areas to display values
        self.left_values_layout = QVBoxLayout()
        self.right_values_layout = QVBoxLayout()

        self.comparison_layout.addLayout(self.left_values_layout, 3, 0)
        self.comparison_layout.addLayout(self.right_values_layout, 3, 1)

        # Add back buttons for each dictionary
        self.left_back_button = QPushButton("Back (Left Dict)")
        self.left_back_button.clicked.connect(self.go_back_left)
        self.left_back_button.setEnabled(False)
        self.comparison_layout.addWidget(self.left_back_button, 4, 0)

        self.right_back_button = QPushButton("Back (Right Dict)")
        self.right_back_button.clicked.connect(self.go_back_right)
        self.right_back_button.setEnabled(False)
        self.comparison_layout.addWidget(self.right_back_button, 4, 1)

        # Add passive history navigation combo boxes
        self.left_history_combo = QComboBox()
        self.left_history_combo.addItems([item[1] for item in self.passive_history1])
        self.left_history_combo.currentIndexChanged.connect(self.on_select_passive_history_left)
        self.comparison_layout.addWidget(self.left_history_combo, 5, 0)

        self.right_history_combo = QComboBox()
        self.right_history_combo.addItems([item[1] for item in self.passive_history2])
        self.right_history_combo.currentIndexChanged.connect(self.on_select_passive_history_right)
        self.comparison_layout.addWidget(self.right_history_combo, 5, 1)

        # Set up the main layout
        self.layout.addLayout(self.comparison_layout)
        self.setLayout(self.layout)
        self.setWindowTitle('Dynamic Dual Dictionary Viewer')

        # Apply the style to the entire application
        self.apply_styles()
        self.show()

    def apply_styles(self):
        """Apply custom styles to the widgets for a better visual appearance."""
        self.setStyleSheet("""
            QLabel {
                font-family: Arial;
                font-size: 16px;
                font-weight: bold;
                color: #333333;
            }
            QListWidget {
                background-color: #f0f0f0;
                font-family: Arial;
                font-size: 14px;
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 10px;
                margin: 10px 0;
            }
            QTextEdit {
                background-color: #ffffff;
                font-family: Courier;
                font-size: 14px;
                border: 1px solid #cccccc;
                border-radius: 10px;
                padding: 10px;
                margin-bottom: 10px;
            }
            QPushButton {
                background-color: #8B4513;
                font-family: Arial;
                font-size: 14px;
                font-weight: bold;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 5px;
                margin: 10px 0;
            }
            QPushButton:hover {
                background-color: #A0522D;
            }
            QWidget {
                background-color: #f8f9fa;
            }
        """)

    def clear_values(self):
        """Clear all the value display areas."""
        for i in reversed(range(self.left_values_layout.count())):
            self.left_values_layout.itemAt(i).widget().deleteLater()
        for i in reversed(range(self.right_values_layout.count())):
            self.right_values_layout.itemAt(i).widget().deleteLater()

    def display_values(self, dictionary, keys, layout, side):
        """Display the selected values in the provided layout without clearing previous values."""
        self.clear_values()  # Ensure only the current selection is visible
        for key in keys:
            selected_value = dictionary.get(key)
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            if isinstance(selected_value, dict):
                text_edit.setText(f"{key}: <Nested Dictionary>")
                # Add to active and passive history, and allow going deeper
                if side == 'left':
                    self.active_history1.append((self.current_dict1, key))
                    self.passive_history1.append((self.current_dict1, key))
                    self.left_history_combo.addItem(f"{key}")
                    self.current_dict1 = selected_value
                    self.left_list.clear()
                    self.left_list.addItems(self.current_dict1.keys())
                    self.left_back_button.setEnabled(True)
                else:
                    self.active_history2.append((self.current_dict2, key))
                    self.passive_history2.append((self.current_dict2, key))
                    self.right_history_combo.addItem(f"{key}")
                    self.current_dict2 = selected_value
                    self.right_list.clear()
                    self.right_list.addItems(self.current_dict2.keys())
                    self.right_back_button.setEnabled(True)
            else:
                text_edit.setText(f"{key}: {str(selected_value)}")
            layout.addWidget(text_edit)

    def on_select_left(self):
        """Handle the selection of multiple keys in the left dictionary."""
        selected_items = self.left_list.selectedItems()
        selected_keys = [item.text() for item in selected_items]
        self.breadcrumb1 = selected_keys
        self.update_breadcrumbs()
        self.display_values(self.current_dict1, selected_keys, self.left_values_layout, 'left')

    def on_select_right(self):
        """Handle the selection of multiple keys in the right dictionary."""
        selected_items = self.right_list.selectedItems()
        selected_keys = [item.text() for item in selected_items]
        self.breadcrumb2 = selected_keys
        self.update_breadcrumbs()
        self.display_values(self.current_dict2, selected_keys, self.right_values_layout, 'right')

    def update_breadcrumbs(self):
        """Update the breadcrumb labels to show the current path in each dictionary."""
        self.breadcrumb1_label.setText(f'Path: /{" / ".join(self.breadcrumb1)}')
        self.breadcrumb2_label.setText(f'Path: /{" / ".join(self.breadcrumb2)}')

    def go_back_left(self):
        """Go back to the previous dictionary in the left list."""
        if self.active_history1:
            self.current_dict1, last_selected_key = self.active_history1.pop()
            self.left_list.clear()
            self.left_list.addItems(self.current_dict1.keys())
            self.breadcrumb1.pop()  # Remove the last breadcrumb entry
            self.update_breadcrumbs()
            self.clear_values()  # Clear previous values when navigating back
            if not self.active_history1:
                self.left_back_button.setEnabled(False)

    def go_back_right(self):
        """Go back to the previous dictionary in the right list."""
        if self.active_history2:
            self.current_dict2, last_selected_key = self.active_history2.pop()
            self.right_list.clear()
            self.right_list.addItems(self.current_dict2.keys())
            self.breadcrumb2.pop()  # Remove the last breadcrumb entry
            self.update_breadcrumbs()
            self.clear_values()  # Clear previous values when navigating back
            if not self.active_history2:
                self.right_back_button.setEnabled(False)

    def on_select_passive_history_left(self, index):
        """Handle selection from passive history for the left dictionary."""
        if index < len(self.passive_history1):
            self.current_dict1, _ = self.passive_history1[index]
            self.left_list.clear()
            self.left_list.addItems(self.current_dict1.keys())
            self.clear_values()

    def on_select_passive_history_right(self, index):
        """Handle selection from passive history for the right dictionary."""
        if index < len(self.passive_history2):
            self.current_dict2, _ = self.passive_history2[index]
            self.right_list.clear()
            self.right_list.addItems(self.current_dict2.keys())
            self.clear_values()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        default="MiniGrid-Empty-Reduced-12x12-v0",
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
        #render_mode=None,
        agent_pov=args.agent_view,
        agent_view_size=args.agent_view_size,
        screen_size=args.screen_size,
    )
    
    #env = EmptyReducedEnv(render_mode="human", size =16)
    # TODO: check if this can be removed
    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, args.tile_size)
        env = ImgObsWrapper(env)

    agent_1 = MainAgent(env, seed=args.seed)
    agent_2 = AssistiveAgent(env=env, seed=args.seed)
    agent_1.start_simBeta(agent_2)
