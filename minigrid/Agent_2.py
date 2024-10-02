import numpy as np
import pandas as pd 
from minigrid.core.actions import ActionsReduced, ActionsAgent2, WorldSate, GoalState
from gymnasium import Env
from minigrid.core.world_object import Wall
import time
ALL_POSSIBLE_ACTIONS_1 = (ActionsReduced.right, ActionsReduced.left, ActionsReduced.forward, ActionsReduced.backward, ActionsReduced.stay)
#ALL_POSSIBLE_ACTIONS_2 = (ActionsAgent2.nothing, ActionsAgent2.take_key)
ALL_POSSIBLE_ACTIONS_2 = (ActionsAgent2.nothing, ActionsAgent2.take_key1, ActionsAgent2.take_key2)
#ALL_POSSIBLE_WOLRD = (WorldSate.open_door, WorldSate.closed_door)
ALL_POSSIBLE_WOLRD = ((WorldSate.open_door1,WorldSate.open_door2), (WorldSate.open_door1,WorldSate.closed_door2), (WorldSate.closed_door1, WorldSate.open_door2), (WorldSate.closed_door1, WorldSate.closed_door2))

ALL_POSSIBLE_GOAL = (GoalState.green_goal, GoalState.red_goal)
#ALL_POSSIBLE_GOAL = (GoalState.green_goal)
class AssistiveAgent:
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
        self.track_belief = {}
        for i in range(len(ALL_POSSIBLE_GOAL)):
            self.track_belief[ALL_POSSIBLE_GOAL[i]] = []
        discretize_num = 5
        self.discretize_belief = np.linspace(0.0, 1.0, discretize_num)

    def step(self, action: ActionsAgent2):
        #_ , reward, terminated, truncated, _ = self.env.step(action)
        #print(f"step={self.env.step_count}, reward={reward:.2f}")
        if action == ActionsAgent2.take_key:
            self.env.grid.set(self.env.splitIdx, self.env.doorIdx, None)
        elif action == ActionsAgent2.nothing:
            pass
        if action == ActionsAgent2.take_key1:
            self.env.grid.set(self.env.rooms[0].doorPos[0], self.env.rooms[0].doorPos[1], None)
        elif action == ActionsAgent2.take_key2:
            self.env.grid.set(self.env.rooms[1].doorPos[0], self.env.rooms[1].doorPos[1], None)
        #if terminated:
        #    print("terminated!")
        #    self.reset(self.seed)
        #elif truncated:
        #    print("truncated!")
        #    self.reset(self.seed)
        #else:
        self.env.render()
    
    def set_discretize_num(self, discrete_num):
        self.discretize_belief = np.linspace(0.0, 1.0, discrete_num)
    
    def instantiate_policy(self, policy):
        self.computed_policy = policy 
        
    def initializeJ_Q(self, g=GoalState.green_goal):
        states = self.env.get_all_states()
        Q= {}
        J = {}
        big_change ={}
        for belief in self.discretize_belief:
            Q[belief] = {}
            J[belief] = {}
            big_change[belief] = {}
            for w in ALL_POSSIBLE_WOLRD:
                Q[belief][w] = {}
                J[belief][w] = {}
                big_change[belief][w] = 0
                for s in states:
                    self.env.set_state(s)
                    J[belief][w][s]= 0
                    Q[belief][w][s] = {}
                    for a in ALL_POSSIBLE_ACTIONS_2:
                        Q[belief][w][s][a] = 0
        return J, Q, states, big_change 
    
    
    
    '''
    def initializeJ_Q(self, g=GoalState.green_goal):
        states = self.env.get_all_states()
        # Initialize Q and J using nested dictionary comprehensions
        Q = {belief: {w: {s: {a: 0 for a in ALL_POSSIBLE_ACTIONS_2} for s in states} for w in ALL_POSSIBLE_WOLRD} for belief in self.discretize_belief}
        J = {belief: {w: {s: 0 for s in states} for w in ALL_POSSIBLE_WOLRD} for belief in self.discretize_belief}
        
        # Initialize big_change with 0 for each belief and world state
        big_change = {belief: {w: 0 for w in ALL_POSSIBLE_WOLRD} for belief in self.discretize_belief}
        
        return J, Q, states, big_change
    '''
    
    def initialize_variation(self):
        big_change = {}
        for belief in self.discretize_belief:
            big_change[belief] = {}
            for w in ALL_POSSIBLE_WOLRD:
                big_change[belief][w] = 0
        return big_change 
    
    '''
    def initialize_variation(self):
        # Use nested dictionary comprehension to initialize big_change
        big_change = {belief: {w: 0 for w in ALL_POSSIBLE_WOLRD} for belief in self.discretize_belief}
        return big_change
    '''
    '''
    def variation_superiorTothreshold(self, variation):
        breaking_flag = True
        
        for belief in self.discretize_belief:
            for w in ALL_POSSIBLE_WOLRD:
                if variation[belief][w] <= self.threshold:
                    breaking_flag = True * breaking_flag
                else:
                    breaking_flag = False * breaking_flag
        return breaking_flag
    '''
    
    def variation_superiorTothreshold(self, variation):
        for belief in self.discretize_belief:
            for w in ALL_POSSIBLE_WOLRD:
                if variation[belief][w] > self.threshold:
                    return False  # Variation exceeds threshold, immediately return False
        return True  # All variations are within threshold, return True
    
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

    def policy(self, belief, w, s):
        return self.computed_policy[belief][w][s]

    '''
    def world_dynamic_update(self, action, current_world):
        if not self.env.multiple_goal:
            # For a single-goal environment, directly map the action to the world state change.
            if action == ActionsAgent2.take_key:
                # Taking a key opens the door if it's closed, but the state remains the same if it's already open.
                return WorldSate.open_door
            else:
                # For any other action, the world remains in its current state.
                return current_world
        else:
            # For environments with multiple goals, update the state based on the specific action taken.
            if action == ActionsAgent2.take_key1:
                return (WorldSate.open_door1, current_world[1])  # Open door 1 regardless of its previous state.
            elif action == ActionsAgent2.take_key2:
                return (current_world[0], WorldSate.open_door2)  # Open door 2 regardless of its previous state.
            else:
                # For any other action, the world remains in its current state.
                return current_world
    '''
    def value_iteration(self, p_action, g=GoalState.green_goal):
        J, Q_prime, states, big_change = self.initializeJ_Q()
        states = self.env.get_states_non_terminated()
        while True:
            big_change = self.initialize_variation()
            old_J = J
            for w in ALL_POSSIBLE_WOLRD:
                # open the door in Value iteration
                self.env.open_door_manually(w)
                for s in states:
                    self.env.set_state(s)
                    temp = J[w][s][g]
                    for a_2 in ALL_POSSIBLE_ACTIONS_2:
                        #transitions = self.env.get_transition_probs(a, cost_value=1)
                        #  prepare the envrionment using the current action of agent 2
                        # if the action is take key it checks the state of the envrionment 
                        # and open the door it virtually open the door so it has to be virtually
                        # put back to the previous state od the door i
                        self.env.check_move(action=a_2, w=w)
                        next_state_reward = []
                        for a_1 in self.env.get_possible_move(s):
                            transitions = self.env.get_transition_probsA2(w=w, action=a_1, cost_value=1)
                            #print(p_action[w][s][g][a_1])
                            for (prob, r, state_prime) in transitions:
                                world_prime = self.world_dynamic_update(a_2, w)
                                       
                                reward = prob*(p_action[world_prime][s][g][a_1]*r + self.gamma* p_action[world_prime][s][g][a_1]*J[world_prime][state_prime][g])
                                next_state_reward.append(reward)
                        # put back the door
                        self.env.check_move(action=a_2, w=w)
                            
                        Q_prime[w][s][g][a_2]=((np.sum(next_state_reward))+ self.env.get_reward_2(a_2))
                        
                    J[w][s][g] = max(Q_prime[w][s][g].values())
                    big_change[w] = max(big_change[w], np.abs(temp-J[w][s][g]))
                # CLOSE the door in Value iteration
                self.env.open_door_manually(w)
                      
            if self.variation_superiorTothreshold(big_change):
                break
        return J, Q_prime
    
    def approx_prob_to_belief(self, prob):
        
        closest = min(self.discretize_belief, key=lambda y: abs(prob - y))
        return closest
    
    '''
    def expected_reward_over_goal(self, s, w, belief_state, a):
        expected = 0
        current_dist = {ALL_POSSIBLE_GOAL[0]: belief_state, ALL_POSSIBLE_GOAL[1]: 1-belief_state}
        for  i in range(len(ALL_POSSIBLE_GOAL)):
            self.env.set_env_to_goal(ALL_POSSIBLE_GOAL[i])
            r = self.env.check_move(action=a,w=w,cost_value=1)
            expected += r[1]*current_dist[ALL_POSSIBLE_GOAL[i]]
        return expected
    '''
    
    def expected_reward_over_goal(self, s, w, belief_state, p_action, a):
        # Initialize the expected reward.
        expected = 0
        
        # Iterate over the possible goals to calculate the expected reward.
        current_dist = {ALL_POSSIBLE_GOAL[0]: belief_state, ALL_POSSIBLE_GOAL[1]: 1-belief_state}
        for  i in range(len(ALL_POSSIBLE_GOAL)):
            self.env.set_env_to_goal(ALL_POSSIBLE_GOAL[i])
            r = self.env.check_move(action=a,w=w,cost_value=1)
            expected += p_action[ALL_POSSIBLE_GOAL[i]][w][s][a]*r[1]*current_dist[ALL_POSSIBLE_GOAL[i]]
        return expected
    
    
    def expected_prob_over_action(self, belief_state,  p_action, s, a, w):
        expected = 0
        current_dist = {ALL_POSSIBLE_GOAL[0]: belief_state, ALL_POSSIBLE_GOAL[1]: 1-belief_state}
        for  i in range(len(ALL_POSSIBLE_GOAL)):
            expected += p_action[ALL_POSSIBLE_GOAL[i]][w][s][a]*current_dist[ALL_POSSIBLE_GOAL[i]]
        return expected
    
    '''
    def expected_prob_over_action(self, belief_state, p_action, s, a, w):
        # Initialize the expected probability.
        expected = sum(p_action[goal][w][s][a] * (belief_state if goal == ALL_POSSIBLE_GOAL[0] else 1 - belief_state) for goal in ALL_POSSIBLE_GOAL)
        return expected
    '''

    def bellman_equation(self, action2, action1, belief, w, s, p_action, J):
        next_state_reward = []
        transitions = self.env.get_transition_probsA2(w=w, action=action1, cost_value=1)
        for (prob, r, state_prime) in transitions:
            world_prime = self.world_dynamic_update(action2, w)
            next_belief = self.belief_state_discretize(belief=belief, dist_boltzmann=p_action, w=world_prime,s=state_prime, previous_state=s)

            next_belief = self.approx_prob_to_belief(next_belief)

            reward = prob*self.expected_reward_over_goal(s=s,w=world_prime , belief_state=belief, p_action=p_action, a=action1)\
                            + self.gamma* self.expected_prob_over_action(belief_state=belief, p_action=p_action,s=s, a=action1,w=world_prime)*J[next_belief][world_prime][state_prime]
            next_state_reward.append(reward)
        return next_state_reward
    
    def bellman_equation_iterative_game(self, action2, action1, belief, w, s, p_action, J):
        next_state_reward = []
        transitions = self.env.get_transition_probsA2(w=w, action=action1, cost_value=1)
        for (prob, r, state_prime) in transitions:
            world_prime = self.world_dynamic_update(action2, w)
            next_belief = self.belief_state_discretize_iterative_game(belief=belief, dist_boltzmann=p_action, w=world_prime,s=state_prime, previous_state=s)
            #print('precomputed belief')
            #print(next_belief)
            next_belief = self.approx_prob_to_belief(next_belief)
            reward = prob*self.expected_reward_over_goal(s=s,w=world_prime , belief_state=belief, p_action=p_action, a=action1)\
                            + self.gamma* self.expected_prob_over_action(belief_state=belief, p_action=p_action,s=s, a=action1,w=world_prime)*J[next_belief][world_prime][state_prime]
            next_state_reward.append(reward)
        return next_state_reward
    '''
    def bellman_equation(self, action2, action1, belief, w, s, p_action, J):
        next_state_rewards = []
        transitions = self.env.get_transition_probsA2(w=w, action=action1, cost_value=1)
        world_prime = self.world_dynamic_update(action2, w)
        
        # Compute parts of the reward calculation that don't depend on the loop over transitions
        expected_action_prob = self.expected_prob_over_action(belief_state=belief, p_action=p_action, s=s, a=action1, w=world_prime)
        expected_goal_reward = self.expected_reward_over_goal(s=s, w=world_prime, belief_state=belief, a=action1)
        
        for (prob, r, state_prime) in transitions:
            next_belief = self.approx_prob_to_belief(
                self.belief_state_discretize(belief=belief, dist_boltzmann=p_action, w=world_prime, s=state_prime, previous_state=s)
            )
            reward = prob * (expected_action_prob * (expected_goal_reward + self.gamma * J[next_belief][world_prime][state_prime]))
            next_state_rewards.append(reward)
            
        return next_state_rewards
    '''
    def test_variation(self, var):
        for dis in self.discretize_belief:
            if var[dis] > self.threshold:
                return False
        return True
    def value_iteration_baseline(self, p_action):
        J, Q_prime, states, big_change = self.initializeJ_Q()
        number_iter = 0
        status = {}
        #for dis in self.discretize_belief:
        #    status[dis] = 0
        while True:
            big_change = self.initialize_variation()
            initial_time = time.time()
            for belief in self.discretize_belief:
                #ceci s'execute en 1.6s
                for w in ALL_POSSIBLE_WOLRD:
                    self.env.open_door_manually(w)
                    # ceci s'execute en 0.44s
                    for s in states:
                        self.env.set_state(s)
                        temp = J[belief][w][s]
                        #ceci s'execute en 0.002s
                        for a_2 in ALL_POSSIBLE_ACTIONS_2:
                            self.env.check_move(action=a_2, w=w)
                            next_state_reward = []
                            for a_1 in self.env.get_possible_move(s):
                                reward =  self.bellman_equation(a_2, a_1, belief, w, s, p_action, J)
                                next_state_reward.append(sum(reward))
                            # put back the door
                            self.env.check_move(action=a_2, w=w)
                            Q_prime[belief][w][s][a_2]=((np.sum(next_state_reward))+ self.env.get_reward_2(a_2))
                        J[belief][w][s] = max(Q_prime[belief][w][s].values())
                        big_change[belief][w] = max(big_change[belief][w], np.abs(temp-J[belief][w][s]))
                    # close the door in Value iteration
                    self.env.open_door_manually(w)
            
            value_iteration_elapsed_time = initial_time - time.time()
            print('Elpased time for value iteration with multiple goal:')
            print(value_iteration_elapsed_time)
            print(number_iter)
            #print(big_change)
            if self.variation_superiorTothreshold(big_change):
                break
            #if self.test_variation(status):
            #    break
            number_iter += 1
        
        return J, Q_prime
    
    def value_iteration_baseline_iterative_game(self, p_action):
        J, Q_prime, states, big_change = self.initializeJ_Q()
        number_iter = 0
        #for dis in self.discretize_belief:
        #    status[dis] = 0
        while True:
            big_change = self.initialize_variation()
            initial_time = time.time()
            for belief in self.discretize_belief:
                #ceci s'execute en 1.6s
                for w in ALL_POSSIBLE_WOLRD:
                    self.env.open_door_manually(w)
                    # ceci s'execute en 0.44s
                    for s in self.env.get_states_non_terminated():
                        self.env.set_state(s)
                        temp = J[belief][w][s]
                        #ceci s'execute en 0.002s
                        for a_2 in ALL_POSSIBLE_ACTIONS_2:
                            self.env.check_move(action=a_2, w=w)
                            next_state_reward = []
                            for a_1 in self.env.get_possible_move(s):
                                reward =  self.bellman_equation_iterative_game(a_2, a_1, belief, w, s, p_action, J)
                                next_state_reward.append(sum(reward))
                            # put back the door
                            self.env.check_move(action=a_2, w=w)
                            Q_prime[belief][w][s][a_2]=((np.sum(next_state_reward))+ self.env.get_reward_2(a_2))
                        J[belief][w][s] = max(Q_prime[belief][w][s].values())
                        big_change[belief][w] = max(big_change[belief][w], np.abs(temp-J[belief][w][s]))
                    # close the door in Value iteration
                    self.env.open_door_manually(w)
            
            value_iteration_elapsed_time = initial_time - time.time()
            print('Elpased time for value iteration with multiple goal:')
            print(value_iteration_elapsed_time)
            print(number_iter)
            if self.variation_superiorTothreshold(big_change):
                break
            #if self.test_variation(status):
            #    break
            number_iter += 1
        
        return J, Q_prime     
  
  
    '''
   
    def value_iteration_baseline(self, p_action):
        J, Q_prime, states, big_change = self.initializeJ_Q()
        states = self.env.get_states_non_terminated()
        number_iter = 0
        
        # Pre-compute possible moves and rewards if they don't change per state
        possible_moves = {s: self.env.get_possible_move(s) for s in states}
        reward_2_cache = {a_2: self.env.get_reward_2(a_2) for a_2 in ALL_POSSIBLE_ACTIONS_2}
        
        while True:
            big_change = self.initialize_variation()
            for belief in self.discretize_belief:
                #initial_time = time.time()
                # 1.61 s
                for w in ALL_POSSIBLE_WOLRD:
                    self.env.open_door_manually(w)
                    # boucle de 0.4s
                    for s in states:
                        self.env.set_state(s)
                        temp = J[belief][w][s]
                        # boucle de 0.0015s
                        for a_2 in ALL_POSSIBLE_ACTIONS_2:
                            self.env.check_move(action=a_2, w=w)
                            next_state_reward = []
                            for a_1 in possible_moves[s]:
                                reward = self.bellman_equation(a_2, a_1, belief, w, s, p_action, J)
                                next_state_reward.append(sum(reward))
                            self.env.check_move(action=a_2, w=w)  # Restore door state if necessary
                            Q_prime[belief][w][s][a_2] = sum(next_state_reward) + reward_2_cache[a_2]
                        J[belief][w][s] = max(Q_prime[belief][w][s].values())
                        big_change[belief][w] = max(big_change[belief][w], abs(temp - J[belief][w][s]))
                    
                    self.env.open_door_manually(w)  # this closes the door
                #value_iteration_elapsed_time = initial_time - time.time()
                #print('Elpased time for value iteration with multiple goal:')
                #print(value_iteration_elapsed_time)
            print(number_iter)
            if self.variation_superiorTothreshold(big_change):
                break
            number_iter += 1
            
        return J, Q_prime
    '''
    def belief_state_discretize(self, belief, dist_boltzmann, w, s, previous_state, action_2=None):
        # be carful of dynamic of w that needs the action of agent 2
        #PROCESS ENVIRONEMENT IF POSSIBLE
        current_dist = {ALL_POSSIBLE_GOAL[0]: belief, ALL_POSSIBLE_GOAL[1]: 1-belief}
        current_dist = self.belief_state(previous_dist_g=current_dist, dist_boltzmann=dist_boltzmann, w=w, s=s, previous_state=previous_state)
        return current_dist[ALL_POSSIBLE_GOAL[0]]
    
    def belief_state_discretize_iterative_game(self, belief, dist_boltzmann, w, s, previous_state, action_2=None):
        # be carful of dynamic of w that needs the action of agent 2
        #PROCESS ENVIRONEMENT IF POSSIBLE
        current_dist = {ALL_POSSIBLE_GOAL[0]: belief, ALL_POSSIBLE_GOAL[1]: 1-belief}
        current_dist = self.belief_state_iterative_game(previous_dist_g=current_dist, dist_boltzmann=dist_boltzmann, w=w, s=s, previous_state=previous_state)
        return current_dist[ALL_POSSIBLE_GOAL[0]]
    
    
    def belief_state(self, previous_dist_g, dist_boltzmann, w, s, previous_state, action_2=None):
        # be carful of dynamic of w that needs the action of agent 2
        #PROCESS ENVIRONEMENT IF POSSIBLE 
        current_dist = previous_dist_g
        normalizing_factor = 0
        for i in range(len(ALL_POSSIBLE_GOAL)):
            conditional_state_world = 0
            self.env.set_state(previous_state)
            for a in self.env.get_possible_move(previous_state):
                transition = self.env.get_transition_probs(a, cost_value=1)
                for (_,_,state_prime) in transition:
                    if state_prime == s:
                        conditional_state_world += dist_boltzmann[ALL_POSSIBLE_GOAL[i]][w][previous_state][a]
            current_dist[ALL_POSSIBLE_GOAL[i]] = conditional_state_world*previous_dist_g[ALL_POSSIBLE_GOAL[i]]
            normalizing_factor += current_dist[ALL_POSSIBLE_GOAL[i]]
        if normalizing_factor > 0:
            current_dist = {ALL_POSSIBLE_GOAL[i]: current_dist[ALL_POSSIBLE_GOAL[i]]/normalizing_factor for i in range(len(ALL_POSSIBLE_GOAL))}
        return current_dist
    
    def belief_state_iterative_game(self, previous_dist_g, dist_boltzmann, w, s, previous_state, action_2=None):
        # be carful of dynamic of w that needs the action of agent 2
        #PROCESS ENVIRONEMENT IF POSSIBLE 
        current_dist = previous_dist_g
        normalizing_factor = 0
        for i in range(len(ALL_POSSIBLE_GOAL)):
            conditional_state_world = 0
            self.env.set_state(previous_state)
            for a in self.env.get_possible_move(previous_state):
                transition = self.env.get_transition_probs(a, cost_value=1)
                for (_,_,state_prime) in transition:
                    if state_prime == s:
                        conditional_state_world += dist_boltzmann[ALL_POSSIBLE_GOAL[i]][previous_dist_g[ALL_POSSIBLE_GOAL[0]]][w][previous_state][a]
            current_dist[ALL_POSSIBLE_GOAL[i]] = conditional_state_world*previous_dist_g[ALL_POSSIBLE_GOAL[i]]
            normalizing_factor += current_dist[ALL_POSSIBLE_GOAL[i]]
        if normalizing_factor > 0:
            current_dist = {ALL_POSSIBLE_GOAL[i]: current_dist[ALL_POSSIBLE_GOAL[i]]/normalizing_factor for i in range(len(ALL_POSSIBLE_GOAL))}
        return current_dist
    '''
    def belief_state(self, previous_dist_g, dist_boltzmann, w, s, previous_state, action_2=None):
        # Set the environment state once if necessary
        self.env.set_state(previous_state)
    
        # Initialize a new distribution to accumulate probabilities
        new_dist = {}
        normalizing_factor = 0
    
        for goal in ALL_POSSIBLE_GOAL:
            conditional_state_world = sum(
                dist_boltzmann[goal][w][previous_state][a] for a in self.env.get_possible_move(previous_state)
                if s in {state_prime for (_, _, state_prime) in self.env.get_transition_probs(a, cost_value=1)}
            )
    
            # Update the belief for the current goal based on observed transitions
            updated_belief = conditional_state_world * previous_dist_g[goal]
            new_dist[goal] = updated_belief
            normalizing_factor += updated_belief
    
        # Normalize the new distribution
        if normalizing_factor > 0:
            new_dist = {goal: belief / normalizing_factor for goal, belief in new_dist.items()}
    
        return new_dist          
    '''
    def extract_J(self, text_File, discretize_test):
        with open(text_File, 'rb') as f: 
            data = f.read()     
        # reconstructing the data as a dictionary 
        Q2 = pd.read_csv("file.txt", delimiter="  ", header = None).to_dict()
        print(Q2)
        J, _, _, _ = self.initializeJ_Q()
        states = self.env.get_states_non_terminated() 
        J_test = {num : J for num in discretize_test}
        for num in discretize_test:
            self.set_discretize_num(discrete_num=num)
            for belief in self.discretize_belief:
                for w in ALL_POSSIBLE_WOLRD:
                    for s in states:
                        J[num][belief][w][s] = max(Q2[num][belief][w][s].values())
        
        return J_test
    
    def deduce_policy(self, J, p_action):
        policy = {}
        g= GoalState.green_goal
        for w in ALL_POSSIBLE_WOLRD:
            policy[w] = {}
            for s in self.env.get_states_non_terminated():
                policy[w][s] = {}
                policy[w][s][g] = np.random.choice(ALL_POSSIBLE_ACTIONS_2)
        
        for w in  ALL_POSSIBLE_WOLRD:
            # open the door in Value iteration
            self.env.open_door_manually(w)
            for s in self.env.get_states_non_terminated():
                self.env.set_state(s)
                
                Q_table = np.zeros(len(ALL_POSSIBLE_ACTIONS_2))
                for action in ALL_POSSIBLE_ACTIONS_2 :
                    self.env.check_move(action=action, w=w)
                    for a_1 in self.env.get_possible_move(s):
                        transitions = self.env.get_transition_probsA2(w=w, action=a_1, cost_value=1)
                        for (prob, r, state_prime) in transitions:
                            world_prime = self.world_dynamic_update(action, w)
                            Q_table[int(action)] += prob*(p_action[world_prime][s][g][a_1]*r + self.gamma* p_action[world_prime][s][g][a_1]*J[world_prime][state_prime][g])
                    # put back the door
                    self.env.check_move(action=action, w=w)
                    Q_table[int(action)] += self.env.get_reward_2(action)
                policy[w][s][g] = ActionsAgent2(np.argmax(Q_table))
            self.env.open_door_manually(w)
        return policy
    
    def find_Action(self, number):
        if number == 0:
            return ActionsAgent2.nothing
        elif number == 1:
            return ActionsAgent2.take_key1
        elif number == 2:
            return ActionsAgent2.take_key2
        
    def deduce_policy_multiple_goal(self, J, p_action):
        policy = {}
        for belief in self.discretize_belief:
            policy[belief] = {}
            for w in ALL_POSSIBLE_WOLRD:
                policy[belief][w] = {}
                for s in self.env.get_states_non_terminated(all=True):
                    policy[belief][w][s] =  np.random.choice(ALL_POSSIBLE_ACTIONS_2)
                    
        for belief in self.discretize_belief:
            for w in  ALL_POSSIBLE_WOLRD:
                # open the door in Value iteration
                self.env.open_door_manually(w)
                for s in self.env.get_states_non_terminated(all=False):
                    self.env.set_state(s) 
                    Q_table = np.zeros(len(ALL_POSSIBLE_ACTIONS_2))
                    for action in ALL_POSSIBLE_ACTIONS_2 :
                        self.env.check_move(action=action, w=w)
                        for a_1 in self.env.get_possible_move(s):
                            next_state_reward = self.bellman_equation(action2=action, action1=a_1, belief=belief, w=w, s=s, p_action=p_action, J=J)
                        Q_table[int(action)] = np.sum(next_state_reward) 
                        # put back the door
                        self.env.check_move(action=action, w=w)
                        Q_table[int(action)] += self.env.get_reward_2(action)
                    policy[belief][w][s] = self.find_Action(np.argmax(Q_table))
                self.env.open_door_manually(w)
        self.instantiate_policy(policy=policy)
        return policy
    
    def deduce_policy_iterative_game(self, J, p_action):
        policy = {}
        for belief in self.discretize_belief:
            policy[belief] = {}
            for w in ALL_POSSIBLE_WOLRD:
                policy[belief][w] = {}
                for s in self.env.get_states_non_terminated():
                    policy[belief][w][s] =  np.random.choice(ALL_POSSIBLE_ACTIONS_2)
                    
        for belief in self.discretize_belief:
            for w in  ALL_POSSIBLE_WOLRD:
                # open the door in Value iteration
                self.env.open_door_manually(w)
                for s in self.env.get_states_non_terminated():
                    self.env.set_state(s) 
                    Q_table = np.zeros(len(ALL_POSSIBLE_ACTIONS_2))
                    for action in ALL_POSSIBLE_ACTIONS_2 :
                        self.env.check_move(action=action, w=w)
                        for a_1 in self.env.get_possible_move(s):
                            next_state_reward = self.bellman_equation(action2=action, action1=a_1, belief=belief, w=w, s=s, p_action=p_action, J=J)
                        Q_table[int(action)] = np.sum(next_state_reward) 
                        # put back the door
                        self.env.check_move(action=action, w=w)
                        Q_table[int(action)] += self.env.get_reward_2(action)
                    policy[belief][w][s] = self.find_Action(np.argmax(Q_table))
                self.env.open_door_manually(w)
        return policy
    
    
    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()
    

