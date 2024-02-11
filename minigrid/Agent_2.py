
import numpy as np
from minigrid.core.actions import ActionsReduced, ActionsAgent2, WorldSate, GoalState
from gymnasium import Env
from minigrid.core.world_object import Wall

ALL_POSSIBLE_ACTIONS_1 = (ActionsReduced.right, ActionsReduced.left, ActionsReduced.forward, ActionsReduced.backward, ActionsReduced.stay)
ALL_POSSIBLE_ACTIONS_2 = (ActionsAgent2.nothing, ActionsAgent2.take_key)
#ALL_POSSIBLE_WOLRD = (WorldSate.open_door, WorldSate.closed_door)
ALL_POSSIBLE_WOLRD = ((WorldSate.open_door1,WorldSate.open_door2), (WorldSate.open_door1,WorldSate.closed_door2), (WorldSate.closed_door1, WorldSate.open_door2), (WorldSate.closed_door1, WorldSate.closed_door2))
ALL_POSSIBLE_GOAL = (GoalState.green_goal, GoalState.red_goal)

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
        self.threshold = 1e-6
        self.track_belief = {}
        for i in range(len(ALL_POSSIBLE_GOAL)):
            self.track_belief[ALL_POSSIBLE_GOAL[i]] = []

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
    
    def initializeJ_Q(self, g=GoalState.green_goal, all_goal=False):
        states = self.env.get_all_states()
        Q= {}
        J = {}
        big_change ={}
        if not all_goal:
        #for g in ALL_POSSIBLE_GOAL:
            for w in ALL_POSSIBLE_WOLRD:
                Q[w] = {}
                J[w] = {}
                big_change[w] = 0
                for s in states:
                    self.env.set_state(s)
                    J[w][s]= {}
                    Q[w][s] = {}
                    J[w][s][g] = 0
                    Q[w][s][g] = {}
                    for a in ALL_POSSIBLE_ACTIONS_2:
                        Q[w][s][g][a] = 0
        else:
            for w in ALL_POSSIBLE_WOLRD:
                Q[w] = {}
                J[w] = {}
                big_change[w] = 0
                for s in states:
                    self.env.set_state(s)
                    J[w][s]= {}
                    Q[w][s] = {}
                    for g in ALL_POSSIBLE_GOAL:
                        big_change[w][g] = 0
                        J[w][s][g] = 0
                        Q[w][s][g] = {}
                    for a in ALL_POSSIBLE_ACTIONS_2:
                        Q[w][s][g][a] = 0
        return J, Q, states, big_change
    
    def initialize_variation(self, all_goal=False):
        big_change = {}
        for w in ALL_POSSIBLE_WOLRD:
            big_change[w] = 0
            if all_goal:
                big_change[w] = {}
                for g in GoalState:
                    big_change[w][g] = 0
        return big_change 
    
    def variation_superiorTothreshold(self, variation, all_goal=False):
        breaking_flag = True
        if not all_goal:
            for w in ALL_POSSIBLE_WOLRD:
                if variation[w] <= self.threshold:
                    breaking_flag = True * breaking_flag
                else:
                    breaking_flag = False * breaking_flag
        else:
            for g in ALL_POSSIBLE_GOAL:
                for w in ALL_POSSIBLE_WOLRD:
                    if variation[w][g] <= self.threshold:
                        breaking_flag = True * breaking_flag
                    else:
                        breaking_flag = False * breaking_flag
        return breaking_flag
    
    def world_dynamic_update(self, action, current_world):
        if action == ActionsAgent2.take_key and current_world == WorldSate.closed_door :
            world_prime = WorldSate.open_door
        if action == ActionsAgent2.take_key and current_world == WorldSate.open_door :
            world_prime = WorldSate.open_door
        if action == ActionsAgent2.nothing:
            world_prime = current_world
        return world_prime
    
    def value_iteration(self, p_action, g=GoalState.green_goal):
        J, Q_prime, states, big_change = self.initializeJ_Q()
        while True:
            big_change = self.initialize_variation()
            old_J = J
            for w in ALL_POSSIBLE_WOLRD:
                # open the door in Value iteration
                self.env.open_door_manually(w)
                for s in self.env.get_states_non_terminated():
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
                        #print(a_2)
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
    
    """
    def best_action_value(self, J, s, p_action, Q=False):# NOT UPDATED WITH WORLD AND GOAL STATE
        best_a = None
        best_value = float('-inf')
        self.env.set_state(s)
        q = {a:0 for a in ALL_POSSIBLE_ACTIONS_2}
        for a_2 in ALL_POSSIBLE_ACTIONS_2:
            #transitions = self.env.get_transition_probs(a, cost_value=1)
            #  prepare the envrionment using the current action of agent 2
            # if the action is take key it checks the state of the envrionment 
            # and open the door it virtually open the door so it has to be virtually
            # put back to the previous state od the door i
            self.env.check_move(a_2)
            expected_v =  0
            expected_r = 0
            for a_1 in ALL_POSSIBLE_ACTIONS_1:
                transitions = self.env.get_transition_probs(a_1, cost_value=1)
                for (prob, r, state_prime) in transitions:
                    expected_r += prob*p_action[s][a_1]*r
                    expected_v += p_action[s][a_1]*J[state_prime]
            # put back the door
            self.env.check_move(a_2)
            v = expected_r + self.env.get_reward_2(a_2) + self.gamma*expected_v
            q[a_2]=v
            if v > best_value:
                best_value = v
                best_a = a_2
        if not Q:
            return best_a, best_value
        else:
            return q, best_a, best_value
    
    def calculate_Values(self, p_action, Q=False):# NOT UPDATED WITH WORLD AND GOAL STATE
        states = self.env.get_all_states()
        if not Q:
            J = {}
            for s in states:
                J[s] = 0
            while True:
                big_change = 0
                old_v = J.copy()       
                for s in self.env.get_states_non_terminated(): 
                    _, new_v = self.best_action_value(old_v, s, p_action=p_action)
                    J[s] = new_v
                    big_change = max(big_change, np.abs(old_v[s]-new_v))
                if big_change < self.threshold :
                    break
            return J
        else:
            Q_prime= {}
            J = {}
            for s in states:
                Q_prime[s] = {}
                J[s] = 0
                for a in ALL_POSSIBLE_ACTIONS_2:
                    Q_prime[s][a] = 0
            while True:
                big_change = 0
                old_v = J.copy()       
                for s in self.env.get_states_non_terminated(): 
                    Q_prime[s], _, new_v = self.best_action_value(old_v, s, p_action=p_action, Q=Q)
                    J[s] = new_v
                    big_change = max(big_change, np.abs(old_v[s]-new_v))
                if big_change < self.threshold :
                    break
            return Q_prime
    
    def initialize_random_policy(self):# NOT UPDATED WITH WORLD AND GOAL STATE
        policy = {}
        for s in self.env.get_states_non_terminated():
            policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS_2)
        return policy
    
    def calculate_greedy_policy(self, J, p_action):# NOT UPDATED WITH WORLD AND GOAL STATE
        policy = self.initialize_random_policy()
        for s in policy.keys():
            self.env.set_state(s)
            best_a , temp = self.best_action_value(J, s, p_action=p_action)
            policy[s] = best_a
        return policy
"""
    
    def belief_state(self,previous_dist_g, dist_boltzmann, w, s):
        # be carful of dynamic of w that needs the action of agent 2
        current_dist = np.copy(previous_dist_g)
        normalizing_factor = 0
        for i in range(len(ALL_POSSIBLE_GOAL)):
            conditional_state_world = 0
            for a in self.env.get_possible_move(s):
                conditional_state_world += dist_boltzmann[w][s][ALL_POSSIBLE_GOAL[i]][a]
            current_dist[ALL_POSSIBLE_GOAL[i]] = conditional_state_world*previous_dist_g[ALL_POSSIBLE_GOAL[i]]
        normalizing_factor += conditional_state_world*previous_dist_g[ALL_POSSIBLE_GOAL[i]]
        
        current_dist = {ALL_POSSIBLE_GOAL[i]: current_dist[ALL_POSSIBLE_GOAL[i]]/normalizing_factor for i in range(len(ALL_POSSIBLE_GOAL))}
        
        return current_dist
              
         
    
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
    
    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()
    

