
import numpy as np
from minigrid.core.actions import ActionsReduced, ActionsAgent2
from gymnasium import Env
from minigrid.core.world_object import Wall

ALL_POSSIBLE_ACTIONS_1 = (ActionsReduced.right, ActionsReduced.left, ActionsReduced.forward, ActionsReduced.backward, ActionsReduced.stay)
ALL_POSSIBLE_ACTIONS_2 = (ActionsAgent2.nothing, ActionsAgent2.take_key)


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

    def step(self, action: ActionsAgent2):
        #_ , reward, terminated, truncated, _ = self.env.step(action)
        #print(f"step={self.env.step_count}, reward={reward:.2f}")
        if action == ActionsAgent2.take_key:
            self.env.grid.set(self.env.splitIdx, self.env.doorIdx, None)
        elif action == ActionsAgent2.nothing:
            pass
        #if terminated:
        #    print("terminated!")
        #    self.reset(self.seed)
        #elif truncated:
        #    print("truncated!")
        #    self.reset(self.seed)
        #else:
        self.env.render()
    
    def best_action_value(self, J, s, p_action, Q=False):
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
            # put back
            self.env.check_move(a_2)
            v = expected_r + self.env.get_reward_2(a_2) + self.gamma*expected_v
            q[a_2]=v
            #print(a_2)
            #print(v)
            if v > best_value:
                best_value = v
                best_a = a_2
        if not Q:
            return best_a, best_value
        else:
            return q, best_a, best_value
    
    def calculate_Values(self, p_action, Q=False):
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
    
    def initialize_random_policy(self):
        policy = {}
        for s in self.env.get_states_non_terminated():
            policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS_2)
        return policy
    
    def calculate_greedy_policy(self, J, p_action):
        policy = self.initialize_random_policy()
        for s in policy.keys():
            self.env.set_state(s)
            best_a , temp = self.best_action_value(J, s, p_action=p_action)
            #print(s)
            #print(best_a)
            policy[s] = best_a
        return policy

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()
    

