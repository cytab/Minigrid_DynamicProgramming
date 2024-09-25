# Enumeration of possible actions
from __future__ import annotations
import os.path
import pomdp_py
from enum import IntEnum
import math
import random
import time
from pomdp_py import sarsop
from pomdp_py.utils import TreeDebugger
import multiprocessing
from pomdp_py.utils.interfaces.conversion import (
    to_pomdp_file,
    PolicyGraph,
    AlphaVectorPolicy,
    parse_pomdp_solve_output,
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
Number_room = 1

class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    # Pick up an object
    pickup = 3
    # Drop an object
    drop = 4
    # Toggle/activate an object
    toggle = 5

    # Done completing task
    done = 6

class ActionsReduced(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    backward = 3
    stay = 4

class ActionsAgent2(IntEnum):
    nothing = 0
    take_key1 = 1
    take_key2 = 2
    take_key = 3

class WorldSate(IntEnum):
    open_door = 0
    closed_door = 1
    open_door1 = 2
    closed_door1 = 3
    open_door2 = 4
    closed_door2 = 5
              
class GoalState(IntEnum):
    green_goal = 0
    red_goal = 1

PROB_SIM_GREEN_GOAL = 0.5
ALL_POSSIBLE_ACTIONS = (ActionsReduced.right, ActionsReduced.left, ActionsReduced.forward, ActionsReduced.backward, ActionsReduced.stay)
#ALL_POSSIBLE_WOLRD = (WorldSate.open_door, WorldSate.closed_door)
#RED-eta9-beliefg1-(-10_999)-
ALL_POSSIBLE_WOLRD = ((WorldSate.open_door1,WorldSate.open_door2), (WorldSate.open_door1,WorldSate.closed_door2), (WorldSate.closed_door1, WorldSate.open_door2), (WorldSate.closed_door1, WorldSate.closed_door2))
ALL_POSSIBLE_GOAL = (GoalState.green_goal,GoalState.red_goal)
#ALL_POSSIBLE_GOAL = GoalState.green_goal
class ActionPOMDP(pomdp_py.Action):
    """Mos action; Simple named action."""

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, ActionPOMDP):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Action(%s)" % self.name
    
STEP_SIZE = 1
class MotionAction(ActionPOMDP):
    # scheme 1 (vx,vy,th)
    SCHEME_XYTH = "xyth"
    EAST = (STEP_SIZE, 0, 0)  # x is horizontal; x+ is right. y is vertical; y+ is down.
    WEST = (-STEP_SIZE, 0, math.pi)
    NORTH = (0, -STEP_SIZE, 3 * math.pi / 2)
    SOUTH = (0, STEP_SIZE, math.pi / 2)

    # scheme 2 (vt, vw) translational, rotational velocities.
    SCHEME_VW = "vw"
    FORWARD = (STEP_SIZE, 0)
    BACKWARD = (-STEP_SIZE, 0)
    LEFT = (0, -math.pi / 4)  # left 45 deg
    RIGHT = (0, math.pi / 4)  # right 45 deg

    # scheme 3 (vx,vy)
    SCHEME_XY = "xy"
    EAST2D = (STEP_SIZE, 0)  # x is horizontal; x+ is right. y is vertical; y+ is down.
    WEST2D = (-STEP_SIZE, 0)
    NORTH2D = (0, -STEP_SIZE)
    SOUTH2D = (0, STEP_SIZE)
    HATL2D = (0, 0)

    SCHEMES = {"xyth", "xy", "vw"}
    MOTION_SCHEME = "xy"  # can be either xy or vw
    def __init__(self, motion, scheme=MOTION_SCHEME, distance_cost=1, motion_name=None):
        """
        motion (tuple): a tuple of floats that describes the motion;
        scheme (str): description of the motion scheme; Either
                      "xy" or "vw"
        """
        if scheme not in MotionAction.SCHEMES:
            raise ValueError("Invalid motion scheme %s" % scheme)

        if scheme == MotionAction.SCHEME_XYTH:
            if motion not in {
                MotionAction.EAST,
                MotionAction.WEST,
                MotionAction.NORTH,
                MotionAction.SOUTH,
            }:
                raise ValueError("Invalid move motion %s" % str(motion))
        elif scheme == MotionAction.SCHEME_VW:
            if motion not in {
                MotionAction.FORWARD,
                MotionAction.BACKWARD,
                MotionAction.LEFT,
                MotionAction.RIGHT,
            }:
                raise ValueError("Invalid move motion %s" % str(motion))
        elif scheme == MotionAction.SCHEME_XY:
            if motion not in {
                MotionAction.EAST2D,
                MotionAction.WEST2D,
                MotionAction.NORTH2D,
                MotionAction.SOUTH2D,
                MotionAction.HATL2D,
            }:
                raise ValueError("Invalid move motion %s" % str(motion))

        self.motion = motion
        self.scheme = scheme
        self.distance_cost = distance_cost
        if motion_name is None:
            motion_name = str(motion)
        super().__init__("%s" % ( motion_name))
    
MoveEast2D = MotionAction(
MotionAction.EAST2D, scheme=MotionAction.SCHEME_XY, motion_name="East2D"
)
MoveWest2D = MotionAction(
    MotionAction.WEST2D, scheme=MotionAction.SCHEME_XY, motion_name="West2D"
)
MoveNorth2D = MotionAction(
    MotionAction.NORTH2D, scheme=MotionAction.SCHEME_XY, motion_name="North2D"
)
MoveSouth2D = MotionAction(
    MotionAction.SOUTH2D, scheme=MotionAction.SCHEME_XY, motion_name="South2D"
)
MoveHalt2D = MotionAction(
    MotionAction.HATL2D, scheme=MotionAction.SCHEME_XY, motion_name="HALT2D"
)

ALL_MOTION_ACTIONS = [MoveEast2D, MoveWest2D, MoveNorth2D, MoveSouth2D, MoveHalt2D]

DoNothing = ActionPOMDP(name='DoNothing')
Take_Key_1 = ActionPOMDP(name='TakeKey1')
Take_Key_2 = ActionPOMDP(name='TakeKey2')
Take_Key = ActionPOMDP(name='Take_key')

def WrappedAction2Motion(mot):
    if mot == ActionsReduced.forward:
        return MoveNorth2D
    elif mot == ActionsReduced.backward:
        return MoveSouth2D
    elif mot == ActionsReduced.left:
        return MoveWest2D
    elif mot == ActionsReduced.right:
        return MoveEast2D
    elif mot == ActionsReduced.stay:
        return MoveHalt2D
    
def WrappedMotion2Action(action):
    if action == MoveNorth2D:
        return ActionsReduced.forward
    elif action == MoveSouth2D:
        return ActionsReduced.backward 
    elif action == MoveWest2D :
        return ActionsReduced.left 
    elif action == MoveEast2D:
        return ActionsReduced.right 
    elif action == MoveHalt2D:
        return ActionsReduced.stay 

def WrappedAction2RobotAction(action):
    if action == DoNothing:
        return ActionsAgent2.nothing
    elif action == Take_Key_1:
        return ActionsAgent2.take_key1
    elif action == Take_Key_2:
        return ActionsAgent2.take_key2
    elif action == Take_Key:
        return ActionsAgent2.take_key
    
def WrappedRobotAction2Action(robot_action):
    if robot_action == ActionsAgent2.nothing:
        return DoNothing
    elif robot_action == ActionsAgent2.take_key1:
        return Take_Key_1
    elif robot_action == ActionsAgent2.take_key2:
        return Take_Key_2
    elif robot_action == ActionsAgent2.take_key:
        return Take_Key
    

class StatePOMDP(pomdp_py.State):
    def __init__(self, world, world2=None, pose=None, goal=None):
        
        self.p = pose
        self.world1 = world
        self.world2 = world2
        self.goal = goal
        #self.name = str(world) + "-" + str(world2) + "-" +  str(pose[0]) + '-' + str(pose[1])  + "-" + str(goal)
        self.name = "case" + str(pose[0]) + 'e' + str(pose[1])
        if self.goal == GoalState.red_goal:
            self.name += 'Goal_Red'
        elif self.goal == GoalState.green_goal:
            self.name += 'Goal_Green'
        if self.world1 == WorldSate.closed_door1:
            self.name += 'W1_Close'
        elif self.world1 == WorldSate.open_door1:
            self.name += 'W1_Open'
        elif self.world1 == WorldSate.open_door:
            self.name += 'W1_Open'
        elif self.world1 == WorldSate.closed_door:
            self.name += 'W1_Closed'
        else:
            self.name = 'W1_None'
        if self.world2 == WorldSate.closed_door2:
            self.name += 'W2_Close'
        elif self.world2 == WorldSate.open_door2 :
            self.name += 'W2_Open'
        else:
            self.name += 'W2_None' 
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, StatePOMDP):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "(%s)" % self.name
    
    def set_goal(self, goal):
        self.goal=goal
    @property
    def pose(self):
        return self.pose

    @property
    def world(self):
        return (self.world1, self.world2)
    
    def get_color(self):
        if self.goal == GoalState.green_goal:
            return 'green'
        elif self.goal == GoalState.red_goal:
            return 'red'
        
    def get_other_color(self):
        if self.goal == GoalState.green_goal:
            return 'red'
        elif self.goal == GoalState.red_goal:
            return 'green'
    
# Human Definition problem -------------------------------------------------------------------- #########################

class HumanTransitionModel(pomdp_py.TransitionModel):
    """We assume that the robot control is perfect and transitions are deterministic."""

    def __init__(self, dim, env, epsilon=1e-5, multiple_goal=True):
        """
        dim (tuple): a tuple (width, length) for the dimension of the world
        """
        # this is used to determine objects found for FindAction
        self.env = env
        self._dim = dim
        self._epsilon = epsilon
        self.multiple_goal = multiple_goal

    #@classmethod
    def if_move_by(self, state, action):
        """Defines the dynamics of robot motion;
        dim (tuple): the width, length of the search world."""
        self.env.set_state(state.p)
        if self.multiple_goal:
            self.env.open_door_manually((state.world1,state.world2))
            self.env.set_env_to_goal(state.goal)
            
        else:
            self.env.open_door_manually(state.world1)
        if not isinstance(action, MotionAction):
            raise ValueError("Cannot move robot with %s action" % str(type(action)))

        human_pose = state.p
        rx, ry = human_pose
        flag, s_prime = self.env.state_dynamic_human(previous_state=human_pose, action_human=WrappedMotion2Action(action))
        #state_world = self.env.get_world_state()
        if self.multiple_goal:
            self.env.open_door_manually((state.world1,state.world2))
        else:
            self.env.open_door_manually(state.world1)
        state_world = state.world
        return s_prime, state_world

    def probability(self, next_robot_state, state, action):
        self.env.set_state(state.p)
        self.env.set_env_to_goal(state.goal)
#        if WrappedMotion2Action(action) in self.env.get_possible_move(state.p):
        if next_robot_state.p == self.argmax(state, action).p:
            return 1.0 
        else:
            return 0.0

    def argmax(self, state, action):
        #import copy
        """Returns the most likely next robot_state"""
        if isinstance(state, StatePOMDP):
            human_state = state
        p, world = self.if_move_by(state, action)
        if self.multiple_goal:
            next_human_state = StatePOMDP(world=world[0], world2=world[1], goal=state.goal, pose=p)
            self.env.set_env_to_goal(state.goal)
        else:
            next_human_state = StatePOMDP(world=world[0], goal=state.goal, pose=p)
        
        return next_human_state

    def sample(self, state, action):
        """Returns next_robot_state"""
        self.env.set_env_to_goal(state.goal)
        return self.argmax(state, action)
    
    def get_all_states(self):
        S = []
        if self.multiple_goal:
            for  pose in self.env.get_all_states():
                for world in ALL_POSSIBLE_WOLRD:
                    for goal in ALL_POSSIBLE_GOAL:
                        S.append(StatePOMDP(world=world[0], world2=world[1], pose=pose, goal=goal))
        else:
            for  pose in self.env.get_all_states():
                for world in ALL_POSSIBLE_WOLRD:
                        S.append(StatePOMDP(world=world, pose=pose, goal=ALL_POSSIBLE_GOAL))
        return S

class HumanObservationPOMDP(pomdp_py.Observation):
    def __init__(self, world, world2, pose, goal):
        self.p = pose
        self.world1 = world
        self.world2 = world2
        self.goal = goal
        #self.name = str(world) + "-" + str(world2) +  "-"  + str(pose[0]) + '-' + str(pose[1])  + "-" + str(goal)
        self.name = "case" + str(pose[0]) + 'e' + str(pose[1]) 
        if self.goal == GoalState.red_goal:
            self.name += 'Goal_Red'
        elif self.goal == GoalState.green_goal:
            self.name += 'Goal_Green'
        if self.world1 == WorldSate.closed_door1:
            self.name += 'W1_Close'
        elif self.world1 == WorldSate.open_door1:
            self.name += 'W1_Open'
        if self.world2 == WorldSate.closed_door2:
            self.name += 'W2_Close'
        elif self.world2 == WorldSate.open_door2 :
            self.name += 'W2_Open'
        
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, HumanObservationPOMDP):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "(%s)" % self.name
    @property
    def pose(self):
        return self.p

    @property
    def world(self):
        return (self.world1, self.world2)

class HumanObservationModel(pomdp_py.ObservationModel):
    def __init__(self, env, dim,epsilon=1e-5):
        self.env = env
        self.dim = dim
        self.epsilon = epsilon
        
    def probability(self, observation, next_state, action, **kwargs):
        """
        Returns the probability of Pr (observation | next_state, action).

        Args:
            observation (ObjectObservation)
            next_state (State)
            action (Action)
        """
        if observation.world1 == next_state.world1 and observation.world2 == next_state.world2 and observation.p == next_state.p and observation.goal == next_state.goal:
            prob = 1.0
        else:
            prob = 0.0
        return prob

    def sample(self, next_state, action):
        """Returns observation"""

        return HumanObservationPOMDP(world=next_state.world1, world2=next_state.world2, pose=next_state.p, goal=next_state.goal)

    def argmax(self, next_state, action):
        return next_state
    
    def get_all_observations(self):
        S = []
        for  pose in self.env.get_all_states():
            #world = self.env.get_world_state()
            for world in ALL_POSSIBLE_WOLRD:
                for goal in ALL_POSSIBLE_GOAL:
                    S.append(StatePOMDP(world=world[0], world2=world[1], pose=pose, goal=goal))
        return S

class HumanRewardModel(pomdp_py.RewardModel):
    def __init__(self, env, multiple_goal=True):
        self.env = env
        self.multiple_goal = multiple_goal
    def _reward_func(self, state, action):
        self.env.set_state(state.p)
        if self.multiple_goal:
            self.env.open_door_manually((state.world1,state.world2))
            self.env.set_env_to_goal(state.goal)
            
        else:
            self.env.open_door_manually(state.world1)
        if isinstance(action, MotionAction):
            reward = 0
            next_state, reward = self.env.check_move(action=WrappedMotion2Action(action), w=None, cost_value=1)
        if self.multiple_goal:
            self.env.open_door_manually((state.world1,state.world2))
        else:
            self.env.open_door_manually(state.world1)
        return reward

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)

class PolicyModel(pomdp_py.RolloutPolicy):
    
    """Simple policy model. All actions are possible at any state."""

    def __init__(self, env, grid_map=None):
        """FindAction can only be taken after LookAction"""
        self._grid_map = grid_map
        self.env = env

    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]

    def probability(self, action, state, **kwargs):
        raise NotImplementedError

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplementedError

    def get_all_actions(self, state=None, history=None):
        """note: find can only happen after look."""
        
        #if state is None:
        return ALL_MOTION_ACTIONS
  
        
    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

class HumanAgent(pomdp_py.Agent):
    def __init__(self, env, init_human_state, dim, epsilon=1e-5, grid_map=None, multiple_goal=True):
        human_transition = HumanTransitionModel(dim, env, epsilon=epsilon, multiple_goal=multiple_goal)
        human_reward = HumanRewardModel(env, multiple_goal=multiple_goal)
        Human_observation = HumanObservationModel(env,dim, epsilon)
        human_policy = PolicyModel(env)
        init_belief = pomdp_py.Histogram( {s: 0.0 for s in human_transition.get_all_states()})
        init_belief[init_human_state] = 1.0
        super().__init__(
            init_belief,
            human_policy,
            transition_model=human_transition,
            observation_model=Human_observation,
            reward_model=human_reward,
        )
    def clear_history(self):
        """Custum function; clear history"""
        self._history = None

class Hproblem(pomdp_py.POMDP):
    def __init__(self, word1, world2, pose, goal, env, dim, epsilon=1e-5, grid_map=None, multiple_goal=True):
        if multiple_goal:
            init_human_state = StatePOMDP(world=word1, world2=world2, pose=pose, goal=goal)
        else:
            init_human_state = StatePOMDP(world=word1, pose=pose, goal=goal)
        
        human_agent = HumanAgent(env, init_human_state, dim, epsilon=1e-5, grid_map=None, multiple_goal=multiple_goal)
        grid_env = GridEnvironment(env=env, init_true_state=init_human_state, dim=dim, epsilon=epsilon, multiple_goal=multiple_goal)
        super().__init__(
            human_agent,
            grid_env,
            name="GRID(%d,%d)" % (env.size, env.size),
        )

class GridEnvironment(pomdp_py.Environment):
    def __init__(self, env, init_true_state, dim, epsilon, multiple_goal=True):
        self.env = env
        transition_model = HumanTransitionModel(dim, env, epsilon=epsilon, multiple_goal=multiple_goal)
        reward_model = HumanRewardModel(env, multiple_goal=multiple_goal)
        super().__init__(init_true_state, transition_model, reward_model)
    
    def state_transition(self,action, execute=True):
        next_state = self.transition_model.sample(self.state, action)
        reward = self.reward_model.sample(self.state, action, next_state)
        terminated = False
        if execute:
            self.apply_transition(next_state)
            _ , _, terminated, truncated, _ = self.env.step(WrappedMotion2Action(action))
            return next_state, reward, terminated
        else:
            return next_state, reward, terminated
# Robot Definitin problem --------------------------------------------------------------------- ##########################

class RobotObservationPOMDP(pomdp_py.Observation):
    def __init__(self, world, world2=None, pose=None, multiple_goal=True):
        self.p = pose
        self.world1 = world
        self.world2 = world2
        self.name = "case" + str(pose[0]) + 'e' + str(pose[1]) 
        if self.world1 == WorldSate.closed_door1:
            self.name += 'W1_Close'
        elif self.world1 == WorldSate.open_door1:
            self.name += 'W1_Open'
        elif self.world1 == WorldSate.open_door:
            self.name += 'W1_Open'
        elif self.world1 == WorldSate.closed_door:
            self.name += 'W1_Closed'
        else:
            self.name = 'W1_None'
        if self.world2 == WorldSate.closed_door2:
            self.name += 'W2_Close'
        elif self.world2 == WorldSate.open_door2 :
            self.name += 'W2_Open'
        else:
            self.name += 'W2_None'  
        
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, RobotObservationPOMDP):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "(%s)" % self.name
    @property
    def pose(self):
        return self.p

    @property
    def world(self):
        return (self.world1, self.world2)
      
class RobotTransitionModel(pomdp_py.TransitionModel):
    """We assume that the robot control is perfect and transitions are deterministic."""

    def __init__(self, dim, env, probability, epsilon=1e-5, multiple_goal=True):
        """
        dim (tuple): a tuple (width, length) for the dimension of the world
        """
        # this is used to determine objects found for FindAction
        self.env = env
        self._dim = dim
        self._epsilon = epsilon
        self.prob = probability
        self.multiple_goal = multiple_goal

    #@classmethod
    def if_move_by(self, state, action):
        """Defines the dynamics of robot motion;
        dim (tuple): the width, length of the search world."""
        self.env.set_state(state.p)
        #self.env.open_door_manually((state.world1,state.world2))
        if self.multiple_goal:
            self.env.open_door_manually((state.world1,state.world2))
            self.env.set_env_to_goal(state.goal)
            
            
        else:
            self.env.open_door_manually(state.world1)
        if not isinstance(action, MotionAction):
            raise ValueError("Cannot move robot with %s action" % str(type(action)))

        human_pose = state.p
        rx, ry = human_pose
        flag, s_prime = self.env.state_dynamic_human(previous_state=human_pose, action_human=WrappedMotion2Action(action))
        if self.multiple_goal:
            self.env.open_door_manually((state.world1,state.world2))
        else:
            self.env.open_door_manually(state.world1)
        state_world = state.world

        return s_prime, state_world

    def probability(self, next_robot_state, state, action):
        prob = 0
        self.env.set_state(state.p)
        if self.multiple_goal:
            self.env.open_door_manually((state.world1,state.world2))
            self.env.check_move(action=WrappedAction2RobotAction(action), w=state.world)
            self.env.set_env_to_goal(state.goal)
            
        else:
            self.env.open_door_manually(state.world1)
            self.env.check_move(action=WrappedAction2RobotAction(action), w=state.world1)
        
        #if WrappedMotion2Action(action) in self.env.get_possible_move(state.p):
        for a_1 in self.env.get_possible_move(state.p):
            intermediate_state = self.argmax(state, action) # only change the world becasue it is the action of the robot
            finale_state = self.human_transition(intermediate_state,WrappedAction2Motion(a_1))
            
            if self.multiple_goal:
                world_p = self.env.world_dynamic_update(action, current_world=state.world)
                if next_robot_state.p == finale_state.p and next_robot_state.world1 == finale_state.world1 and next_robot_state.world2 == finale_state.world2 and next_robot_state.goal == finale_state.goal:
                    prob += self.prob[next_robot_state.goal][intermediate_state.world][intermediate_state.p][a_1]
                else:
                    prob += self._epsilon
            else:
                world_p = self.env.world_dynamic_update(action, current_world=state.world1)
                if next_robot_state.p == finale_state.p and next_robot_state.world1 == finale_state.world1 and next_robot_state.goal == finale_state.goal:
                    prob += self.prob[intermediate_state.world1][intermediate_state.p][next_robot_state.goal][a_1]
                else:
                    prob += self._epsilon
                
        if self.multiple_goal:
            self.env.check_move(action=WrappedAction2RobotAction(action), w=state.world)
            self.env.open_door_manually((state.world1,state.world2))
            
        else:
            self.env.check_move(action=WrappedAction2RobotAction(action), w=state.world1)
            self.env.open_door_manually(state.world1)
        
        return prob

    def human_transition(self, state, action):
        p, world = self.if_move_by(state, action)
        if self.multiple_goal:
            next_human_state = StatePOMDP(world=world[0], world2=world[1], goal=state.goal, pose=p)
        else:
            next_human_state = StatePOMDP(world=world[0], goal=state.goal, pose=p)
        
        return next_human_state
        
    def argmax(self, state, action):
        """Returns the most likely next robot_state"""
        
        if self.multiple_goal:
            self.env.set_env_to_goal(state.goal)
            world = self.env.world_dynamic_update(WrappedAction2RobotAction(action), current_world =state.world)
            next_human_state = StatePOMDP(world=world[0], world2=world[1], goal=state.goal, pose=state.p)
        else:
            world = self.env.world_dynamic_update(WrappedAction2RobotAction(action), current_world =state.world1)
            next_human_state = StatePOMDP(world=world, goal=state.goal, pose=state.p)
        return next_human_state

    def sample(self, state, action):
        """Returns next_robot_state"""
        return self.argmax(state, action)
    
    def get_all_states(self):
        S = []
        if self.multiple_goal:
            for  pose in self.env.get_all_states():
                for world in ALL_POSSIBLE_WOLRD:
                    for goal in ALL_POSSIBLE_GOAL:
                        S.append(StatePOMDP(world=world[0], world2=world[1], pose=pose, goal=goal))
        else:
            for  pose in self.env.get_all_states():
                for world in ALL_POSSIBLE_WOLRD:
                        S.append(StatePOMDP(world=world, pose=pose, goal=ALL_POSSIBLE_GOAL))
        return S
    
# Reward Model
class RobotRewardModel(pomdp_py.RewardModel):
    def __init__(self, env, probability, multiple_goal=True):
        self.env = env
        self.probability = probability
        self.multiple_goal = multiple_goal

    def _reward_func(self, state, action):
        reward = 0
        self.env.set_state(state.p)
        ac = WrappedAction2RobotAction(action)
        if self.multiple_goal:
            self.env.open_door_manually((state.world1,state.world2))
            self.env.check_move(action=WrappedAction2RobotAction(action), w=state.world)
            world_prime = self.env.world_dynamic_update(action=ac, current_world=(state.world1, state.world2))
            self.env.set_env_to_goal(state.goal)
        else:
            self.env.open_door_manually(state.world1)
            self.env.check_move(action=WrappedAction2RobotAction(action), w=state.world1)
            world_prime = self.env.world_dynamic_update(action=ac, current_world=state.world1)
        
    
        for a_1 in ALL_POSSIBLE_ACTIONS:
                next_state, reward_t = self.env.check_move(action=a_1, w=world_prime, cost_value=1)
                if self.multiple_goal:
                    reward +=  self.probability[state.goal][world_prime][state.p][a_1]*reward_t
                else:
                    reward +=  self.probability[world_prime][state.p][state.goal][a_1]*reward_t
                    
                
        if self.multiple_goal:
            self.env.check_move(action=WrappedAction2RobotAction(action), w=state.world)
            self.env.open_door_manually((state.world1,state.world2))
            
        else:
            self.env.check_move(action=WrappedAction2RobotAction(action), w=state.world1)
            self.env.open_door_manually(state.world1)
        reward += self.env.get_reward_2(WrappedAction2RobotAction(action))
        return reward

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)   

class RobotObservationModel(pomdp_py.ObservationModel):
    def __init__(self, env, dim, epsilon=1e-5, multiple_goal=True):
        self.env = env
        self.dim = dim
        self.epsilon = epsilon
        self.multiple_goal = multiple_goal
        
    def probability(self, observation, next_state, action, **kwargs):
        """
        Returns the probability of Pr (observation | next_state, action).

        Args:
            observation (ObjectObservation)
            next_state (State)
            action (Action)
        """
        if observation.world1 == next_state.world1 and observation.world2 == next_state.world2 and observation.p == next_state.p:
            prob = 1.0 - self.epsilon
        else:
            prob = self.epsilon
        return prob

    def sample(self, next_state, action):
        """Returns observation"""
        if self.multiple_goal:
            return RobotObservationPOMDP(world=next_state.world1, world2=next_state.world2, pose=next_state.p)
        else:
            return RobotObservationPOMDP(world=next_state.world1, pose=next_state.p)

    def argmax(self, next_state, action):
        return next_state
    
    def get_all_observations(self):
        S = []
        if self.multiple_goal:
            for  pose in self.env.get_all_states():
                for world in ALL_POSSIBLE_WOLRD:
                #    for goal in ALL_POSSIBLE_GOAL:
                        S.append(RobotObservationPOMDP(world=world[0], world2=world[1], pose=pose))
        else:
            for  pose in self.env.get_all_states():
                for world in ALL_POSSIBLE_WOLRD:
                #    for goal in ALL_POSSIBLE_GOAL:
                        S.append(RobotObservationPOMDP(world=world, pose=pose))
        return S

class RobotPolicyModel(pomdp_py.RolloutPolicy):
    
    """Simple policy model. All actions are possible at any state."""

    def __init__(self, env, grid_map=None, multiple_goal=True):
        """FindAction can only be taken after LookAction"""
        self._grid_map = grid_map
        self.env = env
        self.multiple_goal = multiple_goal

    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]

    def probability(self, action, state, **kwargs):
        raise NotImplementedError

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplementedError

    def get_all_actions(self, state=None, history=None):
        """note: find can only happen after look."""
        
        #if state is None:
        if self.multiple_goal:
            return [DoNothing, Take_Key_1, Take_Key_2]
        else:
            return [DoNothing, Take_Key]
        
    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

class RobotGridEnvironment(pomdp_py.Environment):
    def __init__(self, env, init_true_state, dim, epsilon, human_probability, multiple_goal=True):
        self.env = env
        transition_model = RobotTransitionModel(dim, env, epsilon=epsilon, probability=human_probability, multiple_goal=multiple_goal)
        reward_model = RobotRewardModel(env, probability=human_probability, multiple_goal=multiple_goal)
        self.human_transition_model = HumanTransitionModel(dim, env, epsilon=epsilon, multiple_goal=multiple_goal)
        self.human_reward_model = HumanRewardModel(env, multiple_goal=multiple_goal)
        self.multiple_goal=multiple_goal
        super().__init__(init_true_state, transition_model, reward_model)
    
    def state_transition(self,action, execute=True):
        next_state = self.transition_model.sample(self.state, action)
        reward = self.reward_model.sample(self.state, action, next_state)
        terminated = False
        if execute:
            self.apply_transition(next_state)
            self.robot_step(WrappedAction2RobotAction(action))
            return next_state, reward, terminated
        else:
            return next_state, reward, terminated
    
    def state_transition_human(self,action, execute=True):
        next_state = self.human_transition_model.sample(self.state, action)
        reward = self.human_reward_model.sample(self.state, action, next_state)
        terminated = False
        if execute:
            self.apply_transition(next_state)
            _ , _, terminated, truncated, _ = self.env.step(WrappedMotion2Action(action))
            return next_state, reward, terminated
        else:
            return next_state, reward, terminated
    
    def robot_step(self, action):
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
        self.env.render()
           
class RobotAgent(pomdp_py.Agent):
    def __init__(self, env, init_robot_state, initial_prob, dim, human_probability, epsilon=1e-5, multiple_goal=True):
        robot_transition = RobotTransitionModel(dim, env, epsilon=epsilon, probability=human_probability, multiple_goal=multiple_goal)
        robot_reward = RobotRewardModel(env, probability=human_probability, multiple_goal=multiple_goal)
        robot_observation = RobotObservationModel(env,dim, epsilon, multiple_goal=multiple_goal)
        robot_policy = RobotPolicyModel(env, multiple_goal=multiple_goal)
        init_belief = pomdp_py.Histogram( {s: 0.0 for s in robot_transition.get_all_states()})
        rob = StatePOMDP(world=init_robot_state.world1, world2=init_robot_state.world2, pose=init_robot_state.p, goal=GoalState.green_goal)
        init_belief[rob] = initial_prob
        rob = StatePOMDP(world=init_robot_state.world1, world2=init_robot_state.world2, pose=init_robot_state.p, goal=GoalState.red_goal)
        #init_robot_state.set_goal(goal=GoalState.red_goal)
        init_belief[rob] = 1-initial_prob
        super().__init__(
            init_belief,
            robot_policy,
            transition_model=robot_transition,
            observation_model=robot_observation,
            reward_model=robot_reward,
        )
    def clear_history(self):
        """Custum function; clear history"""
        self._history = None
                
class Robotproblem(pomdp_py.POMDP):
    def __init__(self, word1, world2, pose, goal, env, dim, human_probability, epsilon=1e-9, initial_prob=0.5, multiple_goal=True):
        if multiple_goal:
            init_human_state = StatePOMDP(world=word1, world2=world2, pose=pose, goal=goal)
        else:
            init_human_state = StatePOMDP(world=word1, pose=pose, goal=goal)
        
        robot_agent = RobotAgent(env=env, init_robot_state=init_human_state, dim=dim, epsilon=1e-9, initial_prob=initial_prob, human_probability=human_probability, multiple_goal=multiple_goal)
        grid_env = RobotGridEnvironment(env=env, init_true_state=init_human_state, dim=dim, epsilon=epsilon, human_probability=human_probability, multiple_goal=multiple_goal)
        super().__init__(
            robot_agent,
            grid_env,
            name="GRID(%d,%d)" % (env.size, env.size),
        )       
         
'''        
def belief_update(agent, real_action, real_observation, next_human_state, planner):
    """Updates the agent's belief; The belief update may happen
    through planner update (e.g. when planner is POMCP)."""
    # Updates the planner; In case of POMCP, agent's belief is also updated.
    #planner.update(agent, real_action, real_observation)

    # Update agent's belief, when planner is not POMCP
    if not isinstance(planner, pomdp_py.POMCP):
        # Update belief for every object
            if isinstance(belief_obj, pomdp_py.Histogram):
                    # Assuming the agent can observe its own state:
                    new_belief = pomdp_py.Histogram({next_human_state: 1.0})
                
            else:
                raise ValueError(
                    "Unexpected program state."
                    "Are you using the appropriate belief representation?"
                )

            #agent.cur_belief.set_object_belief(new_belief)
'''            
def run_sarsop(args):
        agent, pomdpsol_path, discount_factor, timeout, memory, precision, remove_generated_files, pomdp_name = args
        planner = sarsop(agent, pomdpsol_path, discount_factor=discount_factor,
                        timeout=timeout, memory=memory, precision=precision,
                        remove_generated_files=remove_generated_files, pomdp_name=pomdp_name)
        return planner

def generate_action(state, worldState, goal, dist, multiple_goal=True):
    if multiple_goal:
        possible_action = [a for a in dist[goal][worldState][state].keys()]
        prob = [dist[goal][worldState][state][a] for a in dist[goal][worldState][state].keys()]
        generated_action = np.random.choice(possible_action, p=prob)
    else:
        possible_action = [a for a in dist[worldState][state][goal].keys()]
        prob = [dist[worldState][state][goal][a] for a in dist[worldState][state][goal].keys()]
        generated_action = np.random.choice(possible_action, p=prob)
    return generated_action


plt.style.use('fivethirtyeight')
step = []
belief_State_Tracker = {ALL_POSSIBLE_GOAL[i]: [] for i in range(len(ALL_POSSIBLE_GOAL))}
#belief_State_Tracker = {ALL_POSSIBLE_GOAL: []}

def animate(state):
    plt.cla()  # Clear the current axes
    
    # Plot the belief states based on the goal
    if state.goal == GoalState.green_goal:
        plt.plot(step, belief_State_Tracker[ALL_POSSIBLE_GOAL[0]], label='Green goal', color=state.get_color())
        plt.plot(step, belief_State_Tracker[ALL_POSSIBLE_GOAL[1]], label='Red goal', color=state.get_other_color())
    elif state.goal == GoalState.red_goal:
        plt.plot(step, belief_State_Tracker[ALL_POSSIBLE_GOAL[0]], label='Green goal', color=state.get_other_color())
        plt.plot(step, belief_State_Tracker[ALL_POSSIBLE_GOAL[1]], label='Red goal', color=state.get_color())
    
    plt.legend(loc='upper left')  # Add a legend
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Force x-axis to show only integer values
    
    plt.tight_layout()  # Adjust subplots to fit in figure area
    plt.draw()  # Redraw the current figure
    plt.pause(0.05)  # Pause for a short time to allow for dynamic updates

        
def solve(
    problem,
    max_depth=2000,  # planning horizon
    discount_factor=0.99,
    planning_time=40.0,  # amount of time (s) to plan each step
    exploration_const=10000,  # exploration constant
    visualize=True,
    max_time=120,  # maximum amount of time allowed to solve the problem
    max_steps=2000,
    solver_type='sarsop',
    humanproblem=False,
    human_intent=GoalState.green_goal,
    dist=None,
    computed_policy=None,
    agent2=None
):  # maximum number of planning steps the agent can take.
    """
    This function terminates when:
    - maximum time (max_time) reached; This time includes planning and updates
    - agent has planned `max_steps` number of steps
    - agent has taken n FindAction(s) where n = number of target objects.

    Args:
        visualize (bool) if True, show the pygame visualization.
    """
    
    
    if solver_type == 'sarsop':
        if humanproblem:
            if not os.path.exists('./temp-pomdp-human.policy') :
                #temp-pomdp-multiple-goalRED-eta9-beliefg1-(-10_1500)-.
                start = time.time()
                print('.......... sarsop solver used ...................')
                agent = problem.agent  # Define or import your agent as needed
                pomdpsol_path = "/home/cyrille/Desktop/Minigrid_DynamicProgramming/sarsop/src/pomdpsol"
                discount_factor = 0.99
                timeout = 100
                memory = 2000000
                precision = 0.000001
                remove_generated_files = False
                pomdp_name = 'temp-pomdp'
                args = (agent, pomdpsol_path, discount_factor, timeout, memory, precision, remove_generated_files, pomdp_name)
                pool = multiprocessing.Pool(6)
                print('........ processing')
                planner = pool.apply(run_sarsop,(args,))
                end = time.time() - start
                print('......... finished processing')
                print('Time spent :')
                print(end)
            
                '''
                pomdpsol_path = "/home/cyrille/Desktop/Minigrid_DynamicProgramming/sarsop/src/pomdpsol"
                planner = sarsop(problem.agent, pomdpsol_path, discount_factor=0.99,
                            timeout=100, memory=200, precision=0.000001,
                            remove_generated_files=False)
                '''
            else:
                policy_path = './temp-pomdp-human.policy'
                planner = AlphaVectorPolicy.construct(policy_path, problem.agent.all_states , problem.agent.all_actions)
        else:
            if not os.path.exists('./temp-pomdp-multiple-goal.policy') :
                # time spent for 8x8 eta = 5 : 103s
                # time spent for 8x8 eta = 9
                start = time.time()
                print('.......... sarsop solver used ...................')
                agent = problem.agent  # Define or import your agent as needed
                pomdpsol_path = "/home/cyrille/Desktop/Minigrid_DynamicProgramming/sarsop/src/pomdpsol"
                discount_factor = 0.99
                timeout = 100
                memory = 2000000
                precision = 0.000001
                remove_generated_files = False
                pomdp_name= 'temp-pomdp-multiple-goal'
                args = (agent, pomdpsol_path, discount_factor, timeout, memory, precision, remove_generated_files, pomdp_name)
                pool = multiprocessing.Pool(6)
                print('........ processing')
                planner = pool.apply(run_sarsop,(args,))
                end = time.time() - start
                print('......... finished processing')
                print('Time spent :')
                print(end)
            
                '''
                pomdpsol_path = "/home/cyrille/Desktop/Minigrid_DynamicProgramming/sarsop/src/pomdpsol"
                planner = sarsop(problem.agent, pomdpsol_path, discount_factor=0.99,
                            timeout=100, memory=200, precision=0.000001,
                            remove_generated_files=False)
                '''
            else :
                # -- define planner for robot 
                policy_path = './temp-pomdp-multiple-goal.policy'
                planner = AlphaVectorPolicy.construct(policy_path, problem.agent.all_states , problem.agent.all_actions)
                
    elif solver_type == 'POUCT':
        planner = pomdp_py.POUCT(
        max_depth=max_depth,
        discount_factor=discount_factor,
        planning_time=planning_time,
        exploration_const=exploration_const,
        rollout_policy=problem.agent.policy_model,
        )
        
    elif solver_type == 'VIPruning':
        pomdp_solve_path = "/home/cyrille/Desktop/Minigrid_DynamicProgramming/pomdp-solve/src/pomdp-solve"
        planner = pomdp_py.vi_pruning(problem.agent, pomdp_solve_path, discount_factor=0.99,
                    options=["-horizon", "100"],
                    remove_generated_files=False,
                    return_policy_graph=False)
    
    prior = {ALL_POSSIBLE_GOAL[0]: PROB_SIM_GREEN_GOAL, ALL_POSSIBLE_GOAL[1]: 1-PROB_SIM_GREEN_GOAL}
    belief_State_Tracker[ALL_POSSIBLE_GOAL[0]].append(prior[ALL_POSSIBLE_GOAL[0]])
    belief_State_Tracker[ALL_POSSIBLE_GOAL[1]].append(prior[ALL_POSSIBLE_GOAL[1]])
    
    #belief_State_Tracker[ALL_POSSIBLE_GOAL].append(1)
    count = 0
    step.append(count)
    
    
    _time_used = 0
    _total_reward = 0  # total, undiscounted reward
    for i in range(max_steps):
        problem.env.env.set_env_to_goal(human_intent)
        plt.ion()
        # Plan action
        _start = time.time()
        if computed_policy is not None and agent2 is not None:
            #w = problem.env.state.world
            w = problem.env.state.world
            state_current = problem.env.state.p
            if problem.env.state.goal == GoalState.green_goal:
                print(problem.agent.cur_belief[problem.env.state])
                approx_belief = agent2.approx_prob_to_belief(problem.agent.cur_belief[problem.env.state])
                print(approx_belief)
                print("Compared Action to baseline: %s" % str(computed_policy[approx_belief][w][state_current]))
            else:
                print(problem.agent.cur_belief[problem.env.state])
                approx_belief = agent2.approx_prob_to_belief(1-problem.agent.cur_belief[problem.env.state])
                print(approx_belief)
                print("Compared Action of baseline: %s" % str(computed_policy[approx_belief][w][state_current]))
        real_action = planner.plan(problem.agent)
        _time_used += time.time() - _start
        #if _time_used > max_time:
        #    break  # no more time to update.
        # Execute action
        next_state, reward, terminated = problem.env.state_transition(
            real_action, execute=True
        )
        
        
        # Receive observation
        _start = time.time()
        
        #print(problem.agent.cur_belief)
        # Updates
        #print(problem._agent.tree)
        
        
        # Info and render
        print('Robot action')
        _total_reward += reward
        print("==== Step %d ====" % (i + 1))
        print("Action: %s" % str(real_action))
        
        print("State: %s" % str(next_state))
        print("Reward: %s" % str(reward))
        print("Reward (Cumulative): %s" % str(_total_reward))
        #print("Find Actions Count: %d" % _find_actions_count)
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)

        if not humanproblem:
            # update agent pose
            current_agent_pose = (problem.env.env.agent_pos[0], problem.env.env.agent_pos[1])
            current_world = problem.env.env.get_world_state()
            
            # Generate action for human
            human_action = ActionsReduced(generate_action(state=current_agent_pose, worldState=current_world, goal=human_intent,dist=dist, multiple_goal=problem.env.multiple_goal))
            next_state, reward_h, terminated = problem.env.state_transition_human(WrappedAction2Motion(human_action))
        
        
        real_observation = problem.agent.observation_model.sample(next_state, real_action)
        print(belief_State_Tracker)
        problem.agent.clear_history()  # truncate history
        problem.agent.update_history(real_action, real_observation)
        if solver_type == 'sarsop':
            new_belief = pomdp_py.update_histogram_belief(problem.agent.cur_belief,
                                                    real_action, real_observation,
                                                    problem.agent.observation_model,
                                                    problem.agent.transition_model)
        else:
            if isinstance(planner, pomdp_py.PolicyGraph):
                # No belief update needed. Just update the policy graph
                planner.update(problem.agent, real_action, real_action)
            else:
                # belief update is needed for AlphaVectorPolicy
                new_belief = pomdp_py.update_histogram_belief(problem.agent.cur_belief,
                                                            real_action, real_observation,
                                                            problem.agent.observation_model,
                                                            problem.agent.transition_model)
        problem.agent.set_belief(new_belief)
        if human_intent == GoalState.green_goal:
            belief_State_Tracker[ALL_POSSIBLE_GOAL[0]].append(problem.agent.cur_belief[next_state])
            belief_State_Tracker[ALL_POSSIBLE_GOAL[1]].append((1-problem.agent.cur_belief[next_state]))
        elif human_intent == GoalState.red_goal:
            belief_State_Tracker[ALL_POSSIBLE_GOAL[0]].append((1-problem.agent.cur_belief[next_state]))
            belief_State_Tracker[ALL_POSSIBLE_GOAL[1]].append((problem.agent.cur_belief[next_state]))
        count += 1
        step.append(count)
        #belief_State_Tracker[ALL_POSSIBLE_GOAL].append(problem.agent.cur_belief[next_state])
        animate(state=next_state)
        _time_used += time.time() - _start
            
        print(' Human action')
        print("Action: %s" % str(WrappedAction2Motion(human_action)))
        print("State: %s" % str(next_state))
        print("Reward: %s" % str(reward_h))
        print("Belief about previous state: %s" % str(problem.agent.cur_belief[next_state]))
        print("Belief about the other goal: %s" % str(1-problem.agent.cur_belief[next_state]))
        
        _total_reward += reward_h
        print("Reward (Cumulative): %s" % str(_total_reward))
            
    

        # Termination check
        if problem.env.env.multiple_goal == True:
            if (problem.env.state.goal == GoalState.red_goal):
                if (problem.env.state.p[0] == problem.env.env.goal_[0][0]) and (problem.env.state.p[1] == problem.env.env.goal_[0][1]) :
                    print("Green Done!")
                    break
            elif (problem.env.state.goal == GoalState.green_goal):
                
                if (problem.env.state.p[0] == problem.env.env.goal_[1][0]) and (problem.env.state.p[1] == problem.env.env.goal_[1][1]) :
                    print("Red Done!")
                    break
        if terminated:
            print('Terminated')
            break
    plt.ioff() 
            
        