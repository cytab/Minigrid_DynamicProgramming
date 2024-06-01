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
    
ALL_POSSIBLE_WOLRD = ((WorldSate.open_door1,WorldSate.open_door2), (WorldSate.open_door1,WorldSate.closed_door2), (WorldSate.closed_door1, WorldSate.open_door2), (WorldSate.closed_door1, WorldSate.closed_door2))
ALL_POSSIBLE_GOAL = (GoalState.green_goal,GoalState.red_goal)
ALL_POSSIBLE_ACTIONS = (ActionsReduced.right, ActionsReduced.left, ActionsReduced.forward, ActionsReduced.backward, ActionsReduced.stay)

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
    
def WrappedRobotAction2Action(robot_action):
    if robot_action == ActionsAgent2.nothing:
        return DoNothing
    elif robot_action == ActionsAgent2.take_key1:
        return Take_Key_1
    elif robot_action == ActionsAgent2.take_key2:
        return Take_Key_2
    

class StatePOMDP(pomdp_py.State):
    def __init__(self, world, world2, pose=None, goal=None):
        
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
        if self.world2 == WorldSate.closed_door2:
            self.name += 'W2_Close'
        elif self.world2 == WorldSate.open_door2 :
            self.name += 'W2_Open' 
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
    
# Human Definition problem -------------------------------------------------------------------- #########################

class HumanTransitionModel(pomdp_py.TransitionModel):
    """We assume that the robot control is perfect and transitions are deterministic."""

    def __init__(self, dim, env, epsilon=1e-12):
        """
        dim (tuple): a tuple (width, length) for the dimension of the world
        """
        # this is used to determine objects found for FindAction
        self.env = env
        self._dim = dim
        self._epsilon = epsilon

    #@classmethod
    def if_move_by(self, state, action):
        """Defines the dynamics of robot motion;
        dim (tuple): the width, length of the search world."""
        self.env.set_state(state.p)
        self.env.open_door_manually((state.world1,state.world2))
        if not isinstance(action, MotionAction):
            raise ValueError("Cannot move robot with %s action" % str(type(action)))

        human_pose = state.p
        rx, ry = human_pose
        flag, s_prime = self.env.state_dynamic_human(previous_state=human_pose, action_human=WrappedMotion2Action(action))
        #state_world = self.env.get_world_state()
        state_world = state.world
        return s_prime, state_world

    def probability(self, next_robot_state, state, action):
        self.env.set_state(state.p)
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

        next_human_state = StatePOMDP(world=state.world1, world2=state.world2, goal=state.goal, pose=state.p)
        next_human_state.p, world = self.if_move_by(state, action)
        next_human_state.world1 = world[0]
        next_human_state.world2 = world[1]
        return next_human_state

    def sample(self, state, action):
        """Returns next_robot_state"""
        return self.argmax(state, action)
    
    def get_all_states(self):
        S = []
        for  pose in self.env.get_all_states():
            #world = self.env.get_world_state()
            for world in ALL_POSSIBLE_WOLRD:
                for goal in ALL_POSSIBLE_GOAL:
                    S.append(StatePOMDP(world=world[0], world2=world[1], pose=pose, goal=goal))
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
    def __init__(self, env, dim, epsilon=1):
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
    def __init__(self, env):
        self.env = env
    def _reward_func(self, state, action):
        self.env.set_state(state.p)
        self.env.open_door_manually((state.world1,state.world2))
        if isinstance(action, MotionAction):
            reward = 0
            next_state, reward = self.env.check_move(action=WrappedMotion2Action(action), w=None, cost_value=1)
        self.env.open_door_manually((state.world1, state.world2))
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
    def __init__(self, env, init_human_state, dim, epsilon=1, grid_map=None):
        human_transition = HumanTransitionModel(dim, env, epsilon=epsilon)
        human_reward = HumanRewardModel(env)
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
    def __init__(self, word1, world2, pose, goal, env, dim, epsilon=1, grid_map=None):
        init_human_state = StatePOMDP(world=word1, world2=world2, pose=pose, goal=goal)
        
        human_agent = HumanAgent(env, init_human_state, dim, epsilon=1, grid_map=None)
        grid_env = GridEnvironment(env=env, init_true_state=init_human_state, dim=dim, epsilon=epsilon)
        super().__init__(
            human_agent,
            grid_env,
            name="GRID(%d,%d)" % (env.size, env.size),
        )

class GridEnvironment(pomdp_py.Environment):
    def __init__(self, env, init_true_state, dim, epsilon):
        self.env = env
        transition_model = HumanTransitionModel(dim, env, epsilon=epsilon)
        reward_model = HumanRewardModel(env)
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
    def __init__(self, world, world2, pose):
        self.p = pose
        self.world1 = world
        self.world2 = world2
        self.name = "case" + str(pose[0]) + 'e' + str(pose[1]) 
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

    def __init__(self, dim, env, probability, epsilon=1e-12):
        """
        dim (tuple): a tuple (width, length) for the dimension of the world
        """
        # this is used to determine objects found for FindAction
        self.env = env
        self._dim = dim
        self._epsilon = epsilon
        self.prob = probability

    #@classmethod
    def if_move_by(self, state, action):
        """Defines the dynamics of robot motion;
        dim (tuple): the width, length of the search world."""
        self.env.set_state(state.p)
        #self.env.open_door_manually((state.world1,state.world2))
        if not isinstance(action, MotionAction):
            raise ValueError("Cannot move robot with %s action" % str(type(action)))

        human_pose = state.p
        rx, ry = human_pose
        flag, s_prime = self.env.state_dynamic_human(previous_state=human_pose, action_human=WrappedMotion2Action(action))
        state_world = state.world

        return s_prime, state_world

    def probability(self, next_robot_state, state, action):
        prob = 0
        self.env.set_state(state.p)
        self.env.open_door_manually((state.world1,state.world2))
        self.env.check_move(action=WrappedAction2RobotAction(action), w=state.world)
        #if WrappedMotion2Action(action) in self.env.get_possible_move(state.p):
        for a_1 in self.env.get_possible_move(state.p):
            intermediate_state = self.argmax(state, action) # only change the world becasue it is the action of the robot
            finale_state = self.human_transition(intermediate_state,WrappedAction2Motion(a_1))
            world_p = self.env.world_dynamic_update(action, current_world=state.world)
            if next_robot_state.p == finale_state.p and next_robot_state.world1 == finale_state.world1 and next_robot_state.world2 == finale_state.world2 and next_robot_state.goal == finale_state.goal:
                prob += self.prob[next_robot_state.goal][intermediate_state.world][intermediate_state.p][a_1]
            else:
                prob += 0.0
        self.env.check_move(action=WrappedAction2RobotAction(action), w=state.world)
        self.env.open_door_manually((state.world1,state.world2))
        
        return prob

    def human_transition(self, state, action):
        next_human_state = StatePOMDP(world=state.world1, world2=state.world2, goal=state.goal, pose=state.p)
        next_human_state.p, world = self.if_move_by(state, action)
        next_human_state.world1 = world[0]
        next_human_state.world2 = world[1]
        return next_human_state
        
    def argmax(self, state, action):
        """Returns the most likely next robot_state"""
        next_human_state = StatePOMDP(world=state.world1, world2=state.world2, goal=state.goal, pose=state.p)
        world = self.env.world_dynamic_update(action, current_world =state.world)
        next_human_state.world1 = world[0]
        next_human_state.world2 = world[1]
        return next_human_state

    def sample(self, state, action):
        """Returns next_robot_state"""
        return self.argmax(state, action)
    
    def get_all_states(self):
        S = []
        for  pose in self.env.get_all_states():
            #world = self.env.get_world_state()
            for world in ALL_POSSIBLE_WOLRD:
                for goal in ALL_POSSIBLE_GOAL:
                    S.append(StatePOMDP(world=world[0], world2=world[1], pose=pose, goal=goal))
        return S
    
# Reward Model
class RobotRewardModel(pomdp_py.RewardModel):
    def __init__(self, env, probability):
        self.env = env
        self.probability = probability

    def _reward_func(self, state, action):
        reward = 0
        self.env.set_state(state.p)
        self.env.open_door_manually((state.world1,state.world2))
        self.env.check_move(action=WrappedAction2RobotAction(action), w=state.world)
        ac = WrappedAction2RobotAction(action)
        world_prime = self.env.world_dynamic_update(action=ac, current_world=(state.world1, state.world2))
        self.env.check_move(action=WrappedAction2RobotAction(action), w=state.world)
        for a_1 in ALL_POSSIBLE_ACTIONS:
                next_state, reward_t = self.env.check_move(action=a_1, w=world_prime, cost_value=1)
                reward +=  self.probability[state.goal][world_prime][state.p][a_1]*reward_t
        self.env.check_move(action=WrappedAction2RobotAction(action), w=state.world)
        self.env.open_door_manually((state.world1,state.world2))
        reward += self.env.get_reward_2(WrappedAction2RobotAction(action))
        return reward

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)   

class RobotObservationModel(pomdp_py.ObservationModel):
    def __init__(self, env, dim, epsilon=1):
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
        if observation.world1 == next_state.world1 and observation.world2 == next_state.world2 and observation.p == next_state.p:
            prob = 1.0
        else:
            prob = 0.0
        return prob

    def sample(self, next_state, action):
        """Returns observation"""
        return RobotObservationPOMDP(world=next_state.world1, world2=next_state.world2, pose=next_state.p)

    def argmax(self, next_state, action):
        return next_state
    
    def get_all_observations(self):
        S = []
        for  pose in self.env.get_all_states():
            for world in ALL_POSSIBLE_WOLRD:
            #    for goal in ALL_POSSIBLE_GOAL:
                S.append(RobotObservationPOMDP(world=world[0], world2=world[1], pose=pose))
        return S

class RobotPolicyModel(pomdp_py.RolloutPolicy):
    
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
        return [DoNothing, Take_Key_1, Take_Key_2]
        
    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

class RobotGridEnvironment(pomdp_py.Environment):
    def __init__(self, env, init_true_state, dim, epsilon, human_probability):
        self.env = env
        transition_model = RobotTransitionModel(dim, env, epsilon=epsilon, probability=human_probability)
        reward_model = RobotRewardModel(env, probability=human_probability)
        super().__init__(init_true_state, transition_model, reward_model)
    
    def state_transition(self,action, execute=True):
        next_state = self.transition_model.sample(self.state, action)
        reward = self.reward_model.sample(self.state, action, next_state)
        terminated = False
        if execute:
            self.apply_transition(next_state)
            self.robot_step(WrappedAction2RobotAction(action))
            return next_state, reward
        else:
            return next_state, reward
    
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
    def __init__(self, env, init_robot_state, initial_prob, dim, human_probability, epsilon=1):
        robot_transition = RobotTransitionModel(dim, env, epsilon=epsilon, probability=human_probability)
        robot_reward = RobotRewardModel(env, probability=human_probability)
        robot_observation = RobotObservationModel(env,dim, epsilon)
        robot_policy = RobotPolicyModel(env)
        init_belief = pomdp_py.Histogram( {s: 0.0 for s in robot_transition.get_all_states()})
        init_belief[init_robot_state] = 0.5
        rob = StatePOMDP(world=init_robot_state.world1, world2=init_robot_state.world2, pose=init_robot_state.p, goal=GoalState.red_goal)
        #init_robot_state.set_goal(goal=GoalState.red_goal)
        init_belief[rob] = 0.5
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
    def __init__(self, word1, world2, pose, goal, env, dim, human_probability, epsilon=1, initial_prob=0.5):
        init_human_state = StatePOMDP(world=word1, world2=world2, pose=pose, goal=goal)
        
        robot_agent = RobotAgent(env=env, init_robot_state=init_human_state, dim=dim, epsilon=1, initial_prob=initial_prob, human_probability=human_probability)
        grid_env = RobotGridEnvironment(env=env, init_true_state=init_human_state, dim=dim, epsilon=epsilon, human_probability=human_probability)
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
        agent, pomdpsol_path, discount_factor, timeout, memory, precision, remove_generated_files = args
        planner = sarsop(agent, pomdpsol_path, discount_factor=discount_factor,
                        timeout=timeout, memory=memory, precision=precision,
                        remove_generated_files=remove_generated_files)
        return planner
             
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
    humanproblem=False
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
            if not os.path.exists('./temp-pomdp-human.policy'):
                
                start = time.time()
                print('.......... sarsop solver used ...................')
                agent = problem.agent  # Define or import your agent as needed
                pomdpsol_path = "/home/cyrille/Desktop/Minigrid_DynamicProgramming/sarsop/src/pomdpsol"
                discount_factor = 0.99
                timeout = 100
                memory = 2000000
                precision = 0.000001
                remove_generated_files = False
                args = (agent, pomdpsol_path, discount_factor, timeout, memory, precision, remove_generated_files)
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
            start = time.time()
            print('.......... sarsop solver used ...................')
            agent = problem.agent  # Define or import your agent as needed
            pomdpsol_path = "/home/cyrille/Desktop/Minigrid_DynamicProgramming/sarsop/src/pomdpsol"
            discount_factor = 0.99
            timeout = 100
            memory = 2000000
            precision = 0.000001
            remove_generated_files = False
            args = (agent, pomdpsol_path, discount_factor, timeout, memory, precision, remove_generated_files)
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

    _time_used = 0
    _total_reward = 0  # total, undiscounted reward
    for i in range(max_steps):
        # Plan action
        _start = time.time()
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
        real_observation = problem.agent.observation_model.sample(problem.env.state, real_action)
        #print(problem.agent.cur_belief)
        # Updates
        #print(problem._agent.tree)
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
        _time_used += time.time() - _start

        # Info and render
        _total_reward += reward
        print("==== Step %d ====" % (i + 1))
        print("Action: %s" % str(real_action))
        print("State: %s" % str(next_state))
        print("Reward: %s" % str(reward))
        print("Reward (Cumulative): %s" % str(_total_reward))
        #print("Find Actions Count: %d" % _find_actions_count)
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)


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
                
            
        #if _time_used > max_time:
        #    print("Maximum time reached.")
        #    break
        #TreeDebugger(problem.agent.tree).pp
        