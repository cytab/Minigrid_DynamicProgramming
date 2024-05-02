# Enumeration of possible actions
from __future__ import annotations
import pomdp_py
from enum import IntEnum
import math
import random
import time
from pomdp_py import sarsop
from pomdp_py.utils import TreeDebugger

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
        super().__init__("move-%s-%s" % (scheme, motion_name))
    
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
    

class StatePOMDP(pomdp_py.State):
    def __init__(self, world, world2, pose= None, goal=None):
        
        self.p = pose
        self.world1 = world
        self.world2 = world2
        self.goal = goal
        self.name = str(world) + "-" + str(world2) + "-" +  str(pose)  + "-" + str(goal)

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
    @property
    def pose(self):
        return self.pose

    @property
    def robot_pose(self):
        return self.pose

class HumanObservationPOMDP(pomdp_py.Observation):
    def __init__(self, world, world2, pose, goal):
        self.p = pose
        self.world1 = world
        self.world2 = world2
        self.goal = goal
        self.name = str(world) + "-" + str(world2) +  "-"  + str(pose) + ';' + str(pose)  + "-" + str(goal)

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
    @property
    def pose(self):
        return self.p

    @property
    def human_pose(self):
        return self.p
    
class ObservationPOMDP(pomdp_py.Observation):
    def __init__(self, world, world2, state):
        self.state = state
        self.world = world
        self.world2 = world2
        self.name = str(world) + "-" + str(world2) + "-" + str(state)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, ObservationPOMDP):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "(%s)" % self.name
    
class HumanTransitionModel(pomdp_py.TransitionModel):
    """We assume that the robot control is perfect and transitions are deterministic."""

    def __init__(self, dim, env, epsilon=1e-9):
        """
        dim (tuple): a tuple (width, length) for the dimension of the world
        """
        # this is used to determine objects found for FindAction
        self.env = env
        self._dim = dim
        self._epsilon = epsilon

    #@classmethod
    def if_move_by(self, state, action, dim, check_collision=True):
        """Defines the dynamics of robot motion;
        dim (tuple): the width, length of the search world."""
        self.env.set_state(state.p)
        self.env.open_door_manually((state.world1,state.world2))
        if not isinstance(action, MotionAction):
            raise ValueError("Cannot move robot with %s action" % str(type(action)))

        human_pose = state.p
        rx, ry = human_pose
        #if action.scheme == MotionAction.SCHEME_XY:
        #    dx, dy = action.motion
        #    rx += dx
        #    ry += dy
        #print(self.env.state_dynamic_human(previous_state=human_pose, state=(rx,ry), action_human=WrappedMotion2Action(action)))
        flag, s_prime = self.env.state_dynamic_human(previous_state=human_pose, action_human=WrappedMotion2Action(action))
        #if flag:
        #    #print(action)
        #    #state_1 = s_prime
        #    #print(s_prime)
        #    return s_prime
        #else:
       #     return human_pose  # no change because change results in invalid pose
        return s_prime

    def probability(self, next_robot_state, state, action):
        if next_robot_state != self.argmax(state, action):
            return 0.01
        else:
            return 1 - 0.01

    def argmax(self, state, action):
        #import copy
        """Returns the most likely next robot_state"""
        if isinstance(state, StatePOMDP):
            human_state = state

        next_human_state = StatePOMDP(world=state.world1, world2=state.world2, goal=state.goal, pose=state.p)
        # camera direction is only not None when looking
        #if isinstance(action, MotionAction):
            # motion action
        next_human_state.p = self.if_move_by(state, action,dim=self._dim)
        #print(self.if_move_by(state, action,dim=self._dim))
        #print('previous')
        #print(state.p)
        #print('next')
        #print(next_human_state)
        return next_human_state

    def sample(self, state, action):
        """Returns next_robot_state"""
        return self.argmax(state, action)
    
    def get_all_states(self):
        S = []
        for  pose in self.env.get_all_states():
            S.append(StatePOMDP(world=WorldSate.open_door1, world2=WorldSate.open_door2, pose=pose, goal=GoalState.green_goal))
        return S
    
    
# Reward Model
class HumanRewardModel(pomdp_py.RewardModel):
    def __init__(self, env):
        self.env = env
    def _reward_func(self, state, action):
        self.env.set_state(state.p)
        self.env.open_door_manually((state.world1,state.world2))
        if isinstance(action, MotionAction):
            next_state, reward = self.env.check_move(action=WrappedMotion2Action(action), w=None, cost_value=1)
        return reward

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)
    
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
            prob = 1
        else:
            prob = 0
        return prob

    def sample(self, next_state, action):
        """Returns observation"""

        return HumanObservationPOMDP(world=next_state.world1, world2=next_state.world2, pose=next_state.p, goal=next_state.goal)

    def argmax(self, next_state, action):
        return next_state
    
    def get_all_observations(self):
        S = []
        for  pose in self.env.get_all_states():
            S.append(HumanObservationPOMDP(world=WorldSate.open_door1, world2=WorldSate.open_door2, pose=pose, goal=GoalState.green_goal))
        return S

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
        
        if state is None:
            return ALL_MOTION_ACTIONS
        else:
            valid_motions = self.env.get_possible_move(state.p)
            custom_motions = list()
            for mot in valid_motions:
                if mot == ActionsReduced.forward:
                    custom_motions.append(MoveNorth2D)
                elif mot == ActionsReduced.backward:
                    custom_motions.append(MoveSouth2D)
                elif mot == ActionsReduced.left:
                    custom_motions.append(MoveWest2D)
                elif mot == ActionsReduced.right:
                    custom_motions.append(MoveEast2D)
                elif mot == ActionsReduced.stay:
                    custom_motions.append(MoveHalt2D)
            return custom_motions
            
    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]


class GridEnvironment(pomdp_py.Environment):
    def __init__(self, env, init_true_state, dim, epsilon):
        self.env = env
        transition_model = HumanTransitionModel(dim, env, epsilon=epsilon)
        reward_model = HumanRewardModel(env)
        super().__init__(init_true_state, transition_model, reward_model)
    
    def state_transition(self,action, execute=True):
        next_state = self.transition_model.sample(self.state, action)
        #print(self.state)
        reward = self.reward_model.sample(self.state, action, next_state)
        if execute:
            #print(next_state)
            self.apply_transition(next_state)
            #print(self.state)
            self.env.step(WrappedMotion2Action(action))
            return next_state, reward
        else:
            return next_state, reward
    
class HumanAgent(pomdp_py.Agent):
    def __init__(self, env, init_human_state, dim, epsilon=1, grid_map=None):
        human_transition = HumanTransitionModel(dim, env, epsilon=epsilon)
        human_reward = HumanRewardModel(env)
        Human_observation = HumanObservationModel(env,dim, epsilon)
        human_policy = PolicyModel(env)
        init_belief = pomdp_py.Histogram( {init_human_state: 1.0})
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
#def belief_update(agent, real_action, real_observation, next_human_state, planner):
    """Updates the agent's belief; The belief update may happen
    through planner update (e.g. when planner is POMCP)."""
    # Updates the planner; In case of POMCP, agent's belief is also updated.
    #planner.update(agent, real_action, real_observation)

    # Update agent's belief, when planner is not POMCP
    #if not isinstance(planner, pomdp_py.POMCP):
        # Update belief for every object
            #if isinstance(belief_obj, pomdp_py.Histogram):
            #        # Assuming the agent can observe its own state:
            #        new_belief = pomdp_py.Histogram({next_human_state: 1.0})
                
            #else:
            #    raise ValueError(
            #        "Unexpected program state."
            #        "Are you using the appropriate belief representation?"
            #    )

            #agent.cur_belief.set_object_belief(new_belief)
            
            
def solve(
    problem,
    max_depth=10,  # planning horizon
    discount_factor=0.99,
    planning_time=1.0,  # amount of time (s) to plan each step
    exploration_const=1000,  # exploration constant
    visualize=True,
    max_time=120,  # maximum amount of time allowed to solve the problem
    max_steps=500,
):  # maximum number of planning steps the agent can take.
    """
    This function terminates when:
    - maximum time (max_time) reached; This time includes planning and updates
    - agent has planned `max_steps` number of steps
    - agent has taken n FindAction(s) where n = number of target objects.

    Args:
        visualize (bool) if True, show the pygame visualization.
    """

    #if isinstance(random_object_belief, pomdp_py.Histogram):
    # Use POUCT
    pomdpsol_path = "/home/cyrille/Desktop/Minigrid_DynamicProgramming/sarsop/src/pomdpsol"
    
    #planner = pomdp_py.POUCT(
    #    max_depth=max_depth,
    #    discount_factor=discount_factor,
    #    planning_time=planning_time,
    #    exploration_const=exploration_const,
    #    rollout_policy=problem.agent.policy_model,
    #    )
    planner = pomdp_py.POUCT(max_depth=5, discount_factor=0.99,
                       planning_time=.5, exploration_const=110,
                       rollout_policy=problem.agent.policy_model, show_progress=True)
    
    #planner = pomdp_py.vi_pruning(problem.agent, pomdp_solve_path, discount_factor=0.95,
    #                options=["-horizon", "100"],
    #                remove_generated_files=False,
    #                return_policy_graph=False)
    #planner = sarsop(problem.agent, pomdpsol_path, discount_factor=0.95,
    #            timeout=10, memory=20, precision=0.000001,
    #            remove_generated_files=True)
    #planner = pomdp_py.ValueIteration(horizon=2, discount_factor=0.99)
    # Random by default
    #elif isinstance(random_object_belief, pomdp_py.Particles):
    #    # Use POMCP
    #    planner = pomdp_py.POMCP(
    #        max_depth=max_depth,
    #        discount_factor=discount_factor,
    #        planning_time=planning_time,
    #        exploration_const=exploration_const,
    #        rollout_policy=problem.agent.policy_model,
    #    )  # Random by default
    #else:
    #    raise ValueError(
    #        "Unsupported object belief type %s" % str(type(random_object_belief))
    #    )

    _time_used = 0
    _total_reward = 0  # total, undiscounted reward
    for i in range(max_steps):
        # Plan action
        _start = time.time()
        real_action = planner.plan(problem.agent)
        _time_used += time.time() - _start
        if _time_used > max_time:
            break  # no more time to update.
        # Execute action
        next_state, reward = problem.env.state_transition(
            real_action, execute=True
        )

        # Receive observation
        _start = time.time()
        real_observation = problem.env.provide_observation(
            problem.agent.observation_model, real_action
        )

        # Updates
        #print(problem._agent.tree)
        problem.agent.clear_history()  # truncate history
        problem.agent.update_history(real_action, real_observation)
        #belief_update(
        #    problem.agent,
        #    real_action,
        #    real_observation,
        #    problem.env.state,
        #    planner,
        #)
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
        if (problem.env.state.goal == GoalState.red_goal):
            if (problem.env.state.p[0] == problem.env.env.goal_[0][0]) and (problem.env.state.p[1] == problem.env.env.goal_[0][1]) :
                print("Done!")
                break
        elif (problem.env.state.goal == GoalState.green_goal):
            if (problem.env.state.p[0] == problem.env.env.goal_[1][0]) and (problem.env.state.p[1] == problem.env.env.goal_[1][1]) :
                print("Done!")
                break
            
        if _time_used > max_time:
            print("Maximum time reached.")
            break
        #TreeDebugger(problem.agent.tree).pp
        

    
#graphe baysian cercle pour variable et fleche variable depend dune autre 
#t et t prime, t+1, t+2