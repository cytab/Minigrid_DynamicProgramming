# Enumeration of possible actions
from __future__ import annotations
import pomdp_py
from enum import IntEnum
import math
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
    HATL2D = (0, STEP_SIZE)

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
MoveSouth2D = MotionAction(
    MotionAction.HATL2D, scheme=MotionAction.SCHEME_XY, motion_name="HALT2D"
)

DoNothing = ActionPOMDP(name='DoNothing')
Take_Key_1 = ActionPOMDP(name='TakeKey1')
Take_Key_2 = ActionPOMDP(name='TakeKey2')

class StatePOMDP(pomdp_py.State):
    def __init__(self, world, state):
        self.state = state
        self.world = world
        self.name = str(world) + "-" + str(state)

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
        return self.state

    @property
    def robot_pose(self):
        return self.state

    #def other(self):
    #    if self.name.endswith("left"):
    #        return TigerState("tiger-right")
    #    else:
    #        return TigerState("tiger-left")
    
class ObservationPOMDP(pomdp_py.Observation):
    def __init__(self, world, state):
        self.state = state
        self.world = world
        self.name = str(world) + "-" + str(state)

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
    

# Reward Model
class RewardModel(pomdp_py.RewardModel):
    def __init__(self, env):
        self.env = env
    def _reward_func(self,w , state, action):
        

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)