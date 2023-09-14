from __future__ import annotations
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.actions import Actions, ActionsReduced
from minigrid.core.world_object import Point, WorldObj





class EmptyReducedEnv(MiniGridEnv):
    """
    ## Description


    This environment is an empty room, and the goal of the agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is
    subtracted for the number of steps to reach the goal. This environment is
    useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agent starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.


    ## Mission Space


    "get to the green goal square"


    ## Action Space


    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |


    ## Observation Encoding


    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked


    ## Rewards


    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.


    ## Termination


    The episode ends if any one of the following conditions is met:


    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).


    ## Registered Configurations

    - `MiniGrid-Empty-Reduced-8x8-v0`
    - `MiniGrid-Empty-Reduced-16x16-v0`


    """


    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.lastaction = None  # for rendering
        self.size = size
        self.goal_pose = np.array([0, 0])
        self.obey_prob = 1
        self.num_goal = 0
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.i = 0
        self.j = 0


        if max_steps is None:
            max_steps = 4 * size**2


        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    def get_states_non_terminated(self, all=False):
        s = list()
        for i in range(1, self.size -1) :
            for j in range(1, self.size-1):
                if i == self.goal_pose[0][0] and j == self.goal_pose[0][1] and all == False:
                    break
                elif i == self.goal_pose[1][0] and j == self.goal_pose[1][1] and all == False:
                    break
                else:
                    s.append((i,j))
        return s
    
    def get_all_states(self):
        return self.get_states_non_terminated(all=True)

    def get_possible_move(self, pose=0):
        if type(pose) == int :
            i = self.agent_pos[0]
            j = self.agent_pos[1]
        else:
            i = pose[0]
            j = pose[1]
            
        if i == 1:
            if j== 1:
                return (ActionsReduced.right, ActionsReduced.backward) 
            elif j > 1 and j < self.size -2 :
                return (ActionsReduced.right, ActionsReduced.forward, ActionsReduced.backward)
            elif j == self.size -2:
                return (ActionsReduced.right, ActionsReduced.forward)
        elif  i > 1 and i < self.size -2 :
            if j == 1:
                return (ActionsReduced.right, ActionsReduced.backward, ActionsReduced.left) 
            elif j> 1 and j < self.size -2 :
                return (ActionsReduced.right, ActionsReduced.left, ActionsReduced.forward ,ActionsReduced.backward)
            elif j == self.size -2:
                return (ActionsReduced.left, ActionsReduced.forward, ActionsReduced.right)
        elif i== self.size -2:
            if j == 1:
                return (ActionsReduced.left, ActionsReduced.backward)
            elif j > 1 and j < self.size -2 :
                return (ActionsReduced.left, ActionsReduced.forward ,ActionsReduced.backward)
            elif j == self.size -2:
                return (ActionsReduced.left, ActionsReduced.forward)
    
    def get_reward(self, i, j, cost_value=0):
        r = 0
        if i == self.goal_pose[0][0] and j == self.goal_pose[0][1]:
            r = 1
        elif i == self.goal_pose[1][0] and j == self.goal_pose[1][1]:
            r = 1
        else:
            r = 0 - cost_value
        return r

    def check_move(self, action, cost_value=0):
        # check if legal move first
        i = self.i
        j = self.j
        
        if action in self.get_possible_move(pose=np.array((i,j))):
            if action == ActionsReduced.forward:
                j -= 1
            elif action == ActionsReduced.backward:
                j += 1
            elif action == ActionsReduced.right:
                i += 1
            elif action == ActionsReduced.left:
                i -= 1
        # return a reward (if any)
        reward = self.get_reward(i, j, cost_value=cost_value)
        return ((i, j), reward)
    
    def get_transition_probs(self, action, cost_value=0):
        probs = []
        next_state, reward = self.check_move(action, cost_value=cost_value)
        probs.append((self.obey_prob, reward, next_state))
        return probs

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]
    
    @staticmethod
    def _gen_mission():
        return "get to the green goal square"


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width , height)
        self.num_goal = 2
        self.goal_pose = list()
        goal_X_1 = width - 4
        goal_Y_1 = height - 4
        
        goal_X_2 = width - 2
        goal_Y_2 = height - 2
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)


        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), goal_X_1, goal_Y_1)
        self.goal_pose.append(np.array([goal_X_1, goal_Y_1]))
        g1 = Goal()
        g1.change_color("red")
        self.put_obj(g1, goal_X_2, goal_Y_2)
        self.goal_pose.append(np.array([goal_X_2, goal_Y_2]))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
            
        self.i = self.agent_pos[0]
        self.j = self.agent_pos[1]

        self.mission = "get to the green goal square"