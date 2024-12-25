from __future__ import annotations
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Door, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.actions import Actions, ActionsReduced, ActionsAgent2, WorldSate, GoalState
from minigrid.core.world_object import Point, WorldObj

class LockedRoom:
    def __init__(self, top, size, doorPos):
        self.top = top
        self.size = size
        self.doorPos = doorPos
        self.color = None
        self.locked = False

    def rand_pos(self, env):
        topX, topY = self.top
        sizeX, sizeY = self.size
        return env._rand_pos(topX + 1, topX + sizeX - 1, topY + 1, topY + sizeY - 1)
    
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
    - `MiniGrid-Empty-Reduced-12x12-v0`
    - `MiniGrid-Empty-Reduced-32x32-v0`
    


    """


    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        door=True,
        multiple_goal=True,
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
        self.splitIdx = 0
        self.doorIdx = 0
        self.target_door = {}
        self.door = door
        self.multiple_goal = multiple_goal  
        self.goal_ = list() # list of every goal postion
        self.goal_pose = list() # current_goal
        self.goal_achieved = 0


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
        if self.multiple_goal:
            for i in range(1, self.size -1) :
                for j in range(1, self.size-1):
                    if i == self.goal_pose[0][0] and j == self.goal_pose[0][1] and all == False:
                        pass
                    else:
                        s.append((i,j))
        else:
            for i in range(1, self.size -1) :
                for j in range(1, self.size-1):
                    if i == self.goal_pose[0][0] and j == self.goal_pose[0][1] and all == False:
                        break
                    #elif i == self.goal_pose[1][0] and j == self.goal_pose[1][1] and all == False:
                    #    break
                    else:
                        s.append((i,j))
        return s
    
    def get_all_states(self):
        return self.get_states_non_terminated(all=True)
    
    def obstacle_pos(self, agent_pos, robot_action=None):
        # get if there are obstacle around the agent : up , below, on right, on left
        #if there is an obstacle (wall, door) then 1 and if not 0:
        obstacle = [0,0,0,0]
        i = agent_pos[0]
        j = agent_pos[1]
        
        #check forward future pose
        cell = []
        cell.append(self.grid.get(*(i,j-1)))
        cell.append(self.grid.get(*(i,j+1)))
        cell.append(self.grid.get(*(i+1,j)))
        cell.append(self.grid.get(*(i-1,j)))
        for index, case in enumerate(cell):
        #check forward future pose
            if case is not None:
                if not self.multiple_goal:
                    if (case.type == "door" and self.target_door[(self.splitIdx, self.doorIdx)]):
                        obstacle[index] = 1
                    elif case.type == "wall":
                        obstacle[index] = 1
                    elif case.type == "door" and not self.target_door[(self.splitIdx, self.doorIdx)]:
                        obstacle[index] = 0
                else:
                    
                    if (case.type == "door"):
                        if self.target_door[case.cur_pos]:
                            obstacle[index] = 1
                        else:
                            obstacle[index] = 0
                    elif case.type == "wall":
                        obstacle[index] = 1
        return obstacle
    
    def get_possible_move(self, pose=0, robot_action=None):
        # get the obstacle position

        if type(pose) == int :
            #i = self.agent_pos[0]
            #j = self.agent_pos[1]
            obstacle = self.obstacle_pos(self.agent_pos)
        else:
            #i = pose[0]
            #j = pose[1]
            obstacle = self.obstacle_pos(pose)
        
        possible_action = [ActionsReduced.stay] # this action is always possible
        
        for index, value in enumerate(obstacle):
            if value == 0:
                if index == 0:
                        possible_action.append(ActionsReduced.forward)
                elif index ==  1:
                        possible_action.append(ActionsReduced.backward)
                elif index == 2:
                        possible_action.append(ActionsReduced.right)
                elif index == 3:
                        possible_action.append(ActionsReduced.left)
        return possible_action   
    
    def get_reward_1(self, i, j, action, cost_value=1):
        r = 0
#        elif i == self.goal_pose[1][0] and j == self.goal_pose[1][1]:
 #           r = 100
        if action == ActionsReduced.stay:
            r += 0
        else:
            r += - cost_value
        
        if i == self.goal_pose[0][0] and j == self.goal_pose[0][1]:
            r += 300
        return r 

    def get_reward_2(self, action: ActionsAgent2):
        r = 0
        if action == ActionsAgent2.nothing:
            r = 0
        elif action == ActionsAgent2.take_key:
            r = -10
        elif action == ActionsAgent2.take_key1:
            r = -10
        elif action == ActionsAgent2.take_key2:
            r = -10
        return r
    
    def check_move(self, action, w, cost_value=1):
        # check if legal move first
        i = self.i
        j = self.j
        if isinstance(action, ActionsReduced):
                #print((i,j))
                #print(self.get_possible_move(pose=np.array((i,j))))
                if action in self.get_possible_move(pose=np.array((i,j))):
                    if action == ActionsReduced.forward:
                        j -= 1
                    elif action == ActionsReduced.backward:
                        j += 1
                    elif action == ActionsReduced.right:
                        i += 1
                    elif action == ActionsReduced.left:
                        i -= 1
                    elif action == ActionsReduced.stay:
                        pass # ne rien faire
                # return a reward (if any)
                reward = self.get_reward_1(i, j, action, cost_value=cost_value)
                return ((i, j), reward)
        elif isinstance(action, ActionsAgent2):
            if action == ActionsAgent2.nothing:
                pass
            if not self.multiple_goal:
                if action == ActionsAgent2.take_key and w is WorldSate.closed_door:
                    if self.door:
                        # considered opened, thus it is not considered an obstacle in solving the bellman equation
                        if self.target_door[(self.splitIdx, self.doorIdx)]:
                            self.target_door[(self.splitIdx, self.doorIdx)] = False
                        else:
                            self.target_door[(self.splitIdx, self.doorIdx)] = True
                    else: 
                        pass
                elif action == ActionsAgent2.take_key and w is WorldSate.open_door:
                    pass
            else:
                if action == ActionsAgent2.nothing:
                    pass
                elif action == ActionsAgent2.take_key1 and w[0] is WorldSate.closed_door1:
                    if self.target_door[self.rooms[0].doorPos]:
                        self.target_door[self.rooms[0].doorPos] = False
                    else:
                        self.target_door[self.rooms[0].doorPos] = True
                elif action == ActionsAgent2.take_key1 and w[0] is WorldSate.open_door1:
                    pass
                elif action == ActionsAgent2.take_key2 and w[1] is WorldSate.closed_door2:
                    if self.target_door[self.rooms[1].doorPos]:
                        self.target_door[self.rooms[1].doorPos] = False
                    else:
                        self.target_door[self.rooms[1].doorPos] = True
                elif action == ActionsAgent2.take_key2 and w[1] is WorldSate.open_door2:
                    pass
                
    # def check_move(self, action, w=None, cost_value=0):
    #     """
    #     Validate and execute the specified action. Returns the updated position and reward.

    #     Parameters:
    #         action (ActionsReduced | ActionsAgent2): The action to validate and perform.
    #         world_state: The current world state (used for ActionsAgent2).
    #         cost_value (int): The cost associated with taking an action.

    #     Returns:
    #         tuple: Updated position (i, j) and associated reward.
    #     """
    #     # Helper function to determine new position based on action
    #     def get_new_position(i, j, action):
    #         if action == ActionsReduced.forward:
    #             return i, j - 1
    #         elif action == ActionsReduced.backward:
    #             return i, j + 1
    #         elif action == ActionsReduced.right:
    #             return i + 1, j
    #         elif action == ActionsReduced.left:
    #             return i - 1, j
    #         return i, j  # Stay or invalid action

    #     # Validate reduced actions
    #     if isinstance(action, ActionsReduced):
    #         possible_moves = self.get_possible_move(pose=np.array((self.i,self.j)))
    #         if action not in possible_moves:
    #             # Invalid action: No movement or reward
    #             return (self.i, self.j), 0
            
    #         # Compute new position and reward
    #         new_i, new_j = get_new_position(self.i, self.j, action)
    #         reward = self.get_reward_1(new_i, new_j, action, cost_value=cost_value)
    #         return (new_i, new_j), reward

    #     # Validate agent-specific actions (ActionsAgent2)
    #     elif isinstance(action, ActionsAgent2):
    #         if action == ActionsAgent2.nothing:
    #             return (self.i, self.j), 0

    #         if not self.multiple_goal:
    #             if action == ActionsAgent2.take_key and w == WorldSate.closed_door:
    #                 self.target_door[(self.splitIdx, self.doorIdx)] = not self.target_door[(self.splitIdx, self.doorIdx)]
    #             # No position change for door interaction
    #             return (self.i, self.j), self.get_reward_2(action)

    #         else:
    #             room_states = [WorldSate.closed_door1, WorldSate.closed_door2]
    #             for idx, state in enumerate(room_states):
    #                 door_pos = self.rooms[idx].doorPos
    #                 if action == getattr(ActionsAgent2, f"take_key{idx+1}") and w[idx] == state:
    #                     self.target_door[door_pos] = not self.target_door[door_pos]
                        
    #             return (self.i, self.j), self.get_reward_2(action)

    #     # Invalid action type
    #     return (self.i, self.j), 0
    # def check_move(self, action, w=None, cost_value=0):
    #     """
    #     Validate and execute the specified action. Returns the updated position and reward.

    #     Parameters:
    #         action (ActionsReduced | ActionsAgent2): The action to validate and perform.
    #         w: The current world state (used for ActionsAgent2).
    #         cost_value (int): The cost associated with taking an action.

    #     Returns:
    #         tuple: Updated position (i, j) and associated reward.
    #     """
    #     def get_new_position(i, j, action):
    #         """
    #         Determine the new position based on the action.
    #         """
    #         movement_map = {
    #             ActionsReduced.forward: (0, -1),
    #             ActionsReduced.backward: (0, 1),
    #             ActionsReduced.right: (1, 0),
    #             ActionsReduced.left: (-1, 0),
    #             ActionsReduced.stay: (0, 0),
    #         }
    #         di, dj = movement_map.get(action, (0, 0))
    #         return i + di, j + dj

    #     def toggle_target_door(target_key):
    #         """
    #         Ensures only one target door can be `False` by setting all others to `True`.
    #         """
    #         for key in self.target_door:
    #             self.target_door[key] = True  # Lock all doors
    #             self.target_door[target_key] = False  # Unlock the selected door

    #     # Handle reduced movement actions
    #     if isinstance(action, ActionsReduced):
    #         possible_moves = self.get_possible_move(pose=(self.i, self.j))
    #         if action not in possible_moves:
    #             return (self.i, self.j), 0  # Invalid action: No movement or reward
            
    #         new_i, new_j = get_new_position(self.i, self.j, action)
    #         reward = self.get_reward_1(new_i, new_j, action, cost_value=cost_value)
    #         return (new_i, new_j), reward

    #     # Handle agent-specific actions
    #     elif isinstance(action, ActionsAgent2):
    #         if action == ActionsAgent2.nothing:
    #             return (self.i, self.j), 0  # No action taken

    #         if not self.multiple_goal:
    #             if action == ActionsAgent2.take_key and w == WorldSate.closed_door:
    #                 toggle_target_door((self.splitIdx, self.doorIdx))
    #             return (self.i, self.j), self.get_reward_2(action)

    #         else:  # Handle multiple goals
    #             room_states = [WorldSate.closed_door1, WorldSate.closed_door2]
    #             for idx, state in enumerate(room_states):
    #                 door_pos = self.rooms[idx].doorPos
    #                 if action == getattr(ActionsAgent2, f"take_key{idx+1}") and w[idx] == state:
    #                     toggle_target_door(door_pos)
    #             return (self.i, self.j), self.get_reward_2(action)

    #     # Invalid action type
    #     return (self.i, self.j), 0

                
    def open_door_manually(self, worldState):
        if not self.multiple_goal:
            if worldState == WorldSate.closed_door:
                    pass
            elif worldState == WorldSate.open_door:
                    if self.door:
                        # considered opened, thus it is not considered an obstacle in solving the bellman equation
                        if self.target_door[(self.splitIdx, self.doorIdx)]:
                            self.target_door[(self.splitIdx, self.doorIdx)] = False
                        else:
                            self.target_door[(self.splitIdx, self.doorIdx)] = True
                    else:
                        pass
        else: 
            if worldState[0] == WorldSate.closed_door1 or worldState[1] == WorldSate.closed_door2:
                pass
            for i in range(len(worldState)):
                if worldState[i] == WorldSate.open_door1 or worldState[i] == WorldSate.open_door2:
                    if self.target_door[self.rooms[i].doorPos]:
                        self.target_door[self.rooms[i].doorPos] = False
                    else:
                        self.target_door[self.rooms[i].doorPos] = True
        
    def get_world_state(self):
        #return simply if the door is open (case none) or close (case door)
        if not self.multiple_goal:
            return  WorldSate.open_door if self.grid.get(*(self.splitIdx,self.doorIdx)) is None else WorldSate.closed_door
        else:
            world_state = [WorldSate.closed_door1, WorldSate.closed_door2]
            for i, room in enumerate(self.rooms):
                if self.grid.get(*room.doorPos) is None and i == 0:
                    world_state[i] = WorldSate.open_door1
                if self.grid.get(*room.doorPos) is None and i == 1:
                    world_state[i] = WorldSate.open_door2
        
            return tuple(world_state) 
    
    def state_dynamic_human(self, previous_state, action_human):
        self.set_state(previous_state)
        transition = self.get_transition_probs(action_human, cost_value=1)
        for (_,_,state_prime) in transition:
                return True, state_prime
        #print(previous_state)
        return False, previous_state      
    
    def world_dynamic_update(self, action, current_world=None):
        world_prime = None
        if current_world == None:
            current_world = self.get_world_state()
        if not self.multiple_goal:
            if action == ActionsAgent2.take_key and current_world == WorldSate.closed_door :
                world_prime = WorldSate.open_door
            if action == ActionsAgent2.take_key and current_world == WorldSate.open_door :
                world_prime = WorldSate.open_door
            if action == ActionsAgent2.nothing:
                world_prime = current_world
        else:
            if action == ActionsAgent2.take_key1 and current_world[0] == WorldSate.closed_door1:
                world_prime = (WorldSate.open_door1, current_world[1])
                #world_prime = (WorldSate.open_door1, WorldSate.closed_door2)
            elif action == ActionsAgent2.take_key2 and current_world[1] == WorldSate.closed_door2:
                world_prime = (current_world[0], WorldSate.open_door2)
                #world_prime = (WorldSate.closed_door1, WorldSate.open_door2)
            elif action == ActionsAgent2.nothing:
                world_prime = current_world
            else:
                world_prime = current_world
        return world_prime
    # def world_dynamic_update(self, action, current_world=None):
    #     """
    #     Update the world state dynamically based on the action performed.

    #     Parameters:
    #         action (ActionsAgent2): The action that triggers the world state update.
    #         current_world: The current world state. If None, fetches the current world state.

    #     Returns:
    #         Updated world state.
    #     """
    #     if current_world is None:
    #         current_world = self.get_world_state()

    #     # Initialize the updated world state
    #     world_prime = current_world

    #     # Handle single goal scenario
    #     if not self.multiple_goal:
    #         if action == ActionsAgent2.take_key and current_world == WorldSate.closed_door:
    #             world_prime = WorldSate.open_door
    #         elif action == ActionsAgent2.nothing:
    #             world_prime = current_world

    #     # Handle multiple goal scenario
    #     else:
    #         if isinstance(current_world, tuple):  # Ensure the current world state is iterable
    #             room_states = [WorldSate.closed_door1, WorldSate.closed_door2]
    #             updated_states = list(room_states)

    #             for idx, state in enumerate(room_states):
    #                 if action == getattr(ActionsAgent2, f"take_key{idx+1}") and current_world[idx] == state:
    #                     updated_states[idx] = getattr(WorldSate, f"open_door{idx+1}")

    #             world_prime = tuple(updated_states)
    #         elif action == ActionsAgent2.nothing:
    #             world_prime = current_world

    #     return world_prime
        
    def get_transition_probs(self, action=None, cost_value=0):
        probs = []
        next_state, reward = self.check_move(action=action, w=None, cost_value=cost_value)
        probs.append((self.obey_prob, reward, next_state))
        return probs
    
    def get_transition_probsA2(self, w, action=None, cost_value=0):
        probs = []
        next_state, reward = self.check_move(action=action, w=w, cost_value=cost_value)
        probs.append((self.obey_prob, reward, next_state))
        return probs

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]
    
    def set_env_to_goal(self, goal):
        self.goal_pose = list()
        if goal == GoalState.green_goal:
            self.goal_pose.append(self.goal_[0])
        elif goal == GoalState.red_goal:
            self.goal_pose.append(self.goal_[1])
    
    @staticmethod
    def _gen_mission():
        return "get to the green goal square"


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width , height)
        self.num_goal = 2
        goal_X_1 = width - 5
        goal_Y_1 = height - 5
        
        goal_X_2 = width - 2
        goal_Y_2 = height - 2
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)


        # Place a goal square in the bottom-right corner
        
        #g1 = Goal()
        #g1.change_color("red")
        #self.put_obj(g1, goal_X_2, goal_Y_2)
        #self.goal_pose.append(np.array([goal_X_2, goal_Y_2]))
        if not self.multiple_goal:
            self.put_obj(Goal(), goal_X_2, goal_Y_2)
            self.goal_pose.append(np.array([goal_X_2, goal_Y_2]))
            if self.door:
                # Extension to add wall and door around the goal\
                # Comment if not necessary  
                #self.splitIdx = self._rand_int(2, width - 2)
                self.splitIdx = 5
                self.grid.vert_wall(self.splitIdx, 0)
                # Place a door in the wall
                #self.doorIdx = self._rand_int(1, width - 2)
                self.doorIdx = 5
                self.target_door = {(self.splitIdx, self.doorIdx):  True}
                self.put_obj(Door("yellow", is_locked=True), self.splitIdx, self.doorIdx)
            else:
                pass
        else :
            self.rooms = []
            # Extension to add wall and door around the goal\
            # Comment if not necessary  
            rWallIdx = width // 2 + 2
            for j in range(0, height):
                self.grid.set(rWallIdx, j, Wall())
            
            # Room splitting walls
            for n in range(0, 2):
                j = n * (height // 2)
                for i in range(rWallIdx, width):
                    self.grid.set(i, j, Wall())

                roomW = 5 + 1
                roomH = height // 3 + 1
                self.rooms.append(
                    LockedRoom((rWallIdx, j), (roomW, roomH), (rWallIdx, j + 4))
                )
                self.target_door[((rWallIdx, j + 4))] = True
                self.put_obj(Door("yellow", is_locked=True), rWallIdx, j+4)
                #self.grid.set(self.rooms[n].doorPos[0], self.rooms[n].doorPos[1], None)
                goalPos = self.rooms[n].rand_pos(self)
                if n == 1: 
                    goal = Goal()
                    goal.change_color("red")
                    #self.grid.set(*goalPos, goal)
                    if self.size == 16:
                        self.grid.set(14,7, goal)
                        self.goal_.append(np.array([14,7]))
                    elif self.size == 12:
                        self.grid.set(10,1, goal)
                        self.goal_.append(np.array([10,1]))
                    elif self.size == 10:
                        self.grid.set(8,1, goal)
                        self.goal_.append(np.array([8,1]))
                    elif self.size == 32:
                        self.grid.set(30,1, goal)
                        self.goal_.append(np.array([30,1]))
                        
                else:
                    #self.grid.set(*goalPos, Goal())
                    if self.size == 16:
                        self.grid.set(14,14, Goal())
                        self.goal_.append(np.array([14,14]))
                    elif self.size == 12:
                        self.grid.set(10,10, Goal())
                        self.goal_.append(np.array([10,10]))
                    elif self.size == 10:
                        self.grid.set(8,8, Goal())
                        self.goal_.append(np.array([8,8]))
                    elif self.size == 32:
                        self.grid.set(30,30, Goal())
                        self.goal_.append(np.array([30,30]))
                #self.goal_.append(np.array([*goalPos]))
            
            
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
            
        self.i = self.agent_pos[0]
        self.j = self.agent_pos[1]

        self.mission = "get to somewhere"