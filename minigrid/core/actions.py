# Enumeration of possible actions
from __future__ import annotations

from enum import IntEnum

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
    take_key = 1


class WorldSate(IntEnum):
    if Number_room == 1:
        open_door = 0
        closed_door = 1
    elif Number_room == 2:
        open_door1 = 0
        closed_door1 = 1
        open_door2 = 2
        closed_door2 = 3
        
        
class GoalState(IntEnum):
    green_goal = 0
    red_goal = 1