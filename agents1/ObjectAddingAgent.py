import sys, random, enum, ast, time, csv
import numpy as np
from brains1.ArtificialBrain import ArtificialBrain
from actions1.CustomActions import *
from matrx import utils
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject, RemoveObject
from matrx.actions.move_actions import MoveNorth
from matrx.messages.message import Message
from matrx.messages.message_manager import MessageManager
from actions1.CustomActions import RemoveObjectTogether, CarryObjectTogether, DropObjectTogether, CarryObject, Drop, AddObject

class ObjectAddingAgent(ArtificialBrain):
    def __init__(self, slowdown, condition):
        super().__init__(slowdown, condition)
        # Initialization of some relevant variables
        self._slowdown = slowdown
        self._condition = condition

    def initialize(self):
        # Initialization of the state tracker and navigation algorithm
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_observations(self, state):
        # Filtering of the world state before deciding on an action 
        return state

    def decide_on_actions(self, state):
        self._tick = state['World']['nr_ticks']
        add_objects = []
        action = None
        action_kwargs = {}
        #if self._tick == 1000:
        if self._tick == 200:
            for loc in [(6,6),(6,7),(6,8),(6,9),(6,10),(6,11),(12,12),(12,13),(12,14),(12,15),(12,16),(12,17),(12,18)]:
                obj_kwargs = {}
                obj_kwargs['location'] = loc
                obj_kwargs['img_name'] = "/images/pool20.svg"
                add_objects+=[obj_kwargs]
            action_kwargs["add_objects"] = add_objects
            return AddObject.__name__, action_kwargs
        
        #if self._tick == 1900:
        if self._tick == 350:
            for loc in [(5,5),(4,5),(3,5),(2,5),(1,5),(13,6),(14,6),(15,6),(16,6),(17,6),(18,6),(19,6)]:
                obj_kwargs = {}
                obj_kwargs['location'] = loc
                obj_kwargs['img_name'] = "/images/lake2.svg"
                add_objects+=[obj_kwargs]
            action_kwargs["add_objects"] = add_objects
            return AddObject.__name__, action_kwargs
        
        #if self._tick == 2800:
        if self._tick == 500:
            for loc in [(20,9),(21,17),(22,17),(23,17)]:
                obj_kwargs = {}
                obj_kwargs['location'] = loc
                obj_kwargs['img_name'] = "/images/lake2.svg"
                add_objects+=[obj_kwargs]
            action_kwargs["add_objects"] = add_objects
            return AddObject.__name__, action_kwargs
        
        #if self._tick == 2802:
        if self._tick == 502:
            for loc in [(20,9),(20,10),(20,11),(20,12),(20,13),(20,14),(20,15),(20,16)]:
                obj_kwargs = {}
                obj_kwargs['location'] = loc
                obj_kwargs['img_name'] = "/images/pool20.svg"
                add_objects+=[obj_kwargs]
            action_kwargs["add_objects"] = add_objects
            return AddObject.__name__, action_kwargs
        
        else: 
            return None, {}