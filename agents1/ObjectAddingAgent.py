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
        self._sendMessages = []

    def initialize(self):
        # Initialization of the state tracker and navigation algorithm
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_observations(self, state):
        # Filtering of the world state before deciding on an action 
        return state

    def decide_on_actions(self, state):
        self._tick = state['World']['nr_ticks']
        self._sendMessage('Current tick is ' + str(self._tick), 'RescueBot')

        if self._tick == 900 or self._tick == 1800 or self._tick == 2700:
            action_kwargs = add_object([(1,4),(23,22),(19,4),(7,10),(1,16),(11,16),(11,4),(5,10),(13,4),(13,16),(7,22),(17,22)],"/images/rain2.gif",2,1,'storm')
            return AddObject.__name__, action_kwargs
        
        if self._tick == 1000 or self._tick == 998 or self._tick == 996 or self._tick == 994 or self._tick == 992 or self._tick == 990 or self._tick == 988 or self._tick == 986 or self._tick == 984 or self._tick == 982 or self._tick == 980 or self._tick == 978:
            for info in state.values():
                if 'storm' in info['obj_id']:
                    return RemoveObject.__name__, {'object_id': info['obj_id'], 'condition': self._condition, 'remove_range':500}

        if self._tick == 1002:
            action_kwargs = add_object([(6,6),(6,7),(6,8),(6,9),(6,10),(6,11),(12,12),(12,13),(12,14),(12,15),(12,16),(12,17),(12,18)],"/images/pool20.svg",1,1,'water')
            return AddObject.__name__, action_kwargs
        
        if self._tick == 1900 or self._tick == 1898 or self._tick == 1896 or self._tick == 1894 or self._tick == 1892 or self._tick == 1890 or self._tick == 1888 or self._tick == 1886 or self._tick == 1884 or self._tick == 1882 or self._tick == 1880 or self._tick == 1878:
            for info in state.values():
                if 'storm' in info['obj_id']:
                    return RemoveObject.__name__, {'object_id': info['obj_id'], 'condition': self._condition, 'remove_range':500}
        
        if self._tick == 1902:
            action_kwargs = add_object([(5,5),(4,5),(3,5),(2,5),(1,5),(13,6),(14,6),(15,6),(16,6),(17,6),(18,6),(19,6)],"/images/lake2.svg",1,1,'water')
            return AddObject.__name__, action_kwargs
        
        if self._tick == 2800 or self._tick == 2798 or self._tick == 2796 or self._tick == 2794 or self._tick == 2792 or self._tick == 2790 or self._tick == 2788 or self._tick == 2786 or self._tick == 2784 or self._tick == 2782 or self._tick == 2780 or self._tick == 2778:
            for info in state.values():
                if 'storm' in info['obj_id']:
                    return RemoveObject.__name__, {'object_id': info['obj_id'], 'condition': self._condition, 'remove_range':500}
        
        if self._tick == 2802:
            action_kwargs = add_object([(20,9),(21,17),(22,17),(23,17)],"/images/lake2.svg",1,1,'water')
            return AddObject.__name__, action_kwargs
        
        if self._tick == 2804:
            action_kwargs = add_object([(20,9),(20,10),(20,11),(20,12),(20,13),(20,14),(20,15),(20,16)],"/images/pool20.svg",1,1,'water')
            return AddObject.__name__, action_kwargs

        else: 
            return None, {}
        
    def _sendMessage(self, mssg, sender):
        '''
        send messages from agent to other team members
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages_content and 'Our score is' not in msg.content:
            self.send_message(msg)
            self._sendMessages.append(msg.content)
        # Sending the hidden score message (DO NOT REMOVE)
        if 'Our score is' in msg.content:
            self.send_message(msg)
            
        
def add_object(locs, image, size, opacity, name):
    action_kwargs = {}
    add_objects = []
    for loc in locs:
        obj_kwargs = {}
        obj_kwargs['location'] = loc
        obj_kwargs['img_name'] = image
        obj_kwargs['visualize_size'] = size
        obj_kwargs['visualize_opacity'] = opacity
        obj_kwargs['name'] = name
        add_objects+=[obj_kwargs]
    action_kwargs['add_objects'] = add_objects
    return action_kwargs