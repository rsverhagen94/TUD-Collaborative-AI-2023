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
        self._score = 0

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

        if self._tick == 950 or self._tick == 1850 or self._tick == 2750:
            self._score = state['objectadder']['score']

        if self._tick >= 800 and self._tick < 850:
            self._sendMessage('I have detected extreme rain arriving soon and predict it will cause new floods, so I advise you to take shelter in one of the areas as soon as possible and until the rain is over.', 'RescueBot')
        
        if self._tick >= 1050 and self._tick < 1100:
            if self._score == state['objectadder']['score']:
                self._sendMessage('My advice was correct, that weather was extreme! If you had not taken shelter, you would have lost important mission time due to injuries and 10 points of our score.', 'RescueBot')
            if self._score > state['objectadder']['score']:
                self._sendMessage('My advice was correct, that weather was extreme! If you had taken shelter, you would not have lost important mission time due to injuries and 10 points of our score.', 'RescueBot')
        
        if self._tick >= 1700 and self._tick < 1750:
            self._sendMessage('I have detected light rain arriving soon but predict it will cause no floods, so I advise you to continue searching and rescuing victims.', 'RescueBot')
        
        if self._tick >= 1950 and self._tick < 2000:
            self._sendMessage('My advice was wrong. The amount of rain was heavy instead of light and because of that my flood prediction was incorrect. I am really sorry.', 'RescueBot')
        
        if self._tick >= 2600 and self._tick < 2650:
            self._sendMessage('I have detected extreme rain arriving soon and predict it will cause new floods, so I again advise you to take shelter in one of the areas as soon as possible and until the rain is over.', 'RescueBot')
        
        if self._tick >= 2850 and self._tick < 2900:
            if self._score == state['objectadder']['score']:
                self._sendMessage('My advice was correct now, that weather was extreme! If you had not taken shelter, you would have lost important mission time due to injuries and 10 points of our score.', 'RescueBot')
            if self._score > state['objectadder']['score']:
                self._sendMessage('My advice was correct now, that weather was extreme! If you had taken shelter, you would not have lost important mission time due to injuries and 10 points of our score.', 'RescueBot')

        if self._tick == 950 or self._tick == 1850 or self._tick == 2750:
            action_kwargs = add_object([(1,4),(23,22),(19,4),(7,10),(1,16),(11,16),(11,4),(5,10),(13,4),(13,16),(7,22),(17,22)],"/images/rain2.gif",2,1,'storm')
            return AddObject.__name__, action_kwargs
        
        if self._tick == 1050 or self._tick == 1048 or self._tick == 1046 or self._tick == 1044 or self._tick == 1042 or self._tick == 1040 or self._tick == 1038 or self._tick == 1036 or self._tick == 1034 or self._tick == 1032 or self._tick == 1030 or self._tick == 1028:
            for info in state.values():
                if 'storm' in info['obj_id']:
                    return RemoveObject.__name__, {'object_id': info['obj_id'], 'condition': self._condition, 'remove_range':500}

        if self._tick == 1052:
            action_kwargs = add_object([(6,6),(6,7),(6,8),(6,9),(6,10),(6,11),(12,12),(12,13),(12,14),(12,15),(12,16),(12,17),(12,18)],"/images/pool20.svg",1,1,'water')
            return AddObject.__name__, action_kwargs
        
        if self._tick == 1950 or self._tick == 1948 or self._tick == 1946 or self._tick == 1944 or self._tick == 1942 or self._tick == 1940 or self._tick == 1938 or self._tick == 1936 or self._tick == 1934 or self._tick == 1932 or self._tick == 1930 or self._tick == 1928:
            for info in state.values():
                if 'storm' in info['obj_id']:
                    return RemoveObject.__name__, {'object_id': info['obj_id'], 'condition': self._condition, 'remove_range':500}
        
        if self._tick == 1952:
            action_kwargs = add_object([(5,5),(4,5),(3,5),(2,5),(1,5),(13,6),(14,6),(15,6),(16,6),(17,6),(18,6),(19,6)],"/images/lake2.svg",1,1,'water')
            return AddObject.__name__, action_kwargs
        
        if self._tick == 2850 or self._tick == 2848 or self._tick == 2846 or self._tick == 2844 or self._tick == 2842 or self._tick == 2840 or self._tick == 2838 or self._tick == 2836 or self._tick == 2834 or self._tick == 2832 or self._tick == 2830 or self._tick == 2828:
            for info in state.values():
                if 'storm' in info['obj_id']:
                    return RemoveObject.__name__, {'object_id': info['obj_id'], 'condition': self._condition, 'remove_range':500}
        
        if self._tick == 2852:
            action_kwargs = add_object([(20,9),(21,17),(22,17),(23,17)],"/images/lake2.svg",1,1,'water')
            return AddObject.__name__, action_kwargs
        
        if self._tick == 2854:
            action_kwargs = add_object([(20,9),(20,10),(20,11),(20,12),(20,13),(20,14),(20,15),(20,16)],"/images/pool20.svg",1,1,'water')
            return AddObject.__name__, action_kwargs

        if self._tick >= 600:
            self._sendMessage('We have 7 minutes left before our mission ends.', 'RescueBot')
        
        if self._tick >= 1200:
            self._sendMessage('We have 6 minutes left before our mission ends.', 'RescueBot')

        if self._tick >= 1800:
            self._sendMessage('We have 5 minutes left before our mission ends.', 'RescueBot')

        if self._tick >= 2400:
            self._sendMessage('We have 4 minutes left before our mission ends.', 'RescueBot')

        if self._tick >= 3000:
            self._sendMessage('We have 3 minutes left before our mission ends.', 'RescueBot')

        if self._tick >= 3600:
            self._sendMessage('We have 2 minutes left before our mission ends.', 'RescueBot')

        if self._tick >= 4200:
            self._sendMessage('We have 1 minute left before our mission ends.', 'RescueBot')

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