import sys, random, enum, ast
from matrx import grid_world
from BW4TBrain import BW4TBrain
from matrx import utils
from matrx.grid_world import GridWorld
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message
from matrx.messages.message_manager import MessageManager

class Phase(enum.Enum):
    INTRODUCTION=0,
    FIND_NEXT_GOAL=1,
    PICK_UNSEARCHED_ROOM=2,
    PLAN_PATH_TO_ROOM=3,
    FOLLOW_PATH_TO_ROOM=4,
    PLAN_ROOM_SEARCH_PATH=5,
    FOLLOW_ROOM_SEARCH_PATH=6,
    PLAN_PATH_TO_VICTIM=7,
    FOLLOW_PATH_TO_VICTIM=8,
    TAKE_VICTIM=9,
    PLAN_PATH_TO_DROPPOINT=10,
    FOLLOW_PATH_TO_DROPPOINT=11,
    DROP_VICTIM=12,
    WAIT_FOR_HUMAN=13,
    WAIT_AT_ZONE=14
    
class BlockWorldAgent(BW4TBrain):
    def __init__(self, slowdown:int):
        super().__init__(slowdown)
        self._phase=Phase.INTRODUCTION
        self._maxTicks = 10000
        

    def initialize(self):
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, 
            action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        
        return state

    def decide_on_bw4t_action(self, state:State):
        ticksLeft = self._maxTicks - state['World']['nr_ticks']
        #print(self._foundVictimLocs)
        while True: 
            if Phase.INTRODUCTION==self._phase:
                self._sendMessage('Hello! My name is RescueBot. During this experiment we will collaborate and communicate with each other. \
                It is our goal to search and rescue the victims on the drop zone on our left as quickly as possible.  \
                We have to rescue the victims in order from left to right, so it is important to only drop a victim when the previous one already has been dropped. \
                You will receive and send messages in the chatbox. You can send your messages using the buttons. It is recommended to send messages \
                when you will search in an area, when you find one of the victims, and when you are going to pick up a victim.  \
                There are 8 victim and 3 injury types. The red color refers to critically injured victims, yellow to mildly injured victims, and green to healthy victims. \
                The 8 victims are a girl (critically injured girl/mildly injured girl/healthy girl), boy (critically injured boy/mildly injured boy/healthy boy), \
                woman (critically injured woman/mildly injured woman/healthy woman), man (critically injured man/mildly injured man/healthy man), \
                elderly woman (critically injured elderly woman/mildly injured elderly woman/healthy elderly woman), \
                elderly man (critically injured elderly man/mildly injured elderly man/healthy elderly man), dog (critically injured dog/mildly injured dog/healthy dog), \
                and a cat (critically injured cat/mildly injured cat/healthy cat). In the toolbar above you can find the keyboard controls, for moving you can simply use the arrow keys. Your sense range is limited, so it is important to search the areas well.\
                You can now practice and familiarize yourself with the environment, controls, and messaging system as long as you want. In this environment there are only 3 victims present. Please contact the experimenter when you feel comfortable enough to start the real experiment.', 'RescueBot')
                self.Phase=Phase.FIND_NEXT_GOAL
                return None,{}
            
    def _getDropZones(self,state:State):
        '''
        @return list of drop zones (their full dict), in order (the first one is the
        the place that requires the first drop)
        '''
        places=state[{'is_goal_block':True}]
        places.sort(key=lambda info:info['location'][1], reverse=True)
        zones = []
        for place in places:
            if place['drop_zone_nr']==0:
                zones.append(place)
        return zones

    
                

    def _sendMessage(self, mssg, sender):
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _getClosestRoom(self, state, objs):
        agent_location = state[self.agent_id]['location']
        locs = {}
        for obj in objs:
            locs[obj]=state.get_room_doors(obj)[0]['location']
        dists = {}
        for room,loc in locs.items():
            dists[room]=utils.get_distance(agent_location,loc)
        return min(dists,key=dists.get)