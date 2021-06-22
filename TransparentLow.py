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
        self._roomVics = []
        self._searchedRooms = []
        self._foundVictims = []
        self._collectedVictims = []
        self._foundVictimLocs = {}
        self._maxTicks = 10000
        

    def initialize(self):
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, 
            action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        self._processMessages(state)
        return state

    def decide_on_bw4t_action(self, state:State):
        ticksLeft = self._maxTicks - state['World']['nr_ticks']
        #print(self._foundVictimLocs)
        while True: 
            if Phase.INTRODUCTION==self._phase:
                self._sendMessage('Hello! My name is RescueBot. Together we will collaborate and try to search and rescue the 8 victims on our left as quickly as possible. \
                We have to rescue the 8 victims in order from left to right, so it is important to only drop a victim when the previous one already has been dropped. \
                Unfortunately, I am not allowed to carry the critically injured victims critically injured elderly woman and critically injured man. \
                Moreover, I am not able to distinguish between critically injured girl and critically injured boy or mildly injured girl and mildly injured boy. \
                We have X minutes to successfully collect all 8 victims in the correct order. \
                If you understood everything I just told you, please type yes. We will then start our mission!', 'RescueBot')
                if self.received_messages and self.received_messages[-1]=='yes':
                    self._phase=Phase.FIND_NEXT_GOAL
                else:
                    return None,{}

            if Phase.FIND_NEXT_GOAL==self._phase:
                zones = self._getDropZones(state)
                locs = [zone['location'] for zone in zones]
                self._firstVictim = str(zones[0]['img_name'])[8:-4]
                remainingZones = []
                for info in zones:
                    if str(info['img_name'])[8:-4] not in self._collectedVictims:
                        remainingZones.append(info)
                if remainingZones:
                    self._goalVic = str(remainingZones[0]['img_name'])[8:-4]
                    self._goalLoc = remainingZones[0]['location']
                else:
                    return None,{}
                if self._goalVic not in self._foundVictims:
                    self._sendMessage('Next victim to rescue: ' + self._goalVic ,'RescueBot')
                    self._phase=Phase.PICK_UNSEARCHED_ROOM
                if self._goalVic in self._foundVictims and 'location' in self._foundVictimLocs[self._goalVic].keys():
                    self._phase=Phase.PLAN_PATH_TO_VICTIM
                    if self._collectedVictims:
                        self._sendMessage('Next victim to rescue is' + self._goalVic + ' in ' + self._foundVictimLocs[self._goalVic]['room'] ,'RescueBot')
                if self._goalVic in self._foundVictims and 'location' not in self._foundVictimLocs[self._goalVic].keys():
                    self._phase=Phase.PLAN_PATH_TO_ROOM
                    if self._collectedVictims:
                        self._sendMessage('Next victim to rescue is' + self._goalVic + ' in ' + self._foundVictimLocs[self._goalVic]['room'] ,'RescueBot')

            if Phase.PICK_UNSEARCHED_ROOM==self._phase:
                unsearchedRooms=[room['room_name'] for room in state.values()
                if 'class_inheritance' in room
                and 'Door' in room['class_inheritance']
                and room['room_name'] not in self._searchedRooms]
                self._door = state.get_room_doors(self._getClosestRoom(state,unsearchedRooms))[0]
                #self._sendMessage('Areas still to search: ' + ', '.join([i.split()[-1] for i in unsearchedRooms]), 'RescueBot')
                self._phase = Phase.PLAN_PATH_TO_ROOM

            if Phase.PLAN_PATH_TO_ROOM==self._phase:
                self._navigator.reset_full()
                if self._goalVic in self._foundVictims and 'location' not in self._foundVictimLocs[self._goalVic].keys():
                    print('TEST')
                    self._door = state.get_room_doors(self._foundVictimLocs[self._goalVic]['room'])[0]
                    doorLoc = self._door['location']
                else:
                    doorLoc = self._door['location']
                self._navigator.add_waypoints([doorLoc])
                self._currentTick = state['World']['nr_ticks']
                self._phase=Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM==self._phase:
                if self._goalVic in self._collectedVictims:
                    self._phase=Phase.FIND_NEXT_GOAL
                else:
                    self._state_tracker.update(state)
                    if state['World']['nr_ticks'] > self._currentTick + 10:
                        self._sendMessage('Moving to ' + str(self._door['room_name']), 'RescueBot')
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        return action,{}
                    self._phase=Phase.PLAN_ROOM_SEARCH_PATH

            if Phase.PLAN_ROOM_SEARCH_PATH==self._phase:
                roomTiles = [info['location'] for info in state.values()
                    if 'class_inheritance' in info 
                    and 'AreaTile' in info['class_inheritance']
                    and 'room_name' in info
                    and info['room_name'] == self._door['room_name']
                ]
                self._roomtiles=roomTiles     
                self._navigator.reset_full()
                self._navigator.add_waypoints(roomTiles)
                self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH

            if Phase.FOLLOW_ROOM_SEARCH_PATH==self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    self._sendMessage('Searching through whole ' + str(self._door['room_name']), 'RescueBot')
                    for info in state.values():
                        if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance']:
                            vic = str(info['img_name'][8:-4])
                            if vic not in self._roomVics:
                                self._roomVics.append(vic)

                            if vic in self._foundVictims and 'location' not in self._foundVictimLocs[vic].keys():
                                self._foundVictimLocs[vic] = {'location':info['location'],'room':self._door['room_name'],'obj_id':info['obj_id']}
                                self._sendMessage('Found '+ vic + ' in ' + self._door['room_name'], 'RescueBot')# + ' like you said', 'RescueBot')
                                self._phase=Phase.FIND_NEXT_GOAL

                            if 'healthy' not in vic and vic not in self._foundVictims:
                                self._foundVictims.append(vic)
                                self._foundVictimLocs[vic] = {'location':info['location'],'room':self._door['room_name'],'obj_id':info['obj_id']}
                                self._sendMessage('Found '+ vic + ' in ' + self._door['room_name'], 'RescueBot')
                    return action,{}
                if self._goalVic not in self._foundVictims:
                    self._sendMessage(self._goalVic.capitalize() + ' not present in ' + str(self._door['room_name']), 'RescueBot')
                if self._goalVic in self._foundVictims and self._goalVic not in self._roomVics and self._foundVictimLocs[self._goalVic]['room']!=self._door['room_name']:
                    self._sendMessage(self._goalVic.capitalize() + ' not present in ' + str(self._door['room_name']), 'RescueBot')
                    self._foundVictimLocs.pop(self._goalVic, None)
                    self._foundVictims.remove(self._goalVic)
                    self._roomVics = []  
                self._searchedRooms.append(self._door['room_name'])
                self._phase=Phase.FIND_NEXT_GOAL
                
            if Phase.PLAN_PATH_TO_VICTIM==self._phase:
                self._sendMessage('Picking up ' + self._goalVic + ' in ' + self._foundVictimLocs[self._goalVic]['room'], 'RescueBot')
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._foundVictimLocs[self._goalVic]['location']])
                self._phase=Phase.FOLLOW_PATH_TO_VICTIM
                    
            if Phase.FOLLOW_PATH_TO_VICTIM==self._phase:
                if self._goalVic in self._collectedVictims:
                    self._phase=Phase.FIND_NEXT_GOAL
                else:
                    self._state_tracker.update(state)
                    action=self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        return action,{}
                    self._phase=Phase.TAKE_VICTIM

            if Phase.TAKE_VICTIM==self._phase:
                self._phase=Phase.PLAN_PATH_TO_DROPPOINT
                self._collectedVictims.append(self._goalVic)
                return GrabObject.__name__,{'object_id':self._foundVictimLocs[self._goalVic]['obj_id']}

            if Phase.PLAN_PATH_TO_DROPPOINT==self._phase:
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._goalLoc])
                self._phase=Phase.FOLLOW_PATH_TO_DROPPOINT

            if Phase.FOLLOW_PATH_TO_DROPPOINT==self._phase:
                self._sendMessage('Transporting '+ self._goalVic + ' to the drop zone', 'RescueBot')
                self._state_tracker.update(state)
                action=self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action,{}
                self._phase=Phase.DROP_VICTIM

            if Phase.DROP_VICTIM == self._phase:
                if state[{'is_collectable':True}] or self._goalVic==self._firstVictim:
                    self._sendMessage('Delivered '+ self._goalVic + ' at the drop zone', 'RescueBot')
                    self._phase=Phase.FIND_NEXT_GOAL
                    return DropObject.__name__,{}
                if not state[{'is_collectable':True}] and self._goalVic!=self._firstVictim:
                    self._sendMessage('Waiting for human operator at drop zone', 'RescueBot')
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

    def _processMessages(self, state):
        '''
        process incoming messages. 
        Reported blocks are added to self._blocks
        '''
        for msg in self.received_messages:
            if msg.startswith("Search:"):
                area = 'area '+ msg.split()[-1]
                if area not in self._searchedRooms:
                    self._searchedRooms.append(area)
            if msg.startswith("Found:"):
                if len(msg.split()) == 6:
                    foundVic = ' '.join(msg.split()[1:4])
                else:
                    foundVic = ' '.join(msg.split()[1:5]) 
                loc = 'area '+ msg.split()[-1]
                if foundVic not in self._foundVictims:
                    self._foundVictims.append(foundVic)
                    self._foundVictimLocs[foundVic] = {'room':loc}
            if msg.startswith('Collect:'):
                if len(msg.split()) == 6:
                    collectVic = ' '.join(msg.split()[1:4])
                else:
                    collectVic = ' '.join(msg.split()[1:5]) 
                if collectVic not in self._collectedVictims:
                    self._collectedVictims.append(collectVic)
                    #self.received_messages = []
                ##if collectedVictim==self._goalVic:
                 #   self._sendMessage('Copy that, switching to next victim to rescue', 'RescueBot')
                

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