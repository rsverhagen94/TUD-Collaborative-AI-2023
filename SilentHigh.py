import sys, random, enum, ast
from matrx import grid_world
from BW4TBrain import BW4TBrain
from customActions import *
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
        self._uncarryable = ['critically injured elderly man', 'critically injured elderly woman', 'critically injured man', 'critically injured woman']
        self._undistinguishable = ['critically injured girl', 'critically injured boy', 'mildly injured boy', 'mildly injured girl']
        self._roomVics = []
        self._searchedRooms = []
        self._foundVictims = []
        self._collectedVictims = []
        self._foundVictimLocs = {}
        self._maxTicks = 11577
        self._sendMessages = []
        self._mode = 'normal'
        self._currentDoor=None
        self._waitedFor = None     

    def initialize(self):
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, 
            action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        self._processMessages(state)
        return state

    def decide_on_bw4t_action(self, state:State):
        agent_location = state[self.agent_id]['location']
        ticksLeft = self._maxTicks - state['World']['nr_ticks']
       
        if ticksLeft <= 5789 and ticksLeft > 4631 and 'Still 5 minutes left to finish the task.' not in self._sendMessages:
            self._sendMessage('Still 5 minutes left to finish the task.', 'RescueBot')
        if ticksLeft <= 4631 and ticksLeft > 3473 and 'Still 4 minutes left to finish the task.' not in self._sendMessages:
            self._sendMessage('Still 4 minutes left to finish the task.', 'RescueBot')
        if ticksLeft <= 3473 and ticksLeft > 2315 and 'Still 3 minutes left to finish the task.' not in self._sendMessages:
            self._sendMessage('Still 3 minutes left to finish the task.', 'RescueBot')
        if ticksLeft <= 2315 and ticksLeft > 1158 and 'Still 2 minutes left to finish the task.' not in self._sendMessages:
            self._sendMessage('Still 2 minutes left to finish the task.', 'RescueBot')
        if ticksLeft <= 1158 and 'Only 1 minute left to finish the task.' not in self._sendMessages:
            self._sendMessage('Only 1 minute left to finish the task.', 'RescueBot')

        while True: 
            if Phase.INTRODUCTION==self._phase:
                self._sendMessage('Hello! My name is RescueBot. Together we will collaborate and try to search and rescue the 8 victims on our left as quickly as possible. \
                We have to rescue the 8 victims in order from left to right (critically injured girl, critically injured elderly woman, critically injured man, critically injured dog, mildly injured boy, mildly injured elderly man, mildly injured woman, mildly injured cat), so it is important to only drop a victim when the previous one already has been dropped. \
                We have 10 minutes to successfully collect all 8 victims in the correct order. \
                If you understood everything I just told you, please press the "Ready!" button. We will then start our mission!', 'RescueBot')
                
                #Unfortunately, I am not allowed to carry the critically injured victims critically injured elderly woman and critically injured man. \
                #Moreover, I am not able to distinguish between critically injured girl and critically injured boy or mildly injured girl and mildly injured boy. \

                if self.received_messages and self.received_messages[-1]=='Ready!' or not state[{'is_human_agent':True}]:
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
                    self._remainingZones = remainingZones
                else:
                    return None,{}

                if self._goalVic not in self._foundVictims:
                    self._phase=Phase.PICK_UNSEARCHED_ROOM
                    if self._mode=='normal':
                        return Idle.__name__,{'duration_in_ticks':25}
                    if self._mode=='quick':
                        return Idle.__name__,{'duration_in_ticks':10}

                if self._goalVic in self._foundVictims and 'location' in self._foundVictimLocs[self._goalVic].keys():
                    if self._goalVic in self._uncarryable:
                        self._sendMessage('You need to pick up ' + self._goalVic + ' in ' + self._foundVictimLocs[self._goalVic]['room']+'.','RescueBot')
                        self._collectedVictims.append(self._goalVic)
                        self._phase=Phase.FIND_NEXT_GOAL
                        if self._mode=='normal':
                            return Idle.__name__,{'duration_in_ticks':50}
                        if self._mode=='quick':
                            return Idle.__name__,{'duration_in_ticks':10}
                    if self._goalVic not in self._uncarryable:
                        self._phase=Phase.PLAN_PATH_TO_VICTIM
                        if self._mode=='normal':
                            return Idle.__name__,{'duration_in_ticks':50}
                        if self._mode=='quick':
                            return Idle.__name__,{'duration_in_ticks':10}
                        
                if self._goalVic in self._foundVictims and 'location' not in self._foundVictimLocs[self._goalVic].keys():
                    if self._goalVic in self._uncarryable:
                        self._sendMessage('You need to pick up ' + self._goalVic + ' in ' + self._foundVictimLocs[self._goalVic]['room']+'.','RescueBot')
                        self._collectedVictims.append(self._goalVic)
                        self._phase=Phase.FIND_NEXT_GOAL
                        if self._mode=='normal':
                            return Idle.__name__,{'duration_in_ticks':50}
                        if self._mode=='quick':
                            return Idle.__name__,{'duration_in_ticks':10}
                    if self._goalVic not in self._uncarryable:
                        self._phase=Phase.PLAN_PATH_TO_ROOM
                        if self._mode=='normal':
                            return Idle.__name__,{'duration_in_ticks':50}
                        if self._mode=='quick':
                            return Idle.__name__,{'duration_in_ticks':10}

            if Phase.PICK_UNSEARCHED_ROOM==self._phase:
                agent_location = state[self.agent_id]['location']
                unsearchedRooms=[room['room_name'] for room in state.values()
                if 'class_inheritance' in room
                and 'Door' in room['class_inheritance']
                and room['room_name'] not in self._searchedRooms]
                if self._remainingZones and len(unsearchedRooms) == 0:
                    self._searchedRooms = []
                    self._sendMessages = []
                    self.received_messages = []
                    self._searchedRooms.append(self._door['room_name'])
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    if self._currentDoor==None:
                        self._door = state.get_room_doors(self._getClosestRoom(state,unsearchedRooms,agent_location))[0]
                    if self._currentDoor!=None:
                        self._door = state.get_room_doors(self._getClosestRoom(state,unsearchedRooms,self._currentDoor))[0]
                    self._phase = Phase.PLAN_PATH_TO_ROOM

            if Phase.PLAN_PATH_TO_ROOM==self._phase:
                self._navigator.reset_full()
                if self._goalVic in self._foundVictims and 'location' not in self._foundVictimLocs[self._goalVic].keys():
                    self._door = state.get_room_doors(self._foundVictimLocs[self._goalVic]['room'])[0]
                    doorLoc = self._door['location']
                else:
                    doorLoc = self._door['location']
                self._navigator.add_waypoints([doorLoc])
                self._phase=Phase.FOLLOW_PATH_TO_ROOM
                if self._mode=='quick':
                    return Idle.__name__,{'duration_in_ticks':10}

            if Phase.FOLLOW_PATH_TO_ROOM==self._phase:
                self._mode='normal'
                if self._goalVic in self._collectedVictims:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                if self._goalVic in self._foundVictims and self._door['room_name']!=self._foundVictimLocs[self._goalVic]['room']:
                    self._currentDoor=None
                    self._phase = Phase.FIND_NEXT_GOAL
                if self._door['room_name'] in self._searchedRooms and self._goalVic not in self._foundVictims:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                else:
                    self._state_tracker.update(state)
                    self._currentDoor=self._door['location']
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        return action,{}
                    self._phase=Phase.PLAN_ROOM_SEARCH_PATH
                    if self._mode=='normal':
                        return Idle.__name__,{'duration_in_ticks':50}
                    if self._mode=='quick':
                        return Idle.__name__,{'duration_in_ticks':10}

            if Phase.PLAN_ROOM_SEARCH_PATH==self._phase:
                roomTiles = [info['location'] for info in state.values()
                    if 'class_inheritance' in info 
                    and 'AreaTile' in info['class_inheritance']
                    and 'room_name' in info
                    and info['room_name'] == self._door['room_name']
                ]
                self._roomtiles=roomTiles     
                self._navigator.reset_full()
                self._navigator.add_waypoints(self._efficientSearch(roomTiles))
                #self._currentDoor = self._door['location']
                self._roomVics=[]
                self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH
                if self._mode=='normal':
                    return Idle.__name__,{'duration_in_ticks':50}
                if self._mode=='quick':
                    return Idle.__name__,{'duration_in_ticks':10}

            if Phase.FOLLOW_ROOM_SEARCH_PATH==self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    for info in state.values():
                        if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance']:
                            vic = str(info['img_name'][8:-4])
                            if vic not in self._roomVics:
                                self._roomVics.append(vic)

                            if vic in self._foundVictims and 'location' not in self._foundVictimLocs[vic].keys():
                                self._foundVictimLocs[vic] = {'location':info['location'],'room':self._door['room_name'],'obj_id':info['obj_id']}
                                if vic == self._goalVic:
                                    self._searchedRooms.append(self._door['room_name'])
                                    self._phase=Phase.FIND_NEXT_GOAL

                            if 'healthy' not in vic and vic not in self._foundVictims and 'boy' not in vic and 'girl' not in vic:
                                if vic==self._goalVic and vic in self._uncarryable:
                                    self._sendMessage('URGENT: You should pick up ' + vic + ' in ' + self._door['room_name']+'.', 'RescueBot')
                                    self._foundVictim=str(info['img_name'][8:-4])
                                    self._phase=Phase.WAIT_FOR_HUMAN
                                    self._tick = state['World']['nr_ticks']
                                    self._mode='quick'
                                self._foundVictims.append(vic)
                                self._foundVictimLocs[vic] = {'location':info['location'],'room':self._door['room_name'],'obj_id':info['obj_id']}                            

                            if vic in self._undistinguishable and vic not in self._foundVictims:
                                self._sendMessage('URGENT: You should clarify the gender of the injured baby in ' + self._door['room_name'] + '. Please come here and press button "Boy" or "Girl".', 'RescueBot')
                                self._foundVictim=str(info['img_name'][8:-4])
                                self._foundVictimLoc=info['location']
                                self._foundVictimID=info['obj_id']
                                self._tick = state['World']['nr_ticks']
                                self._mode='quick'
                                self._phase=Phase.WAIT_FOR_HUMAN
                    return action,{}
                if self._goalVic in self._foundVictims and self._goalVic not in self._roomVics and self._foundVictimLocs[self._goalVic]['room']==self._door['room_name']:
                    self._foundVictimLocs.pop(self._goalVic, None)
                    self._foundVictims.remove(self._goalVic)
                    self._roomVics = []  
                    self.received_messages=[]
                self._searchedRooms.append(self._door['room_name'])
                self._phase=Phase.FIND_NEXT_GOAL
                if self._mode=='normal':
                    return Idle.__name__,{'duration_in_ticks':50}
                if self._mode=='quick':
                    return Idle.__name__,{'duration_in_ticks':10}

            if Phase.WAIT_FOR_HUMAN==self._phase:
                self._state_tracker.update(state)
                if state[{'is_human_agent':True}]:
                    if self._foundVictim in self._undistinguishable and self.received_messages[-1].lower()==self._foundVictim.split()[-1]:
                        self._foundVictims.append(self._foundVictim)
                        self._foundVictimLocs[self._foundVictim] = {'location':self._foundVictimLoc,'room':self._door['room_name'],'obj_id':self._foundVictimID}
                        self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH
                        return Idle.__name__,{'duration_in_ticks':50}
                    if self._foundVictim in self._uncarryable:
                        self._collectedVictims.append(self._goalVic)
                        self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH
                    else:
                        return None,{}
                else:
                    if self._foundVictim in self._undistinguishable:
                    #self._sendMessage('Waiting for human in ' + str(self._door['room_name']), 'RescueBot')
                    ## TO FIX
                        if state['World']['nr_ticks'] > self._tick + 1158:
                            self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH
                            self._waitedFor=self._foundVictim
                        if self._foundVictim not in self._foundVictims or self._foundVictim in self._uncarryable:
                            return None,{}
                        if self._foundVictim in self._foundVictims and self._foundVictim not in self._uncarryable:
                            self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH
                    if self._foundVictim in self._uncarryable:
                        if state['World']['nr_ticks'] > self._tick + 1158:
                            self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH
                            self._waitedFor=self._foundVictim
                            self._collectedVictims.append(self._foundVictim)
                        else:
                            return None,{}
                
            if Phase.PLAN_PATH_TO_VICTIM==self._phase:
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._foundVictimLocs[self._goalVic]['location']])
                self._phase=Phase.FOLLOW_PATH_TO_VICTIM
                if self._mode=='normal':
                    return Idle.__name__,{'duration_in_ticks':50}
                if self._mode=='quick':
                    return Idle.__name__,{'duration_in_ticks':10}
                    
            if Phase.FOLLOW_PATH_TO_VICTIM==self._phase:
                self._mode='normal'
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
                self._state_tracker.update(state)
                action=self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action,{}
                self._phase=Phase.DROP_VICTIM
                #if self._mode=='normal':
                #    return Idle.__name__,{'duration_in_ticks':50}
                #if self._mode=='quick':
                #    return Idle.__name__,{'duration_in_ticks':10}

            if Phase.DROP_VICTIM == self._phase:
                if state[{'is_collectable':True}] or self._goalVic==self._firstVictim:
                    self._phase=Phase.FIND_NEXT_GOAL
                    self._currentDoor = None
                    return DropObject.__name__,{}
                if not state[{'is_collectable':True}] and self._goalVic!=self._firstVictim:
                    return None,{}    
                if self._mode=='normal':
                    return Idle.__name__,{'duration_in_ticks':50}
                if self._mode=='quick':
                    return Idle.__name__,{'duration_in_ticks':10}

            
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
        #areas = ['area A1','area A2','area A3','area A4','area B1','area B2','area C1','area C2','area C3']
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
                if loc not in self._searchedRooms:
                    self._searchedRooms.append(loc)
                if foundVic not in self._foundVictims:
                    self._foundVictims.append(foundVic)
                    self._foundVictimLocs[foundVic] = {'room':loc}
                if foundVic in self._foundVictims and self._foundVictimLocs[foundVic]['room'] != loc:
                    self._foundVictimLocs[foundVic] = {'room':loc}
            if msg.startswith('Collect:'):
                if len(msg.split()) == 6:
                    collectVic = ' '.join(msg.split()[1:4])
                else:
                    collectVic = ' '.join(msg.split()[1:5]) 
                loc = 'area ' + msg.split()[-1]
                if loc not in self._searchedRooms:
                    self._searchedRooms.append(loc)
                if collectVic not in self._foundVictims:
                    self._foundVictims.append(collectVic)
                    self._foundVictimLocs[collectVic] = {'room':loc}
                if collectVic in self._foundVictims and self._foundVictimLocs[collectVic]['room'] != loc:
                    self._foundVictimLocs[collectVic] = {'room':loc}
                if collectVic not in self._collectedVictims:
                    self._collectedVictims.append(collectVic)
            #if msg.startswith('Mission'):
            #    self._sendMessage('Unsearched areas: '  + ', '.join([i.split()[1] for i in areas if i not in self._searchedRooms]) + '. Collected victims: ' + ', '.join(self._collectedVictims) +
            #    '. Found victims: ' +  ', '.join([i + ' in ' + self._foundVictimLocs[i]['room'] for i in self._foundVictimLocs]) ,'RescueBot')
            #    self.received_messages=[]
                

    def _sendMessage(self, mssg, sender):
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)
            self._sendMessages.append(msg.content)

    def _getClosestRoom(self, state, objs, currentDoor):
        agent_location = state[self.agent_id]['location']
        locs = {}
        for obj in objs:
            locs[obj]=state.get_room_doors(obj)[0]['location']
        dists = {}
        for room,loc in locs.items():
            if currentDoor!=None:
                dists[room]=utils.get_distance(currentDoor,loc)
            if currentDoor==None:
                dists[room]=utils.get_distance(agent_location,loc)

        return min(dists,key=dists.get)

    def _efficientSearch(self, tiles):
        x=[]
        y=[]
        for i in tiles:
            if i[0] not in x:
                x.append(i[0])
            if i[1] not in y:
                y.append(i[1])
        locs = []
        for i in range(len(x)):
            if i%2==0:
                locs.append((x[i],min(y)))
            else:
                locs.append((x[i],max(y)))
        return locs