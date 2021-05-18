import sys, random, enum, ast

from matrx import grid_world
from BW4TBrain import BW4TBrain
from BlockPositionsHigh import BlockPositions, sameAppearance
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
    PLAN_PATH_ALONG_DROPZONE=0,
    FOLLOW_PATH_ALONG_DROPZONE=1,
    FIND_NEXT_GOAL=2,
    PICK_SOME_UNSEARCHED_ROOM_DOOR=3,
    PLAN_PATH_TO_UNSEARCHED_ROOM_DOOR=4,
    FOLLOW_PATH_TO_UNSEARCHED_ROOM_DOOR=5,
    PLAN_ROOM_SEARCH_PATH=6,
    FOLLOW_ROOM_SEARCH_PATH=7,
    PLAN_PATH_TO_VICTIM=8,
    FOLLOW_PATH_TO_VICTIM=9,
    TAKE_VICTIM=10,
    PLAN_PATH_TO_DROPPOINT=11,
    FOLLOW_PATH_TO_DROPPOINT=12,
    DROP_VICTIM=13,
    WAIT_FOR_HUMAN=14,
    WAIT_AT_ZONE=15
    
class BlockWorldAgent(BW4TBrain):
    def __init__(self, slowdown:int):
        super().__init__(slowdown)
        self._phase=Phase.PLAN_PATH_ALONG_DROPZONE
        self._blockpositions = BlockPositions()
        self._searchedRoomDoors = []
        self._foundVictims = {}
        self._handovers = []
        self._foundVictimsHuman = {}
        self._correctJudgements = []
        self._humansQuests = {}
        self._collectedHuman = {}
        self._uncarryable = ['critically injured elderly man', 'critically injured elderly woman', 'critically injured man', 'critically injured woman']
        self._undistinguishable = ['critically injured girl', 'critically injured boy', 'mildly injured boy', 'mildly injured girl']
        self._maxTicks = 4000

    def initialize(self):
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, 
            action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        self._processMessages(state)
        return state

    def decide_on_bw4t_action(self, state:State):
        oldblocks=self._blockpositions
        self._blockpositions=self._blockpositions.update(state)
        changes=self._blockpositions.getDifference(oldblocks)
        ticksLeft = self._maxTicks - state['World']['nr_ticks']
        #print(ticksLeft)
        
        while True: 
            if Phase.PLAN_PATH_ALONG_DROPZONE==self._phase:
                self._navigator.reset_full()
                waypoints = map(lambda info:info['location'], self._getDropZones(state))
                self._navigator.add_waypoints(waypoints)
                self._phase = Phase.FOLLOW_PATH_ALONG_DROPZONE

            if Phase.FOLLOW_PATH_ALONG_DROPZONE==self._phase:
                self._sendMessage('Exploring the drop zone', 'RescueBot')
                self._state_tracker.update(state)
                action=self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action,{}
                self._phase=Phase.FIND_NEXT_GOAL

            if Phase.FIND_NEXT_GOAL==self._phase:
                self._goalZone=None
                updated_zones = []
                zones = self._getDropZones(state)
                self._firstVictim = str(zones[0]['img_name'])[8:-4]
                for info in zones:
                    if str(info['img_name'])[8:-4] not in self._handovers and str(info['img_name'])[8:-4] not in self._collectedHuman.keys():
                        updated_zones.append(info)
                for info in updated_zones:
                    goodblocks = [blockinfo 
                        for blockinfo in self._blockpositions.getBlocksAt(info['location'])
                        if sameAppearance(blockinfo['img_name'], info['img_name'])]
                    if len(goodblocks)==0:
                        self._goalZone=info
                        break
                if self._goalZone==None:
                    self._phase=Phase.PLAN_PATH_ALONG_DROPZONE
                else:
                    self._goalVictim = str(self._goalZone['img_name'])[8:-4]
                    if self._goalVictim not in self._foundVictims.keys() and self._goalVictim not in self._foundVictimsHuman.keys():
                        self._sendMessage('Going to search for the ' + self._goalVictim, 'RescueBot')
                    # all known blocks with required appearance that are not in dropzone
                    options=self._blockpositions.getAppearance(self._goalZone['img_name'])
                    droplocs=[info['location'] for info in self._getDropZones(state)]
                    options=[info for info in options if not info['location'] in droplocs]
                    if len(options)==0 or self._goalVictim in self._foundVictimsHuman.keys() and self._goalVictim not in self._correctJudgements:
                        self._phase=Phase.PICK_SOME_UNSEARCHED_ROOM_DOOR
                    else:
                        self._block = random.choice(options)
                        self._phase=Phase.PLAN_PATH_TO_VICTIM
            
            if Phase.PICK_SOME_UNSEARCHED_ROOM_DOOR==self._phase:
                unsearchedRoomDoors=[door for door in state.values()
                    if 'class_inheritance' in door 
                    and 'Door' in door['class_inheritance']
                    and door not in self._searchedRoomDoors
                    and door['room_name'] not in self._humansQuests.values()]
                if len(unsearchedRoomDoors)==0:
                    self._phase=Phase.PLAN_PATH_ALONG_DROPZONE
                else:
                    if self._goalVictim in self._foundVictimsHuman.keys() and self._goalVictim not in self._foundVictims.keys() and self._goalVictim not in self._collectedHuman.keys():
                        self._door = state.get_room_doors(self._foundVictimsHuman[self._goalVictim])[0]
                        self._searchedRoomDoors.append(self._door)
                        self._phase=Phase.PLAN_PATH_TO_UNSEARCHED_ROOM_DOOR
                    if self._goalVictim in self._collectedHuman.keys():
                        self._phase = Phase.FIND_NEXT_GOAL
                    if self._goalVictim not in self._foundVictims.keys() and self._goalVictim not in self._foundVictimsHuman.keys():
                        #self._door=random.choice(unsearchedRoomDoors)
                        self._door=state.get_room_doors(self._getClosestRoom(state, unsearchedRoomDoors))[0]
                        print(self._door)
                        self._searchedRoomDoors.append(self._door)
                        self._phase = Phase.PLAN_PATH_TO_UNSEARCHED_ROOM_DOOR
                    
            if Phase.PLAN_PATH_TO_UNSEARCHED_ROOM_DOOR == self._phase:
                self._navigator.reset_full()
                doorLoc = state.get_room_objects(self._door['room_name'])[0]['doormat']
                self._sendMessage('Moving to the ' + str(self._door['room_name']), 'RescueBot')
                self._navigator.add_waypoints([doorLoc])
                self._phase=Phase.FOLLOW_PATH_TO_UNSEARCHED_ROOM_DOOR
            
            if Phase.FOLLOW_PATH_TO_UNSEARCHED_ROOM_DOOR == self._phase:
                self._state_tracker.update(state)
                if self._goalVictim in self._collectedHuman.keys():
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    action=self._navigator.get_move_action(self._state_tracker)
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
            
            if Phase.FOLLOW_ROOM_SEARCH_PATH == self._phase:
                self._state_tracker.update(state)
                if self._goalVictim in self._collectedHuman.keys():
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    action=self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        if len(changes)>0:
                            self._foundVictim = str(changes[0]['img_name'][8:-4])
                            self._foundVictims[self._foundVictim] = str(self._door['room_name'])
                        if state[self.agent_id]['location'] == self._door['location']:
                            self._sendMessage('Searching through the ' + str(self._door['room_name']), 'RescueBot')
                        if len(changes)>0 and 'healthy' not in str(changes[0]['img_name']) and 'boy' not in str(changes[0]['img_name']) and 'girl' not in str(changes[0]['img_name']):
                            self._sendMessage('Found '+self._foundVictim + ' in ' + str(self._door['room_name']), 'RescueBot')
                        if len(changes)>0 and self._foundVictim==self._goalVictim and self._foundVictim in self._uncarryable:
                            self._sendMessage('URGENT: ASSISTANCE REQUIRED! I need you to pick up ' + self._foundVictim + ' in ' + self._foundVictims[self._foundVictim], 'RescueBot')
                            self._handovers.append(self._foundVictim)
                            self._phase=Phase.WAIT_FOR_HUMAN
                        if len(changes)>0 and self._foundVictim in self._undistinguishable and self._goalVictim in self._undistinguishable:
                            self._sendMessage('URGENT: ASSISTANCE REQUIRED! I need you to clarify the gender of the injured baby in ' + self._foundVictims[self._foundVictim], 'RescueBot')
                            self._phase=Phase.WAIT_FOR_HUMAN
                        return action,{}
                    if self._goalVictim not in self._foundVictims.keys():
                        self._sendMessage(self._goalVictim.capitalize() + ' not present in ' + str(self._door['room_name']), 'RescueBot')
                        if self._goalVictim in self._foundVictimsHuman.keys() and str(self._door['room_name'])==self._foundVictimsHuman[self._goalVictim]:
                            self._foundVictimsHuman.pop(self._goalVictim, None)
                        else:  
                            self._correctJudgements.append(self._goalVictim)
                    self._phase=Phase.FIND_NEXT_GOAL

            if Phase.WAIT_FOR_HUMAN==self._phase:
                self._state_tracker.update(state)
                if self._goalVictim in self._collectedHuman.keys():
                    self._phase = Phase.FIND_NEXT_GOAL
                if state[{'is_human_agent':True}]:
                    if self._goalVictim in self._uncarryable:
                        self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH
                    if self._goalVictim in self._undistinguishable and self.received_messages[-1]==self._goalVictim.split()[-1]:
                        self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH
                    if self._goalVictim in self._undistinguishable and self.received_messages[-1]=='boy' and self._goalVictim.split()[-1]!='boy':
                        self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH
                    if self._goalVictim in self._undistinguishable and self.received_messages[-1]=='girl' and self._goalVictim.split()[-1]!='girl':
                        self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH
                    else:
                        return None,{}
                else:
                    self._sendMessage('Waiting for human operator to arrive in ' + str(self._door['room_name']), 'RescueBot')
                    return None,{}

            if Phase.PLAN_PATH_TO_VICTIM==self._phase:
                if self._goalVictim not in self._uncarryable:
                    self._sendMessage('Picking up ' + self._goalVictim + ' in ' + self._foundVictims[self._goalVictim], 'RescueBot')
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self._block['location']])
                    self._phase=Phase.FOLLOW_PATH_TO_VICTIM
                else:
                    self._handovers.append(self._goalVictim)
                    self._phase=Phase.FIND_NEXT_GOAL
            
            if Phase.FOLLOW_PATH_TO_VICTIM == self._phase:
                self._state_tracker.update(state)
                if self._goalVictim in self._collectedHuman.keys():
                    self._phase = Phase.FIND_NEXT_GOAL
                if self._goalVictim not in self._uncarryable:
                    action=self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        return action,{}
                    self._phase=Phase.TAKE_VICTIM
                else:
                    self._handovers.append(self._goalVictim)
                    self._phase=Phase.FIND_NEXT_GOAL
    
            if Phase.TAKE_VICTIM == self._phase:
                if self._goalVictim in self._collectedHuman.keys():
                    self._phase = Phase.FIND_NEXT_GOAL
                if self._goalVictim not in self._uncarryable:
                    self._sendMessage('Transporting '+str(self._block['img_name'])[8:-4] + ' to the drop zone', 'RescueBot')
                    self._phase=Phase.PLAN_PATH_TO_DROPPOINT
                    return GrabObject.__name__,{'object_id':self._block['obj_id']}
                else:
                    self._handovers.append(self._goalVictim)
                    self._phase=Phase.FIND_NEXT_GOAL
            
            if Phase.PLAN_PATH_TO_DROPPOINT==self._phase:
                if self._goalVictim not in self._uncarryable:
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self._goalZone['location']])
                    self._phase=Phase.FOLLOW_PATH_TO_DROPPOINT
                else:
                    self._handovers.append(self._goalVictim)
                    self._phase=Phase.FIND_NEXT_GOAL

            if Phase.FOLLOW_PATH_TO_DROPPOINT==self._phase:
                self._state_tracker.update(state)
                if self._goalVictim not in self._uncarryable:
                    action=self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        return action,{}
                    self._phase=Phase.DROP_VICTIM
                else:
                    self._handovers.append(self._goalVictim)
                    self._phase=Phase.FIND_NEXT_GOAL

            if Phase.DROP_VICTIM == self._phase:
                if state[{'is_collectable':True}] or str(self._block['img_name'])[8:-4]==self._firstVictim:
                    if self._goalVictim not in self._uncarryable:
                        self._sendMessage('Delivered '+str(self._block['img_name'])[8:-4] + ' at the drop zone', 'RescueBot')
                        self._phase=Phase.FIND_NEXT_GOAL
                        return DropObject.__name__,{}
                    else:
                        self._handovers.append(self._goalVictim)
                        self._phase=Phase.FIND_NEXT_GOAL
                if not state[{'is_collectable':True}] and str(self._block['img_name'])[8:-4]!=self._firstVictim:
                    zones = self._getDropZones(state)
                    previous_victims = []
                    for info in zones:
                        if str(info['img_name'])[8:-4] in self._uncarryable:
                            previous_victims.append(str(info['img_name'])[8:-4])
                    self._sendMessage('Waiting for human operator to deliver ' + ' and '.join(previous_victims), 'RescueBot')
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
    
    def _isCarrying(self, state:State,appearance:dict):
        """
        @param state the current State
        @param appearance a dict with the required block appearance
        @return true iff we are carrying a block of given appearance
        """
        for carrying in state[self.agent_id]['is_carrying']:
            if sameAppearance(carrying['img_name'], appearance):
                return True
        return False
        

    def _getDropOff(self, state:State,y:int)-> tuple:
        """
        @param y the y location of the required drop off location
        @return the drop off location (x,y) given the y.
        @throws out of index error if there is no drop zone at given y position.
        """
        for id in state.keys():
            if 'is_drop_zone' in state[id] and state[id]['location'][1] == y:
                return state[id]['location']
        raise ValueError("There is no block at y location "+str(y))
        

    def _findLocationOfBlock(self, state:State,appearance:dict):
        """
        @param state the current State
        @param appearance the 'visualization' settings. Must contain
        'size', 'shape' and  color.
        @return (id, x,y) of a block of requested appearance,
        that is not already on a dropoff point or being carried.
        """
        droplocations=[state[id]['location'] 
           for id in state.keys() 
           if 'is_goal_block' in state[id] and state[id]['is_goal_block']]

        # find locations of all blocks of given appearance that are not already
        # on a droplocation
        locs=[(id,)+state[id]['location'] for id in state.keys() 
            if 'is_collectable' in state[id] 
            and state[id]['is_collectable'] 
            and state[id]['img_name']==appearance['img_name']
            and not state[id]['location'] in droplocations
            and len(state[id]['carried_by'])==0
            ]
        if len(locs)==0:
            return None
        return random.choice(locs)

    def _findRoomContaining(self, state:State, loc:tuple):
        """
        @param loc the (x,y) location 
        @return a (should be unique) room name 
        that contains that location, or None if no such room.
        Basically we look for an AreaTile at given loc. 
        NOTICE: room name is a label used by the room designer,
        it's not the ID. I assume that properly designed 
        worlds use this label consistently to tag other objects in the same room,
        typically the doors tiles and walls.
        """
        locs=[state[id]['room_name'] for id in state.keys() 
            if 'class_inheritance' in state[id]
            and 'AreaTile' in state[id]['class_inheritance']
            and state[id]['location'] == loc
        ]
        if (len(locs)==0):
            return None
        return locs[0]

    def _processMessages(self, state):
        '''
        process incoming messages. 
        Reported blocks are added to self._blocks
        '''
        room_names = state.get_all_room_names()
        for msg in self.received_messages:
            if msg.startswith("found"):
                try:
                    content = msg[6:-1].split(',')
                    for room in room_names:
                        if content[2] in room.split() and content[3] in room.split():
                            self._foundVictimsHuman[content[0]+'ly injured '+content[1]]=room
                    self.received_messages.remove(msg)
                except:
                    self.received_messages=[]
                    self._sendMessage('I did not understand your message "' +msg+'", please try again','RescueBot')
            if msg.startswith("search"):
                try:
                    content = msg[7:-1].split(',')
                    for room in room_names:
                        if content[2] in room.split() and content[3] in room.split():
                            self._humansQuests[content[0]+'ly injured ' + content[1]]=room
                    self.received_messages.remove(msg)
                except:
                    self.received_messages=[]
                    self._sendMessage('I did not understand your message "' +msg+'", please try again','RescueBot')
            if msg.startswith("collect"):
                try:
                    content = msg[8:-1].split(',')
                    for room in room_names:
                        if content[2] in room.split() and content[3] in room.split():
                            self._collectedHuman[content[0] + 'ly injured ' + content[1]] = room
                    self.received_messages.remove(msg)
                except:
                    self.received_messages=[]
                    self._sendMessage('I did not understand your message "' +msg+'", please try again','RescueBot')

    def _sendMessage(self, mssg, sender):
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _getClosestRoom(self, state, objs):
        agent_location = state[self.agent_id]['location']
        locs = {}
        for obj in objs:
            locs[obj['room_name']]=obj['location']
        dists = {}
        for room,loc in locs.items():
            dists[room]=utils.get_distance(agent_location,loc)
        print(dists)
        return min(dists,key=dists.get)