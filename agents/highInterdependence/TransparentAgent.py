import sys
sys.path.append("./worlds")
from BW4TBrain import BW4TBrain
#from ColorBlindBW4TBrain import ColorBlindBW4TBrain
from highInterdependence.BlockPositions import BlockPositions, sameAppearance
import enum
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
import random
from matrx.messages.message import Message
from matrx.messages.message_manager import MessageManager
import ast

class Phase(enum.Enum):
    PLAN_PATH_ALONG_DROPZONE=0, # to update our knowledge about placed blocks
    FOLLOW_PATH_ALONG_DROPZONE=1,
    FIND_NEXT_GOAL=2,# determine next goal zone
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
    WAIT_AT_ZONE=15,
    PROCESS_MESSAGES=16
    
class BlockWorldAgent(BW4TBrain):
    """
    This extends planningagent2 with communication.
    """
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

    #override
    def initialize(self):
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, 
            action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    #override
    def filter_bw4t_observations(self, state):
        self._processMessages(state)
        return state

    #override
    def decide_on_bw4t_action(self, state:State):
        print('Found:', self._foundVictimsHuman)
        print('Quests:', self._humansQuests)
        self._uncarryable = ['critically injured elderly man', 'critically injured elderly woman', 'critically injured man', 'critically injured woman']
        self._undistinguishable = ['critically injured girl', 'critically injured boy', 'mildly injured boy', 'mildly injured girl']
        oldblocks=self._blockpositions
        self._blockpositions=self._blockpositions.update(state)

        changes=self._blockpositions.getDifference(oldblocks)
        #to_id=self._teamMembers(state) does not work ok yet
        #    self.send_message(msg)

        while True: 
            if Phase.PLAN_PATH_ALONG_DROPZONE==self._phase:
                self._navigator.reset_full()
                waypoints = map(lambda info:info['location'], self._getDropZones(state))
                self._navigator.add_waypoints(waypoints)
                self._phase = Phase.FOLLOW_PATH_ALONG_DROPZONE

            if Phase.FOLLOW_PATH_ALONG_DROPZONE==self._phase:
                # This explores the area so we know the needed blocks afterwards
                msg = Message(content='Exploring the drop zone', from_id='RescueBot')
                if msg.content not in self.received_messages:
                    self.send_message(msg)
                self._state_tracker.update(state)
                # execute the  path steps as planned in self._navigator
                action=self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action,{}
                # If we get here, we're there
                self._phase=Phase.FIND_NEXT_GOAL

            if Phase.FIND_NEXT_GOAL==self._phase:
                self._goalZone=None
                updated_zones = []
                zones = self._getDropZones(state)
                self._firstVictim = str(zones[0]['img_name'])[8:-4]
                for info in zones:
                    if str(info['img_name'])[8:-4] not in self._handovers:
                        updated_zones.append(info)
                for info in updated_zones:
                    goodblocks = [blockinfo 
                        for blockinfo in self._blockpositions.getBlocksAt(info['location'])
                        if sameAppearance(blockinfo['img_name'], info['img_name'])]
                    if len(goodblocks)==0:
                        self._goalZone=info
                        break
                if self._goalZone==None:
                    # all blocks are in place. can't handle this situation.
                    self._phase=Phase.PLAN_PATH_ALONG_DROPZONE
                else:
                    if str(self._goalZone['img_name'])[8:-4] not in self._foundVictims.keys() and str(self._goalZone['img_name'])[8:-4] not in self._foundVictimsHuman.keys():
                        msg = Message(content='Going to search for the ' + str(self._goalZone['img_name'])[8:-4], from_id='RescueBot')
                        if msg.content not in self.received_messages:
                            self.send_message(msg)
                    # all known blocks with required appearance that are not in dropzone
                    options=self._blockpositions.getAppearance(self._goalZone['img_name'])
                    droplocs=[info['location'] for info in self._getDropZones(state)]
                    options=[info for info in options if not info['location'] in droplocs]
                    if len(options)==0 or str(self._goalZone['img_name'])[8:-4] in self._foundVictimsHuman.keys() and str(self._goalZone['img_name'])[8:-4] not in self._correctJudgements:
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
                #print(unsearchedRoomDoors)
                if len(unsearchedRoomDoors)==0:
                    # can't handle this situation. 
                    self._phase=Phase.PLAN_PATH_ALONG_DROPZONE
                else:
                    if str(self._goalZone['img_name'])[8:-4] in self._foundVictimsHuman.keys():
                        self._door = state.get_room_doors(self._foundVictimsHuman[str(self._goalZone['img_name'])[8:-4]])[0]
                        self._searchedRoomDoors.append(self._door)
                        self._phase=Phase.PLAN_PATH_TO_UNSEARCHED_ROOM_DOOR
                    else:
                        self._door=random.choice(unsearchedRoomDoors)
                        self._searchedRoomDoors.append(self._door)
                        self._phase = Phase.PLAN_PATH_TO_UNSEARCHED_ROOM_DOOR
                    
            if Phase.PLAN_PATH_TO_UNSEARCHED_ROOM_DOOR == self._phase:
                # self._door must be set to target door
                self._navigator.reset_full()
                #doorLoc:tuple = self._door['location']
                # HACK we assume door is at south side of room
                #doorLoc = doorLoc[0],doorLoc[1]+1
                doorLoc = state.get_room_objects(self._door['room_name'])[0]['doormat']
                #doorLoc = self._door[0]['location']
                #print(state.get_room_objects(self._door['room_name'])[0]['doormat'])
                #print(state[self.agent_id]['location'])
                msg = Message(content='Moving to the ' + str(self._door['room_name']), from_id='RescueBot')
                if msg.content not in self.received_messages:
                    self.send_message(msg)
                self._navigator.add_waypoints([doorLoc])
                self._phase=Phase.FOLLOW_PATH_TO_UNSEARCHED_ROOM_DOOR
            
            if Phase.FOLLOW_PATH_TO_UNSEARCHED_ROOM_DOOR == self._phase:
                self._state_tracker.update(state)
                # self._door must be set
                # execute the  path steps as planned in self._navigator
                action=self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    #while state[self.agent_id]['location']!=state.get_room_objects(self._door['room_name'])[0]['doormat']:
                    return action,{}
                # If we get here, we're there
                self._phase=Phase.PLAN_ROOM_SEARCH_PATH

            if Phase.PLAN_ROOM_SEARCH_PATH==self._phase:
                # self._door must be set
                roomTiles = [info['location'] for info in state.values()
                    if 'class_inheritance' in info 
                    and 'AreaTile' in info['class_inheritance']
                    #notice, in matrx a room can go to only 1 door
                    # because of the 'room_name' property of doors
                    and 'room_name' in info
                    and info['room_name'] == self._door['room_name']
                ]
                self._roomtiles=roomTiles     
                # FIXME we want to sort these tiles for efficient search...
                # CHECK rooms don't need to be square I assume?
                self._navigator.reset_full()
                self._navigator.add_waypoints(roomTiles)
                self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH
            
            if Phase.FOLLOW_ROOM_SEARCH_PATH == self._phase:

                self._state_tracker.update(state)
                action=self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    if len(changes)>0:
                        self._foundVictims[str(changes[0]['img_name'][8:-4])] = str(self._door['room_name'])
                    if state[self.agent_id]['location'] == self._door['location']:
                        msg = Message(content='Searching through the ' + str(self._door['room_name']), from_id='RescueBot')
                        if msg.content not in self.received_messages:
                            self.send_message(msg)
                    if len(changes)>0 and 'healthy' not in str(changes[0]['img_name']) and 'boy' not in str(changes[0]['img_name']) and 'girl' not in str(changes[0]['img_name']):
                        #msg = Message(content='Found '+str(changes[0]['img_name'][8:-4]) + ' in ' + self._foundVictims[str(changes[0]['img_name'][8:-4])], from_id='RescueBot' )
                        msg = Message(content='Found '+str(changes[0]['img_name'][8:-4]) + ' in ' + str(self._door['room_name']), from_id='RescueBot' )
                        if msg.content not in self.received_messages:
                            self.send_message(msg)
                    #if len(changes)>0 and 'critically' in str(changes[0]['img_name']) and 'man' in str(changes[0]['img_name']) and str(changes[0]['img_name'][8:-4])==str(self._goalZone['img_name'])[8:-4]:
                    if len(changes)>0 and str(changes[0]['img_name'][8:-4])==str(self._goalZone['img_name'])[8:-4] and str(changes[0]['img_name'][8:-4]) in self._uncarryable:
                        msg = Message(content='URGENT: ASSISTANCE REQUIRED! I need you to pick up ' + str(changes[0]['img_name'][8:-4]) + ' in ' + self._foundVictims[str(changes[0]['img_name'][8:-4])], from_id='RescueBot' )
                        if msg.content not in self.received_messages:
                            self.send_message(msg)
                        self._handovers.append(str(changes[0]['img_name'][8:-4]))
                        self._phase=Phase.WAIT_FOR_HUMAN
                    if len(changes)>0 and str(changes[0]['img_name'][8:-4]) in self._undistinguishable:
                        msg = Message(content='URGENT: ASSISTANCE REQUIRED! I need you to clarify the gender of the injured baby in ' + self._foundVictims[str(changes[0]['img_name'][8:-4])], from_id='RescueBot' )
                        if msg.content not in self.received_messages:
                            self.send_message(msg)
                        #self._handovers.append(str(changes[0]['img_name'][8:-4]))
                        self._phase=Phase.WAIT_FOR_HUMAN
                    return action,{}
                # If we get here, we're done
                if str(self._goalZone['img_name'])[8:-4] not in self._foundVictims.keys():
                    msg = Message(content=str(self._goalZone['img_name'])[8:-4].capitalize() + ' not present in ' + str(self._door['room_name']), from_id='RescueBot')
                    if msg.content not in self.received_messages:
                        self.send_message(msg)
                    self._lastSearched=str(self._door['room_name'])
                # CONTINUE HERE
                if str(self._goalZone['img_name'])[8:-4] in self._foundVictimsHuman.keys() and str(self._goalZone['img_name'])[8:-4] in self._foundVictims.keys():
                    if self._foundVictims[str(self._goalZone['img_name'])[8:-4]]!=self._foundVictimsHuman[str(self._goalZone['img_name'])[8:-4]] and str(self._door['room_name'])==self._foundVictimsHuman[str(self._goalZone['img_name'])[8:-4]]:
                        self._foundVictimsHuman.pop(str(self._goalZone['img_name'])[8:-4], None)
                    else:
                        self._correctJudgements.append(str(self._goalZone['img_name'])[8:-4])
                #for k,v in self._foundVictimsHuman.copy().items():
                self._phase=Phase.FIND_NEXT_GOAL

            if Phase.WAIT_FOR_HUMAN==self._phase:
                self._state_tracker.update(state)
                if 'human_0_in_team_0' in state.keys():
                    if str(self._goalZone['img_name'])[8:-4] in self._uncarryable:
                        self._phase=Phase.FIND_NEXT_GOAL
                    if str(self._goalZone['img_name'])[8:-4] in self._undistinguishable and self.received_messages[-1]==str(self._goalZone['img_name'])[8:-4].split()[-1]:
                        self._phase=Phase.FIND_NEXT_GOAL
                    if str(self._goalZone['img_name'])[8:-4] in self._undistinguishable and self.received_messages[-1]=='boy' and str(self._goalZone['img_name'])[8:-4].split()[-1]!='boy':
                        self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH
                    if str(self._goalZone['img_name'])[8:-4] in self._undistinguishable and self.received_messages[-1]=='girl' and str(self._goalZone['img_name'])[8:-4].split()[-1]!='girl':
                        self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH
                    else:
                        return None,{}
                else:
                    msg = Message(content='Waiting for human operator to arrive in ' + str(self._door['room_name']), from_id='RescueBot' )
                    if msg.content not in self.received_messages:
                        self.send_message(msg)
                    return None,{}

            if Phase.PLAN_PATH_TO_VICTIM==self._phase:
                # self._block must be set to info of target block 
                # self._goalZone must be set to goalzone needing that block
                # we assume door to room containing block is open
                if str(self._goalZone['img_name'])[8:-4] not in self._uncarryable:
                    msg = Message(content='Picking up ' + str(self._goalZone['img_name'])[8:-4] + ' in ' + self._foundVictims[str(self._goalZone['img_name'])[8:-4]], from_id='RescueBot')
                    if msg.content not in self.received_messages:
                        self.send_message(msg)
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self._block['location']])
                    self._phase=Phase.FOLLOW_PATH_TO_VICTIM
                else:
                    self._handovers.append(str(self._goalZone['img_name'])[8:-4])
                    self._phase=Phase.FIND_NEXT_GOAL
            
            if Phase.FOLLOW_PATH_TO_VICTIM == self._phase:
                # self._block must be set to info of target block 
                # self._goalZone must be set to goalzone needing that block
                self._state_tracker.update(state)
                if str(self._goalZone['img_name'])[8:-4] not in self._uncarryable:
                    action=self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        return action,{}
                #if self._navigator.is_done:
                    self._phase=Phase.TAKE_VICTIM
                else:
                    self._handovers.append(str(self._goalZone['img_name'])[8:-4])
                    self._phase=Phase.FIND_NEXT_GOAL
    
            if Phase.TAKE_VICTIM == self._phase:
                # self._block must be set to info of target block 
                # self._goalZone must be set to goalzone needing that block
                if str(self._goalZone['img_name'])[8:-4] not in self._uncarryable:
                    msg = Message(content='Transporting '+str(self._block['img_name'])[8:-4] + ' to the drop zone', from_id='RescueBot' )
                    if msg.content not in self.received_messages:
                        self.send_message(msg)
                    self._phase=Phase.PLAN_PATH_TO_DROPPOINT
                    return GrabObject.__name__,{'object_id':self._block['obj_id']}
                else:
                    self._handovers.append(str(self._goalZone['img_name'])[8:-4])
                    self._phase=Phase.FIND_NEXT_GOAL
            
            if Phase.PLAN_PATH_TO_DROPPOINT==self._phase:
                # self._block must be set to info of target block 
                # self._goalZone must be set to goalzone needing that block
                if str(self._goalZone['img_name'])[8:-4] not in self._uncarryable:
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self._goalZone['location']])
                    self._phase=Phase.FOLLOW_PATH_TO_DROPPOINT
                else:
                    self._handovers.append(str(self._goalZone['img_name'])[8:-4])
                    self._phase=Phase.FIND_NEXT_GOAL

            if Phase.FOLLOW_PATH_TO_DROPPOINT==self._phase:
                self._state_tracker.update(state)
                if str(self._goalZone['img_name'])[8:-4] not in self._uncarryable:
                # execute the  path steps as planned in self._navigator
                    action=self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        return action,{}
                # If we get here, we're there
                    self._phase=Phase.DROP_VICTIM
                else:
                    self._handovers.append(str(self._goalZone['img_name'])[8:-4])
                    self._phase=Phase.FIND_NEXT_GOAL

            #if Phase.WAIT_AT_ZONE==self._phase:
            #    self._state_tracker.update(state)
                
            if Phase.DROP_VICTIM == self._phase:
                if state[{'is_collectable':True}] or str(self._block['img_name'])[8:-4]==self._firstVictim:
                #print("Dropped box ",self._block)
                    if str(self._goalZone['img_name'])[8:-4] not in self._uncarryable:
                        msg = Message(content='Delivered '+str(self._block['img_name'])[8:-4] + ' at the drop zone', from_id='RescueBot' )
                        if msg.content not in self.received_messages:
                            self.send_message(msg)
                        self._phase=Phase.FIND_NEXT_GOAL
                # don't use 'object_id':self._nextBlock[0],
                # there seems a bug in MATRX DropObject #7
                # Maybe it's fixed now
                        return DropObject.__name__,{}
                    else:
                        self._handovers.append(str(self._goalZone['img_name'])[8:-4])
                        self._phase=Phase.FIND_NEXT_GOAL
                if not state[{'is_collectable':True}] and str(self._block['img_name'])[8:-4]!=self._firstVictim:
                    #for block in state[{'is_goal_block': True}]:
                    #    if block['location'] == (self._goalZone['location'][0]-1, self._goalZone['location'][1]):
                    #        self._previousVictim = str(block['img_name'])[8:-4]
                    zones = self._getDropZones(state)
                    previous_victims = []
                    for info in zones:
                        if str(info['img_name'])[8:-4] in self._uncarryable:
                            previous_victims.append(str(info['img_name'])[8:-4])
                    msg = Message(content='Waiting for human operator to deliver ' + ' and '.join(previous_victims), from_id='RescueBot')
                    if msg.content not in self.received_messages:
                        self.send_message(msg)
                    return None,{}

    
    def _getDropZones(self,state:State):
        '''
        @return list of drop zones (their full dict), in order (the first one is the
        the place that requires the first drop)
        '''
        # we must use is_goal_block, not is_drop_zone, to collect also the
        # correct appearance
        places=state[{'is_goal_block':True}]
        #print(places)
        # sort does in-place sorting
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
            #and state[id]['visualization']['size']==appearance['size']
            #and state[id]['visualization']['shape']==appearance['shape']
            #and state[id]['visualization']['colour']==appearance['colour']
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
        #if len(self.received_messages)>0:
        #    print(self.agent_name+" recvd msgs:",self.received_messages )
        room_names = state.get_all_room_names()
        for msg in self.received_messages:
            if msg.startswith("found"):
                content = msg[6:-1].split(',')
                for room in room_names:
                    if content[2] in room.split() and content[3] in room.split():
                        self._foundVictimsHuman[content[0]+'ly injured '+content[1]]=room
                self.received_messages.remove(msg)
            if msg.startswith("search"):
                content = msg[7:-1].split(',')
                for room in room_names:
                    if content[2] in room.split() and content[3] in room.split():
                        self._humansQuests[content[0]+'ly injured ' + content[1]]=room
                self.received_messages.remove(msg)
                #try:
                #    content=msg[6:]
                #    infos=ast.literal_eval(content)
                #    for blockinfo in infos:
                #        self._blockpositions = self._blockpositions.updateInfo(blockinfo)
                #except:
                #    print("Warning. parsing err "+str(sys.exc_info())+": "+content)
        #workaround for bug
        #self.received_messages=[]

    def _teamMembers(self, state):
        '''
        @param state the State
        @return team members, except me
        '''
        #Due to bug #241 this does not work properly now.
        return [info['name'] for info in state.values()
                if 'isAgent' in info 
                and info['name']!=self.agent_name]
    