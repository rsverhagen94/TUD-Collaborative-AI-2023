import sys, random, enum, ast, time
from matrx import grid_world
from brains1.BW4TBrain import BW4TBrain
from actions1.customActions import *
from matrx import utils
from matrx.grid_world import GridWorld
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject, RemoveObject
from matrx.actions.move_actions import MoveNorth
from matrx.messages.message import Message
from matrx.messages.message_manager import MessageManager
from actions1.customActions import RemoveObjectTogether, CarryObjectTogether, DropObjectTogether, CarryObject, Drop

class Phase(enum.Enum):
    INTRO0 = 0,
    INTRO1 = 1,
    INTRO2 = 2,
    INTRO3 = 3,
    INTRO4 = 4,
    INTRO5 = 5,
    INTRO6 = 6,
    INTRO7 = 7,
    INTRO8 = 8,
    INTRO9 = 9,
    INTRO10 = 10,
    INTRO11 = 11,
    FIND_NEXT_GOAL = 12,
    PICK_UNSEARCHED_ROOM = 13,
    PLAN_PATH_TO_ROOM = 14,
    FOLLOW_PATH_TO_ROOM = 15,
    PLAN_ROOM_SEARCH_PATH = 16,
    FOLLOW_ROOM_SEARCH_PATH = 17,
    PLAN_PATH_TO_VICTIM = 18,
    FOLLOW_PATH_TO_VICTIM = 19,
    TAKE_VICTIM = 20,
    PLAN_PATH_TO_DROPPOINT = 21,
    FOLLOW_PATH_TO_DROPPOINT = 22,
    DROP_VICTIM = 23,
    WAIT_FOR_HUMAN = 24,
    WAIT_AT_ZONE = 25,
    FIX_ORDER_GRAB = 26,
    FIX_ORDER_DROP = 27,
    REMOVE_OBSTACLE_IF_NEEDED = 28,
    ENTER_ROOM = 29


class BaselineAgent(BW4TBrain):
    def __init__(self, slowdown: int, condition: str):
        super().__init__(slowdown, condition)
        self._slowdown = slowdown
        self._condition = condition
        self._phase = Phase.INTRO0
        self._roomVics = []
        self._searchedRooms = []
        self._foundVictims = []
        self._collectedVictims = []
        self._foundVictimLocs = {}
        self._maxTicks = 9600
        self._sendMessages = []
        self._currentDoor = None
        self._providedExplanations = []
        self._teamMembers = []
        self._carryingTogether = False
        self._remove = False
        self._goalVic = None
        self._goalLoc = None
        self._second = None
        self._humanLoc = None
        self._distanceHuman = None
        self._distanceDrop = None
        self._agentLoc = None
        self._todo = []
        self._answered = False
        self._tosearch = []
        self._ignored = 0
        self._followed = 0
        self._noSuggestions = 0
        self._suggestion = []
        self._carrying = False
        self._waiting = False
        self._rescue = None
        

    def initialize(self):
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state: State):
        self._criticalFound = 0
        self._criticalRescued = 0
        for vic in self._foundVictims:
            if 'critical' in vic:
                self._criticalFound += 1
        for vic in self._collectedVictims:
            if 'critical' in vic:
                self._criticalRescued += 1

        if self._criticalFound == 1:
            self._vicString = 'victim'
        else:
            self._vicString = 'victims'

        if self._criticalRescued == 1:
            self._vicString2 = 'victim'
        else:
            self._vicString2 = 'victims'

        if state[{'is_human_agent': True}]:
            self._distanceHuman = 'close'
        if not state[{'is_human_agent': True}]:
            if self._agentLoc in [1, 2, 3, 4, 5, 6, 7] and self._humanLoc in [8, 9, 10, 11, 12, 13, 14]:
                self._distanceHuman = 'far'
            if self._agentLoc in [1, 2, 3, 4, 5, 6, 7] and self._humanLoc in [1, 2, 3, 4, 5, 6, 7]:
                self._distanceHuman = 'close'
            if self._agentLoc in [8, 9, 10, 11, 12, 13, 14] and self._humanLoc in [1, 2, 3, 4, 5, 6, 7]:
                self._distanceHuman = 'far'
            if self._agentLoc in [8, 9, 10, 11, 12, 13, 14] and self._humanLoc in [8, 9, 10, 11, 12, 13, 14]:
                self._distanceHuman = 'close'
            #else:
            #    self._distanceHuman = 'close'

        if self._agentLoc in [1, 2, 5, 6, 8, 9, 11, 12]:
            self._distanceDrop = 'far'
        if self._agentLoc in [3, 4, 7, 10, 13, 14]:
            self._distanceDrop = 'close'

        self._second = state['World']['tick_duration'] * state['World']['nr_ticks']

        for info in state.values():
            if 'is_human_agent' in info and 'Human' in info['name'] and len(info['is_carrying']) > 0 and 'critical' in info['is_carrying'][0]['obj_id'] or \
                'is_human_agent' in info and 'Human' in info['name'] and len(info['is_carrying']) > 0 and 'mild' in info['is_carrying'][0]['obj_id'] and self._rescue=='together':
                if info['is_carrying'][0]['img_name'][8:-4] not in self._collectedVictims:
                    self._collectedVictims.append(info['is_carrying'][0]['img_name'][8:-4])
                self._carryingTogether = True
            if 'is_human_agent' in info and 'Human' in info['name'] and len(info['is_carrying']) == 0:
                self._carryingTogether = False
        if self._carryingTogether == True:
            return None, {}
        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
        # Process messages from team members
        self._processMessages(state, self._teamMembers, self._condition)

        self._sendMessage('Our score is ' + str(state['rescuebot']['score']) + '.', 'RescueBot')
        if self._noSuggestions > 0:
            state['rescuebot']['ignored'] = round(self._ignored / self._noSuggestions, 2)
            self._sendMessage('You ignored me ' +str(state['rescuebot']['ignored']), 'RescueBot')

        while True:
            if Phase.INTRO0 == self._phase:
                self._sendMessage('Hello! My name is RescueBot. Together we will collaborate and try to search and rescue the 8 victims on our right as quickly as possible. \
                Each critical victim (critically injured girl/critically injured elderly woman/critically injured man/critically injured dog) adds 6 points to our score, \
                each mild victim (mildly injured boy/mildly injured elderly man/mildly injured woman/mildly injured cat) 3 points. \
                If you are ready to begin our mission, you can simply start moving.', 'RescueBot')
                #if self.received_messages_content and self.received_messages_content[-1] == 'Ready!':
                #    self._phase = Phase.FIND_NEXT_GOAL
                if not state[{'is_human_agent': True}]:
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    return None, {}

            if Phase.FIND_NEXT_GOAL == self._phase:
                self._answered = False
                self._advice = False
                self._goalVic = None
                self._goalLoc = None
                zones = self._getDropZones(state)
                remainingZones = []
                remainingVics = []
                remaining = {}
                for info in zones:
                    if str(info['img_name'])[8:-4] not in self._collectedVictims:
                        remainingZones.append(info)
                        remainingVics.append(str(info['img_name'])[8:-4])
                        remaining[str(info['img_name'])[8:-4]] = info['location']
                if remainingZones:
                    self._remainingZones = remainingZones
                    self._remaining = remaining
                if not remainingZones:
                    return None, {}

                for vic in remainingVics:
                    if vic in self._foundVictims and vic in self._todo and len(self._searchedRooms)==0:
                        self._goalVic = vic
                        self._goalLoc = remaining[vic]
                        self._sendMessage('Moving to ' + self._foundVictimLocs[vic]['room'] + ' to pick up ' + self._goalVic +'. Please come there as well to help me carry ' + self._goalVic + ' to the drop zone.', 'RescueBot')
                        if 'location' in self._foundVictimLocs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__, {'duration_in_ticks': 25}
                        if 'location' not in self._foundVictimLocs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                            return Idle.__name__, {'duration_in_ticks': 25}
                    if vic in self._foundVictims and vic not in self._todo:
                        self._goalVic = vic
                        self._goalLoc = remaining[vic]
                        if 'location' in self._foundVictimLocs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__, {'duration_in_ticks': 25}
                        if 'location' not in self._foundVictimLocs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                            return Idle.__name__, {'duration_in_ticks': 25}
                self._phase = Phase.PICK_UNSEARCHED_ROOM

            if Phase.PICK_UNSEARCHED_ROOM == self._phase:
                self._advice = False
                agent_location = state[self.agent_id]['location']
                unsearchedRooms = [room['room_name'] for room in state.values()
                                   if 'class_inheritance' in room
                                   and 'Door' in room['class_inheritance']
                                   and room['room_name'] not in self._searchedRooms
                                   and room['room_name'] not in self._tosearch]
                if self._remainingZones and len(unsearchedRooms) == 0:
                    self._tosearch = []
                    #self._todo = []
                    self._searchedRooms = []
                    self._sendMessages = []
                    self.received_messages = []
                    self.received_messages_content = []
                    #self._searchedRooms.append(self._door['room_name'])
                    self._sendMessage('Going to re-search all areas.', 'RescueBot')
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    if self._currentDoor == None:
                        self._door = state.get_room_doors(self._getClosestRoom(state, unsearchedRooms, agent_location))[0]
                        self._doormat = state.get_room(self._getClosestRoom(state, unsearchedRooms, agent_location))[-1]['doormat']
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3, 5)
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    if self._currentDoor != None:
                        self._door = state.get_room_doors(self._getClosestRoom(state, unsearchedRooms, self._currentDoor))[0]
                        self._doormat = state.get_room(self._getClosestRoom(state, unsearchedRooms, self._currentDoor))[-1]['doormat']
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3, 5)
                        self._phase = Phase.PLAN_PATH_TO_ROOM

            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                self._navigator.reset_full()
                if self._goalVic and self._goalVic in self._foundVictims and 'location' not in self._foundVictimLocs[self._goalVic].keys():
                    self._door = state.get_room_doors(self._foundVictimLocs[self._goalVic]['room'])[0]
                    self._doormat = state.get_room(self._foundVictimLocs[self._goalVic]['room'])[-1]['doormat']
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3, 5)
                    doorLoc = self._doormat
                else:
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3, 5)
                    doorLoc = self._doormat
                self._navigator.add_waypoints([doorLoc])
                self._tick = state['World']['nr_ticks']
                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                if self._goalVic and self._goalVic in self._collectedVictims:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                if self._goalVic and self._goalVic in self._foundVictims and self._door['room_name'] != self._foundVictimLocs[self._goalVic]['room']:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                if self._door['room_name'] in self._searchedRooms and self._goalVic not in self._foundVictims:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    self._state_tracker.update(state)
                    if self._goalVic in self._foundVictims and str(self._door['room_name']) == self._foundVictimLocs[self._goalVic]['room'] and not self._remove:
                        if self._condition=='weak':
                            self._sendMessage('Moving to ' + str(self._door['room_name']) + ' to pick up ' + self._goalVic + ' together with you.', 'RescueBot')
                        else:
                            self._sendMessage('Moving to ' + str(self._door['room_name']) + ' to pick up ' + self._goalVic + '.', 'RescueBot')
                    if self._goalVic not in self._foundVictims and not self._remove or not self._goalVic and not self._remove :
                        self._sendMessage('Moving to ' + str(self._door['room_name']) + ' because it is the closest unsearched area.', 'RescueBot')
                    self._currentDoor = self._door['location']
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action != None:
                        for info in state.values():
                            if 'class_inheritance' in info and 'ObstacleObject' in info[
                                'class_inheritance'] and 'stone' in info['obj_id'] and info['location'] not in [(9, 4), (9, 7), (9, 19), (21, 19)]:
                                self._sendMessage('Reaching ' + str(self._door['room_name']) + ' will take a bit longer because I found stones blocking my path.', 'RescueBot')
                                return RemoveObject.__name__, {'object_id': info['obj_id']}
                        return action, {}
                    self._phase = Phase.REMOVE_OBSTACLE_IF_NEEDED

            if Phase.REMOVE_OBSTACLE_IF_NEEDED == self._phase:
                objects = []
                agent_location = state[self.agent_id]['location']
                for info in state.values():
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'rock' in info['obj_id']:
                        objects.append(info)
                        if self._answered == False and not self._remove and not self._waiting:                  
                            self._sendMessage('Found rock blocking ' + str(self._door['room_name']) + '. Please decide whether to "Remove" or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(self._collectedVictims) + ' \n explore - areas searched: area ' + str(self._searchedRooms).replace('area ','') + ' \
                                \n clock - removal time: 5 seconds \n afstand - distance between us: ' + self._distanceHuman ,'RescueBot')
                            self._waiting = True                          
                        
                        if self.received_messages_content and self.received_messages_content[-1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            self._tosearch.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove' or self._remove:
                            if not self._remove:
                                self._answered = True
                            if not state[{'is_human_agent': True}]:
                                self._sendMessage('Please come to ' + str(self._door['room_name']) + ' to remove rock.','RescueBot')
                                return None, {}
                            if state[{'is_human_agent': True}]:
                                self._sendMessage('Lets remove rock blocking ' + str(self._door['room_name']) + '!','RescueBot')
                                return None, {}
                        else:
                            return None, {}

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'tree' in info['obj_id']:
                        objects.append(info)
                        if self._answered == False and not self._remove and not self._waiting:
                            self._sendMessage('Found tree blocking  ' + str(self._door['room_name']) + '. Please decide whether to "Remove" or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(self._collectedVictims) + '\n explore - areas searched: area ' + str(self._searchedRooms).replace('area ','') + ' \
                                \n clock - removal time: 10 seconds','RescueBot')
                            self._waiting = True
                            
                        if self.received_messages_content and self.received_messages_content[-1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            self._tosearch.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove' or self._remove:
                            if not self._remove:
                                self._answered = True
                                self._waiting = False
                                self._sendMessage('Removing tree blocking ' + str(self._door['room_name']) + '.','RescueBot')
                            if self._remove:
                                self._sendMessage('Removing tree blocking ' + str(self._door['room_name']) + ' because you asked me to.', 'RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}
                        else:
                            return None, {}

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'stone' in info['obj_id']:
                        objects.append(info)
                        if self._answered == False and not self._remove and not self._waiting:
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '. Please decide whether to "Remove together", "Remove alone", or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(self._collectedVictims) + ' \n explore - areas searched: area ' + str(self._searchedRooms).replace('area','') + ' \
                                \n clock - removal time together: 3 seconds \n afstand - distance between us: ' + self._distanceHuman + '\n clock - removal time alone: 20 seconds','RescueBot')
                            self._waiting = True
                                                        
                        if self.received_messages_content and self.received_messages_content[-1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            self._tosearch.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove alone' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            self._sendMessage('Removing stones blocking ' + str(self._door['room_name']) + '.','RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove together' or self._remove:
                            if not self._remove:
                                self._answered = True
                            if not state[{'is_human_agent': True}]:
                                self._sendMessage(
                                    'Please come to ' + str(self._door['room_name']) + ' to remove stones together.','RescueBot')
                                return None, {}
                            if state[{'is_human_agent': True}]:
                                self._sendMessage('Lets remove stones blocking ' + str(self._door['room_name']) + '!','RescueBot')
                                return None, {}
                        else:
                            return None, {}

                if len(objects) == 0:
                    self._answered = False
                    self._remove = False
                    self._waiting = False
                    self._phase = Phase.ENTER_ROOM

            if Phase.ENTER_ROOM == self._phase:
                self._answered = False
                if self._goalVic in self._collectedVictims:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                if self._goalVic in self._foundVictims and self._door['room_name'] != self._foundVictimLocs[self._goalVic]['room']:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                if self._door['room_name'] in self._searchedRooms and self._goalVic not in self._foundVictims:
                    self._currentDoor = None
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    self._state_tracker.update(state)
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action != None:
                        return action, {}
                    self._phase = Phase.PLAN_ROOM_SEARCH_PATH

            if Phase.PLAN_ROOM_SEARCH_PATH == self._phase:
                self._agentLoc = int(self._door['room_name'].split()[-1])
                roomTiles = [info['location'] for info in state.values()
                             if 'class_inheritance' in info
                             and 'AreaTile' in info['class_inheritance']
                             and 'room_name' in info
                             and info['room_name'] == self._door['room_name']]
                self._roomtiles = roomTiles
                self._navigator.reset_full()
                self._navigator.add_waypoints(self._efficientSearch(roomTiles))
                self._roomVics = []
                self._phase = Phase.FOLLOW_ROOM_SEARCH_PATH

            if Phase.FOLLOW_ROOM_SEARCH_PATH == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    for info in state.values():
                        if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance']:
                            vic = str(info['img_name'][8:-4])
                            if vic not in self._roomVics:
                                self._roomVics.append(vic)

                            if vic in self._foundVictims and 'location' not in self._foundVictimLocs[vic].keys():
                                self._foundVictimLocs[vic] = {'location': info['location'],'room': self._door['room_name'], 'obj_id': info['obj_id']}
                                if vic == self._goalVic:
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + ' because you told me ' + vic + ' was located here.','RescueBot')
                                    if self._door['room_name'] not in self._searchedRooms:
                                        self._searchedRooms.append(self._door['room_name'])
                                    self._phase = Phase.FIND_NEXT_GOAL

                            if 'healthy' not in vic and vic not in self._foundVictims:
                                self._advice = True
                                self._recentVic = vic
                                self._foundVictims.append(vic)
                                self._foundVictimLocs[vic] = {'location': info['location'],'room': self._door['room_name'], 'obj_id': info['obj_id']}
                                if 'mild' in vic and self._answered == False and not self._waiting:
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. Please decide whether to "Rescue together", "Rescue alone", or "Continue" searching. \n \n \
                                        Important features to consider are: \n safe - victims rescued: ' + str(self._collectedVictims) + '\n explore - areas searched: area ' + str(self._searchedRooms).replace('area ','') + '\n \
                                        clock - extra time when rescuing alone: 15 seconds \n afstand - distance between us: ' + self._distanceHuman,'RescueBot')
                                    self._waiting = True
                                        
                                if 'critical' in vic and self._answered == False and not self._waiting:
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. Please decide whether to "Rescue" or "Continue" searching. \n\n \
                                        Important features to consider are: \n explore - areas searched: area ' + str(self._searchedRooms).replace('area','') + ' \n safe - victims rescued: ' + str(self._collectedVictims) + '\n \
                                        afstand - distance between us: ' + self._distanceHuman,'RescueBot')
                                    self._waiting = True    
                    return action, {}

                if self._goalVic in self._foundVictims and self._goalVic not in self._roomVics and self._foundVictimLocs[self._goalVic]['room'] == self._door['room_name']:
                    self._sendMessage(self._goalVic + ' not present in ' + str(self._door['room_name']) + ' because I searched the whole area without finding ' + self._goalVic + '.','RescueBot')
                    self._foundVictimLocs.pop(self._goalVic, None)
                    self._foundVictims.remove(self._goalVic)
                    self._roomVics = []
                    self.received_messages = []
                    self.received_messages_content = []
                if self._door['room_name'] not in self._searchedRooms:
                    self._searchedRooms.append(self._door['room_name'])
                if self.received_messages_content and self.received_messages_content[-1] == 'Rescue' and 'critical' in self._recentVic:
                    self._answered = True
                    self._waiting = False
                    if not state[{'is_human_agent': True}]:
                        self._sendMessage('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(self._recentVic) + ' together.', 'RescueBot')
                    if state[{'is_human_agent': True}]:
                        self._sendMessage('Lets carry ' + str(self._recentVic) + ' together!', 'RescueBot')
                    self._phase = Phase.FIND_NEXT_GOAL
                if self.received_messages_content and self.received_messages_content[-1] == 'Rescue together' and 'mild' in self._recentVic:
                    self._rescue = 'together'
                    self._answered = True
                    self._waiting = False
                    if not state[{'is_human_agent': True}]:
                        self._sendMessage('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(self._recentVic) + ' together.', 'RescueBot')
                    if state[{'is_human_agent': True}]:
                        self._sendMessage('Lets carry ' + str(self._recentVic) + ' together!', 'RescueBot')
                    self._phase = Phase.FIND_NEXT_GOAL
                if self.received_messages_content and self.received_messages_content[-1] == 'Rescue alone' and 'mild' in self._recentVic:
                    self._sendMessage('Picking up ' + self._recentVic + ' in ' + self._door['room_name'] + '.','RescueBot')
                    self._rescue = 'alone'
                    self._answered = True
                    self._waiting = False
                    self._phase = Phase.FIND_NEXT_GOAL

                if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                    self._answered = True
                    self._waiting = False
                    self._todo.append(self._recentVic)
                    self._phase = Phase.FIND_NEXT_GOAL
                if self.received_messages_content and self._advice and self.received_messages_content[-1] != 'Rescue' and self.received_messages_content[-1] != 'Continue':
                    return None, {}
                if not self._advice:
                    self._phase = Phase.FIND_NEXT_GOAL
                return Idle.__name__, {'duration_in_ticks': 25}

            if Phase.PLAN_PATH_TO_VICTIM == self._phase:
                #if 'mild' in self._goalVic:
                    #self._sendMessage('Picking up ' + self._goalVic + ' in ' + self._foundVictimLocs[self._goalVic]['room'] + '.','RescueBot')
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._foundVictimLocs[self._goalVic]['location']])
                self._phase = Phase.FOLLOW_PATH_TO_VICTIM

            if Phase.FOLLOW_PATH_TO_VICTIM == self._phase:
                if self._goalVic and self._goalVic in self._collectedVictims:
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    self._state_tracker.update(state)
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action != None:
                        return action, {}
                    self._phase = Phase.TAKE_VICTIM

            if Phase.TAKE_VICTIM == self._phase:
                roomTiles = [info['location'] for info in state.values()
                             if 'class_inheritance' in info
                             and 'AreaTile' in info['class_inheritance']
                             and 'room_name' in info
                             and info['room_name'] == self._foundVictimLocs[self._goalVic]['room']]
                self._roomtiles = roomTiles
                objects = []
                for info in state.values():
                    if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] and 'critical' in info['obj_id'] and info['location'] in self._roomtiles or \
                        'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] and 'mild' in info['obj_id'] and info['location'] in self._roomtiles and self._rescue=='together' or \
                        self._goalVic in self._foundVictims and self._goalVic in self._todo and len(self._searchedRooms)==0 and 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] and 'critical' in info['obj_id'] and info['location'] in self._roomtiles or \
                        self._goalVic in self._foundVictims and self._goalVic in self._todo and len(self._searchedRooms)==0 and 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] and 'mild' in info['obj_id'] and info['location'] in self._roomtiles:
                        objects.append(info)
                        if self._goalVic not in self._collectedVictims:
                            self._collectedVictims.append(self._goalVic)
                        if not 'Human' in info['name']:
                            return None, {}
                if len(objects) == 0 and 'critical' in self._goalVic or len(objects) == 0 and 'mild' in self._goalVic and self._rescue=='together':
                    if self._goalVic not in self._collectedVictims:
                        self._collectedVictims.append(self._goalVic)
                    self._phase = Phase.PLAN_PATH_TO_DROPPOINT
                if 'mild' in self._goalVic and self._rescue=='alone':
                    self._phase = Phase.PLAN_PATH_TO_DROPPOINT
                    if self._goalVic not in self._collectedVictims:
                        self._collectedVictims.append(self._goalVic)
                    self._carrying = True
                    return CarryObject.__name__, {'object_id': self._foundVictimLocs[self._goalVic]['obj_id']}

            if Phase.PLAN_PATH_TO_DROPPOINT == self._phase:
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._goalLoc])
                self._phase = Phase.FOLLOW_PATH_TO_DROPPOINT

            if Phase.FOLLOW_PATH_TO_DROPPOINT == self._phase:
                if 'mild' in self._goalVic and self._rescue=='alone':
                    self._sendMessage('Transporting ' + self._goalVic + ' to the drop zone.', 'RescueBot')
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.DROP_VICTIM

            if Phase.DROP_VICTIM == self._phase:
                if 'mild' in self._goalVic and self._rescue=='alone':
                    self._sendMessage('Delivered ' + self._goalVic + ' at the drop zone.', 'RescueBot')
                self._phase = Phase.FIND_NEXT_GOAL
                self._rescue = None
                self._currentDoor = None
                self._tick = state['World']['nr_ticks']
                self._carrying = False
                return Drop.__name__, {}

    def _getDropZones(self, state: State):
        '''
        @return list of drop zones (their full dict), in order (the first one is the
        the place that requires the first drop)
        '''
        places = state[{'is_goal_block': True}]
        places.sort(key=lambda info: info['location'][1])
        zones = []
        for place in places:
            if place['drop_zone_nr'] == 0:
                zones.append(place)
        return zones

    def _processMessages(self, state, teamMembers, condition):
        '''
        process incoming messages.
        Reported blocks are added to self._blocks
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)

        for mssgs in receivedMessages.values():
            for msg in mssgs:
                if msg.startswith("Search:"):
                    area = 'area ' + msg.split()[-1]
                    if area not in self._searchedRooms:
                        self._searchedRooms.append(area)
                if msg.startswith("Found:"):
                    if len(msg.split()) == 6:
                        foundVic = ' '.join(msg.split()[1:4])
                    else:
                        foundVic = ' '.join(msg.split()[1:5])
                    loc = 'area ' + msg.split()[-1]
                    if loc not in self._searchedRooms:
                        self._searchedRooms.append(loc)
                    if foundVic not in self._foundVictims:
                        self._foundVictims.append(foundVic)
                        self._foundVictimLocs[foundVic] = {'room': loc}
                    if foundVic in self._foundVictims and self._foundVictimLocs[foundVic]['room'] != loc:
                        self._foundVictimLocs[foundVic] = {'room': loc}
                    if condition=='weak':
                        self._rescue = 'together'
                    if 'mild' in foundVic and condition!='weak':
                        self._todo.append(foundVic)
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
                        self._foundVictimLocs[collectVic] = {'room': loc}
                    if collectVic in self._foundVictims and self._foundVictimLocs[collectVic]['room'] != loc:
                        self._foundVictimLocs[collectVic] = {'room': loc}
                    if condition=='weak' and 'mild' not in collectVic and collectVic not in self._collectedVictims:
                        self._collectedVictims.append(collectVic)
                    if condition!='weak' and collectVic not in self._collectedVictims:
                        self._collectedVictims.append(collectVic)
                    if condition=='weak':
                        self._rescue = 'together'
                if msg.startswith('Remove:'):
                    if not self._carrying:
                        area = 'area ' + msg.split()[-1]
                        self._door = state.get_room_doors(area)[0]
                        self._doormat = state.get_room(area)[-1]['doormat']
                        if area in self._searchedRooms:
                            self._searchedRooms.remove(area)
                        self.received_messages = []
                        self.received_messages_content = []
                        self._remove = True
                        self._sendMessage(
                            'Moving to ' + str(self._door['room_name']) + ' to help you remove an obstacle.',
                            'RescueBot')
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        area = 'area ' + msg.split()[-1]
                        self._sendMessage('Will come to ' + area + ' after dropping ' + self._goalVic + '.',
                                          'RescueBot')
            if mssgs and mssgs[-1].split()[-1] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']:
                self._humanLoc = int(mssgs[-1].split()[-1])


    def _sendMessage(self, mssg, sender):
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages_content and 'Our score is' not in msg.content:
            self.send_message(msg)
            self._sendMessages.append(msg.content)
        if 'Our score is' in msg.content:
            self.send_message(msg)

    def _getClosestRoom(self, state, objs, currentDoor):
        agent_location = state[self.agent_id]['location']
        locs = {}
        for obj in objs:
            locs[obj] = state.get_room_doors(obj)[0]['location']
        dists = {}
        for room, loc in locs.items():
            if currentDoor != None:
                dists[room] = utils.get_distance(currentDoor, loc)
            if currentDoor == None:
                dists[room] = utils.get_distance(agent_location, loc)

        return min(dists, key=dists.get)

    def _efficientSearch(self, tiles):
        x = []
        y = []
        for i in tiles:
            if i[0] not in x:
                x.append(i[0])
            if i[1] not in y:
                y.append(i[1])
        locs = []
        for i in range(len(x)):
            if i % 2 == 0:
                locs.append((x[i], min(y)))
            else:
                locs.append((x[i], max(y)))
        return locs

    def _dynamicMessage(self, mssg1, mssg2, explanation, sender):
        if explanation not in self._providedExplanations:
            self._sendMessage(mssg1, sender)
            self._providedExplanations.append(explanation)
        if 'Searching' in mssg1:
            if explanation in self._providedExplanations and mssg1 not in self._sendMessages[-5:]:
                self._sendMessage(mssg2, sender)
        if 'Found' in mssg1:
            history = [mssg2[:-1] in mssg for mssg in self._sendMessages]
            if explanation in self._providedExplanations and True not in history:
                self._sendMessage(mssg2, sender)
        if 'Searching' not in mssg1 and 'Found' not in mssg1:
            if explanation in self._providedExplanations and self._sendMessages[-1] != mssg1:
                self._sendMessage(mssg2, sender)