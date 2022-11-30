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
import numpy as np
from agents1.BaselineAgent import BaselineAgent, Phase


class WorkloadAgent(BaselineAgent):

    def __init__(self, slowdown:int):
        super(WorkloadAgent, self).__init__(slowdown)
        self._workload = [0,0,1/3,0]
        self._workloadVal = 0
        self._explanationChoice = 3
        self._humanLastWorkTick = 0
        self._humanWorkTime = 0
        self._humanLastTask = ''
        self._taskSwitch = 0
        self._min = 0
        self._lastSec = 0


    def decide_on_bw4t_action(self, state: State):
        self._updateTimePressure()
        self._updateTimeOccupied_TaskSeverity_TaskSwitch(state)
        self._computeWorkload()
        
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
            else:
                self._distanceHuman = 'close'

        if self._agentLoc in [1, 2, 5, 6, 8, 9, 11, 12]:
            self._distanceDrop = 'far'
        if self._agentLoc in [3, 4, 7, 10, 13, 14]:
            self._distanceDrop = 'close'

        self._second = state['World']['tick_duration'] * state['World']['nr_ticks']

        for info in state.values():
            if 'is_human_agent' in info and 'Human' in info['name'] and len(info['is_carrying']) > 0 and 'critical' in info['is_carrying'][0]['obj_id']:
                self._collectedVictims.append(info['is_carrying'][0]['img_name'][8:-4])
                self._carryingTogether = True
            if 'is_human_agent' in info and 'Human' in info['name'] and len(info['is_carrying']) == 0:
                self._carryingTogether = False
        if self._carryingTogether == True:
            return None, {}
        agent_name = state[self.agent_id]['obj_id']
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
        self._processMessages(state, self._teamMembers)

        self._sendMessage('Our score is ' + str(state['rescuebot']['score']) + '.', 'RescueBot')
        if self._noSuggestions > 0:
            state['rescuebot']['ignored'] = round(self._ignored / self._noSuggestions, 2)
            self._sendMessage('You ignored me ' + str(state['rescuebot']['ignored']), 'RescueBot')
        workloadRound = round(self._workloadVal, 2)
        self._sendMessage('Your workload is ' +  str(round(self._workload[0],2)) + ' ' + str(round(self._workload[1],2)) + ' ' + str(round(self._workload[2],2)) + ' ' + str(round(self._workload[3],2)) + ' ' + str(workloadRound), 'RescueBot')

        while True:
            if Phase.INTRO0 == self._phase:
                self._sendMessage('Hello! My name is RescueBot. Together we will collaborate and try to search and rescue the 8 victims on our right as quickly as possible. \
                        We have 8 minutes to successfully collect all victims. \
                        Each critical victim (critically injured girl/critically injured elderly woman/critically injured man/critically injured dog) adds 6 points to our score, each mild victim (mildly injured boy/mildly injured elderly man/mildly injured woman/mildly injured cat) 3 points. \
                        If you are ready to begin our mission, press the "Ready!" button.', 'RescueBot')
                if self.received_messages_content and self.received_messages_content[
                    -1] == 'Ready!':  # or not state[{'is_human_agent':True}]:
                    self._phase = Phase.FIND_NEXT_GOAL
                if not state[{'is_human_agent': True}]:
                    self._sendMessage('Ready!', 'RescueBot')
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
                    self._todo = []
                    self._searchedRooms = []
                    self._sendMessages = []
                    self.received_messages = []
                    self.received_messages_content = []
                    self._searchedRooms.append(self._door['room_name'])
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
                        self._sendMessage('Moving to ' + str(self._door['room_name']) + ' to pick up ' + self._goalVic + '.', 'RescueBot')
                    if self._goalVic not in self._foundVictims and not self._remove or not self._goalVic and not self._remove:
                        self._sendMessage('Moving to ' + str(self._door['room_name']) + ' because it is the closest unsearched area.', 'RescueBot')
                    self._currentDoor = self._door['location']
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action != None:
                        for info in state.values():
                            if 'class_inheritance' in info and 'ObstacleObject' in info[
                                'class_inheritance'] and 'stone' in info['obj_id'] and info['location'] not in [(9, 4),(9, 7),(9, 19),(21,19)]:
                                self._sendMessage('Reaching ' + str(self._door['room_name']) + ' will take a bit longer because I found stones blocking my path.','RescueBot')
                                return RemoveObject.__name__, {'object_id': info['obj_id']}
                        return action, {}
                    self._phase = Phase.REMOVE_OBSTACLE_IF_NEEDED

            if Phase.REMOVE_OBSTACLE_IF_NEEDED == self._phase:
                objects = []
                agent_location = state[self.agent_id]['location']
                for info in state.values():
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'rock' in info['obj_id']:
                        objects.append(info)
                        if self._distanceHuman == 'close' and self._second < 240 and self._criticalFound < 2 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 5/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If we had found more than 1 critical victim, I would have suggested to remove rock.']
                            self._sendMessage('Found rock blocking ' + str(self._door['room_name']) + '. \
                                        I suggest to continue searching instead of removing rock' + messages[self._explanationChoice] + ' Select your decision using the buttons "Remove" or "Continue".','RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True

                        if self._distanceHuman == 'close' and self._second > 240 and self._criticalFound < 2 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 7/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If we had found more than 1 critical victim, I would have suggested to remove rock.']
                            self._sendMessage('Found rock blocking  ' + str(self._door['room_name']) + '. \
                                        I suggest to continue searching instead of removing rock' + messages[self._explanationChoice] + ' Select your decision using the buttons "Remove" or "Continue".','RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True

                        if self._distanceHuman == 'close' and self._second < 240 and self._criticalFound > 1 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + ' and have around ' + str(round((480 - self._second) / 60)) + ' minutes left.',
                                        ': 7/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + ' and have around ' + str(round((480 - self._second) / 60)) + ' minutes left. If we had found less than 2 critical victims, I would have suggested to continue searching.']
                            self._sendMessage('Found rock blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to remove rock instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Remove" or "Continue".','RescueBot')
                            self._suggestion = ['Remove']
                            self._waiting = True

                        if self._distanceHuman == 'close' and self._second > 240 and self._criticalFound > 1 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 5/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If we had found less than 2 critical victims, I would have suggested to continue searching.']
                            self._sendMessage('Found rock blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to remove rock instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Remove" or "Continue".','RescueBot')
                            self._suggestion = ['Remove']
                            self._waiting = True

                        if self._distanceHuman == 'far' and self._second < 240 and self._criticalFound < 2 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 8/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If we had found less than 1 critical victim, I would have suggested to remove rock.']
                            self._sendMessage('Found rock blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to continue searching instead of removing rock' + messages[self._explanationChoice] + ' Select your decision using the buttons "Remove" or "Continue".','RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True

                        if self._distanceHuman == 'far' and self._second > 240 and self._criticalFound < 2 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because the distance between us is large.',
                                        ': 8/9 rescuers would decide the same, because the distance between us is large. If we had found more than 1 critical victim, I would have suggested to remove rock.']
                            self._sendMessage('Found rock blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to continue searching instead of removing rock' + messages[self._explanationChoice] + ' Select your decision using the buttons "Remove" or "Continue".','RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True

                        if self._distanceHuman == 'far' and self._second < 240 and self._criticalFound > 1 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 6/9 rescuers would decide the same.',': 6/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 6/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If we had found less than 2 critical victims, I would have suggested to continue searching.']
                            self._sendMessage('Found rock blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to remove rock instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Remove" or "Continue".','RescueBot')
                            self._suggestion = ['Remove']
                            self._waiting = True

                        if self._distanceHuman == 'far' and self._second > 240 and self._criticalFound > 1 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left.',
                                        ': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left. If we had found less than 2 critical victims, I would have suggested to continue searching.']
                            self._sendMessage('Found rock blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to remove rock instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Remove" or "Continue".','RescueBot')
                            self._suggestion = ['Remove']
                            self._waiting = True

                        if self.received_messages_content and self.received_messages_content[-1] == 'Continue' and not self._remove:
                            if self.received_messages_content[-1] not in self._suggestion:
                                self._ignored += 1
                            self._noSuggestions += 1
                            self._answered = True
                            self._waiting = False
                            self._tosearch.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove' or self._remove:
                            self._changeWorkLoadMsg("Remove")
                            if self.received_messages_content[-1] not in self._suggestion and not self._remove:
                                self._ignored += 1
                            if not self._remove:
                                self._noSuggestions += 1
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
                        if self._second < 240 and self._criticalFound < 2 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because removing tree only takes around 10 seconds.',
                                        ': 5/9 rescuers would decide the same, because removing tree only takes around 10 seconds. If we had less than 4 minutes left, I would have suggested to continue searching.']
                            self._sendMessage('Found tree blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to remove tree instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Remove" or "Continue".','RescueBot')
                            self._suggestion = ['Remove']
                            self._waiting = True

                        if self._second < 240 and self._criticalFound > 1 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because removing tree only takes around 10 seconds.',
                                        ': 8/9 rescuers would decide the same, because removing tree only takes around 10 seconds. If we had less than 4 minutes left, I would have suggested to continue searching.']
                            self._sendMessage('Found tree blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to remove tree instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Remove" or "Continue".','RescueBot')
                            self._suggestion = ['Remove']
                            self._waiting = True

                        if self._second > 240 and self._criticalFound < 2 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 8/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If we had more than 4 minutes left, I would have suggested to remove tree.']
                            self._sendMessage('Found tree blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to continue searching instead of removing tree' + messages[self._explanationChoice] + ' Select your decision using the buttons "Remove" or "Continue".','RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True

                        if self._second > 240 and self._criticalFound > 1 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left.',
                                        ': 7/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left. If we had more than 4 minutes left, I would have suggested to remove tree.']
                            self._sendMessage('Found tree blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to continue searching instead of removing tree' + messages[self._explanationChoice] + ' Select your decision using the buttons "Remove" or "Continue".','RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True

                        if self.received_messages_content and self.received_messages_content[-1] == 'Continue' and not self._remove:
                            if self.received_messages_content[-1] not in self._suggestion:
                                self._ignored += 1
                            self._noSuggestions += 1
                            self._answered = True
                            self._waiting = False
                            self._tosearch.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove' or self._remove:
                            self._changeWorkLoadMsg("Remove")
                            if self.received_messages_content[-1] not in self._suggestion and not self._remove:
                                self._ignored += 1
                            if not self._remove:
                                self._noSuggestions += 1
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
                        if self._distanceHuman == 'far' and self._criticalFound < 2 and self._second < 240 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left.',
                                        ': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left. If we had found more than 1 critical victim, I would have suggested to remove alone. If the distance between us had been small, I would have suggested to remove together.']
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to continue searching instead of removing stones' + messages[self._explanationChoice] + ' Select your decision using the buttons "Continue", "Remove alone" or "Remove together".','RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True

                        if self._distanceHuman == 'far' and self._criticalFound > 1 and self._second < 240 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 8/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If the distance between us had been small, I would have suggested to remove together. If we had found less than 2 critical victims, I would have suggested to continue searching.']
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to remove stones alone instead of continue searching or removing together' +
                                              messages[self._explanationChoice] + ' Select your decision using the buttons "Continue", "Remove alone" or "Remove together".','RescueBot')
                            self._suggestion = ['Remove alone']
                            self._waiting = True

                        if self._distanceHuman == 'far' and self._criticalFound < 2 and self._second > 240 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 7/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If we had found more than 1 critical victim, I would have suggested to remove alone. If the distance between us had been small, and we had more than 4 minutes left or found more than 1 critical victim, I would have suggested to remove together.']
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to continue searching instead of removing stones' + messages[self._explanationChoice] + ' Select your decision using the buttons "Continue", "Remove alone" or "Remove together".','RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True

                        if self._distanceHuman == 'far' and self._criticalFound > 1 and self._second > 240 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left and the distance between us is large.',
                                        ': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left and the distance between us is large. If the distance between us had been small, I would have suggested to remove together. If we had found less than 2 critical victims, I would have suggested to continue searching.']
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to remove stones alone instead of continue searching or removing together' + messages[self._explanationChoice] + ' Select your decision using the buttons "Continue", "Remove alone" or "Remove together".','RescueBot')
                            self._suggestion = ['Remove alone']
                            self._waiting = True

                        if self._distanceHuman == 'close' and self._criticalFound < 2 and self._second < 240 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 8/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If the distance between us had been large and we had found more than 1 critical victim, I would have suggested to remove alone.']
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to remove stones together or to continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Continue", "Remove alone" or "Remove together".','RescueBot')
                            self._suggestion = ['Remove together', 'Continue']
                            self._waiting = True

                        if self._distanceHuman == 'close' and self._criticalFound > 1 and self._second < 240 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 6/9 rescuers would decide the same.',': 6/9 rescuers would decide the same, because the distance between us is small and removing together only takes around 3 seconds.',
                                        ': 6/9 rescuers would decide the same, because the distance between us is small and removing together only takes around 3 seconds. If the distance between us had been large, I would have suggested to remove alone. If we had found less than 2 critical victims, I would have suggested to continue searching.']
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to remove stones together instead of continue searching or removing alone' + messages[self._explanationChoice] + ' Select your decision using the buttons "Continue", "Remove alone" or "Remove together".','RescueBot')
                            self._suggestion = ['Remove together']
                            self._waiting = True

                        if self._distanceHuman == 'close' and self._criticalFound < 2 and self._second > 240 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 5/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If the distance between us had been large and we had found more than 1 critical victim, I would have suggested to remove alone. If we had found more than 1 critical victim, I would have suggested to remove together.']
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to continue searching instead of removing stones' + messages[self._explanationChoice] + ' Select your decision using the buttons "Continue", "Remove alone" or "Remove together".','RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True

                        if self._distanceHuman == 'close' and self._criticalFound > 1 and self._second > 240 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because the distance between us is small.',
                                        ': 7/9 rescuers would decide the same, because the distance between us is small. If the distance between us had been large, I would have suggested to remove alone. If we had found less than 2 critical victims, I would have suggested to continue searching.']
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '.  \
                                        I suggest to remove stones together instead of continue searching or removing alone' + messages[self._explanationChoice] + ' Select your decision using the buttons "Continue", "Remove alone" or "Remove together".','RescueBot')
                            self._suggestion = ['Remove together']
                            self._waiting = True

                        if self.received_messages_content and self.received_messages_content[-1] == 'Continue' and not self._remove:
                            if self.received_messages_content[-1] not in self._suggestion:
                                self._ignored += 1
                            self._noSuggestions += 1
                            self._answered = True
                            self._waiting = False
                            self._tosearch.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove alone' and not self._remove:
                            self._changeWorkLoadMsg("Remove alone")
                            if self.received_messages_content[-1] not in self._suggestion:
                                self._ignored += 1
                            self._noSuggestions += 1
                            self._answered = True
                            self._waiting = False
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove together' or self._remove:
                            self._changeWorkLoadMsg("Remove together")
                            if self.received_messages_content[-1] not in self._suggestion and not self._remove:
                                self._ignored += 1
                            if not self._remove:
                                self._noSuggestions += 1
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
                             and info['room_name'] == self._door['room_name']
                             ]
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
                                    self._searchedRooms.append(self._door['room_name'])
                                    self._phase = Phase.FIND_NEXT_GOAL

                            if 'healthy' not in vic and vic not in self._foundVictims:
                                self._advice = True
                                self._recentVic = vic
                                self._foundVictims.append(vic)
                                self._foundVictimLocs[vic] = {'location': info['location'],'room': self._door['room_name'], 'obj_id': info['obj_id']}
                                if 'mild' in vic and self._second < 240 and self._criticalRescued < 2 and self._distanceDrop == 'close' and self._answered == False and not self._waiting:
                                    messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we only rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '.',
                                                ': 5/9 rescuers would decide the same, because we only rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '. If we had rescued  more than 1 critical victim, I would have suggested to rescue ' + vic + '.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                                I suggest to continue searching instead of rescuing ' + vic + messages[self._explanationChoice] + ' Select your decision using the buttons "Rescue" or "Continue".','RescueBot')
                                    self._suggestion = ['Continue']
                                    self._waiting = True

                                if 'mild' in vic and self._second < 240 and self._criticalRescued > 1 and self._distanceDrop == 'close' and self._answered == False and not self._waiting:
                                    messages = ['.', ': 5/9 rescuers would decide the same',': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left and the distance to the drop zone is small.',
                                                ': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left and the distance to the drop zone is small. If we had rescued less than 2 critical victims, I would have suggested to continue searching.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                                I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Rescue" or "Continue".','RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True

                                if 'mild' in vic and self._second > 240 and self._criticalRescued < 2 and self._distanceDrop == 'close' and self._answered == False and not self._waiting:
                                    messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because we only rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '.',
                                                ': 8/9 rescuers would decide the same, because we only rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '. If we had rescued  more than 1 critical victim, I would have suggested to rescue ' + vic + '.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                                I suggest to continue searching instead of rescuing ' + vic + messages[self._explanationChoice] + ' Select your decision using the buttons "Rescue" or "Continue".','RescueBot')
                                    self._suggestion = ['Continue']
                                    self._waiting = True

                                if 'mild' in vic and self._second > 240 and self._criticalRescued > 1 and self._distanceDrop == 'close' and self._answered == False and not self._waiting:
                                    messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because we already rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '.',
                                                ': 7/9 rescuers would decide the same, because we already rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '. If we had rescued less than 2 critical victims, I would have suggested to continue searching.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                                I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Rescue" or "Continue".','RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True

                                if 'mild' in vic and self._second < 240 and self._criticalRescued < 2 and self._distanceDrop == 'far' and self._answered == False and not self._waiting:
                                    messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we only rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '.',
                                                ': 5/9 rescuers would decide the same, because we only rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '. If we had rescued  more than 1 critical victim, I would have suggested to rescue ' + vic + '.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                                I suggest to continue searching instead of rescuing ' + vic + messages[self._explanationChoice] + ' Select your decision using the buttons "Rescue" or "Continue".','RescueBot')
                                    self._suggestion = ['Continue']
                                    self._waiting = True

                                if 'mild' in vic and self._second < 240 and self._criticalRescued > 1 and self._distanceDrop == 'far' and self._answered == False and not self._waiting:
                                    messages = ['.', ': 6/9 rescuers would decide the same.',': 6/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left.',
                                                ': 6/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left. If we had rescued less than 2 critical victims, I would have suggested to continue searching.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                                I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Rescue" or "Continue".','RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True

                                if 'mild' in vic and self._second > 240 and self._criticalRescued < 2 and self._distanceDrop == 'far' and self._answered == False and not self._waiting:
                                    messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because we only rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '.',
                                                ': 7/9 rescuers would decide the same, because we only rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '. If we had rescued  more than 1 critical victim, I would have suggested to rescue ' + vic + '.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                                I suggest to continue searching instead of rescuing ' + vic + messages[self._explanationChoice] + ' Select your decision using the buttons "Rescue" or "Continue".','RescueBot')
                                    self._suggestion = ['Continue']
                                    self._waiting = True

                                if 'mild' in vic and self._second > 240 and self._criticalRescued > 1 and self._distanceDrop == 'far' and self._answered == False and not self._waiting:
                                    messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we already rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + ' and have around ' + str(round((480 - self._second) / 60)) + ' minutes left.',
                                                ': 5/9 rescuers would decide the same, because we already rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + ' and have around ' + str(round((480 - self._second) / 60)) + ' minutes left. If we had rescued less than 2 critical victims, I would have suggested to continue searching.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                                I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Rescue" or "Continue".','RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True

                                if self._explanationChoice == 3:
                                    self._explanationChoice = 2

                                if 'critical' in vic and self._distanceDrop == 'far' and self._distanceHuman == 'close' and self._second < 240 and self._answered == False and not self._waiting:
                                    messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because the distance between us is small.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                                I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Rescue" or "Continue".','RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True

                                if 'critical' in vic and self._distanceDrop == 'close' and self._distanceHuman == 'far' and self._second < 240 and self._answered == False and not self._waiting:
                                    messages = ['.', ': 6/9 rescuers would decide the same.',': 6/9 rescuers would decide the same, because the distance to the drop zone is small and we have around ' + str(round((480 - self._second) / 60)) + ' minutes left.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                                I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Rescue" or "Continue".','RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True

                                if 'critical' in vic and self._distanceDrop == 'close' and self._distanceHuman == 'close' and self._second > 240 and self._answered == False and not self._waiting:
                                    messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because critical victims have a higher priority.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                                I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Rescue" or "Continue".','RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True

                                if 'critical' in vic and self._distanceDrop == 'close' and self._distanceHuman == 'close' and self._second < 240 and self._answered == False and not self._waiting:
                                    messages = ['.', ': 6/9 rescuers would decide the same.',': 6/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                                I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Rescue" or "Continue".','RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True

                                if 'critical' in vic and self._distanceDrop == 'close' and self._distanceHuman == 'close' and self._second > 240 and self._answered == False and not self._waiting:
                                    messages = ['.', ': 6/9 rescuers would decide the same.',': 6/9 rescuers would decide the same, because the distance between us is small.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                                I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Rescue" or "Continue".','RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True

                                if 'critical' in vic and self._distanceDrop == 'far' and self._distanceHuman == 'close' and self._second > 240 and self._answered == False and not self._waiting:
                                    messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because critical victims have a higher priority.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                                I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Rescue" or "Continue".','RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True

                                if 'critical' in vic and self._distanceDrop == 'far' and self._distanceHuman == 'far' and self._second < 240 and self._answered == False and not self._waiting:
                                    messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                                I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Rescue" or "Continue".','RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True

                                if 'critical' in vic and self._distanceDrop == 'far' and self._distanceHuman == 'far' and self._second > 240 and self._answered == False and not self._waiting:
                                    messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                                I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._explanationChoice] + ' Select your decision using the buttons "Rescue" or "Continue".','RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True
                    return action, {}

                if self._goalVic in self._foundVictims and self._goalVic not in self._roomVics and self._foundVictimLocs[self._goalVic]['room'] == self._door['room_name']:
                    self._sendMessage(self._goalVic + ' not present in ' + str(self._door['room_name']) + ' because I searched the whole area without finding ' + self._goalVic + '.','RescueBot')
                    self._foundVictimLocs.pop(self._goalVic, None)
                    self._foundVictims.remove(self._goalVic)
                    self._roomVics = []
                    self.received_messages = []
                    self.received_messages_content = []
                self._searchedRooms.append(self._door['room_name'])
                if self.received_messages_content and self.received_messages_content[-1] == 'Rescue':
                    self._changeWorkLoadMsg("Rescue")
                    if self.received_messages_content[-1] not in self._suggestion:
                        self._ignored += 1
                    self._noSuggestions += 1
                    self._answered = True
                    self._waiting = False
                    if 'critical' in self._recentVic:
                        if not state[{'is_human_agent': True}]:
                            self._sendMessage('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(self._recentVic) + ' together.', 'RescueBot')
                        if state[{'is_human_agent': True}]:
                            self._sendMessage('Lets carry ' + str(self._recentVic) + ' together!', 'RescueBot')
                    self._phase = Phase.FIND_NEXT_GOAL
                if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                    if self.received_messages_content[-1] not in self._suggestion:
                        self._ignored += 1
                    self._noSuggestions += 1
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
                if 'mild' in self._goalVic:
                    self._sendMessage('Picking up ' + self._goalVic + ' in ' + self._foundVictimLocs[self._goalVic]['room'] + '.','RescueBot')
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
                objects = []
                for info in state.values():
                    if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] and 'critical' in info['obj_id'] and info['location'] in self._roomtiles:
                        objects.append(info)
                        if self._goalVic not in self._collectedVictims:
                            self._collectedVictims.append(self._goalVic)
                        if not 'Human' in info['name']:
                            return None, {}
                if len(objects) == 0 and 'critical' in self._goalVic:
                    if self._goalVic not in self._collectedVictims:
                        self._collectedVictims.append(self._goalVic)
                    self._phase = Phase.PLAN_PATH_TO_DROPPOINT
                if 'mild' in self._goalVic:
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
                if 'mild' in self._goalVic:
                    self._sendMessage('Transporting ' + self._goalVic + ' to the drop zone.', 'RescueBot')
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.DROP_VICTIM

            if Phase.DROP_VICTIM == self._phase:
                if 'mild' in self._goalVic:
                    self._sendMessage('Delivered ' + self._goalVic + ' at the drop zone.', 'RescueBot')
                if 'critical' in self._goalVic:
                    self._workload[3] = self._workload[3]-0.1
                    if self._workload[3] < 0:
                        self._workload[3] = 0
                self._phase = Phase.FIND_NEXT_GOAL
                self._currentDoor = None
                self._tick = state['World']['nr_ticks']
                self._carrying = False
                return Drop.__name__, {}
    
    def filter_bw4t_observations(self, state):
        return super(WorkloadAgent, self).filter_bw4t_observations(state)

    def _computeWorkload(self):
        cognitiveLoad = 0.2 * self._workload[0] + 0.8 * self._workload[1]
        affectiveLoad = 0.5 * self._workload[2] + 0.5 * self._workload[3]

        self._workloadVal =  0.5 * cognitiveLoad + 0.5 * affectiveLoad

        if self._workloadVal > 0.75:
            self._explanationChoice = 0
        elif self._workloadVal > 0.5:
            self._explanationChoice = 1
        elif self._workloadVal > 0.25:
            self._explanationChoice = 2
        else:
            self._explanationChoice = 3

    def _updateTimePressure(self):
        second = (0 if self._second is None else self._second)

        if second > self._lastSec:
            self._workload[3] += (second - self._lastSec)/480
            self._lastSec = second
        if self._workload[3] > 1:
            self._workload[3] = 1

    def _changeWorkLoadMsg(self, msg):
        if msg == 'Remove' or 'Remove alone' or 'Remove together':
            self._taskSwitch = self._taskSwitch + 1
            self._workload[2] = 2/3
            self._humanLastTask = 'remove'

        if msg == 'Rescue':
            self._taskSwitch = self._taskSwitch + 1
            self._workload[2] = 3 / 3
            self._humanLastTask = 'rescue'

    def _updateTimeOccupied_TaskSeverity_TaskSwitch(self, state ):
        self._second = int(0 if self._second is None else self._second)

        if self._second > 0:
            if self._second / 60 > self._min:
                self._min += 1
                self._taskSwitch = 0
                self._humanWorkTime = 0

        if state[{'is_human_agent':True}]:
            if state['human'] is not None:
                if state['human']['current_action'] is not None:
                    if self._humanLastWorkTick != state['human']['current_action_started_at_tick']:
                        self._humanLastWorkTick = state['human']['current_action_started_at_tick']

                        self._humanWorkTime += state['human']['current_action_duration'] / 10
                        self._workload[0] = self._humanWorkTime/60

                        if (state['human']['current_action'] == 'MoveSouthEast' or 'MoveWest' or 'MoveNorthEast' or 'MoveSouth' or 'Move' or 'MoveEast' or 'MoveNorth'or 'MoveNorthWest'):
                            current = 'move'
                            self._workload[2] = 1 / 3
                        elif state['human']['current_action'] == ('RemoveObject' or 'RemoveObjectTogether'):
                            current = 'remove'
                            self._workload[2] = 2 / 3
                        else:
                            current = 'rescue'
                            self._workload[2] = 1

                        if self._humanLastTask != current:
                            self._humanLastTask = current

                            self._taskSwitch += 1

                            self._workload[1] = self._taskSwitch/ 6
                            if self._workload[1] > 1:
                                self._workload[1] = 1