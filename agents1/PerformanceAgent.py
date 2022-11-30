import enum
from matrx import utils
from matrx.actions.object_actions import RemoveObject
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.messages.message import Message
from actions1.customActions import *
from actions1.customActions import CarryObject, Drop
from brains1.BW4TBrain import BW4TBrain

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

class PerformanceAgent(BW4TBrain):
    def __init__(self, slowdown: int):
        super().__init__(slowdown)
        self._slowdown = slowdown
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
        self._confidence = True
        self._waitingHuman = False
        self._complySpeed = 800
        self._lateComply = 0
        self._responseSpeed = 800
        self._lateResponse = 0
        self._reminderMessage = ""
        self._searchSpeed = 58
        self._searchTookLong = 0
        self._reminder = True
        self._performanceMetric = 0
        self._progressMessage = ""

    def initialize(self):
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

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
            else:
                self._distanceHuman = 'close'

        if self._agentLoc in [1, 2, 5, 6, 8, 9, 11, 12]:
            self._distanceDrop = 'far'
        if self._agentLoc in [3, 4, 7, 10, 13, 14]:
            self._distanceDrop = 'close'

        self._second = state['World']['tick_duration'] * state['World']['nr_ticks']

        if (self._responseSpeed != 800) and ((self._second - self._responseSpeed) > 11):
            if self._lateResponse <= 0:
                self._sendMessage("I am still waiting for an answer from you in order to proceed.", 'RescueBot')
            elif self._lateResponse == 1:
                self._sendMessage("You still haven't given an answer to my question, please do, so I can proceed.",'RescueBot')
            else:
                self._sendMessage("While waiting for an answer, we are losing crucial time.", 'RescueBot')

        for info in state.values():
            if 'is_human_agent' in info and 'Human' in info['name'] and self._waitingHuman and self._complySpeed != 0:
                self._waitingHuman = False
                self._sendMessage('Your performance comply speed is ' + str(self._second - self._complySpeed), 'RescueBot')
                if (self._second - self._complySpeed) > 15:
                    self._complySpeed = 800
                    self._lateComply += 1
                else:
                    self._complySpeed = 800
                    self._lateComply = 0

            if 'is_human_agent' in info and 'Human' in info['name'] and len(info['is_carrying']) > 0 and 'critical' in info['is_carrying'][0]['obj_id']:
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
        self._processMessages(state, self._teamMembers)

        if self._second < 1:
            predictedScore = 0
        else:
            predictedScore = round(((480/self._second)*state['rescuebot']['score']), 0)
        progressMessagesGood = ["", "", "", " We are doing great! Continuing like this, we will probably complete the mission with " + str(predictedScore) + "/36 point(s), because we have " + str(round((480 - self._second)/60, 1)) + " minute(s) left and " + str(len(self._collectedVictims)) + " victim(s) rescued."]
        progressMessagesBad = [" We should be a little faster! Continuing like this, we will probably complete the mission with " + str(predictedScore) + "/36 point(s), because we have " + str(round((480 - self._second)/60, 1)) + " minute(s) left and " + str(8-len(self._collectedVictims)) + " victim(s) to rescue.", " It is not recommended to reject high confidence suggestions, because it will likely decrease our performance.", "", ""]
        self._sendMessage('Your performance gap is ' + str(self._second/480 - state['rescuebot']['score']/36), 'RescueBot')
        if (self._second/480 - state['rescuebot']['score']/36) >= 0.20:
            self._performanceMetric = 3
            self._progressMessage = progressMessagesBad[random.randint(0, 3)]
        elif (self._second/480 - state['rescuebot']['score']/36) >= 0.133333:
            self._performanceMetric = 2
            self._progressMessage = progressMessagesBad[random.randint(0, 3)]
        elif (self._second/480 - state['rescuebot']['score']/36) >= 0.066666:
            self._performanceMetric = 1
            self._progressMessage = progressMessagesGood[random.randint(0, 3)]
        else:
            self._performanceMetric = 0
            self._progressMessage = progressMessagesGood[random.randint(0, 3)]

        self._sendMessage('Our score is ' + str(state['rescuebot']['score']) + '.', 'RescueBot')
        if self._noSuggestions > 0:
            state['rescuebot']['ignored'] = round(self._ignored / self._noSuggestions, 2)
            self._sendMessage('You ignored me ' + str(state['rescuebot']['ignored']), 'RescueBot')

        while True:
            if Phase.INTRO0 == self._phase:
                self._sendMessage('Hello! My name is RescueBot. Together we will collaborate and try to search and rescue the 8 victims on our right as quickly as possible. \
                We have 8 minutes to successfully collect all victims. \
                Each critical victim (critically injured girl/critically injured elderly woman/critically injured man/critically injured dog) adds 6 points to our score, each mild victim (mildly injured boy/mildly injured elderly man/mildly injured woman/mildly injured cat) 3 points. \
                If you are ready to begin our mission, press the "Ready!" button.', 'RescueBot')
                if self.received_messages_content and self.received_messages_content[-1] == 'Ready!':
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
                if self._goalVic and self._goalVic in self._foundVictims and 'location' not in self._foundVictimLocs[
                    self._goalVic].keys():
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
                    if self._second > 8 and (self._reminder or (
                            ((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._searchSpeed) > 63)):
                        reminderMessages = [
                            " Do not forget to inform me about the areas you will search (or have searched), because this improves our performance.",
                            " To complete the mission successfully, we have to be a little faster and communicate efficiently.",
                            " Do not forget to keep me informed me about where you will search / who you found / who you will rescue, because this greatly improves efficiency."]
                        self._reminderMessage = reminderMessages[random.randint(0, 2)]
                    elif self._second <= 8:
                        self._reminderMessage = ""
                    else:
                        reminderMessages = ["", " You are fast! Tip: Do not forget to avoid water, because it slows you down.",
                                            " We are doing great! Tip: Do not forget you can search a room with only two steps forward, because you can also see the blocks next to you.",
                                            " We are doing great!", " You are fast!", ""]
                        self._reminderMessage = reminderMessages[random.randint(0, 5)]
                    if self._goalVic in self._foundVictims and str(self._door['room_name']) == self._foundVictimLocs[self._goalVic]['room'] and not self._remove:
                        self._sendMessage('Moving to ' + str(self._door['room_name']) + ' to pick up ' + self._goalVic + '.' + self._reminderMessage,'RescueBot')
                    if self._goalVic not in self._foundVictims and not self._remove or not self._goalVic and not self._remove:
                        self._sendMessage('Moving to ' + str(self._door['room_name']) + ' because it is the closest unsearched area.' + self._reminderMessage,'RescueBot')
                    self._currentDoor = self._door['location']
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action != None:
                        for info in state.values():
                            if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'stone' in info['obj_id'] and info['location'] not in [(9, 4),(9, 7),(9, 19),(21,19)]:
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
                            if self._lateResponse <= 0:
                                responseString = "."
                            elif self._lateResponse == 1:
                                responseString = ", a bit faster than last time, since we lost crucial time."
                            else:
                                responseString = ", please inform me quickly this time, since I am waiting for your response."
                            self._sendMessage('Found rock blocking ' + str(self._door['room_name']) + '. \
                                I suggest to continue searching instead of removing rock' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Remove" or "Continue"' + responseString,'RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = True
                        if self._distanceHuman == 'close' and self._second > 240 and self._criticalFound < 2 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 7/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If we had found more than 1 critical victim, I would have suggested to remove rock.']
                            if self._lateResponse <= 0:
                                responseString = "."
                            elif self._lateResponse == 1:
                                responseString = ", a bit faster than last time, since we lost crucial time."
                            else:
                                responseString = ", please inform me quickly this time, since I am waiting for your response."
                            self._sendMessage('Found rock blocking  ' + str(self._door['room_name']) + '. \
                                I suggest to continue searching instead of removing rock' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Remove" or "Continue"' + responseString,'RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = True
                        if self._distanceHuman == 'close' and self._second < 240 and self._criticalFound > 1 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + ' and have around ' + str(round((480 - self._second) / 60)) + ' minutes left.',
                                        ': 7/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + ' and have around ' + str(round((480 - self._second) / 60)) + ' minutes left. If we had found less than 2 critical victims, I would have suggested to continue searching.']
                            if self._lateResponse <= 0:
                                responseString = "."
                            elif self._lateResponse == 1:
                                responseString = ", a bit faster than last time, since we lost crucial time."
                            else:
                                responseString = ", please inform me quickly this time, since I am waiting for your response."
                            self._sendMessage('Found rock blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to remove rock instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Remove" or "Continue"' + responseString,'RescueBot')
                            self._suggestion = ['Remove']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = True
                        if self._distanceHuman == 'close' and self._second > 240 and self._criticalFound > 1 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 5/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If we had found less than 2 critical victims, I would have suggested to continue searching.']
                            if self._lateResponse <= 0:
                                responseString = "."
                            elif self._lateResponse == 1:
                                responseString = ", a bit faster than last time, since we lost crucial time."
                            else:
                                responseString = ", please inform me quickly this time, since I am waiting for your response."
                            self._sendMessage('Found rock blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to remove rock instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Remove" or "Continue"' + responseString,'RescueBot')
                            self._suggestion = ['Remove']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = False
                        if self._distanceHuman == 'far' and self._second < 240 and self._criticalFound < 2 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 8/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If we had found less than 1 critical victim, I would have suggested to remove rock.']
                            if self._lateResponse <= 0:
                                responseString = "."
                            elif self._lateResponse == 1:
                                responseString = ", a bit faster than last time, since we lost crucial time."
                            else:
                                responseString = ", please inform me quickly this time, since I am waiting for your response."
                            self._sendMessage('Found rock blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to continue searching instead of removing rock' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Remove" or "Continue"' + responseString,'RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = True
                        if self._distanceHuman == 'far' and self._second > 240 and self._criticalFound < 2 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because the distance between us is large.',
                                        ': 8/9 rescuers would decide the same, because the distance between us is large. If we had found more than 1 critical victim, I would have suggested to remove rock.']
                            if self._lateResponse <= 0:
                                responseString = "."
                            elif self._lateResponse == 1:
                                responseString = ", a bit faster than last time, since we lost crucial time."
                            else:
                                responseString = ", please inform me quickly this time, since I am waiting for your response."
                            self._sendMessage('Found rock blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to continue searching instead of removing rock' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Remove" or "Continue"' + responseString,'RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = True
                        if self._distanceHuman == 'far' and self._second < 240 and self._criticalFound > 1 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 6/9 rescuers would decide the same.',': 6/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 6/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If we had found less than 2 critical victims, I would have suggested to continue searching.']
                            if self._lateResponse <= 0:
                                responseString = "."
                            elif self._lateResponse == 1:
                                responseString = ", a bit faster than last time, since we lost crucial time."
                            else:
                                responseString = ", please inform me quickly this time, since I am waiting for your response."
                            self._sendMessage('Found rock blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to remove rock instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Remove" or "Continue"' + responseString,'RescueBot')
                            self._suggestion = ['Remove']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = True
                        if self._distanceHuman == 'far' and self._second > 240 and self._criticalFound > 1 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left.',
                                        ': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left. If we had found less than 2 critical victims, I would have suggested to continue searching.']
                            if self._lateResponse <= 0:
                                responseString = "."
                            elif self._lateResponse == 1:
                                responseString = ", a bit faster than last time, since we lost crucial time."
                            else:
                                responseString = ", please inform me quickly this time, since I am waiting for your response."
                            self._sendMessage('Found rock blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to remove rock instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Remove" or "Continue"' + responseString,'RescueBot')
                            self._suggestion = ['Remove']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = False
                        if self.received_messages_content and self.received_messages_content[-1] == 'Continue' and not self._remove:
                            if self.received_messages_content[-1] not in self._suggestion:
                                self._ignored += 1
                            self._noSuggestions += 1
                            self._answered = True
                            self._sendMessage('Your performance response time is ' + str(((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed)), 'RescueBot')
                            if ((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed) > 8.5:
                                self._lateResponse += 1
                                self._responseSpeed = 800
                            else:
                                self._lateResponse = 0
                                self._responseSpeed = 800
                            self._waiting = False
                            self._tosearch.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove' or self._remove:
                            if self.received_messages_content[-1] not in self._suggestion and not self._remove:
                                self._ignored += 1
                            if not self._remove:
                                self._noSuggestions += 1
                                self._answered = True
                            self._sendMessage('Your performance response time is ' + str(((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed)), 'RescueBot')
                            if ((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed) > 8.5:
                                self._lateResponse += 1
                                self._responseSpeed = 800
                            else:
                                self._lateResponse = 0
                                self._responseSpeed = 800
                            if not state[{'is_human_agent': True}]:
                                if self._lateComply <= 0:
                                    self._sendMessage(
                                        'Please come to ' + str(self._door['room_name']) + ' to remove rock.','RescueBot')
                                elif self._lateComply == 1:
                                    self._sendMessage('Please come to ' + str(self._door['room_name']) + ' to remove rock, a bit faster than last time, because we lost important time then.','RescueBot')
                                else:
                                    self._sendMessage('Please come to ' + str(self._door['room_name']) + ' to remove rock. You have to be quicker this time, because I am waiting for you.','RescueBot')
                                self._waitingHuman = True
                                self._complySpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                return None, {}
                            if state[{'is_human_agent': True}]:
                                self._sendMessage('Lets remove rock blocking ' + str(self._door['room_name']) + '!','RescueBot')
                                return None, {}
                        else:
                            return None, {}

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'tree' in info['obj_id']:
                        objects.append(info)
                        if self._second < 240 and self._criticalFound < 2 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 5/9 rescuers would decide the same.', ': 5/9 rescuers would decide the same, because removing tree only takes around 10 seconds.',
                                        ': 5/9 rescuers would decide the same, because removing tree only takes around 10 seconds. If we had less than 4 minutes left, I would have suggested to continue searching.']
                            if self._lateResponse <= 0:
                                responseString = "."
                            elif self._lateResponse == 1:
                                responseString = ", a bit faster than last time, since we lost crucial time."
                            else:
                                responseString = ", please inform me quickly this time, since I am waiting for your response."
                            self._sendMessage('Found tree blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to remove tree instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Remove" or "Continue"' + responseString,'RescueBot')
                            self._suggestion = ['Remove']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = False
                        if self._second < 240 and self._criticalFound > 1 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because removing tree only takes around 10 seconds.',
                                        ': 8/9 rescuers would decide the same, because removing tree only takes around 10 seconds. If we had less than 4 minutes left, I would have suggested to continue searching.']
                            if self._lateResponse <= 0:
                                responseString = "."
                            elif self._lateResponse == 1:
                                responseString = ", a bit faster than last time, since we lost crucial time."
                            else:
                                responseString = ", please inform me quickly this time, since I am waiting for your response."
                            self._sendMessage('Found tree blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to remove tree instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Remove" or "Continue"' + responseString,'RescueBot')
                            self._suggestion = ['Remove']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = True
                        if self._second > 240 and self._criticalFound < 2 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 8/9 rescuers would decide the same.', ': 8/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 8/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If we had more than 4 minutes left, I would have suggested to remove tree.']
                            if self._lateResponse <= 0:
                                responseString = "."
                            elif self._lateResponse == 1:
                                responseString = ", a bit faster than last time, since we lost crucial time."
                            else:
                                responseString = ", please inform me quickly this time, since I am waiting for your response."
                            self._sendMessage('Found tree blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to continue searching instead of removing tree' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Remove" or "Continue"' + responseString,'RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = True
                        if self._second > 240 and self._criticalFound > 1 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left.',
                                        ': 7/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left. If we had more than 4 minutes left, I would have suggested to remove tree.']
                            if self._lateResponse <= 0:
                                responseString = "."
                            elif self._lateResponse == 1:
                                responseString = ", a bit faster than last time, since we lost crucial time."
                            else:
                                responseString = ", please inform me quickly this time, since I am waiting for your response."
                            self._sendMessage('Found tree blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to continue searching instead of removing tree' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Remove" or "Continue"' + responseString,'RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = True
                        if self.received_messages_content and self.received_messages_content[-1] == 'Continue' and not self._remove:
                            if self.received_messages_content[-1] not in self._suggestion:
                                self._ignored += 1
                            self._noSuggestions += 1
                            self._answered = True
                            self._sendMessage('Your performance response time is ' + str(((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed)), 'RescueBot')
                            if ((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed) > 8.5:
                                self._lateResponse += 1
                                self._responseSpeed = 800
                            else:
                                self._lateResponse = 0
                                self._responseSpeed = 800
                            self._waiting = False
                            self._tosearch.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove' or self._remove:
                            if self.received_messages_content[-1] not in self._suggestion and not self._remove:
                                self._ignored += 1
                            if not self._remove:
                                self._noSuggestions += 1
                                self._answered = True
                                self._sendMessage('Your performance response time is ' + str(((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed)), 'RescueBot')
                                if ((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed) > 13:
                                    self._lateResponse += 1
                                    self._responseSpeed = 800
                                else:
                                    self._lateResponse = 0
                                    self._responseSpeed = 800
                                self._waiting = False
                                self._sendMessage('Removing tree blocking ' + str(self._door['room_name']) + '.', 'RescueBot')
                            if self._remove:
                                self._sendMessage('Removing tree blocking ' + str(self._door['room_name']) + ' because you asked me to.', 'RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}
                        else:
                            return None, {}

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'stone' in info['obj_id']:
                        objects.append(info)
                        if self._lateResponse <= 0:
                            responseString = "."
                        elif self._lateResponse == 1:
                            responseString = ", a bit faster than last time, since we lost crucial time."
                        else:
                            responseString = ", please inform me quickly this time, since I am waiting for your response."
                        if self._distanceHuman == 'far' and self._criticalFound < 2 and self._second < 240 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left.',
                                        ': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left. If we had found more than 1 critical victim, I would have suggested to remove alone. If the distance between us had been small, I would have suggested to remove together.']
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to continue searching instead of removing stones' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Continue", "Remove alone" or "Remove together"' + responseString,'RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = False

                        if self._distanceHuman == 'far' and self._criticalFound > 1 and self._second < 240 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 8/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If the distance between us had been small, I would have suggested to remove together. If we had found less than 2 critical victims, I would have suggested to continue searching.']
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to remove stones alone instead of continue searching or removing together' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Continue", "Remove alone" or "Remove together"' + responseString,'RescueBot')
                            self._suggestion = ['Remove alone']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = True
                        if self._distanceHuman == 'far' and self._criticalFound < 2 and self._second > 240 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 7/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If we had found more than 1 critical victim, I would have suggested to remove alone. If the distance between us had been small, and we had more than 4 minutes left or found more than 1 critical victim, I would have suggested to remove together.']
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to continue searching instead of removing stones' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Continue", "Remove alone" or "Remove together"' + responseString,'RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = True
                        if self._distanceHuman == 'far' and self._criticalFound > 1 and self._second > 240 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left and the distance between us is large.',
                                        ': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left and the distance between us is large. If the distance between us had been small, I would have suggested to remove together. If we had found less than 2 critical victims, I would have suggested to continue searching.']
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to remove stones alone instead of continue searching or removing together' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Continue", "Remove alone" or "Remove together"' + responseString,'RescueBot')
                            self._suggestion = ['Remove alone']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = False
                        if self._distanceHuman == 'close' and self._criticalFound < 2 and self._second < 240 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 8/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If the distance between us had been large and we had found more than 1 critical victim, I would have suggested to remove alone.']
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to remove stones together or to continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Continue", "Remove alone" or "Remove together"' + responseString,'RescueBot')
                            self._suggestion = ['Remove together', 'Continue']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = True
                        if self._distanceHuman == 'close' and self._criticalFound > 1 and self._second < 240 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 6/9 rescuers would decide the same.',': 6/9 rescuers would decide the same, because the distance between us is small and removing together only takes around 3 seconds.',
                                        ': 6/9 rescuers would decide the same, because the distance between us is small and removing together only takes around 3 seconds. If the distance between us had been large, I would have suggested to remove alone. If we had found less than 2 critical victims, I would have suggested to continue searching.']
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to remove stones together instead of continue searching or removing alone' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Continue", "Remove alone" or "Remove together"' + responseString,'RescueBot')
                            self._suggestion = ['Remove together']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = True
                        if self._distanceHuman == 'close' and self._criticalFound < 2 and self._second > 240 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '.',
                                        ': 5/9 rescuers would decide the same, because we found ' + str(self._criticalFound) + ' critical ' + self._vicString + '. If the distance between us had been large and we had found more than 1 critical victim, I would have suggested to remove alone. If we had found more than 1 critical victim, I would have suggested to remove together.']
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to continue searching instead of removing stones' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Continue", "Remove alone" or "Remove together"' + responseString,'RescueBot')
                            self._suggestion = ['Continue']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = False
                        if self._distanceHuman == 'close' and self._criticalFound > 1 and self._second > 240 and self._answered == False and not self._remove and not self._waiting:
                            messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because the distance between us is small.',
                                        ': 7/9 rescuers would decide the same, because the distance between us is small. If the distance between us had been large, I would have suggested to remove alone. If we had found less than 2 critical victims, I would have suggested to continue searching.']
                            self._sendMessage('Found stones blocking  ' + str(self._door['room_name']) + '.  \
                                I suggest to remove stones together instead of continue searching or removing alone' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Continue", "Remove alone" or "Remove together"' + responseString,'RescueBot')
                            self._suggestion = ['Remove together']
                            self._waiting = True
                            self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                            self._confidence = True
                        if self.received_messages_content and self.received_messages_content[-1] == 'Continue' and not self._remove:
                            if self.received_messages_content[-1] not in self._suggestion:
                                self._ignored += 1
                            self._noSuggestions += 1
                            self._answered = True
                            self._sendMessage('Your performance response time is ' + str(((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed)),'RescueBot')
                            if ((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed) > 8.5:
                                self._lateResponse += 1
                                self._responseSpeed = 800
                            else:
                                self._lateResponse = 0
                                self._responseSpeed = 800
                            self._waiting = False
                            self._tosearch.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove alone' and not self._remove:
                            if self.received_messages_content[-1] not in self._suggestion:
                                self._ignored += 1
                            self._noSuggestions += 1
                            self._answered = True
                            self._sendMessage('Your performance response time is ' + str(((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed)),'RescueBot')
                            if ((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed) > 8.5:
                                self._lateResponse += 1
                                self._responseSpeed = 800
                            else:
                                self._lateResponse = 0
                                self._responseSpeed = 800
                            self._waiting = False
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove together' or self._remove:
                            if self.received_messages_content[-1] not in self._suggestion and not self._remove:
                                self._ignored += 1
                            if not self._remove:
                                self._noSuggestions += 1
                                self._answered = True
                            self._sendMessage(
                                'Your performance response time is ' + str(((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed)),'RescueBot')
                            if ((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed) > 8.5:
                                self._lateResponse += 1
                                self._responseSpeed = 800
                            else:
                                self._lateResponse = 0
                                self._responseSpeed = 800
                            if not state[{'is_human_agent': True}]:
                                if self._lateComply <= 0:
                                    self._sendMessage('Please come to ' + str(self._door['room_name']) + ' to remove stones together.','RescueBot')
                                elif self._lateComply == 1:
                                    self._sendMessage('Please come to ' + str(self._door['room_name']) + ' to remove stones together, a bit faster than last time, because we lost important time then.','RescueBot')
                                else:
                                    self._sendMessage('Please come to ' + str(self._door['room_name']) + ' to remove stones together. You have to be quicker this time, because I am waiting for you.','RescueBot')
                                self._waitingHuman = True
                                self._complySpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
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
                if self._goalVic in self._foundVictims and self._door['room_name'] != \
                        self._foundVictimLocs[self._goalVic]['room']:
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
                                if self._lateResponse <= 0:
                                    responseString = "."
                                elif self._lateResponse == 1:
                                    responseString = ", a bit faster than last time, since we lost crucial time."
                                else:
                                    responseString = ", please inform me quickly this time, since I am waiting for your response."
                                if 'mild' in vic and self._second < 240 and self._criticalRescued < 2 and self._distanceDrop == 'close' and self._answered == False and not self._waiting:
                                    messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we only rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '.',
                                                ': 5/9 rescuers would decide the same, because we only rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '. If we had rescued  more than 1 critical victim, I would have suggested to rescue ' + vic + '.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                        I suggest to continue searching instead of rescuing ' + vic + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Rescue" or "Continue"' + responseString,'RescueBot')
                                    self._suggestion = ['Continue']
                                    self._waiting = True
                                    self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                    self._confidence = False
                                if 'mild' in vic and self._second < 240 and self._criticalRescued > 1 and self._distanceDrop == 'close' and self._answered == False and not self._waiting:
                                    messages = ['.', ': 5/9 rescuers would decide the same',': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left and the distance to the drop zone is small.',
                                                ': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left and the distance to the drop zone is small. If we had rescued less than 2 critical victims, I would have suggested to continue searching.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                        I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Rescue" or "Continue"' + responseString,'RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True
                                    self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                    self._confidence = False
                                if 'mild' in vic and self._second > 240 and self._criticalRescued < 2 and self._distanceDrop == 'close' and self._answered == False and not self._waiting:
                                    messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because we only rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '.',
                                                ': 8/9 rescuers would decide the same, because we only rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '. If we had rescued  more than 1 critical victim, I would have suggested to rescue ' + vic + '.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                        I suggest to continue searching instead of rescuing ' + vic + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Rescue" or "Continue"' + responseString,'RescueBot')
                                    self._suggestion = ['Continue']
                                    self._waiting = True
                                    self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                    self._confidence = True
                                if 'mild' in vic and self._second > 240 and self._criticalRescued > 1 and self._distanceDrop == 'close' and self._answered == False and not self._waiting:
                                    messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because we already rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '.',
                                                ': 7/9 rescuers would decide the same, because we already rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '. If we had rescued less than 2 critical victims, I would have suggested to continue searching.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                        I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Rescue" or "Continue"' + responseString,'RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True
                                    self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                    self._confidence = True
                                if 'mild' in vic and self._second < 240 and self._criticalRescued < 2 and self._distanceDrop == 'far' and self._answered == False and not self._waiting:
                                    messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we only rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '.',
                                                ': 5/9 rescuers would decide the same, because we only rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '. If we had rescued  more than 1 critical victim, I would have suggested to rescue ' + vic + '.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                        I suggest to continue searching instead of rescuing ' + vic + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Rescue" or "Continue"' + responseString,'RescueBot')
                                    self._suggestion = ['Continue']
                                    self._waiting = True
                                    self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                    self._confidence = False
                                if 'mild' in vic and self._second < 240 and self._criticalRescued > 1 and self._distanceDrop == 'far' and self._answered == False and not self._waiting:
                                    messages = ['.', ': 6/9 rescuers would decide the same.',': 6/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left.',
                                                ': 6/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left. If we had rescued less than 2 critical victims, I would have suggested to continue searching.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                        I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Rescue" or "Continue"' + responseString,'RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True
                                    self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                    self._confidence = True

                                if 'mild' in vic and self._second > 240 and self._criticalRescued < 2 and self._distanceDrop == 'far' and self._answered == False and not self._waiting:
                                    messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because we only rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '.',
                                                ': 7/9 rescuers would decide the same, because we only rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + '. If we had rescued  more than 1 critical victim, I would have suggested to rescue ' + vic + '.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                        I suggest to continue searching instead of rescuing ' + vic + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Rescue" or "Continue"' + responseString,'RescueBot')
                                    self._suggestion = ['Continue']
                                    self._waiting = True
                                    self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                    self._confidence = True
                                if 'mild' in vic and self._second > 240 and self._criticalRescued > 1 and self._distanceDrop == 'far' and self._answered == False and not self._waiting:
                                    messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we already rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + ' and have around ' + str(round((480 - self._second) / 60)) + ' minutes left.',
                                                ': 5/9 rescuers would decide the same, because we already rescued ' + str(self._criticalFound) + ' critical ' + self._vicString2 + ' and have around ' + str(round((480 - self._second) / 60)) + ' minutes left. If we had rescued less than 2 critical victims, I would have suggested to continue searching.']
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                        I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Rescue" or "Continue"' + responseString,'RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True
                                    self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                    self._confidence = False
                                if 'critical' in vic and self._distanceDrop == 'far' and self._distanceHuman == 'close' and self._second < 240 and self._answered == False and not self._waiting:
                                    messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because the distance between us is small.']
                                    if (self._performanceMetric == 3):
                                        self._performanceMetric = 2
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                        I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Rescue" or "Continue"' + responseString,'RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True
                                    self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                    self._confidence = True
                                if 'critical' in vic and self._distanceDrop == 'close' and self._distanceHuman == 'far' and self._second < 240 and self._answered == False and not self._waiting:
                                    messages = ['.', ': 6/9 rescuers would decide the same.',': 6/9 rescuers would decide the same, because the distance to the drop zone is small and we have around ' + str(round((480 - self._second) / 60)) + ' minutes left.']
                                    if (self._performanceMetric == 3):
                                        self._performanceMetric = 2
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                        I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Rescue" or "Continue"' + responseString,'RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True
                                    self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                    self._confidence = True
                                if 'critical' in vic and self._distanceDrop == 'close' and self._distanceHuman == 'close' and self._second > 240 and self._answered == False and not self._waiting:
                                    messages = ['.', ': 8/9 rescuers would decide the same.',': 8/9 rescuers would decide the same, because critical victims have a higher priority.']
                                    if (self._performanceMetric == 3):
                                        self._performanceMetric = 2
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                        I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Rescue" or "Continue"' + responseString,'RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True
                                    self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                    self._confidence = True
                                if 'critical' in vic and self._distanceDrop == 'close' and self._distanceHuman == 'close' and self._second < 240 and self._answered == False and not self._waiting:
                                    messages = ['.', ': 6/9 rescuers would decide the same.',': 6/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left.']
                                    if (self._performanceMetric == 3):
                                        self._performanceMetric = 2
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                        I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Rescue" or "Continue"' + responseString,'RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True
                                    self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                    self._confidence = True
                                if 'critical' in vic and self._distanceDrop == 'close' and self._distanceHuman == 'close' and self._second > 240 and self._answered == False and not self._waiting:
                                    messages = ['.', ': 6/9 rescuers would decide the same.',': 6/9 rescuers would decide the same, because the distance between us is small.']
                                    if (self._performanceMetric == 3):
                                        self._performanceMetric = 2
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                        I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Rescue" or "Continue"' + responseString,'RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True
                                    self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                    self._confidence = True
                                if 'critical' in vic and self._distanceDrop == 'far' and self._distanceHuman == 'close' and self._second > 240 and self._answered == False and not self._waiting:
                                    messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because critical victims have a higher priority.']
                                    if (self._performanceMetric == 3):
                                        self._performanceMetric = 2
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                        I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Rescue" or "Continue"' + responseString,'RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True
                                    self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                    self._confidence = True
                                if 'critical' in vic and self._distanceDrop == 'far' and self._distanceHuman == 'far' and self._second < 240 and self._answered == False and not self._waiting:
                                    messages = ['.', ': 5/9 rescuers would decide the same.',': 5/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left.']
                                    if (self._performanceMetric == 3):
                                        self._performanceMetric = 2
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                        I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Rescue" or "Continue"' + responseString,'RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True
                                    self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                    self._confidence = False
                                if 'critical' in vic and self._distanceDrop == 'far' and self._distanceHuman == 'far' and self._second > 240 and self._answered == False and not self._waiting:
                                    messages = ['.', ': 7/9 rescuers would decide the same.',': 7/9 rescuers would decide the same, because we have around ' + str(round((480 - self._second) / 60)) + ' minutes left.']
                                    if (self._performanceMetric == 3):
                                        self._performanceMetric = 2
                                    self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '. \
                                        I suggest to rescue ' + vic + ' instead of continue searching' + messages[self._performanceMetric] + self._progressMessage + ' Select your decision using the buttons "Rescue" or "Continue"' + responseString,'RescueBot')
                                    self._suggestion = ['Rescue']
                                    self._waiting = True
                                    self._responseSpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                                    self._confidence = True
                    return action, {}
                if self._goalVic in self._foundVictims and self._goalVic not in self._roomVics and \
                        self._foundVictimLocs[self._goalVic]['room'] == self._door['room_name']:
                    self._sendMessage(self._goalVic + ' not present in ' + str(self._door['room_name']) + ' because I searched the whole area without finding ' + self._goalVic + '.','RescueBot')
                    self._foundVictimLocs.pop(self._goalVic, None)
                    self._foundVictims.remove(self._goalVic)
                    self._roomVics = []
                    self.received_messages = []
                    self.received_messages_content = []
                self._searchedRooms.append(self._door['room_name'])
                if self.received_messages_content and self.received_messages_content[-1] == 'Rescue':
                    if self.received_messages_content[-1] not in self._suggestion:
                        self._ignored += 1
                    self._noSuggestions += 1
                    self._answered = True
                    self._sendMessage('Your performance response time is ' + str(((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed)),'RescueBot')
                    if ((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed) > 8.5:
                        self._lateResponse += 1
                        self._responseSpeed = 800
                    else:
                        self._lateResponse = 0
                        self._responseSpeed = 800
                    self._waiting = False
                    if 'critical' in self._recentVic:
                        if not state[{'is_human_agent': True}]:
                            if self._lateComply <= 0:
                                self._sendMessage('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(self._recentVic) + ' together.', 'RescueBot')
                            elif self._lateComply == 1:
                                self._sendMessage('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(self._recentVic) + ' together, a bit faster than last time, because we lost important time then.', 'RescueBot')
                            else:
                                self._sendMessage('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(self._recentVic) + ' together. You have to be quicker this time, because I am waiting for you.', 'RescueBot')
                            self._waitingHuman = True
                            self._complySpeed = state['World']['tick_duration'] * state['World']['nr_ticks']
                        if state[{'is_human_agent': True}]:
                            self._sendMessage('Lets carry ' + str(self._recentVic) + ' together!', 'RescueBot')
                    self._phase = Phase.FIND_NEXT_GOAL
                if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                    if self.received_messages_content[-1] not in self._suggestion:
                        self._ignored += 1
                    self._noSuggestions += 1
                    self._answered = True
                    self._sendMessage('Your performance response time is ' + str(((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed)),'RescueBot')
                    if ((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._responseSpeed) > 8.5:
                        self._lateResponse += 1
                        self._responseSpeed = 800
                    else:
                        self._lateResponse = 0
                        self._responseSpeed = 800
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
                self._phase = Phase.FIND_NEXT_GOAL
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

    def _processMessages(self, state, teamMembers):
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
                        if ((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._searchSpeed) > 63:
                            self._reminder = True
                            self._searchTookLong += 1
                            self._sendMessage('Your performance search took long is ' + str(self._searchTookLong),'RescueBot')
                        else:
                            self._reminder = False
                        self._searchSpeed = (state['World']['tick_duration'] * state['World']['nr_ticks'])
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
                    if 'mild' in foundVic:
                        self._todo.append(foundVic)
                    if ((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._searchSpeed) > 63:
                        self._reminder = True
                        self._searchTookLong += 1
                        self._sendMessage(
                            'Your performance search took long is ' + str(self._searchTookLong),'RescueBot')
                    else:
                        self._reminder = False
                    self._searchSpeed = (state['World']['tick_duration'] * state['World']['nr_ticks'])
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
                    if collectVic not in self._collectedVictims:
                        self._collectedVictims.append(collectVic)
                    if ((state['World']['tick_duration'] * state['World']['nr_ticks']) - self._searchSpeed) > 63:
                        self._reminder = True
                        self._searchTookLong += 1
                        self._sendMessage('Your performance search took long is ' + str(self._searchTookLong),'RescueBot')
                    else:
                        self._reminder = False
                    self._searchSpeed = (state['World']['tick_duration'] * state['World']['nr_ticks'])
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
                        self._sendMessage('Moving to ' + str(self._door['room_name']) + ' to help you remove an obstacle.','RescueBot')
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        area = 'area ' + msg.split()[-1]
                        self._sendMessage('Will come to ' + area + ' after dropping ' + self._goalVic + '.','RescueBot')
            if mssgs and mssgs[-1].split()[-1] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13','14']:
                self._humanLoc = int(mssgs[-1].split()[-1])

    def _sendMessage(self, mssg, sender):
        msg = Message(content=mssg, from_id=sender)

        if (('Moving to ' not in msg.content) or (msg.content[0:len(msg.content) - (len(self._reminderMessage))] not in self._sendMessages)):
            if msg.content not in self.received_messages_content and 'Our score is' not in msg.content:
                self.send_message(msg)
                if ('Moving to ' in msg.content):
                    self._sendMessages.append((msg.content[0:len(msg.content) - (len(self._reminderMessage))]))
                else:
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