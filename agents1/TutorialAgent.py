import sys, random, enum, ast, time
from matrx import grid_world
from brains1.ArtificialBrain import ArtificialBrain
from actions1.CustomActions import *
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
from actions1.CustomActions import RemoveObjectTogether, CarryObjectTogether, DropObjectTogether, CarryObject, Drop

class Phase(enum.Enum):
    INTRO0=0,
    INTRO1=1,
    INTRO2=2,
    INTRO3=3,
    INTRO4=4,
    INTRO5=5,
    INTRO6=6,
    INTRO7=7,
    INTRO8=8,
    INTRO9=9,
    INTRO10=10,
    INTRO11=11,
    FIND_NEXT_GOAL=12,
    PICK_UNSEARCHED_ROOM=13,
    PLAN_PATH_TO_ROOM=14,
    FOLLOW_PATH_TO_ROOM=15,
    PLAN_ROOM_SEARCH_PATH=16,
    FOLLOW_ROOM_SEARCH_PATH=17,
    PLAN_PATH_TO_VICTIM=18,
    FOLLOW_PATH_TO_VICTIM=19,
    TAKE_VICTIM=20,
    PLAN_PATH_TO_DROPPOINT=21,
    FOLLOW_PATH_TO_DROPPOINT=22,
    DROP_VICTIM=23,
    WAIT_FOR_HUMAN=24,
    WAIT_AT_ZONE=25,
    FIX_ORDER_GRAB=26,
    FIX_ORDER_DROP=27,
    REMOVE_OBSTACLE_IF_NEEDED=28,
    ENTER_ROOM=29
    
class TutorialAgent(ArtificialBrain):
    def __init__(self, slowdown, condition, name, folder):
        super().__init__(slowdown, condition, name, folder)
        # Initialization of some relevant variables
        self._slowdown = slowdown
        self._humanName = name
        self._folder = folder
        self._phase=Phase.INTRO0
        self._roomVics = []
        self._searchedRooms = []
        self._foundVictims = []
        self._collectedVictims = []
        self._foundVictimLocs = {}
        self._maxTicks = 9600
        self._sendMessages = []
        self._currentDoor=None 
        self._condition = condition
        self._providedExplanations = []   
        self._teamMembers = []
        self._carryingTogether = False
        self._remove = False
        self._goalVic = None
        self._goalLoc = None
        self._humanLoc = None
        self._distanceHuman = None
        self._distanceDrop = None
        self._agentLoc = None
        self._todo = []
        self._answered = False
        self._tosearch = []
        self._tutorial = True
        self._recentVic = None

    def initialize(self):
        # Initialization of the state tracker and navigation algorithm
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_observations(self, state):
        # Filtering of the world state before deciding on an action 
        return state

    def decide_on_actions(self, state):
        # Identify team members
        agent_name = state[self.agent_id]['obj_id']
        for member in state['World']['team_members']:
            if member!=agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)       
        # Process messages from team members
        self._processMessages(state, self._teamMembers)
        
        # Check whether human is close in distance
        if state[{'is_human_agent':True}]:
            self._distanceHuman = 'close'
        if not state[{'is_human_agent':True}]: 
            # Define distance between human and agent based on last known area locations
            if self._agentLoc in [1, 2, 3, 4, 5, 6, 7] and self._humanLoc in [8, 9, 10, 11, 12, 13, 14]:
                self._distanceHuman = 'far'
            if self._agentLoc in [1, 2, 3, 4, 5, 6, 7] and self._humanLoc in [1, 2, 3, 4, 5, 6, 7]:
                self._distanceHuman = 'close'
            if self._agentLoc in [8, 9, 10, 11, 12, 13, 14] and self._humanLoc in [1, 2, 3, 4, 5, 6, 7]:
                self._distanceHuman = 'far'
            if self._agentLoc in [8, 9, 10, 11, 12, 13, 14] and self._humanLoc in [8, 9, 10, 11, 12, 13, 14]:
                self._distanceHuman = 'close'

        # Define distance to drop zone based on last known area location
        if self._agentLoc in [1, 2, 5, 6, 8, 9, 11, 12]:
            self._distanceDrop = 'far'
        if self._agentLoc in [3, 4, 7, 10, 13, 14]:
            self._distanceDrop = 'close'

        # Check whether victims are currently being carried together by human and agent
        for info in state.values():
            if 'is_human_agent' in info and self._humanName in info['name'] and len(info['is_carrying'])>0 and 'critical' in info['is_carrying'][0]['obj_id']:
                # Add victim to colleced victims memory
                self._collectedVictims.append(info['is_carrying'][0]['img_name'][8:-4])
                self._carryingTogether = True
            if 'is_human_agent' in info and self._humanName in info['name'] and len(info['is_carrying'])==0:
                self._carryingTogether = False
        # If carrying a victim together, let agent be idle (because joint actions are essentially carried out by the human)
        if self._carryingTogether == True:
            return None, {}
        
        # Send the hidden score message for displaying and logging the score during the task, DO NOT REMOVE THIS
        self._sendMessage('Our score is ' + str(state['rescuebot']['score']) +'.', 'RescueBot')

        # Ongoing loop untill the task is terminated, using different phases for defining the agent's behavior
        while True:           
            # The first phases are all introduction messages explaining the task, environment, etc.
            if Phase.INTRO0==self._phase:
                self._sendMessage('Hello! My name is RescueBot. During this task we will collaborate with each other to search and rescue the victims at the drop zone on our right. \
                For this tutorial there are 4 victims and 3 injury types, during the official task there will be 8 victims to rescue. \
                The red color refers to critically injured victims, yellow to mildly injured victims, and green to healthy victims. Healthy victims do not need to be rescued. \
                The 8 victims are a girl (critically injured girl/mildly injured girl/healthy girl), boy (critically injured boy/mildly injured boy/healthy boy), \
                woman (critically injured woman/mildly injured woman/healthy woman), man (critically injured man/mildly injured man/healthy man), \
                elderly woman (critically injured elderly woman/mildly injured elderly woman/healthy elderly woman), \
                elderly man (critically injured elderly man/mildly injured elderly man/healthy elderly man), dog (critically injured dog/mildly injured dog/healthy dog), \
                and a cat (critically injured cat/mildly injured cat/healthy cat). The environment will also contain different obstacle types with varying removal times. \
                At the top of the world you can find the keyboard controls, for moving you can use the arrow keys. \
                Press the "Continue" button to start the tutorial explaining everything.', 'RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO1
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO1==self._phase:
                self._sendMessage('Lets try out the controls first. You can move with the arrow keys. If you move down twice, you will notice that you can now no longer see me. \
                So you can only see as far as 2 grid cells. Therefore, it is important to search the areas well. If you moved down twice, press the "Continue" button.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO2
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO2==self._phase:
                self._sendMessage('Lets move to area 3 now. When you are going to search an area, it is recommended to inform me about this.  \
                You can do this using the button "03". This way, we can collaborate more efficiently. \
                If you pressed the button "03" and moved to the area entrance, press the "Continue" button.', 'RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO3
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO3==self._phase:
                self._sendMessage('If you search area 3, you will find one of the victims to rescue: critically injured elderly woman. \
                There will be 3 different versions of the official task, manipulating your capabilities and resulting in different interdependence relationships between us. \
                However, in all conditions the critically injured victims have to be carried together. \
                So, let us carry critically injured elderly woman together! To do so, inform me that you found this victim by using the buttons below "I have found:" and selecting "critically injured elderly woman in 03". \
                If you found critically injured elderly woman and informed me about it, press the "Continue" button. I will then come over to help.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.FIND_NEXT_GOAL
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO4==self._phase:
                self._sendMessage('Let us carry ' + self._goalVic + ' together. To do this, move yourself on top, above, or next to ' + self._goalVic + '. \
                Now, press "A" on your keyboard (all keyboard controls can be found at the top of the world). \
                Transport ' + self._goalVic + ' to the drop zone and move yourself on top of the image of '+ self._goalVic + '. \
                Next, press "S" on your keyboard to drop '+ self._goalVic + '. \
                If you completed these steps, press the "Continue" button.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO5
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO5==self._phase:
                self._sendMessage('Nice job! Lets move to area 5 next. Remember to inform me about this. \
                If you are in front of area 5, you see that it is blocked by rock. This is one of the three obstacle types, and can only be removed together. \
                So, let us remove rock together! To do so, inform me that you found this obstacle by using the button "Help remove" and selecting "at 05". \
                I will then come over to help. If you informed me and I arrived at area 5 to help, press the "Continue" button.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO6
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO6==self._phase:
                self._sendMessage('Let us remove rock together now! To do so, remain in front of rock and press "D" on your keyboard. \
                Now, you will see a small busy icon untill rock is successfully removed. If the entrance is cleared, press the "Continue" button.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO7
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO7==self._phase:
                self._sendMessage('Lets move to area 4 next. Remember to inform me about this. \
                If you are in front of area 4, you see that it is blocked by tree. This is another obstacle type, and tree can only be removed by me. \
                So, let me remove tree for you! To do so, inform me that you need help with removing by using the button "Help remove" and selecting "at 04". \
                I will then come over to remove tree for you.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO8
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO8==self._phase:
                self._sendMessage('In area 4 you will find mildly injured elderly man. If you find mildly injured victims, it is recommended to inform me about this. \
                You can do this using the buttons below "I have found:" and selecting "mildly injured elderly man in 04". \
                Depending on the condition of the official task, you can rescue mildly injured victims alone or require my help. In this tutorial, you will carry mildly injured elderly man alone. \
                If you decide to carry mildly injured victims, it is recommended to inform me about it. \
                You can do this using the buttons below "I will pick up:" and selecting "mildly injured elderly man in 04." \
                Next, you can pick up mildly injured elderly man by moving yourself on top, above, or next to mildly injured elderly man. \
                Now, press "Q" on your keyboard and transport mildly injured elderly man to the drop zone. \
                Drop mildly injured elderly man by moving on top of the image and pressing "W" on your keyboard. \
                If you completed these steps, press the "Continue" button.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO9
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO9==self._phase:
                self._sendMessage('Nice job! Lets move to area 8 now. Remember to inform me about this. \
                If you are in front of area 8, you see that it is blocked by stones. \
                Depending on the condition of the official task, you might remove stones alone, require my help, or use my help to remove stones faster than doing it alone. \
                However, when I find stones, removing them together will always be faster than when I remove stones alone. For this tutorial, you will remove stones alone. \
                You can remove stones by pressing "E" on your keyboard. Now, you will see a small busy icon untill stones is successfully removed. \
                When you are busy removing, you can send messages but they will only appear once the action is finished. \
                So, no need to keep clicking buttons! If the entrance is cleared, press the "Continue" button.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO10
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO10==self._phase:
                self._sendMessage('This concludes the tutorial! You can now start the real task.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Found: critically injured girl in 5':
                    self._phase=Phase.FIND_NEXT_GOAL
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None, {}
            
            if Phase.FIND_NEXT_GOAL==self._phase:
                # Definition of some relevant variables
                self._answered = False
                self._goalVic = None
                self._goalLoc = None
                remainingZones = []
                remainingVics = []
                remaining = {}
                # Identification of the location of the drop zones
                zones = self._getDropZones(state)
                # Identification of which victims still need to be rescued and on which location they should be dropped
                for info in zones:
                    if str(info['img_name'])[8:-4] not in self._collectedVictims:
                        remainingZones.append(info)
                        remainingVics.append(str(info['img_name'])[8:-4])
                        remaining[str(info['img_name'])[8:-4]] = info['location']
                if remainingZones:
                    self._remainingZones = remainingZones
                    self._remaining = remaining
                # Remain idle if there are no victims left to rescue
                if not remainingZones:
                    return None,{}

                # Check which victims can be rescued next because human or agent already found them
                for vic in remainingVics:
                    # Define a previously found victim as target victim
                    if vic in self._foundVictims and vic not in self._todo:
                        self._goalVic = vic
                        self._goalLoc = remaining[vic]
                        # Plan path to victim because the exact location is known (i.e., the agent found this victim)
                        if 'location' in self._foundVictimLocs[vic].keys():
                            self._phase=Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__,{'duration_in_ticks':25}  
                        # Plan path to area because the exact victim location is not known, only the area (i.e., human found this victim)
                        if 'location' not in self._foundVictimLocs[vic].keys():
                            self._phase=Phase.PLAN_PATH_TO_ROOM
                            return Idle.__name__,{'duration_in_ticks':25}     
                # If there are no target victims found, visit an unsearched area to search for victims         
                self._phase=Phase.PICK_UNSEARCHED_ROOM

            if Phase.PICK_UNSEARCHED_ROOM==self._phase:
                agent_location = state[self.agent_id]['location']
                # Identify which areas are not explored yet
                unsearchedRooms=[room['room_name'] for room in state.values()
                if 'class_inheritance' in room
                and 'Door' in room['class_inheritance']
                and room['room_name'] not in self._searchedRooms
                and room['room_name'] not in self._tosearch]
                # If all areas have been searched but the task is not finished, start searching areas again
                if self._remainingZones and len(unsearchedRooms) == 0:
                    self._tosearch = []
                    self._todo = []
                    self._searchedRooms = []
                    self._sendMessages = []
                    self.received_messages = []
                    self.received_messages_content = []
                    self._searchedRooms.append(self._door['room_name'])
                    self._sendMessage('Going to re-search all areas.','RescueBot')
                    self._phase = Phase.FIND_NEXT_GOAL
                # If there are still areas to search, define which one to search next
                else:
                    # Identify the closest door when the agent did not search any areas yet
                    if self._currentDoor==None:
                        # Find all area entrance locations
                        self._door = state.get_room_doors(self._getClosestRoom(state,unsearchedRooms,agent_location))[0]
                        self._doormat = state.get_room(self._getClosestRoom(state,unsearchedRooms,agent_location))[-1]['doormat']
                        # Workaround for one area because of some bug
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3,5)
                        # Plan path to area
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    # Identify the closest door when the agent just searched another area
                    if self._currentDoor!=None:
                        self._door = state.get_room_doors(self._getClosestRoom(state,unsearchedRooms,self._currentDoor))[0]
                        self._doormat = state.get_room(self._getClosestRoom(state, unsearchedRooms,self._currentDoor))[-1]['doormat']
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3,5)
                        self._phase = Phase.PLAN_PATH_TO_ROOM

            if Phase.PLAN_PATH_TO_ROOM==self._phase:
                self._navigator.reset_full()
                # Switch to a different area when the human found a victim
                if self._goalVic and self._goalVic in self._foundVictims and 'location' not in self._foundVictimLocs[self._goalVic].keys():
                    self._door = state.get_room_doors(self._foundVictimLocs[self._goalVic]['room'])[0]
                    self._doormat = state.get_room(self._foundVictimLocs[self._goalVic]['room'])[-1]['doormat']
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3,5)
                    doorLoc = self._doormat
                # Otherwise plan the route to the previously identified area to search
                else:
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3,5)
                    doorLoc = self._doormat
                self._navigator.add_waypoints([doorLoc])
                # Follow the route to the next area to search
                self._phase=Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM==self._phase:
                # Find the next victim to rescue if the previously identified target victim was rescued by the human
                if self._goalVic and self._goalVic in self._collectedVictims:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                # Identify which area to move to because the human found the previously identified target victim
                if self._goalVic and self._goalVic in self._foundVictims and self._door['room_name']!=self._foundVictimLocs[self._goalVic]['room']:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                # Identify the next area to search if the human already searched the previously identified area
                if self._door['room_name'] in self._searchedRooms and self._goalVic not in self._foundVictims:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                # Otherwise move to the next area to search 
                else:
                    self._state_tracker.update(state)
                    # Explain why the agent is moving to the specific area, either because it contains the current target victim or because it is the closest unsearched area
                    if self._goalVic in self._foundVictims and str(self._door['room_name']) == self._foundVictimLocs[self._goalVic]['room'] and not self._remove:
                        self._sendMessage('Moving to ' + str(self._door['room_name']) + ' to pick up ' + self._goalVic+'.', 'RescueBot')                 
                    if self._goalVic not in self._foundVictims and not self._remove or not self._goalVic and not self._remove:
                        self._sendMessage('Moving to ' + str(self._door['room_name']) + ' because it is the closest unsearched area.', 'RescueBot')                   
                    self._currentDoor=self._door['location']
                    # Retrieve move actions to execute
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        # Remove obstacles blocking the path to the area
                        for info in state.values():
                            if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'stone' in info['obj_id'] and info['location'] not in [(9,7),(9,19),(21,19)]:
                                return RemoveObject.__name__,{'object_id':info['obj_id']}
                        return action,{}
                    # Identify and remove obstacles if they are blocking the entrance of the area 
                    self._phase=Phase.REMOVE_OBSTACLE_IF_NEEDED     

            if Phase.REMOVE_OBSTACLE_IF_NEEDED==self._phase:
                objects = []
                agent_location = state[self.agent_id]['location']
                # Identify which obstacle is blocking the entrance
                for info in state.values():
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'rock' in info['obj_id']:
                        objects.append(info)
                        # Proceed when human is ready to continue
                        if self._tutorial and self.received_messages_content and self.received_messages_content[-1]=='Continue':
                            self._phase=Phase.INTRO6
                            self.received_messages_content=[]
                            self.received_messages=[]
                        # Otherwise remain idle
                        else:
                            return None,{}
                       
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'tree' in info['obj_id']:
                        objects.append(info)
                        self.received_messages_content=[]
                        self.received_messages=[]
                        self._remove = False
                        self._phase=Phase.INTRO8
                        # Remove the obstacle when it is a tree
                        return RemoveObject.__name__,{'object_id':info['obj_id']}

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'stone' in info['obj_id']:
                        objects.append(info)   
                        # Remain idle when the obstacle is a stone                    
                        return None, {}
                # If no obstacles are blocking the entrance, enter the area
                if len(objects)==0:                    
                    self._answered = False
                    self._remove = False
                    self._phase = Phase.ENTER_ROOM
                    
            if Phase.ENTER_ROOM==self._phase:
                self._answered = False
                # If the target victim is rescued by the human, identify the next victim to rescue
                if self._goalVic in self._collectedVictims:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                # If the target victim is found in a different area, start moving there
                if self._goalVic in self._foundVictims and self._door['room_name']!=self._foundVictimLocs[self._goalVic]['room']:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                # If the human searched the same area, plan searching another area instead
                if self._door['room_name'] in self._searchedRooms and self._goalVic not in self._foundVictims:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                # Otherwise, enter the area and plan to search it
                else:
                    self._state_tracker.update(state)                 
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        return action,{}
                    self._phase=Phase.PLAN_ROOM_SEARCH_PATH

            if Phase.PLAN_ROOM_SEARCH_PATH==self._phase:
                self._agentLoc = int(self._door['room_name'].split()[-1])
                # Store the locations of all area tiles 
                roomTiles = [info['location'] for info in state.values()
                    if 'class_inheritance' in info 
                    and 'AreaTile' in info['class_inheritance']
                    and 'room_name' in info
                    and info['room_name'] == self._door['room_name']]
                self._roomtiles=roomTiles   
                # Make the plan for searching the area            
                self._navigator.reset_full()
                self._navigator.add_waypoints(self._efficientSearch(roomTiles))
                self._roomVics=[]
                self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH

            if Phase.FOLLOW_ROOM_SEARCH_PATH==self._phase:
                # Search the area
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:               
                    # Identify victims present in the area    
                    for info in state.values():
                        if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance']:
                            vic = str(info['img_name'][8:-4])
                            # Remember which victim the agent found in this area
                            if vic not in self._roomVics:
                                self._roomVics.append(vic)

                            # Identify the exact location of the victim that was found by the human earlier
                            if vic in self._foundVictims and 'location' not in self._foundVictimLocs[vic].keys():
                                # Add the exact location to the corresponding dictionary
                                self._foundVictimLocs[vic] = {'location':info['location'],'room':self._door['room_name'],'obj_id':info['obj_id']}
                                if vic == self._goalVic:
                                    # Communicate which victim was found
                                    self._sendMessage('Found '+ vic + ' in ' + self._door['room_name'] + ' because you told me '+vic+ ' was located here.', 'RescueBot')
                                    # Add the area to the list with searched areas
                                    self._searchedRooms.append(self._door['room_name'])
                                    # Do not continue searching the rest of the area but start planning to rescue the victim 
                                    self._phase=Phase.FIND_NEXT_GOAL

                            # Identify injured victims in the area
                            if 'healthy' not in vic and vic not in self._foundVictims:
                                self._recentVic = vic
                                # Add the victim and the location to the corresponding dictionary
                                self._foundVictims.append(vic)
                                self._foundVictimLocs[vic] = {'location':info['location'],'room':self._door['room_name'],'obj_id':info['obj_id']}
                    # Execute move actions to explore the area
                    return action,{}

                # Communicate that the agent did not find the target victim in the area despite the human previously communicating that the victim was located here
                if self._goalVic in self._foundVictims and self._goalVic not in self._roomVics and self._foundVictimLocs[self._goalVic]['room']==self._door['room_name']:
                    self._sendMessage(self._goalVic + ' not present in ' + str(self._door['room_name']) + ' because I searched the whole area without finding ' + self._goalVic+'.', 'RescueBot')
                    # Remove the victim location from memory
                    self._foundVictimLocs.pop(self._goalVic, None)
                    self._foundVictims.remove(self._goalVic)
                    self._roomVics = []
                    # Reset received messages (bug fix)
                    self.received_messages = []
                    self.received_messages_content = []
                # Add the area to the list of searched areas and make a plan what to do next
                self._searchedRooms.append(self._door['room_name'])
                self._recentVic = None
                self._phase=Phase.FIND_NEXT_GOAL
                return Idle.__name__,{'duration_in_ticks':25}
                
            if Phase.PLAN_PATH_TO_VICTIM==self._phase:
                # Communicate which vctim the agent is going to pick up
                if 'mild' in self._goalVic:
                    self._sendMessage('Picking up ' + self._goalVic + ' in ' + self._foundVictimLocs[self._goalVic]['room'] + '.', 'RescueBot')
                # Plan the path to the victim using its location
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._foundVictimLocs[self._goalVic]['location']])
                self._phase=Phase.FOLLOW_PATH_TO_VICTIM
                    
            if Phase.FOLLOW_PATH_TO_VICTIM==self._phase:
                # Start searching for other victims if the human already rescued the target victim
                if self._goalVic and self._goalVic in self._collectedVictims:
                    self._phase=Phase.FIND_NEXT_GOAL
                # Otherwise, move towards the location of the found victim
                else:
                    self._state_tracker.update(state)
                    action=self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        return action,{}
                    self._phase=Phase.TAKE_VICTIM
                    
            if Phase.TAKE_VICTIM==self._phase:
                objects=[]
                # Notify the human when a critically injured victim needs to be carried together
                for info in state.values():
                    if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] and 'critical' in info['obj_id'] and info['location'] in self._roomtiles:
                        objects.append(info)
                        self._collectedVictims.append(self._goalVic)
                        self._phase=Phase.INTRO4
                        # Remain idle until the human arrives
                        if not self._humanName in info['name']:
                            return None, {} 
                # When a critically injured victim is picked up, start planning the path to the drop zone and add the victim to the list of rescued victims
                if len(objects)==0 and 'critical' in self._goalVic:
                    self._collectedVictims.append(self._goalVic)
                    self._phase = Phase.PLAN_PATH_TO_DROPPOINT
                # When rescuing mildly injured victims, pick the victim up and plan the path to the drop zone
                if 'mild' in self._goalVic:
                    self._phase=Phase.PLAN_PATH_TO_DROPPOINT
                    self._collectedVictims.append(self._goalVic)
                    return CarryObject.__name__,{'object_id':self._foundVictimLocs[self._goalVic]['obj_id'], 'human_name': self._humanName}                

            if Phase.PLAN_PATH_TO_DROPPOINT==self._phase:
                self._navigator.reset_full()
                # Plan the path to the drop zone
                self._navigator.add_waypoints([self._goalLoc])
                # Follow the path to the drop zone
                self._phase=Phase.FOLLOW_PATH_TO_DROPPOINT

            if Phase.FOLLOW_PATH_TO_DROPPOINT==self._phase:
                self._state_tracker.update(state)
                # Retrieve the move actions
                action=self._navigator.get_move_action(self._state_tracker)
                # Execute the move actions to the drop zone
                if action!=None:
                    return action,{}
                # Drop the victim at the drop zone
                self._phase=Phase.DROP_VICTIM

            if Phase.DROP_VICTIM == self._phase:
                # Communicate that the agent delivered a mildly injured victim alone to the drop zone
                if 'mild' in self._goalVic:
                    self._sendMessage('Delivered '+ self._goalVic + ' at the drop zone.', 'RescueBot')
                # Identify the next target victim to rescue
                self._phase=Phase.FIND_NEXT_GOAL
                self._currentDoor = None
                self._tick = state['World']['nr_ticks']
                # Drop the victim on the correct location on the drop zone
                return Drop.__name__,{'human_name': self._humanName}

            
    def _getDropZones(self,state:State):
        '''
        @return list of drop zones (their full dict), in order (the first one is the
        the place that requires the first drop)
        '''
        places=state[{'is_goal_block':True}]
        places.sort(key=lambda info:info['location'][1])
        zones = []
        for place in places:
            if place['drop_zone_nr']==0:
                zones.append(place)
        return zones

    def _processMessages(self, state, teamMembers):
        '''
        process incoming messages received from the team members
        '''
        receivedMessages = {}
        # Create a dictionary with a list of received messages from each team member
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content) 
        # Check the content of the received messages
        for mssgs in receivedMessages.values():
            for msg in mssgs:
                # If a received message involves team members searching areas, add these areas to the memory of areas that have been explored
                if msg.startswith("Search:"):
                    area = 'area '+ msg.split()[-1]
                    if area not in self._searchedRooms:
                        self._searchedRooms.append(area)
                # If a received message involves team members finding victims, add these victims and their locations to memory
                if msg.startswith("Found:"):
                    # Identify which victim and area it concerns
                    if len(msg.split()) == 6:
                        foundVic = ' '.join(msg.split()[1:4])
                    else:
                        foundVic = ' '.join(msg.split()[1:5]) 
                    loc = 'area '+ msg.split()[-1]
                    # Add the area to the memory of searched areas
                    if loc not in self._searchedRooms:
                        self._searchedRooms.append(loc)
                    # Add the victim and its location to memory
                    if foundVic not in self._foundVictims:
                        self._foundVictims.append(foundVic)
                        self._foundVictimLocs[foundVic] = {'room':loc}
                    if foundVic in self._foundVictims and self._foundVictimLocs[foundVic]['room'] != loc:
                        self._foundVictimLocs[foundVic] = {'room':loc}
                    # Add the found mildly injured victim to the to do list
                    if 'mild' in foundVic:
                        self._todo.append(foundVic)
                # If a received message involves team members rescuing victims, add these victims and their locations to memory
                if msg.startswith('Collect:'):
                    # Identify which victim and area it concerns
                    if len(msg.split()) == 6:
                        collectVic = ' '.join(msg.split()[1:4])
                    else:
                        collectVic = ' '.join(msg.split()[1:5]) 
                    loc = 'area ' + msg.split()[-1]
                    # Add the area to the memory of searched areas 
                    if loc not in self._searchedRooms:
                        self._searchedRooms.append(loc)
                    # Add the victim and location to the memory of found victims
                    if collectVic not in self._foundVictims:
                        self._foundVictims.append(collectVic)
                        self._foundVictimLocs[collectVic] = {'room':loc}
                    if collectVic in self._foundVictims and self._foundVictimLocs[collectVic]['room'] != loc:
                        self._foundVictimLocs[collectVic] = {'room':loc}
                    # Add the victim to the memory of rescued victims 
                    if collectVic not in self._collectedVictims:
                        self._collectedVictims.append(collectVic)
                # If a received message involves team members asking for help with removing obstacles, add their location to memory and come over
                if msg.startswith('Remove:'):
                    # Identify at which location the human needs help
                    area = 'area ' + msg.split()[-1]
                    self._door = state.get_room_doors(area)[0]
                    self._doormat = state.get_room(area)[-1]['doormat']
                    if area in self._searchedRooms:
                        self._searchedRooms.remove(area)
                    # Clear received messages (bug fix)
                    self.received_messages = []
                    self.received_messages_content = []
                    self._remove = True
                    # Let the human know that the agent is coming over to help
                    self._sendMessage('Moving to ' + str(self._door['room_name']) + ' to help you remove an obstacle.', 'RescueBot')  
                    # Plan the path to the relevant area
                    self._phase = Phase.PLAN_PATH_TO_ROOM
            # Store the current location of the human in memory
            if mssgs and mssgs[-1].split()[-1] in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']:
                self._humanLoc = int(mssgs[-1].split()[-1])

    def _sendMessage(self, mssg, sender):
        '''
        send messages from agent to other team members
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages_content and 'score' not in msg.content:
            self.send_message(msg)
            self._sendMessages.append(msg.content)
        # Sending the hidden score message (DO NOT REMOVE)
        if 'score' in msg.content:
            self.send_message(msg)

    def _getClosestRoom(self, state, objs, currentDoor):
        '''
        calculate which area is closest to the agent's location
        '''
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
        '''
        efficiently transverse areas instead of moving over every single area tile
        '''
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