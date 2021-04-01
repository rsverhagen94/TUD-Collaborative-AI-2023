import numpy as np
from matrx.agents import AgentBrain
from matrx.messages import Message
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions import *
from matrx import utils
from BW4TBrain import BW4TBrain


class BlockWorldAgent(BW4TBrain):

    def __init__(self, slowdown:int):
        super().__init__(slowdown)

    def initialize(self):
        # Initialize some necessary variables like state_tracker and navigator.
        self.state_tracker = StateTracker(agent_id=self.agent_id)
        self.navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set,
                                   algorithm=Navigator.A_STAR_ALGORITHM)
        #self.send_message(Message(content="Hello World", from_id=self.agent_id))
        self.actions = [[], []]
        self.seen_blocks = {}
        self.entered_rooms = []
        self.loc = None
        self.id = None
        self.collected_blocks = []
        self.block_locs = {}
        self.visited_drop_zones = []
        self.current_action=None

    def filter_bw4t_observations(self, state):
        # Update state.
        self.state_tracker.update(state)
        return state

    def search_first_room_action(self, state, location):
        # Move to the first unseen room.
        self.navigator.add_waypoint(location)
        move_action = self.navigator.get_move_action(self.state_tracker)
        self.actions[0].append(move_action)
        self.actions[1].append({})
        return

    def search_next_room_action(self, state, location):
        # Move to a next unseen room. Navigator was causing problems so for this function I had to first reset the navigator.
        self.navigator.reset_full()
        self.navigator.add_waypoint(location)
        move_action = self.navigator.get_move_action(self.state_tracker)
        self.actions[0].append(move_action)
        self.actions[1].append({})
        return

    def open_door_action(self, state, obj_id):
        # Open a door based on its object id.
        open_kwargs = {}
        open_kwargs['object_id'] = obj_id
        open_action = OpenDoorAction.__name__
        self.actions[0].append(open_action)
        self.actions[1].append(open_kwargs)
        return

    def enter_room_action(self, state, location):
        # Enter a room.
        loc = list(state[self.agent_id]['location'])
        entrance = list(loc)
        entrance[1]-=3
        self.navigator.add_waypoint(tuple(entrance))
        move_action = self.navigator.get_move_action(self.state_tracker)
        self.actions[0].append(move_action)
        self.actions[1].append({})
        return

    def inspect_room_action(self, state):
        # Inspect a room when inside it by adding the block-visualization pairs to self.seen_blocks and block-location pairs to see.block_locs.
        # IMPORTANT: I've changed the variable 'block_sense_range' in builder.py from 2 to 3, in order to make the agent be able to perceive every block when inside a room.
        objects = list(state.keys())
        for obj in objects:
            if 'Block_in_room' in obj and obj not in self.seen_blocks.keys():
                self.seen_blocks[obj] = state[obj]['visualization']
                self.seen_blocks[obj].pop('depth', None)
                self.seen_blocks[obj].pop('opacity', None)
                self.block_locs[obj] = state[obj]['location']
        return

    def move_to_block_action(self, state, location):
        # Move to to be collected block.
        # Navigator was causing problems so had be reset again.
        self.navigator.reset_full()
        self.navigator.add_waypoint(location)
        move_action = self.navigator.get_move_action(self.state_tracker)
        self.actions[0].append(move_action)
        self.actions[1].append({})
        return

    def collect_block_action(self, state, obj_id, location):
        # Move to to be collected block and grab it. Navigator was causing me lots of troubles, so in some cases I had to use 'get_move_action' while in others the solution below.
        # In addition, in somce cases this function did not work so a combination of move_to_block_action and grab_object_action had to be used.
        self.navigator.reset_full()
        self.navigator.add_waypoint(location)
        route_actions = self.navigator._Navigator__get_route(self.state_tracker)
        print(route_actions)
        if route_actions:
            for action in route_actions.values():
                self.actions[0].append(action)
                self.actions[1].append({})
        else:
            self.actions[0].append(None)
            self.actions[1].append({})
        grab_kwargs = {}
        grab_kwargs['object_id'] = obj_id
        #grab_kwargs['grab_range'] = 1
        grab_kwargs['max_objects'] = 5
        grab_action = GrabObject.__name__
        self.actions[0].append(grab_action)
        self.actions[1].append(grab_kwargs)
        return
    
    def grab_block_action(self, state, obj_id):
        # Grab block specified by its object_id.
        grab_kwargs = {}
        grab_kwargs['object_id'] = obj_id
        #grab_kwargs['grab_range'] = 1
        grab_kwargs['max_objects'] = 5
        grab_action = GrabObject.__name__
        self.actions[0].append(grab_action)
        self.actions[1].append(grab_kwargs)
        return

    def drop_block_action(self, state, location):
        # Drop block when at specified location. Again, navigator was causing problems so I had to find a workaround.
        if state[self.agent_id]['is_carrying']:
            self.navigator.add_waypoint(location)
            route_actions = self.navigator._Navigator__get_route(self.state_tracker)
            if route_actions:
                for action in route_actions.values():
                    self.actions[0].append(action)
                    self.actions[1].append({})
            else:
                self.actions[0].append(None)
                self.actions[1].append({})
            drop_action = DropObject.__name__
            self.actions[0].append(drop_action)
            self.actions[1].append({})
        return

    def decide_on_bw4t_action(self, state):
        # First, add the visualization of the goal blocks and location of drop zones to a list. Probably also possible under filter_observations.
        # Definitely much more efficient ways to solve to problem, but for now this provides some useful insights into the task and MATRX.
        objects = list(state.keys())
        goal_blocks = []
        drop_zones = []
        for obj in objects:
            # Only add drop_zone if it is not visited before
            if 'is_drop_zone' in state[obj] and state[obj]['is_drop_zone'] == True and state[obj]['location'] not in self.visited_drop_zones:
                drop_zones.append(state[obj]['location'])
            # Only add goal_block if it is not collected before
            if 'is_goal_block' in state[obj] and state[obj]['is_goal_block'] == True and state[obj]['visualization'] not in goal_blocks and state[obj]['visualization'] not in self.collected_blocks:
                goal_blocks.append(state[obj]['visualization'])
        # Reverse drop_zones since it's in wrong order wrt goal_blocks
        drop_zones.reverse() 

        # Next, get all room names. Probably also possible under filter_observations.
        rooms = state.get_all_room_names()  
        # Remove world_bounds from room_names
        rooms.remove('world_bounds')
        # Next, add the locations and object_ids of doors to dictionaries as room-id and location-room pairs. Probably also possible under filter_observations.
        doors = {}
        door_ids = {}
        for room in rooms:
            loc = state.get_room_doors(room)[0]['location']
            # Make sure the entrance of the door is in front of the door (y+1).
            entrance = list(loc)
            entrance[1]+=1
            doors[tuple(entrance)] = room
            door_ids[room] = state.get_room_doors(room)[0]['obj_id']

        # Create a list with the locations of the entrances belonging to non-visited rooms.
        unseen = []
        for door, room in doors.items():
            if room not in self.entered_rooms:
                unseen.append(door)

        # Create variable with current agent location
        agent_location = state[self.agent_id]['location']

        # Core of the decide on action function. Basically repeat below conditions while there are still unseen rooms, goal blocks to collect, and drop zones to visit.
        if unseen:
            if goal_blocks:
                # Select the next goal block as block to be collected.
                next_block = goal_blocks[0].copy()
                # Remove depth and opacity visualization features since they are different between ghost and real blocks.
                next_block.pop('depth', None)
                next_block.pop('opacity', None)
            if drop_zones:
                # Select drop_zone corresponding to next to be collected goal block.
                drop_zone = drop_zones.copy()[0]
            
            # Select first element of unseen rooms door-locations as next door.
            next_door = unseen[0]
            # Set some relevant variables for ease of reading, e.g.: next room, if next door is open, the entrance location within the room, and the multiple locations within a room.
            next_room = doors[next_door]
            door_open = state[door_ids[next_room]]['is_open']
            room_entrance = list(next_door)
            room_entrance[1]-=1
            room_entrance = tuple(room_entrance)
            # Add locations within room to room area list.
            room_area = []
            for obj in state.get_room_objects(next_room):
                if obj['location'] not in room_area:
                    room_area.append(obj['location'])

            # CORE PART: Condition checking and corresponding actions related to conditions.
            # If the agent is not at the location of the next door, it did not see the next goal block in a room, and did not collect any blocks yet:
            # Search/move to the first unseen door.
            if agent_location != next_door and next_block not in self.seen_blocks.values() and not self.collected_blocks:
                print('searching next door at location: ', next_door)
                msg = Message(content='Searching '+next_room+' because I do not know the location of the goal block', from_id='RescueBot')
                if msg.content not in self.received_messages:
                    self.send_message(msg)
                self.search_first_room_action(state, next_door)

            # If the agent is at the location of the next door, the door is closed, the it did not see the next block in a room:
            # Open the closed door.
            if agent_location == next_door and door_open == False and next_block not in self.seen_blocks.values():
                print('opening door at location: ', next_door)
                msg = Message(content='Opening door of '+next_room+' because it is closed', from_id='RescueBot')
                if msg.content not in self.received_messages:
                    self.send_message(msg)
                self.open_door_action(state, door_ids[next_room])

            # If the agent is at the location of the next door, the door is open, and it did not see the next block in a room:
            # Enter the room.
            if agent_location == next_door and door_open == True and next_block not in self.seen_blocks.values():
                print('entering door at location: ', next_door)
                msg = Message(content='Entering '+next_room+' to search for the next goal block', from_id='RescueBot')
                if msg.content not in self.received_messages:
                    self.send_message(msg)
                self.enter_room_action(state, agent_location)

            # If the agent is inside the room and it did not see the next block yet:
            # Inspect the room by adding visualization and location to dicts.
            if agent_location == room_entrance and next_block not in self.seen_blocks.values():
                print('inspecting room at location: ', room_entrance)
                self.inspect_room_action(state)
                msg = Message(content='Inspecting '+next_room+' to see if it contains the goal block', from_id='RescueBot')
                if msg.content not in self.received_messages:
                    self.send_message(msg)
                # If the next block is not inside this room, add the room to entered rooms so it will no longer be in the unseen list.
                if next_block not in self.seen_blocks.values():
                    msg = Message(content='Current goal block not in '+next_room+' because I just searched it',from_id='RescueBot')
                    if msg.content not in self.received_messages:
                        self.send_message(msg)
                    self.entered_rooms.append(next_room)

            # If the next block is seen, the agent is inside the room, not at the location of the block, and not carrying anything:
            # Move to the block and pick it up.
            if next_block in self.seen_blocks.values() and agent_location in room_area and agent_location!=self.loc and not state[self.agent_id]['is_carrying']:
                key_list = list(self.seen_blocks.keys())
                val_list = list(self.seen_blocks.values())
                block = key_list[val_list.index(next_block)]
                block_loc = self.block_locs[block]
                #block_loc = state[block]['location']
                self.loc = block_loc
                #self.id = state[block]['obj_id']
                self.id = block
                print('moving to goal block at location: ', block_loc)
                msg = Message(content='Picking up goal block in '+next_room+' because I am in the room and not carrying anything',from_id='RescueBot')
                if msg.content not in self.received_messages:
                    self.send_message(msg)
                self.collect_block_action(state, self.id, block_loc)

            # If the agent is carrying something, is not at the location of the drop zone, and the carrying block is not collected yet:
            # Move to the drop zone and drop the block.
            if state[self.agent_id]['is_carrying'] and agent_location != drop_zone and self.seen_blocks[self.id] not in self.collected_blocks:
                print('moving to drop zone at location: ', drop_zone)
                self.drop_block_action(state, drop_zone)
                # Add current action because of problems with passing drop zone when current action is not dropping.
                self.current_action='dropping'
                msg = Message(content='Moving to drop zone at location '+str(drop_zone)+' because I am carrying a goal block', from_id='RescueBot')
                if msg.content not in self.received_messages:
                    self.send_message(msg)

            # If the agent is at the drop zone, the carrying block in not collected yet, and the current action is dropping/the agent just dropped/collected a block:
            # Add block to collected blocks and drop zone to visited drop zones, so they will not be added to drop_zones and goal_blocks lists.
            if agent_location == drop_zone and self.seen_blocks[self.id] not in self.collected_blocks and not state[self.agent_id]['is_carrying'] and self.current_action=='dropping':
                print('dropped block at location: ', drop_zone)
                self.collected_blocks.append(goal_blocks[0])
                self.visited_drop_zones.append(drop_zone)
                msg = Message(content='Dropped block at drop zone '+str(drop_zone),from_id='RescueBot')
                if msg.content not in self.received_messages:
                    self.send_message(msg)

            # If the agent is not carrying, it collected a block, is at the drop zone, and the current action is dropping:
            # Add the room where it found the dropped block to entered rooms, and switch to the next goal block and drop zone.
            if not state[self.agent_id]['is_carrying'] and self.collected_blocks and agent_location==drop_zone and self.current_action=='dropping':
                self.entered_rooms.append(next_room)
                print('switching to next goal')
                goal_blocks.pop(0)
                drop_zones.pop(0)
                msg = Message(content='Switching to next goal block '+str(next_block) +' because I just collected the previous goal block', from_id='RescueBot')
                if msg.content not in self.received_messages:
                    self.send_message(msg)
                # Navigator was causing problems so I had to reset the actions lists
                self.actions = [[], []]             

            # If the agent is not carrying, it collected a block, saw the next goal block in a room, and is not at the location of the next goal block:
            # Move to the location of the next seen goal block.
            if not state[self.agent_id]['is_carrying'] and self.collected_blocks and next_block in self.seen_blocks.values() and agent_location!=self.loc:
                for block, viz in self.seen_blocks.items():
                    if viz == next_block:
                        # Get the location and object id of the next seen goal block.
                        self.loc = self.block_locs[block]
                        self.id = block
                        break
                print('move to next seen block at location: ', self.loc)
                msg = Message(content='Moving to next seen goal block at location '+str(self.loc) +' because I am not carrying anything and remember I saw it there', from_id='RescueBot')
                if msg.content not in self.received_messages:
                    self.send_message(msg)
                # Set current action to moving so when it passes the drop zone it will not change goal block.
                self.current_action='moving'
                # Navigator was causing problems so I had to reset the actions lists.
                self.actions = [[], []]
                self.move_to_block_action(state, self.loc)
            
            # If the next goal block is seen in a room, the agent is at the location of this goal block, is not carrying, and the next goal block is not collected yet:
            # Grab the next goal block based on its object_id.
            if next_block in self.seen_blocks.values() and agent_location==self.loc and not state[self.agent_id]['is_carrying'] and self.seen_blocks[self.id] not in self.collected_blocks:
                print('grabbing block at location: ', self.loc)
                msg = Message(content='Grabbing block at location '+str(self.loc)+' because I will bring it to the drop zone', from_id='RescueBot')
                if msg.content not in self.received_messages:
                    self.send_message(msg)
                self.grab_block_action(state, self.id)
            
            # If the agent is not carrying, it collected a block before, did not see the next goal block in a room yet, and is not at the door location of the next unseen room:
            # Search/move to a next unseen room
            if not state[self.agent_id]['is_carrying'] and self.collected_blocks and next_block not in self.seen_blocks.values() and agent_location!=next_door:
                print('already block(s) collected, now searching next door at location: ', next_door)
                msg = Message(content='Searching '+next_room+' because I just collected a block but do not know the location of the next goal block', from_id='RescueBot')
                if msg.content not in self.received_messages:
                    self.send_message(msg)
                self.search_next_room_action(state, next_door)

        # Execute all actions appended to the action list, and their corresponding arguments.
        if self.actions:
            if self.actions[0] and self.actions[1]:
                action = self.actions[0].pop(0)
                action_kwargs = self.actions[1].pop(0)
                return action, action_kwargs
            else:
                return None, {}