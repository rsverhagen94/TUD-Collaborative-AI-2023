import sys
import itertools
from collections import OrderedDict
from itertools import product

from matrx import WorldBuilder
import numpy as np
from matrx.actions import MoveNorth, OpenDoorAction, CloseDoorAction
from matrx.actions.move_actions import MoveEast, MoveSouth, MoveWest
from matrx.agents import AgentBrain, HumanAgentBrain, SenseCapability
from matrx.grid_world import GridWorld, DropObject, GrabObject, AgentBody
from matrx.objects import EnvObject
from matrx.world_builder import RandomProperty
from matrx.goals import WorldGoal
from explainable_agent import BlockWorldAgent

tick_duration = 0.2
random_seed = 1
verbose = False
key_action_map = {  # For the human agents
    'w': MoveNorth.__name__,
    'd': MoveEast.__name__,
    's': MoveSouth.__name__,
    'a': MoveWest.__name__,
    'q': GrabObject.__name__,
    'e': DropObject.__name__,
    'r': OpenDoorAction.__name__,
    'f': CloseDoorAction.__name__,
}

# Some BW4T settings
room_size = (6, 4)  # width, height
nr_rooms = 9
rooms_per_row = 3
average_blocks_per_room = 3
block_shapes = [0, 1]
block_colors = ['#0008ff', '#ff1500', '#0dff00']
room_colors = ['#0008ff', '#ff1500', '#0dff00']
wall_color = "#8a8a8a"
drop_off_color = "#878787"
block_size = 0.5
nr_drop_zones = 1
nr_blocks_needed = 3
hallway_space = 2
nr_teams = 1
agents_per_team = 2
human_agents_per_team = 1
agent_sense_range = 2  # the range with which agents detect other agents
block_sense_range = 3  # the range with which agents detect blocks
other_sense_range = np.inf  # the range with which agents detect other objects (walls, doors, etc.)
agent_memory_decay = 5  # we want to memorize states for seconds / tick_duration ticks
fov_occlusion = True

agent_slowdown=[3,2,1,1,1]
# Should be list with at least <bots_per_team  * nr_teams> values.
# wehere bots_per_team=agents_per_team - human_agents_per_team.
# 1=normal, 3 means 3x slowdown. allowed: natural numbers.
# This value is used in all actions of the agents.

def calculate_world_size():
    nr_room_rows = np.ceil(nr_rooms / rooms_per_row)

    # calculate the total width
    world_width = max(rooms_per_row * room_size[0] + 2 * hallway_space,
                      (nr_drop_zones + 1) * hallway_space + nr_drop_zones) + 2

    # calculate the total height
    world_height = nr_room_rows * room_size[1] + (nr_room_rows + 1) * hallway_space + nr_blocks_needed + 2
    return int(world_width), int(world_height)


def get_room_loc(room_nr):
    row = np.floor(room_nr / rooms_per_row)
    column = room_nr % rooms_per_row

    # x is: +1 for the edge, +edge hallway, +room width * column nr, +1 off by one
    room_x = int(1 + hallway_space + (room_size[0] * column) + 1)

    # y is: +1 for the edge, +hallway space * (nr row + 1 for the top hallway), +row * room height, +1 off by one
    room_y = int(1 + hallway_space * (row + 1) + row * room_size[1] + 1)

    # door location is always center bottom
    door_x = room_x + int(np.ceil(room_size[0] / 2))
    door_y = room_y + room_size[1] - 1

    return (room_x, room_y), (door_x, door_y)


def add_blocks(builder, room_locations):
    for room_name, locations in room_locations.items():
        for loc in locations:
            # Get the block's name
            name = f"Block in {room_name}"

            # Get the probability for adding a block so we get the on average the requested number of blocks per room
            prob = min(1.0, average_blocks_per_room / len(locations))

            # Create a MATRX random property of shape and color so each block varies per created world.
            # These random property objects are used to obtain a certain value each time a new world is
            # created from this builder.
            colour_property = RandomProperty(values=block_colors)
            shape_property = RandomProperty(values=block_shapes)

            # Add the block; a regular SquareBlock as denoted by the given 'callable_class' which the
            # builder will use to create the object. In addition to setting MATRX properties, we also
            # provide a `is_block` boolean as custom property so we can identify this as a collectible
            # block.
            builder.add_object_prospect(loc, name, callable_class=CollectableBlock, probability=prob,
                                        visualize_shape=shape_property, visualize_colour=colour_property)


def add_drop_off_zones(builder, world_size):
    x = int(np.ceil(world_size[0] / 2)) - (int(np.floor(nr_drop_zones / 2)) * (hallway_space + 1))
    y = world_size[1] - 1 - 1  # once for off by one, another for world bound
    for nr_zone in range(nr_drop_zones):
        # Add the zone's tiles. Area tiles are special types of objects in MATRX that simply function as
        # a kind of floor. They are always traversable and cannot be picked up.
        builder.add_area((x, y - nr_blocks_needed + 1), width=1, height=nr_blocks_needed, name=f"Drop off {nr_zone}",
                         visualize_colour=drop_off_color, drop_zone_nr=nr_zone, is_drop_zone=True, is_goal_block=False,
                         is_collectable=False)

        # Go through all needed blocks
        for nr_block in range(nr_blocks_needed):
            # Create a MATRX random property of shape and color so each world contains different blocks to collect
            colour_property = RandomProperty(values=block_colors)
            shape_property = RandomProperty(values=block_shapes)

            # Add a 'ghost image' of the block that should be collected. This can be seen by both humans and agents to
            # know what should be collected in what order.
            loc = (x, y - nr_block)
            builder.add_object(loc, name="Collect Block", callable_class=GhostBlock,
                               visualize_colour=colour_property, visualize_shape=shape_property,
                               drop_zone_nr=nr_zone)

        # Change the x to the next zone
        x = x + hallway_space + 1


def add_agents(builder):
    # Create the agents sense capability. This is a circular range around the agent that denotes what it can perceive.
    # Here, we separated
    sense_capability = SenseCapability({AgentBody: agent_sense_range,
                                        CollectableBlock: block_sense_range,
                                        None: other_sense_range})

    loc = (0, 1)  # we begin adding agents to the top left, x is zero because we add +1 each time we add an agent
    for team in range(nr_teams):
        team_name = f"Team {team}"
        # Add agents
        nr_agents = agents_per_team - human_agents_per_team
        for agent_nr in range(nr_agents):
            brain = BlockWorldAgent(slowdown=agent_slowdown[agent_nr])
            loc = (loc[0] + 1, loc[1])
            builder.add_agent(loc, brain, team=team_name, name=f"Agent {agent_nr} in {team_name}",
                              sense_capability=sense_capability, img_name="/images/robotics.svg")

        # Add human agents
        for human_agent_nr in range(human_agents_per_team):
            brain = HumanAgentBrain(max_carry_objects=1, grab_range=0, drop_range=0, fov_occlusion=fov_occlusion)
            loc = (loc[0] + 1, loc[1])
            builder.add_human_agent(loc, brain, team=team_name, name=f"Human {human_agent_nr} in {team_name}",
                                    key_action_map=key_action_map, sense_capability=sense_capability, img_name="/images/first-responder.svg")


def add_rooms(builder):
    room_locations = {}
    for room_nr in range(nr_rooms):
        room_top_left, door_loc = get_room_loc(room_nr)

        # We assign a simple random color to each room. Not for any particular reason except to brighting up the place.
        np.random.shuffle(room_colors)
        room_color = room_colors[0]

        # Add the room
        room_name = f"room_{room_nr}"
        builder.add_room(top_left_location=room_top_left, width=room_size[0], height=room_size[1], name=room_name,
                         door_locations=[door_loc], wall_visualize_colour=wall_color,
                         with_area_tiles=True, area_visualize_colour=room_color, area_visualize_opacity=0.1)

        # Find all inner room locations where we allow objects (making sure that the location behind to door is free)
        room_locations[room_name] = builder.get_room_locations(room_top_left, room_size[0], room_size[1])

    return room_locations


def create_builder():

    # Set numpy's random generator
    np.random.seed(random_seed)

    # Get world size
    world_size = calculate_world_size()

    # Create the goal
    goal = CollectionGoal()

    # Create our world builder
    builder = WorldBuilder(shape=[24,25], tick_duration=tick_duration, random_seed=random_seed, run_matrx_api=True,
                           run_matrx_visualizer=True, verbose=verbose, simulation_goal=goal)

    # Add the world bounds (not needed, as agents cannot 'walk off' the grid, but for visual effects)
    builder.add_room(top_left_location=(0, 0), width=world_size[0], height=world_size[1], name="world_bounds")

    # Create the rooms
    room_locations = add_rooms(builder)

    # Add the collectible objects, we do so probabilistically so each world will contain different blocks
    add_blocks(builder, room_locations)

    # Create the drop-off zones, this includes generating the random colour/shape combinations to collect.
    add_drop_off_zones(builder, world_size)

    # Add the agents and human agents to the top row of the world
    add_agents(builder)

    # Return the builder
    return builder


class CollectableBlock(EnvObject):
    def __init__(self, location, name, visualize_colour, visualize_shape):
        super().__init__(location, name, is_traversable=True, is_movable=True,
                         visualize_colour=visualize_colour, visualize_shape=visualize_shape,
                         visualize_size=block_size, class_callable=CollectableBlock,
                         is_drop_zone=False, is_goal_block=False, is_collectable=True)


class GhostBlock(EnvObject):
    def __init__(self, location, drop_zone_nr, name, visualize_colour, visualize_shape):
        super().__init__(location, name, is_traversable=True, is_movable=False,
                         visualize_colour=visualize_colour, visualize_shape=visualize_shape,
                         visualize_size=block_size, class_callable=GhostBlock,
                         visualize_depth=85, drop_zone_nr=drop_zone_nr, visualize_opacity=0.5,
                         is_drop_zone=False, is_goal_block=True, is_collectable=False)


class CollectionGoal(WorldGoal):

    def __init__(self):
        super().__init__()

        # A dictionary of all drop locations. The keys is the drop zone number, the value another dict.
        # This dictionary contains as key the rank of the to be collected object and as value the location
        # of where it should be dropped, the shape and colour of the block, and the tick number the correct
        # block was delivered. The rank and tick number is there so we can check if objects are dropped in
        # the right order.
        self.__drop_off = None

        # We also track the progress
        self.__progress = 0

    def goal_reached(self, grid_world: GridWorld):
        if self.__drop_off is None:  # find all drop off locations, its tile ID's and goal blocks
            self.__find_drop_off_locations(grid_world)

        # Go through each drop zone, and check if the blocks are there in the right order
        is_satisfied, progress = self.__check_completion(grid_world)

        # Progress in percentage
        self.__progress = progress / sum([len(goal_blocks) for goal_blocks in self.__drop_off.values()])

        return is_satisfied

    def __find_drop_off_locations(self, grid_world):

        goal_blocks = {}  # dict with as key the zone nr and values list of ghostly goal blocks
        all_objs = grid_world.environment_objects
        for obj_id, obj in all_objs.items():  # go through all objects
            if "drop_zone_nr" in obj.properties.keys():  # check if the object is part of a drop zone
                zone_nr = obj.properties["drop_zone_nr"]  # obtain the zone number
                if obj.properties["is_goal_block"]:  # check if the object is a ghostly goal block
                    if zone_nr in goal_blocks.keys():  # create or add to the list
                        goal_blocks[zone_nr].append(obj)
                    else:
                        goal_blocks[zone_nr] = [obj]

        self.__drop_off = {}
        for zone_nr in goal_blocks.keys():  # go through all drop of zones and fill the drop_off dict
            # Instantiate the zone's dict.
            self.__drop_off[zone_nr] = {}

            # Obtain the zone's goal blocks.
            blocks = goal_blocks[zone_nr].copy()

            # The number of blocks is the maximum the max number blocks to collect for this zone.
            max_rank = len(blocks)

            # Find the 'bottom' location
            bottom_loc = (-np.inf, -np.inf)
            for block in blocks:
                if block.location[1] > bottom_loc[1]:
                    bottom_loc = block.location

            # Now loop through blocks lists and add them to their appropriate ranks
            for rank in range(max_rank):
                loc = (bottom_loc[0], bottom_loc[1] - rank)

                # find the block at that location
                for block in blocks:
                    if block.location == loc:
                        # Add to self.drop_off
                        self.__drop_off[zone_nr][rank] = [loc, block.visualize_shape, block.visualize_colour, None]

    def __check_completion(self, grid_world):
        # Get the current tick number
        curr_tick = grid_world.current_nr_ticks

        # loop through all zones, check the blocks and set the tick if satisfied
        for zone_nr, goal_blocks in self.__drop_off.items():
            # Go through all ranks of this drop off zone
            for rank, block_data in goal_blocks.items():
                loc = block_data[0]  # the location, needed to find blocks here
                shape = block_data[1]  # the desired shape
                colour = block_data[2]  # the desired colour
                tick = block_data[3]

                # Retrieve all objects, the object ids at the location and obtain all BW4T Blocks from it
                all_objs = grid_world.environment_objects
                obj_ids = grid_world.get_objects_in_range(loc, object_type=EnvObject, sense_range=0)
                blocks = [all_objs[obj_id] for obj_id in obj_ids
                          if obj_id in all_objs.keys() and "is_collectable" in all_objs[obj_id].properties.keys()]
                blocks = [b for b in blocks if b.properties["is_collectable"]]

                # Check if there is a block, and if so if it is the right one and the tick is not yet set, then set the
                # current tick.
                if len(blocks) > 0 and blocks[0].visualize_shape == shape and blocks[0].visualize_colour == colour and \
                        tick is None:
                    self.__drop_off[zone_nr][rank][3] = curr_tick
                # if there is no block, reset its tick to None
                elif len(blocks) == 0:
                    self.__drop_off[zone_nr][rank][3] = None

        # Now check if all blocks are collected in the right order
        is_satisfied = True
        progress = 0
        for zone_nr, goal_blocks in self.__drop_off.items():
            zone_satisfied = True
            ticks = [goal_blocks[r][3] for r in range(len(goal_blocks))]  # list of ticks in rank order

            # check if all ticks are increasing
            for idx, tick in enumerate(ticks[:-1]):
                if tick is None or ticks[idx+1] is None or not tick < ticks[idx+1]:
                    progress += (idx+1) if tick is not None else idx  # increment progress
                    zone_satisfied = False  # zone is not complete or ordered
                    break  # break this loop

            # if all ticks were increasing, check if the last tick is set and set progress to full for this zone
            if zone_satisfied and ticks[-1] is not None:
                progress += len(goal_blocks)

            # update our satisfied boolean
            is_satisfied = is_satisfied and zone_satisfied

        return is_satisfied, progress
