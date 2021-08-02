import os
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
from InstructionsExplainable import BlockWorldAgent
from HumanBrain import HumanBrain
from loggers.action_logger import ActionLogger
from datetime import datetime
from loggers.message_logger import MessageLogger

tick_duration = 0.05
random_seed = 1
verbose = False
key_action_map = {
        'w': MoveNorth.__name__,
        'd': MoveEast.__name__,
        's': MoveSouth.__name__,
        'a': MoveWest.__name__,
        'ArrowUp': MoveNorth.__name__,
        'ArrowRight': MoveEast.__name__,
        'ArrowDown': MoveSouth.__name__,
        'ArrowLeft': MoveWest.__name__,
        'q': GrabObject.__name__,
        'e': DropObject.__name__,
        'b': GrabObject.__name__,
        'n': DropObject.__name__
    }

# Some BW4T settings
nr_rooms = 9
room_colors = ['#0008ff', '#ff1500', '#0dff00']
wall_color = "#8a8a8a"
drop_off_color = "#878787"
block_size = 0.8
nr_drop_zones = 1
nr_teams = 1
agents_per_team = 2
human_agents_per_team = 1
agent_sense_range = 2  # the range with which agents detect other agents
block_sense_range = 1  # the range with which agents detect blocks
other_sense_range = np.inf  # the range with which agents detect other objects (walls, doors, etc.)
agent_memory_decay = 5  # we want to memorize states for seconds / tick_duration ticks
fov_occlusion = True

agent_slowdown=[3,2,1,1,1]
# Should be list with at least <bots_per_team  * nr_teams> values.
# wehere bots_per_team=agents_per_team - human_agents_per_team.
# 1=normal, 3 means 3x slowdown. allowed: natural numbers.
# This value is used in all actions of the agents.


def add_drop_off_zones(builder, world_size):
    x=1
    #x = int(np.ceil(world_size[0] / 2)) - (int(np.floor(nr_drop_zones / 2)) * (hallway_space + 1))
    #y = world_size[1] - 1 - 1  # once for off by one, another for world bound
    for nr_zone in range(nr_drop_zones):
        # Add the zone's tiles. Area tiles are special types of objects in MATRX that simply function as
        # a kind of floor. They are always traversable and cannot be picked up.
        builder.add_area((1,23), width=8, height=1, name=f"Drop off {nr_zone}", visualize_opacity=0.5, visualize_colour=drop_off_color, drop_zone_nr=nr_zone,
        is_drop_zone=True, is_goal_block=False, is_collectable=False)  

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
            brain = BlockWorldAgent(slowdown=25)
            loc = (9,23)
            builder.add_agent(loc, brain, team=team_name, name=f"Agent {agent_nr} in {team_name}",
                              sense_capability=sense_capability, is_traversable=True, img_name="/images/robotics5.svg")

        # Add human agents
        for human_agent_nr in range(human_agents_per_team):
            brain = HumanBrain(max_carry_objects=1, grab_range=0, drop_range=0, fov_occlusion=fov_occlusion)
            loc = (10,23)
            builder.add_human_agent(loc, brain, team=team_name, name=f"Human {human_agent_nr} in {team_name}",
                                    key_action_map=key_action_map, sense_capability=sense_capability, is_traversable=True, img_name="/images/first-responder6.svg")

def create_builder():
    # Set numpy's random generator
    np.random.seed(random_seed)

    # Create the goal
    goal = CollectionGoal(max_nr_ticks=10000000000000000000)
    # Create our world builder
    builder = WorldBuilder(shape=[24,25], tick_duration=tick_duration, run_matrx_api=True,
                           run_matrx_visualizer=False, verbose=verbose, simulation_goal=goal, visualization_bg_img="/images/background_70.svg")
    #current_exp_folder = datetime.now().strftime("exp_TRIAL_at_time_%Hh-%Mm-%Ss_date_%dd-%mm-%Yy")
    #logger_save_folder = os.path.join("experiment_logs", current_exp_folder)
    #builder.add_logger(ActionLogger, log_strategy=1, save_path=logger_save_folder, file_name_prefix="actions_")
    #builder.add_logger(MessageLogger, save_path=logger_save_folder, file_name_prefix="messages_")

    # Add the world bounds (not needed, as agents cannot 'walk off' the grid, but for visual effects)
    builder.add_room(top_left_location=(0, 0), width=24, height=25, name="world_bounds")
    bush_locations=[(1,12),(1,13),(1,14),(1,15),(1,16),(5,16),(5,15),(5,14),(7,14),(7,15),(7,16),(11,14),(11,15),(11,16),
    (3,12),(22,11),(21,11),(20,11),(19,11),(18,11),(17,11),(1,6),(2,6),(3,6),(16,10),(16,11),(5,6),(6,6),(7,6),(8,6),(9,6),(10,6),(11,6),(12,6),(13,6),
    (14,6),(14,7),(14,8),(14,9),(14,10),(14,11),(14,12),(14,13),(14,14),(14,15),(14,16),(14,17),(14,19),(14,20),(14,21),(14,22),(14,23),
    (3,22),(2,22),(5,22),(6,22),(7,22),(8,22),(1,22)]
    path_locations=[(2,5),(3,5),(4,5),(5,5),(6,5),(7,5),(8,5),(9,5),(10,5),(11,5),(12,5),(13,5),(14,5),(15,5),(16,5),(8,1),(12,1),(3,13),(2,13),(2,12),
    (4,6),(4,7),(4,8),(4,9),(4,10),(4,11),(4,12),(4,13),(5,13),(6,13),(7,13),(8,13),(9,13),(1,5),(4,1),(3,14),(3,15),(3,16),(9,14),(9,15),(9,16),
    (10,13),(11,13),(12,13),(13,13),(13,14),(13,15),(13,16),(13,17),(13,18),(14,18),(15,18),(13,19),(13,20),(13,21),(12,21),(11,21),(10,21),(9,21),(8,21),(7,21),(6,21),(5,21),
    (4,21),(4,22),(9,23),(10,23),(9,22),(10,22),(1,1),(1,2),(1,3),(1,4)]
    parquet_locations=[(18,4),(18,3),(18,2),(20,4)]
    fence_locations=[(6,2),(6,3),(6,4),(10,2),(10,3),(10,4),(14,2),(14,3),(14,4),(2,2),(2,3),(2,4),(2,1),(6,1),(10,1),(14,1),(6,20),(6,19),(6,18),(6,17),(6,16),(6,15),(6,14),
    (12,14),(12,15),(12,16),(12,17),(12,18),(12,19),(12,20)]
    plant_locations=[(15,13),(15,14),(15,15),(15,16),(15,17),(15,19),(15,20),(15,21),(15,22),(15,23),(2,16),(2,15),(2,14),(4,14),(4,15),(4,16),(8,14),(8,15),(8,16),
    (10,14),(10,15),(10,16)]
    for bush_loc in bush_locations:
        builder.add_object(bush_loc,'bush',EnvObject,is_traversable=False,is_movable=False,visualize_shape='img',img_name="/images/tree.svg")
    for path_loc in path_locations:
        builder.add_object(path_loc,'path',EnvObject,is_traversable=True,is_movable=False,visualize_shape='img',img_name="/images/paving10.svg")
    for fence_loc in fence_locations:
        builder.add_object(fence_loc,'fence',EnvObject,is_traversable=False,is_movable=False,visualize_shape='img',img_name="/images/fence.svg")
    for plant_loc in plant_locations:
        builder.add_object(plant_loc,'plant',EnvObject,is_traversable=False,is_movable=False,visualize_shape='img',img_name="/images/pool20.svg")
    #for parquet_loc in parquet_locations:
    #    builder.add_object(parquet_loc,'plant',EnvObject,is_traversable=False,is_movable=False,visualize_shape=0,visualize_colour=wall_color)

    builder.add_object((3,1),'bbq',EnvObject,is_traversable=False,is_movable=False,visualize_shape='img',img_name="/images/bbq2.svg") 
    builder.add_object((5,1),'tree',EnvObject,is_traversable=False,is_movable=False,visualize_shape='img',img_name="/images/tree.svg") 
    builder.add_object((7,1),'umbrella',EnvObject,is_traversable=False,is_movable=False,visualize_shape='img',img_name="/images/umbrella2.svg")
    builder.add_object((9,1),'tree',EnvObject,is_traversable=False,is_movable=False,visualize_shape='img',img_name="/images/tree.svg")
    builder.add_object((13,1),'tree',EnvObject,is_traversable=False,is_movable=False,visualize_shape='img',img_name="/images/tree.svg")
    builder.add_object((11,1),'tree',EnvObject,is_traversable=False,is_movable=False,visualize_shape='img',img_name="/images/tree.svg")
    builder.add_object((12,22),'ball',EnvObject,is_traversable=False,is_movable=True,visualize_shape='img',img_name="/images/soccer-ball.svg")

    #builder.add_object((1,16),'car',EnvObject,is_traversable=False,is_movable=False,visualize_shape='img',img_name="/images/car (1).svg")
    builder.add_object((1,23),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/critically injured girl.svg",drop_zone_nr=0)
    builder.add_object((2,23),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/critically injured elderly woman.svg",drop_zone_nr=0)
    builder.add_object((3,23),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/critically injured man.svg",drop_zone_nr=0)
    builder.add_object((4,23),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/critically injured dog.svg",drop_zone_nr=0)
    builder.add_object((5,23),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/mildly injured boy.svg",drop_zone_nr=0)
    builder.add_object((6,23),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/mildly injured elderly man.svg",drop_zone_nr=0)
    builder.add_object((7,23),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/mildly injured woman.svg",drop_zone_nr=0)
    builder.add_object((8,23),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/mildly injured cat.svg",drop_zone_nr=0)

    # Create the rooms
   # room_locations = add_rooms(builder)
    builder.add_room(top_left_location=(17,1), width=6, height=10, name='area A4', door_locations=[(17,5)],doors_open=True, wall_visualize_colour=wall_color, 
    with_area_tiles=True, area_custom_properties={'doormat':(16,5)}, area_visualize_colour=room_colors[0],area_visualize_opacity=0.0)
    builder.add_room(top_left_location=(3,2), width=3, height=3, name='area A1', door_locations=[(4,4)],doors_open=True, wall_visualize_colour=wall_color,
    with_area_tiles=True, area_custom_properties={'doormat':(4,5)}, area_visualize_colour=room_colors[0], area_visualize_opacity=0.0)
    builder.add_room(top_left_location=(7,2), width=3, height=3, name='area A2', door_locations=[(8,4)],doors_open=True, wall_visualize_colour=wall_color,
    with_area_tiles=True, area_custom_properties={'doormat':(9,4)}, area_visualize_colour=room_colors[0], area_visualize_opacity=0.0)
    builder.add_room(top_left_location=(11,2), width=3, height=3, name='area A3', door_locations=[(12,4)],doors_open=True, wall_visualize_colour=wall_color,
    with_area_tiles=True, area_custom_properties={'doormat':(12,5)}, area_visualize_colour=room_colors[0], area_visualize_opacity=0.0)
    builder.add_room(top_left_location=(1,7), width=3, height=5, name='area B1', door_locations=[(2,11)],doors_open=True, wall_visualize_colour=wall_color,
    with_area_tiles=True, area_custom_properties={'doormat':(2,12)}, area_visualize_colour=room_colors[0], area_visualize_opacity=0.0)
    builder.add_room(top_left_location=(5,7), width=9, height=6, name='area B2', door_locations=[(9,12)],doors_open=True, wall_visualize_colour=wall_color,
    with_area_tiles=True, area_custom_properties={'doormat':(9,13)}, area_visualize_colour=room_colors[0], area_visualize_opacity=0.0)
    builder.add_room(top_left_location=(1,17), width=5, height=4, name='area C1', door_locations=[(3,17)],doors_open=True, wall_visualize_colour=wall_color,
    with_area_tiles=True, area_custom_properties={'doormat':(3,16)}, area_visualize_colour=room_colors[0], area_visualize_opacity=0.0)
    builder.add_room(top_left_location=(7,17), width=5, height=4, name='area C2', door_locations=[(9,17)],doors_open=True, wall_visualize_colour=wall_color,
    with_area_tiles=True, area_custom_properties={'doormat':(9,16)}, area_visualize_colour=room_colors[0], area_visualize_opacity=0.0)
    builder.add_room((16,13),7,11,'area C3',door_locations=[(16,18)],doors_open=True,wall_visualize_colour=wall_color, 
    with_area_tiles=True, area_custom_properties={'doormat':(15,18)}, area_visualize_colour=room_colors[0], area_visualize_opacity=0.0)

    builder.add_object((9,18),'critically injured elderly woman in area C2', callable_class=CollectableBlock, 
    visualize_shape='img',img_name="/images/critically injured elderly woman.svg")
    builder.add_object((3,18),'critically injured man in area C1', callable_class=CollectableBlock, 
    visualize_shape='img',img_name="/images/critically injured man.svg")
    builder.add_object((18,18),'critically injured girl in area C3', callable_class=CollectableBlock, 
    visualize_shape='img',img_name="/images/critically injured girl.svg")
    builder.add_object((9,11),'critically injured dog in area B2', callable_class=CollectableBlock, 
    visualize_shape='img',img_name="/images/critically injured dog.svg")
    builder.add_object((2,10),'mildly injured boy in area B1', callable_class=CollectableBlock, 
    visualize_shape='img',img_name="/images/mildly injured boy.svg")
    builder.add_object((4,3),'mildly injured elderly man in area A1', callable_class=CollectableBlock, 
    visualize_shape='img',img_name="/images/mildly injured elderly man.svg")
    builder.add_object((8,3),'mildly injured woman in area A2', callable_class=CollectableBlock, 
    visualize_shape='img',img_name="/images/mildly injured woman.svg")
    builder.add_object((10,19),'mildly injured cat in area C2', callable_class=CollectableBlock, 
    visualize_shape='img',img_name="/images/mildly injured cat.svg")


    builder.add_object((18,17),'healthy man in area C3', callable_class=CollectableBlock, 
    visualize_shape='img',img_name="/images/healthy man.svg")  
    builder.add_object((18,19),'healthy woman in area C3', callable_class=CollectableBlock, 
    visualize_shape='img',img_name="/images/healthy woman.svg")   
    builder.add_object((20,19),'healthy dog in area C3', callable_class=CollectableBlock, 
    visualize_shape='img',img_name="/images/healthy dog.svg")   



    builder.add_object(location=[4,4], is_traversable=True, name="area A1 sign", img_name="/images/area1_new.svg", visualize_depth=110, visualize_size=0.55)
    builder.add_object(location=[8,4], is_traversable=True, name="area A2 sign", img_name="/images/areaA2.svg", visualize_depth=110, visualize_size=0.6)
    builder.add_object(location=[12,4], is_traversable=True, name="area A3 sign", img_name="/images/areaA3.svg", visualize_depth=110, visualize_size=0.6)
    builder.add_object(location=[17,5], is_traversable=True, name="area A4 sign", img_name="/images/areaA4.svg", visualize_depth=110, visualize_size=0.65)
    builder.add_object(location=[2,11], is_traversable=True, name="area B1 sign", img_name="/images/areaB1.svg", visualize_depth=110, visualize_size=0.5)
    builder.add_object(location=[9,12], is_traversable=True, name="area B2 sign", img_name="/images/areaB2.svg", visualize_depth=110, visualize_size=0.6)
    builder.add_object(location=[3,17], is_traversable=True, name="area C1 sign", img_name="/images/areaC1.svg", visualize_depth=110, visualize_size=0.5)
    builder.add_object(location=[9,17], is_traversable=True, name="area C2 sign", img_name="/images/areaC2.svg", visualize_depth=110, visualize_size=0.6)
    builder.add_object(location=[16,18], is_traversable=True, name="area C3 sign", img_name="/images/areaC3.svg", visualize_depth=110, visualize_size=0.6)

    #builder.add_object(location=[10,0], is_traversable=True, name="keyboard sign", img_name="/images/keyboard2.svg", visualize_depth=110, visualize_size=10)
    # Add the collectible objects, we do so probabilistically so each world will contain different blocks
    #add_blocks(builder, room_locations)
    # Create the drop-off zones, this includes generating the random colour/shape combinations to collect.
    add_drop_off_zones(builder, [24,25])

    # Add the agents and human agents to the top row of the world
    add_agents(builder)

    # Return the builder
    return builder

class CollectableBlock(EnvObject):
    def __init__(self, location, name, visualize_shape, img_name):
        super().__init__(location, name, is_traversable=True, is_movable=True,
                         visualize_shape=visualize_shape,img_name=img_name,
                         visualize_size=block_size, class_callable=CollectableBlock,
                         is_drop_zone=False, is_goal_block=False, is_collectable=True)


class GhostBlock(EnvObject):
    def __init__(self, location, drop_zone_nr, name, visualize_shape, img_name):
        super().__init__(location, name, is_traversable=True, is_movable=False,
                         visualize_shape=visualize_shape, img_name=img_name,
                         visualize_size=block_size, class_callable=GhostBlock,
                         visualize_depth=110, drop_zone_nr=drop_zone_nr, visualize_opacity=0.5,
                         is_drop_zone=False, is_goal_block=True, is_collectable=False)


class CollectionGoal(WorldGoal):
    '''
    The goal for BW4T world (the simulator), so determines
    when the simulator should stop.
    '''
    def __init__(self, max_nr_ticks:int):
        '''
        @param max_nr_ticks the max number of ticks to be used for this task
        '''
        super().__init__()
        self.max_nr_ticks = max_nr_ticks

        # A dictionary of all drop locations. The keys is the drop zone number, the value another dict.
        # This dictionary contains as key the rank of the to be collected object and as value the location
        # of where it should be dropped, the shape and colour of the block, and the tick number the correct
        # block was delivered. The rank and tick number is there so we can check if objects are dropped in
        # the right order.
        self.__drop_off:dict = {}
        self.__drop_off_zone:dict = {}

        # We also track the progress
        self.__progress = 0

    def goal_reached(self, grid_world: GridWorld):
        if grid_world.current_nr_ticks >= self.max_nr_ticks:
            return True
        return self.isBlocksPlaced(grid_world)

    def isBlocksPlaced(self, grid_world:GridWorld):
        '''
        @return true if all blocks have been placed in right order
        '''

        if self.__drop_off =={}:  # find all drop off locations, its tile ID's and goal blocks
            self.__find_drop_off_locations(grid_world)

        # Go through each drop zone, and check if the blocks are there in the right order
        is_satisfied, progress = self.__check_completion(grid_world)

        # Progress in percentage
        self.__progress = progress / sum([len(goal_blocks)\
            for goal_blocks in self.__drop_off.values()])
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

        self.__drop_off_zone:dict = {}
        self.__drop_off:dict = {}
        for zone_nr in goal_blocks.keys():  # go through all drop of zones and fill the drop_off dict
            # Instantiate the zone's dict.
            self.__drop_off_zone[zone_nr] = {}
            self.__drop_off[zone_nr] = {}

            # Obtain the zone's goal blocks.
            blocks = goal_blocks[zone_nr].copy()

            # The number of blocks is the maximum the max number blocks to collect for this zone.
            max_rank = len(blocks)

            # Find the 'bottom' location
            bottom_loc = (-np.inf, -np.inf)
            for block in blocks:
                if block.location[0] > bottom_loc[0]:
                    bottom_loc = block.location

            # Now loop through blocks lists and add them to their appropriate ranks
            for rank in range(max_rank):
                loc = (bottom_loc[0] - rank, bottom_loc[1])

                # find the block at that location
                for block in blocks:
                    if block.location == loc:
                        # Add to self.drop_off
                        self.__drop_off_zone[zone_nr][rank] = [loc, block.visualize_shape, block.visualize_colour, None]
                        for i in self.__drop_off_zone.keys():
                            self.__drop_off[i] = {}
                            vals = list(self.__drop_off_zone[i].values())
                            vals.reverse()
                            for j in range(len(self.__drop_off_zone[i].keys())):
                                self.__drop_off[i][j] = vals[j]

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
