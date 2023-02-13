import os
import sys
import numpy as np
import itertools
from collections import OrderedDict
from itertools import product
from matrx import WorldBuilder
from matrx.actions import MoveNorth, OpenDoorAction, CloseDoorAction, GrabObject
from matrx.actions.move_actions import MoveEast, MoveSouth, MoveWest
from matrx.agents import AgentBrain, HumanAgentBrain, SenseCapability
from matrx.grid_world import GridWorld, AgentBody
from actions1.CustomActions import RemoveObjectTogether, DropObject, Idle, CarryObject, Drop, CarryObjectTogether, DropObjectTogether
from matrx.actions.object_actions import RemoveObject
from matrx.objects import EnvObject
from matrx.world_builder import RandomProperty
from matrx.goals import WorldGoal
from agents1.OfficialAgent import BaselineAgent
from agents1.TutorialAgent import TutorialAgent
from actions1.CustomActions import RemoveObjectTogether
from brains1.HumanBrain import HumanBrain
from loggers.ActionLogger import ActionLogger
from datetime import datetime

random_seed = 1
verbose = False
# Tick duration determines the speed of the world. A tick duration of 0.1 means 10 ticks are executed in a second. 
# You can speed up or slow down the world by changing this value without changing behavior. Leave this value at 0.1 during evaluations.
tick_duration = 0.1
# Define the keyboarc controls for the human agent
key_action_map = {
        'ArrowUp': MoveNorth.__name__,
        'ArrowRight': MoveEast.__name__,
        'ArrowDown': MoveSouth.__name__,
        'ArrowLeft': MoveWest.__name__,
        'q': CarryObject.__name__,
        'w': Drop.__name__,
        'd': RemoveObjectTogether.__name__,
        'a': CarryObjectTogether.__name__,
        's': DropObjectTogether.__name__,
        'e': RemoveObject.__name__,
    }

# Some settings
nr_rooms = 9
wall_color = "#8a8a8a"
drop_off_color = "#1F262A"
object_size = 0.9
nr_teams = 1
agents_per_team = 2
human_agents_per_team = 1
agent_sense_range = 2  # the range with which agents detect other agents. Do not change this value.
object_sense_range = 1  # the range with which agents detect blocks. Do not change this value.
other_sense_range = np.inf  # the range with which agents detect other objects (walls, doors, etc.). Do not change this value.
fov_occlusion = True

# Add the drop zones to the world
def add_drop_off_zones(builder, task_type):
    if task_type == "official":
        nr_drop_zones = 1
        for nr_zone in range(nr_drop_zones):
            builder.add_area((23,8), width=1, height=8, name=f"Drop off {nr_zone}", visualize_opacity=0.5, visualize_colour=drop_off_color, drop_zone_nr=nr_zone, is_drop_zone=True, is_goal_block=False, is_collectable=False) 
    if task_type == "tutorial":
        nr_drop_zones = 1
        for nr_zone in range(nr_drop_zones):
            builder.add_area((17,7), width=1, height=4, name=f"Drop off {nr_zone}",visualize_opacity=0.5, visualize_colour=drop_off_color, drop_zone_nr=nr_zone, is_drop_zone=True, is_goal_block=False, is_collectable=False) 

# Add the agents to the world
def add_agents(builder, condition, task_type, name, folder):
    # Define the agent's sense capabilites
    sense_capability_agent = SenseCapability({AgentBody: agent_sense_range, CollectableBlock: object_sense_range, None: other_sense_range, ObstacleObject: 1})
    # Define the human's sense capabilities based on the selected condition
    if condition=='normal' or condition=='weak' or condition=='tutorial':
        sense_capability_human = SenseCapability({AgentBody: agent_sense_range, CollectableBlock: object_sense_range, None: other_sense_range, ObstacleObject: 1})
    if condition=='strong':
        sense_capability_human = SenseCapability({AgentBody: agent_sense_range, CollectableBlock: object_sense_range, None: other_sense_range, ObstacleObject: 10})

    for team in range(nr_teams):
        team_name = f"Team {team}"
        # Add the artificial agents based on condition
        nr_agents = agents_per_team - human_agents_per_team
        for agent_nr in range(nr_agents):
            if task_type=="official":
                brain = BaselineAgent(slowdown=8, condition=condition, name=name, folder=folder) # Slowdown makes the agent a bit slower, do not change value during evaluations
                loc = (22,11)
            if task_type=="tutorial":
                brain = TutorialAgent(slowdown=8, condition=condition, name=name, folder=folder)
                loc = (16,8)
            builder.add_agent(loc, brain, team=team_name, name="RescueBot",customizable_properties = ['score'], score=0, sense_capability=sense_capability_agent, is_traversable=True, img_name="/images/robot-final4.svg")

        # Add human agents based on condition, do not change human brain values
        for human_agent_nr in range(human_agents_per_team):
            if condition=='strong':
                brain = HumanBrain(max_carry_objects=np.inf, grab_range=1, drop_range=0, remove_range=1, fov_occlusion=fov_occlusion, strength=condition, name=name)
            else:
                brain = HumanBrain(max_carry_objects=1, grab_range=1, drop_range=0, remove_range=1, fov_occlusion=fov_occlusion, strength=condition, name=name)
            if task_type=="official":
                loc = (22,12)
            else:
                loc = (16,9)
            builder.add_human_agent(loc, brain, team=team_name, name=name, key_action_map=key_action_map, sense_capability=sense_capability_human, is_traversable=True, img_name="/images/rescue-man-final3.svg", visualize_when_busy=True)

# Create the world
def create_builder(task_type, condition, name, folder):
    # Set numpy's random generator
    np.random.seed(random_seed)
    # Create the collection goal
    goal = CollectionGoal(max_nr_ticks=np.inf)
    # Create the world builder
    if task_type=="official":
        builder = WorldBuilder(shape=[25,24], tick_duration=tick_duration, run_matrx_api=True, run_matrx_visualizer=False, verbose=verbose, simulation_goal=goal, visualization_bg_clr='#9a9083')
    else:
        builder = WorldBuilder(shape=[19,19], tick_duration=tick_duration, run_matrx_api=True,random_seed=random_seed, run_matrx_visualizer=False, verbose=verbose, simulation_goal=goal, visualization_bg_clr='#9a9083')

    # Add all areas and objects to the tutorial world
    if task_type == "tutorial":
        builder.add_room(top_left_location=(0, 0), width=19, height=19, name="world_bounds", wall_visualize_colour="#1F262A")
        builder.add_room(top_left_location=(1,1), width=5, height=4, name='area 1', door_locations=[(3,4)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0, door_open_colour='#9a9083', area_custom_properties={'doormat':(3,5)})
        builder.add_room(top_left_location=(7,1), width=5, height=4, name='area 2', door_locations=[(9,4)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(9,5)})
        builder.add_room(top_left_location=(13,1), width=5, height=4, name='area 3', door_locations=[(15,4)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(15,5)})
        builder.add_room(top_left_location=(1,7), width=5, height=4, name='area 4', door_locations=[(3,7)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(3,6)})
        builder.add_room(top_left_location=(7,7), width=5, height=4, name='area 5', door_locations=[(9,7)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(9,6)})
        builder.add_room(top_left_location=(1,13), width=5, height=4, name='area 6', door_locations=[(3,16)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(3,17)})
        builder.add_room(top_left_location=(7,13), width=5, height=4, name='area 7', door_locations=[(9,16)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(9,17)})
        builder.add_room(top_left_location=(13,13), width=5, height=4, name='area 8', door_locations=[(15,16)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(15,17)})

        builder.add_object((3,4), 'stone',ObstacleObject,visualize_shape='img',img_name="/images/stone-small.svg")
        builder.add_object((3,7),'tree',ObstacleObject,visualize_shape='img',img_name="/images/tree-fallen2.svg")
        builder.add_object((3,16),'tree',ObstacleObject,visualize_shape='img',img_name="/images/tree-fallen2.svg")
        builder.add_object((9,16),'rock',ObstacleObject,visualize_shape='img',img_name="/images/stone.svg")
        builder.add_object((15,16),'stone',ObstacleObject,visualize_shape='img',img_name="/images/stone-small.svg")
        builder.add_object((9,7),'rock',ObstacleObject,visualize_shape='img',img_name="/images/stone.svg")

        builder.add_object((16,3),'critically injured elderly woman in area 3', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/critically injured elderly woman.svg")
        builder.add_object((14,14),'healthy man in area 8', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy man.svg")
        builder.add_object((2,9),'mildly injured elderly man in area 4', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/mildly injured elderly man.svg")
        builder.add_object((2,14),'healthy girl in area 6', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy girl.svg")
        builder.add_object((8,9),'critically injured girl in area 5', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/critically injured girl.svg")
        builder.add_object((16,15),'mildly injured boy in area 8', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/mildly injured boy.svg")
        builder.add_object((10,3),'healthy boy in area 2', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy boy.svg")
        builder.add_object((10,8),'healthy elderly man in area 5', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy elderly man.svg")
        builder.add_object((10,15),'healthy dog in area 7', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy dog.svg")

        builder.add_object((17,7),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/critically injured girl.svg",drop_zone_nr=0)
        builder.add_object((17,8),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/critically injured elderly woman.svg",drop_zone_nr=0)
        builder.add_object((17,9),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/mildly injured boy.svg",drop_zone_nr=0)
        builder.add_object((17,10),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/mildly injured elderly man.svg",drop_zone_nr=0)

        builder.add_object(location=[3,1], is_traversable=True, is_movable=False, name="area 01 sign", img_name="/images/sign01.svg", visualize_depth=110, visualize_size=0.5)
        builder.add_object(location=[9,1], is_traversable=True, is_movable=False, name="area 02 sign", img_name="/images/sign02.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[15,1], is_traversable=True, is_movable=False, name="area 03 sign", img_name="/images/sign03.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[3,10], is_traversable=True, is_movable=False, name="area 04 sign", img_name="/images/sign04.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[9,10], is_traversable=True, is_movable=False, name="area 05 sign", img_name="/images/sign05.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[3,13], is_traversable=True, is_movable=False, name="area 06 sign", img_name="/images/sign06.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[9,13], is_traversable=True, is_movable=False, name="area 07 sign", img_name="/images/sign07.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[15,13], is_traversable=True, is_movable=False, name="area 08 sign", img_name="/images/sign08.svg", visualize_depth=110, visualize_size=0.55)

        builder.add_object(location=[9,0], is_traversable=True, name="keyboard sign", img_name="/images/keyboard-final.svg", visualize_depth=110, visualize_size=15)

        for loc in [(1,1),(2,1),(3,1),(4,1),(5,1),(1,2),(1,3),(1,4),(2,4),(4,4),(5,4),(5,3),(5,2),(7,1),(8,1),
                    (13,1),(14,1),(15,1),(16,1),(17,1),(13,1),(14,1),(15,1),(16,1),(17,1),(13,2),(13,3),(13,4),
                    (1,7),(1,8),(1,9),(1,10),(2,10),(3,10),(4,10),(5,10),(5,9),(5,8),(5,7),(4,7),(2,7),(8,4),
                    (7,7),(7,8),(7,9),(7,10),(8,10),(9,10),(10,10),(11,10),(11,9),(11,8),(11,7),(10,7),(8,7),
                    (1,13),(2,13),(3,13),(4,13),(5,13),(1,14),(1,15),(1,16),(2,16),(4,16),(5,16),(5,15),(5,14),
                    (7,13),(8,13),(9,13),(10,13),(11,13),(7,14),(7,15),(7,16),(8,16),(10,16),(11,16),(11,15),
                    (11,14),(17,16),(16,16),(5,13),(14,4),(16,4),(17,4),(17,3),(17,2),(9,1),(10,1),(11,1),(7,2),
                    (13,13),(14,13),(15,13),(16,13),(17,13),(13,14),(13,15),(13,16),(14,16),(17,14),(17,15),
                    (7,3),(7,4),(11,2),(11,3),(11,4),(10,4)]:
            builder.add_object(loc,'roof', EnvObject,is_traversable=True, is_movable=False, visualize_shape='img',img_name="/images/roof-final5.svg")

    # Create folders where the logs are stored during the official condition
    if task_type=="official":
        current_exp_folder = datetime.now().strftime("exp_"+condition+"_at_time_%Hh-%Mm-%Ss_date_%dd-%mm-%Yy")
        logger_save_folder = os.path.join("logs", current_exp_folder)
        builder.add_logger(ActionLogger, log_strategy=1, save_path=logger_save_folder, file_name_prefix="actions_")
        
    # Add all area and objects to the official world
    if task_type == "official":
        builder.add_room(top_left_location=(0, 0), width=25, height=24, name="world_bounds", wall_visualize_colour="#1F262A")
        builder.add_room(top_left_location=(1,1), width=5, height=4, name='area 1', door_locations=[(3,4)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0, door_open_colour='#9a9083', area_custom_properties={'doormat':(3,5)})
        builder.add_room(top_left_location=(7,1), width=5, height=4, name='area 2', door_locations=[(9,4)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(9,5)})
        builder.add_room(top_left_location=(13,1), width=5, height=4, name='area 3', door_locations=[(15,4)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(15,5)})
        builder.add_room(top_left_location=(19,1), width=5, height=4, name='area 4', door_locations=[(21,4)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(21,5)})
        builder.add_room(top_left_location=(1,7), width=5, height=4, name='area 5', door_locations=[(3,7)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(3,6)})
        builder.add_room(top_left_location=(7,7), width=5, height=4, name='area 6', door_locations=[(9,7)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(9,6)})
        builder.add_room(top_left_location=(13,7), width=5, height=4, name='area 7', door_locations=[(15,7)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(15,6)})
        builder.add_room(top_left_location=(1,13), width=5, height=4, name='area 8', door_locations=[(3,16)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(3,17)})
        builder.add_room(top_left_location=(7,13), width=5, height=4, name='area 9', door_locations=[(9,16)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(9,17)})
        builder.add_room(top_left_location=(13,13), width=5, height=4, name='area 10', door_locations=[(15,16)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(15,17)})
        builder.add_room(top_left_location=(1,19), width=5, height=4, name='area 11', door_locations=[(3,19)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(3,18)})
        builder.add_room(top_left_location=(7,19), width=5, height=4, name='area 12', door_locations=[(9,19)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(9,18)})
        builder.add_room(top_left_location=(13,19), width=5, height=4, name='area 13', door_locations=[(15,19)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(15,18)})
        builder.add_room(top_left_location=(19,19), width=5, height=4, name='area 14', door_locations=[(21,19)],doors_open=True, wall_visualize_colour=wall_color, with_area_tiles=True, area_visualize_colour='#0008ff',area_visualize_opacity=0.0,door_open_colour='#9a9083', area_custom_properties={'doormat':(21,18)})

        builder.add_object((3,4), 'rock',ObstacleObject,visualize_shape='img',img_name="/images/stone.svg")
        builder.add_object((9,4),'stone',ObstacleObject,visualize_shape='img',img_name="/images/stone-small.svg")
        builder.add_object((9,16),'tree',ObstacleObject,visualize_shape='img',img_name="/images/tree-fallen2.svg")
        builder.add_object((15, 7),'tree',ObstacleObject,visualize_shape='img',img_name="/images/tree-fallen2.svg")
        builder.add_object((15,19),'tree',ObstacleObject,visualize_shape='img',img_name="/images/tree-fallen2.svg")
        builder.add_object((3,16),'rock',ObstacleObject,visualize_shape='img',img_name="/images/stone.svg")
        builder.add_object((15,4),'rock',ObstacleObject,visualize_shape='img',img_name="/images/stone.svg")
        builder.add_object((21,19),'stone',ObstacleObject,visualize_shape='img',img_name="/images/stone-small.svg")
        builder.add_object((9,19),'stone',ObstacleObject,visualize_shape='img',img_name="/images/stone-small.svg")
        builder.add_object((9,7),'stone',ObstacleObject,visualize_shape='img',img_name="/images/stone-small.svg")

        builder.add_object((1,12),'plant',EnvObject,is_traversable=True,is_movable=False,visualize_shape='img',img_name="/images/tree.svg", visualize_size=3)
        builder.add_object((21,7),'heli',EnvObject,is_traversable=False,is_movable=False,visualize_shape='img',img_name="/images/helicopter.svg", visualize_size=3) 
        builder.add_object((21,16),'ambulance',EnvObject,is_traversable=False,is_movable=False,visualize_shape='img',img_name="/images/ambulance.svg", visualize_size=2.3) 
    
        builder.add_object((23,8),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/critically injured girl.svg",drop_zone_nr=0)
        builder.add_object((23,9),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/critically injured elderly woman.svg",drop_zone_nr=0)
        builder.add_object((23,10),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/critically injured man.svg",drop_zone_nr=0)
        builder.add_object((23,11),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/critically injured dog.svg",drop_zone_nr=0)
        builder.add_object((23,12),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/mildly injured boy.svg",drop_zone_nr=0)
        builder.add_object((23,13),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/mildly injured elderly man.svg",drop_zone_nr=0)
        builder.add_object((23,14),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/mildly injured woman.svg",drop_zone_nr=0)
        builder.add_object((23,15),name="Collect Block", callable_class=GhostBlock,visualize_shape='img',img_name="/images/mildly injured cat.svg",drop_zone_nr=0)

        builder.add_object((10,15),'critically injured elderly woman in area 9', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/critically injured elderly woman.svg")
        builder.add_object((8,20),'healthy elderly woman in area 12', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy elderly woman.svg")
        builder.add_object((14,14),'healthy man in area 10', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy man.svg")
        builder.add_object((4,15),'critically injured man in area 8', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/critically injured man.svg")
        builder.add_object((2,14),'healthy girl in area 8', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy girl.svg")
        builder.add_object((10,3),'critically injured girl in area 2', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/critically injured girl.svg")
        builder.add_object((2,2),'mildly injured boy in area 1', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/mildly injured boy.svg")
        builder.add_object((16,3),'healthy boy in area 3', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy boy.svg")
        builder.add_object((14,20),'mildly injured elderly man in area 13', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/mildly injured elderly man.svg")
        builder.add_object((10,8),'healthy elderly man in area 6', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy elderly man.svg")
        builder.add_object((14,8),'mildly injured woman in area 7', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/mildly injured woman.svg")
        builder.add_object((16,21),'healthy woman in area 13', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy woman.svg")
        builder.add_object((8,9),'critically injured dog in area 6', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/critically injured dog.svg")
        builder.add_object((4,21),'mildly injured cat in area 11', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/mildly injured cat.svg")
        builder.add_object((10,21),'healthy girl in area 12', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy girl.svg")
        builder.add_object((16,9),'healthy girl in area 7', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy girl.svg")
        builder.add_object((22,3),'healthy boy in area 4', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy boy.svg")
        builder.add_object((2,20),'healthy elderly woman in area 11', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy elderly woman.svg")
        builder.add_object((20,2),'healthy man in area 4', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy man.svg")
        builder.add_object((20,20),'healthy man in area 14', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy man.svg")
        builder.add_object((22,21),'healthy boy in area 14', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy boy.svg")
        builder.add_object((8,14),'healthy boy in area 9', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy boy.svg")
        builder.add_object((4,3),'healthy elderly man in area 1', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy elderly man.svg")
        builder.add_object((14,2),'healthy elderly man in area 3', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy elderly man.svg")
        builder.add_object((16,15),'healthy woman in area 10', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy woman.svg")
        builder.add_object((8,2),'healthy woman in area 2', callable_class=CollectableBlock, visualize_shape='img',img_name="/images/healthy woman.svg")

        builder.add_object(location=[3,1], is_traversable=True, is_movable=False, name="area 01 sign", img_name="/images/sign01.svg", visualize_depth=110, visualize_size=0.5)
        builder.add_object(location=[9,1], is_traversable=True, is_movable=False, name="area 02 sign", img_name="/images/sign02.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[15,1], is_traversable=True, is_movable=False, name="area 03 sign", img_name="/images/sign03.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[21,1], is_traversable=True, is_movable=False, name="area 04 sign", img_name="/images/sign04.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[3,10], is_traversable=True, is_movable=False, name="area 05 sign", img_name="/images/sign05.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[9,10], is_traversable=True, is_movable=False, name="area 06 sign", img_name="/images/sign06.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[15,10], is_traversable=True, is_movable=False, name="area 07 sign", img_name="/images/sign07.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[3,13], is_traversable=True, is_movable=False, name="area 08 sign", img_name="/images/sign08.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[9,13], is_traversable=True, is_movable=False, name="area 09 sign", img_name="/images/sign09.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[15,13], is_traversable=True, is_movable=False, name="area 10 sign", img_name="/images/sign10.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[3,22], is_traversable=True, is_movable=False, name="area 11 sign", img_name="/images/sign11.svg", visualize_depth=110, visualize_size=0.45)
        builder.add_object(location=[9,22], is_traversable=True, is_movable=False, name="area 12 sign", img_name="/images/sign12.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[15,22], is_traversable=True, is_movable=False, name="area 13 sign", img_name="/images/sign13.svg", visualize_depth=110, visualize_size=0.55)
        builder.add_object(location=[21,22], is_traversable=True, is_movable=False, name="area 14 sign", img_name="/images/sign14.svg", visualize_depth=110, visualize_size=0.55)

        builder.add_object(location=[12,0], is_traversable=True, name="keyboard sign", img_name="/images/keyboard-final.svg", visualize_depth=110, visualize_size=20)

        for loc in [(1,1),(2,1),(3,1),(4,1),(5,1),(1,2),(1,3),(1,4),(2,4),(4,4),(5,4),(5,3),(5,2),(7,1),(8,1),(9,1),
                    (10,1),(11,1),(7,2),(7,3),(7,4),(8,4),(11,2),(11,3),(11,4),(10,4),(16,4),(17,4),(17,3),(17,2),
                    (13,1),(14,1),(15,1),(16,1),(17,1),(13,1),(14,1),(15,1),(16,1),(17,1),(13,2),(13,3),(13,4),(14,4),
                    (19,1),(20,1),(21,1),(22,1),(23,1),(19,2),(19,3),(19,4),(20,4),(22,4),(23,4),(23,3),(23,2),(23,1),
                    (1,7),(1,8),(1,9),(1,10),(2,10),(3,10),(4,10),(5,10),(5,9),(5,8),(5,7),(4,7),(2,7),(16,16),(10,19),
                    (13,7),(13,8),(13,9),(13,10),(14,10),(15,10),(16,10),(17,10),(17,9),(17,8),(17,7),(16,7),(14,7),
                    (1,13),(2,13),(3,13),(4,13),(5,13),(1,14),(1,15),(1,16),(2,16),(4,16),(5,16),(5,15),(5,14),(5,13),
                    (7,13),(8,13),(9,13),(10,13),(11,13),(7,14),(7,15),(7,16),(8,16),(10,16),(11,16),(11,15),(11,14),
                    (13,13),(14,13),(15,13),(16,13),(17,13),(13,14),(13,15),(13,16),(14,16),(17,14),(17,15),(17,16),
                    (1,19),(2,19),(4,19),(5,19),(1,20),(1,21),(1,22),(2,22),(3,22),(4,22),(5,22),(5,21),(5,20),(5,19),
                    (7,19),(8,19),(4,19),(5,19),(7,20),(7,21),(7,22),(8,22),(9,22),(10,22),(11,22),(11,21),(11,20),(11,19),
                    (13,19),(14,19),(16,19),(17,19),(13,20),(13,21),(13,22),(14,22),(15,22),(16,22),(17,22),(17,21),(17,20),
                    (19,19),(20,19),(22,19),(23,19),(19,20),(19,21),(19,22),(20,22),(21,22),(22,22),(23,22),(23,21),(23,20),
                    (7,7),(7,8),(7,9),(7,10),(8,10),(9,10),(10,10),(11,10),(11,9),(11,8),(11,7),(10,7),(8,7)]:
            builder.add_object(loc,'roof', EnvObject,is_traversable=True, is_movable=False, visualize_shape='img',img_name="/images/roof-final5.svg")

        for loc in [(6,1),(6,2),(6,3),(6,4),(6,5),(6,12),(6,13),(6,14),(6,15),(6,16),(6,17),(11,12),(11,11),(18,12),(18,21),
                    (3,12),(3,11),(12,6),(12,7),(12,8),(12,9),(12,10),(12,11),(18,11),(18,10),(18,9),(19,9),(19,8),(18,22),
                    (18,13),(18,14),(18,15),(18,16),(18,17),(9,17),(9,18),(20,17),(20,18),(12,1),(12,2),(6,22),(18,20),
                    (19,7),(19,6),(19,5),(10,6),(10,5),(14,17),(14,18),(12,19),(12,20),(12,21),(12,18),(12,22)]:
            builder.add_object(loc,'water',EnvObject,is_traversable=True, is_movable=False, visualize_shape='img',img_name="/images/pool20.svg")

        for loc in [(1,11),(2,11),(3,11),(3,12),(4,12),(5,12),(6,12),(7,12),(8,12),(9,12),(10,12),(11,12),(12,11),(13,11),(20,17),
                    (14,11),(15,11),(16,11),(17,11),(18,11),(6,17),(7,17),(8,17),(9,17),(9,18),(5,17),(4,17),(3,17),(2,17),(1,17),
                    (18,9),(19,9),(19,5),(20,5),(21,5),(22,5),(23,5),(11,6),(12,6),(10,6),(10,5),(9,5),(8,5),(7,5),(6,5),(19,17),
                    (11,11),(18,17),(17,17),(16,17),(15,17),(14,17),(14,18),(13,18),(12,18),(10,18),(11,18)]:
            builder.add_object(loc,'water', EnvObject,is_traversable=True, is_movable=False, visualize_shape='img', img_name="/images/lake2.svg")

        for loc in [(11,5),(13,5),(14,5),(13,6),(14,6),(12,5),(15,5),(15,6),(16,5),(16,6),(17,5),(17,6),(18,5),
                    (8,6),(7,6),(6,6),(5,6),(4,6),(3,6),(2,6),(1,6),(20,9),(21,9),(21,14),(20,14),(19,14),(9,6),
                    (1,5),(2,5),(3,5),(4,5),(5,5),(22,11),(22,12),(19,18),(18,18),(17,18),(16,18),(15,18),(13,17),
                    (11,17),(10,17),(8,18),(7,18),(6,18),(5,18),(4,18),(3,18),(2,18),(1,18),(12,17),(18,6)]:
            builder.add_object(loc,'street',EnvObject,is_traversable=True,is_movable=False,visualize_shape='img',img_name="/images/paving-final20.svg", visualize_size=1) 

        for loc in [(12,3),(12,4),(18,1),(18,2),(18,3),(18,4),(6,19),(6,20),(6,21),(18,19)]:
            builder.add_object(loc,'plant',EnvObject,is_traversable=True,is_movable=False,visualize_shape='img',img_name="/images/tree.svg", visualize_size=1.25) 
            
        for loc in [(21,10),(21,11),(21,12),(21,13),(19,15),(19,16)]:
            builder.add_object(loc,'street',EnvObject,is_traversable=True,is_movable=False,visualize_shape='img',img_name="/images/paving-final15.svg", visualize_size=1) 
    
    add_drop_off_zones(builder, task_type)
    add_agents(builder, condition, task_type, name, folder)

    return builder

class CollectableBlock(EnvObject):
    '''
    Objects that can be collected by agents.
    '''
    def __init__(self, location, name, visualize_shape, img_name):
        super().__init__(location, name, is_traversable=True, is_movable=True,
                         visualize_shape=visualize_shape,img_name=img_name,
                         visualize_size=object_size, class_callable=CollectableBlock,
                         is_drop_zone=False, is_goal_block=False, is_collectable=True)

class ObstacleObject(EnvObject):
    '''
    Obstacles that can be removed by agents
    '''
    def __init__(self, location, name, visualize_shape, img_name):
        super().__init__(location, name, is_traversable=False, is_movable=True,
                         visualize_shape=visualize_shape,img_name=img_name,
                         visualize_size=1.25, class_callable=ObstacleObject,
                         is_drop_zone=False, is_goal_block=False, is_collectable=False)

class GhostBlock(EnvObject):
    '''
    Objects on the drop zone that cannot be carried by agents.
    '''
    def __init__(self, location, drop_zone_nr, name, visualize_shape, img_name):
        super().__init__(location, name, is_traversable=True, is_movable=False,
                         visualize_shape=visualize_shape, img_name=img_name,
                         visualize_size=object_size, class_callable=GhostBlock,
                         visualize_depth=110, drop_zone_nr=drop_zone_nr, visualize_opacity=0.5,
                         is_drop_zone=False, is_goal_block=True, is_collectable=False)

class CollectionGoal(WorldGoal):
    '''
    The goal for world which determines when the simulator should stop.
    '''
    def __init__(self, max_nr_ticks):
        super().__init__()
        self.max_nr_ticks = max_nr_ticks
        self.__drop_off= {}
        self.__drop_off_zone = {}
        self.__progress = 0
        self.__score = 0
    
    def score(self, grid_world):
        return self.__score

    def goal_reached(self, grid_world):
        if grid_world.current_nr_ticks >= self.max_nr_ticks:
            return True
        return self.isVictimPlaced(grid_world)

    def isVictimPlaced(self, grid_world):
        '''
        @return true if all victims have been rescued
        '''
        # find all drop off locations, its tile ID's and goal victims
        if self.__drop_off =={}:
            self.__find_drop_off_locations(grid_world)
        # Go through each drop zone, and check if the victims are there on the right spot
        is_satisfied, progress = self.__check_completion(grid_world)
        # Progress in percentage
        self.__progress = progress / sum([len(goal_vics) for goal_vics in self.__drop_off.values()])

        return is_satisfied

    def progress(self, grid_world):
        # find all drop off locations, its tile ID's and goal blocks
        if self.__drop_off =={}:  
            self.__find_drop_off_locations(grid_world)
        # Go through each drop zone, and check if the victims are there in the right spot
        is_satisfied, progress = self.__check_completion(grid_world)
        # Progress in percentage
        self.__progress = progress / sum([len(goal_vics) for goal_vics in self.__drop_off.values()])
        return self.__progress

    def __find_drop_off_locations(self, grid_world):
        goal_vics = {} 
        all_objs = grid_world.environment_objects
        for obj_id, obj in all_objs.items():  # go through all objects
            if "drop_zone_nr" in obj.properties.keys():  # check if the object is part of a drop zone
                zone_nr = obj.properties["drop_zone_nr"]  # obtain the zone number
                if obj.properties["is_goal_block"]:  # check if the object is a ghostly goal victim
                    if zone_nr in goal_vics.keys():  # create or add to the list
                        goal_vics[zone_nr].append(obj)
                    else:
                        goal_vics[zone_nr] = [obj]

        self.__drop_off_zone = {}
        self.__drop_off = {}
        for zone_nr in goal_vics.keys():  # go through all drop of zones and fill the drop_off dict
            # Instantiate the zone's dict.
            self.__drop_off_zone[zone_nr] = {}
            self.__drop_off[zone_nr] = {}
            # Obtain the zone's goal victims.
            vics = goal_vics[zone_nr].copy()
            # The number of victims is the maximum number of victims to collect for this zone.
            max_rank = len(vics)
            # Find the 'bottom' location
            bottom_loc = (-np.inf, -np.inf)
            for vic in vics:
                if vic.location[1] > bottom_loc[1]:
                    bottom_loc = vic.location
            # Now loop through victim lists and add them to their appropriate ranks
            for rank in range(max_rank):
                loc = (bottom_loc[0], bottom_loc[1]-rank)
                # find the victim at that location
                for vic in vics:
                    if vic.location == loc:
                        # Add to self.drop_off
                        self.__drop_off_zone[zone_nr][rank] = [loc, vic.properties['img_name'][8:-4], None]
                        for i in self.__drop_off_zone.keys():
                            self.__drop_off[i] = {}
                            vals = list(self.__drop_off_zone[i].values())
                            vals.reverse()
                            for j in range(len(self.__drop_off_zone[i].keys())):
                                self.__drop_off[i][j] = vals[j]

    def __check_completion(self, grid_world):
        # Get the current tick number
        curr_tick = grid_world.current_nr_ticks
        # loop through all zones, check the victims and set the tick if satisfied
        for zone_nr, goal_vics in self.__drop_off.items():
            # Go through all ranks of this drop off zone
            for rank, vic_data in goal_vics.items():
                loc = vic_data[0]  # the location, needed to find victims here
                shape = vic_data[1]  # the desired shape
                tick = vic_data[2]

                # Retrieve all objects, the object ids at the location and obtain all victims from it
                all_objs = grid_world.environment_objects
                obj_ids = grid_world.get_objects_in_range(loc, object_type=EnvObject, sense_range=0)
                vics = [all_objs[obj_id] for obj_id in obj_ids
                          if obj_id in all_objs.keys() and "is_collectable" in all_objs[obj_id].properties.keys()]
                vics = [v for v in vics if v.properties["is_collectable"]]

                # Check if there is a victim, and if so if it is the right one and the tick is not yet set, then set the current tick and increase the score.
                if len(vics) > 0 and vics[0].properties['img_name'][8:-4] == shape and tick is None:
                    self.__drop_off[zone_nr][rank][2] = curr_tick
                    if 'critical' in vics[0].properties['img_name'][8:-4]:
                        self.__score+=6
                    if 'mild' in vics[0].properties['img_name'][8:-4]:
                        self.__score+=3
                # Deduct points from the score when victims are picked up from drop zone
                elif len(vics) == 0:
                    if self.__drop_off[zone_nr][rank][2] != None:
                        self.__drop_off[zone_nr][rank][2] = None
                        if rank in [0,1,2,3]:
                            self.__score-=6
                        if rank in [4,5,6,7]:
                            self.__score-=3

        # Now check if all victims are collected
        is_satisfied = True
        progress = 0
        for zone_nr, goal_vics in self.__drop_off.items():
            zone_satisfied = True
            ticks = [goal_vics[r][2] for r in range(len(goal_vics))]  # list of ticks in rank order
            for tick in ticks:
                if tick is not None:
                    progress += 1
            if None in ticks:
                zone_satisfied = False
            # update our satisfied boolean
            is_satisfied = is_satisfied and zone_satisfied
        agent = grid_world.registered_agents['rescuebot']
        agent.change_property('score',self.__score)

        return is_satisfied, progress