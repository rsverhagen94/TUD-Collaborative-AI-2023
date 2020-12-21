import matrx
import os
from matrx import WorldBuilder
from matrx.agents import AgentBrain
from matrx.agents.agent_utils.state_tracker import StateTracker

def create_builder():
    # Create the builder
    builder = WorldBuilder(shape=[21,20], run_matrx_api=True, run_matrx_visualizer=True,
    visualization_bg_img="", tick_duration=0.1)

    # Add the rooms
    builder.add_room(top_left_location=[3,4], width=5, height=5, name="room_blue", door_locations=[(7,6)])
    builder.add_room(top_left_location=[12,4], width=5, height=5, name="room_red", door_locations=[(12,6)])
    builder.add_room(top_left_location=[3,11], width=5, height=5, name="room_yellow", door_locations=[(7,13)])
    builder.add_room(top_left_location=[12,11], width=5, height=5, name="room_green", door_locations=[(12,13)])

    # Add drop-off zone
    builder.add_multiple_objects(locations=[(16,0),(17,0),(18,0),(19,0),(20,0)], names="drop zone", visualize_colours="#707070", is_movable=False)

    # Add the blocks
    builder.add_object(location=[4,7], name="block_blue", visualize_colour="#0000FF", is_traversable=False, is_movable=True)
    builder.add_object(location=[5,6], name="block_blue", visualize_colour="#0000FF", is_traversable=False, is_movable=True)
    builder.add_object(location=[13,5], name="block_red", visualize_colour="FF0000", is_traversable=False, is_movable=True)
    builder.add_object(location=[15,7], name="block_red", visualize_colour="FF0000", is_traversable=False, is_movable=True)
    builder.add_object(location=[4,12], name="block_yellow", visualize_colour="FFFF00", is_traversable=False, is_movable=True)
    builder.add_object(location=[5,13], name="block_yellow", visualize_colour="FFFF00", is_traversable=False, is_movable=True)
    builder.add_object(location=[15,12], name="block_green", visualize_colour="00FF00", is_traversable=False, is_movable=True)
    builder.add_object(location=[15,13], name="block_green", visualize_colour="00FF00", is_traversable=False, is_movable=True)

    #Add door blocks
    builder.add_multiple_objects(locations=[(8,6),(11,6),(8,13),(11,13)], names=["doormat_blue", "doormat_red", "doormat_yellow", "doormat_green"],
    visualize_colours=["#0000FF", "FF0000", "FFFF00", "#00FF00"], visualize_opacities=0.5)

    # Add autonomous agent
    brain = AgentBrain()
    builder.add_agent(location=[2,2], agent_brain=brain, name="Bot_1", is_traversable=False, img_name="/images/machine.png")
    builder.add_agent(location=[10,17], agent_brain=brain, name="Bot_2", is_traversable=False, img_name="/images/machine.png")

    return builder

if __name__ == "__main__":
    # Call the method that creates the builder
    builder = create_builder()

    # Refer to own media folder instead of MATRX's one
    media_folder = os.path.dirname(os.path.join(os.path.realpath("C:/Users/Rsv19/MATRX"), "media"))

    # Start the MATRX API we need for our visualization
    builder.startup(media_folder=media_folder)
    # Create a world
    world = builder.get_world()
    # Run world
    world.run(api_info=builder.api_info)