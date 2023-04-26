import os, requests
import sys
import csv
import glob
import pathlib
from SaR_gui import visualization_server
from worlds1.WorldBuilder import create_builder
from pathlib import Path
from loggers.OutputLogger import output_logger

if __name__ == "__main__":
    fld = os.getcwd()
    print("\nEnter one of the task types 'tutorial' or 'official':")
    choice1=input()
    if choice1=='tutorial':
        builder = create_builder(task_type='tutorial', condition='tutorial')
    else:
        # ADD QUESTION ON CONDITION HERE
        builder = create_builder(task_type='official', condition='baseline')

    # Start overarching MATRX scripts and threads, such as the api and/or visualizer if requested. Here we also link our own media resource folder with MATRX.
    media_folder = pathlib.Path().resolve()
    builder.startup(media_folder=media_folder)
    print("Starting custom visualizer")
    vis_thread = visualization_server.run_matrx_visualizer(verbose=False, media_folder=media_folder)
    world = builder.get_world()
    print("Started world...")
    #builder.api_info['matrx_paused'] = False
    world.run(builder.api_info)
    print("DONE!")
    print("Shutting down custom visualizer")
    r = requests.get("http://localhost:" + str(visualization_server.port) + "/shutdown_visualizer")
    vis_thread.join()
    if choice1=="official":
        # Generate one final output log file for the official task type
        output_logger(fld)
    builder.stop()
