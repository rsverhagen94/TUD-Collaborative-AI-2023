import os, requests
import sys
import csv
import glob
import pathlib
from SaR_gui import visualization_server
from worlds1.worldBuilder import create_builder
from pathlib import Path

if __name__ == "__main__":
    print("\nEnter one of the environments 'trial' or 'experiment':")
    choice1=input()
    if choice1=='trial':
        builder = create_builder(exp_version='trial',condition='tutorial')
    else:
        print("\nEnter one of the human conditions 'normal', 'strong', or 'weak':")
        choice2=input()
        if choice2=='normal' or choice2=='strong' or choice2=='weak':
            builder = create_builder(exp_version=choice1, condition=choice2)
        else:
            print("\nWrong condition name entered")

    # Start overarching MATRX scripts and threads, such as the api and/or visualizer if requested. Here we also link our own media resource folder with MATRX.
    media_folder = pathlib.Path().resolve()
    builder.startup(media_folder=media_folder)
    print("Starting custom visualizer")
    vis_thread = visualization_server.run_matrx_visualizer(verbose=False, media_folder=media_folder)
    world = builder.get_world()
    print("Started world...")
    builder.api_info['matrx_paused'] = False
    world.run(builder.api_info)
    print("DONE!")
    print("Shutting down custom visualizer")
    r = requests.get("http://localhost:" + str(visualization_server.port) + "/shutdown_visualizer")
    vis_thread.join()

    if choice1=="experiment":
        fld = os.getcwd()
        recent_dir = max(glob.glob(os.path.join(fld, '*/')), key=os.path.getmtime)
        recent_dir = max(glob.glob(os.path.join(recent_dir, '*/')), key=os.path.getmtime)
        action_file = glob.glob(os.path.join(recent_dir,'world_1/action*'))[0]
        message_file = glob.glob(os.path.join(recent_dir,'world_1/message*'))[0]
        action_header = []
        action_contents=[]
        message_header = []
        message_contents=[]
        unique_agent_moves = []
        unique_human_moves = []
        human_moves = []
        idle_agent = 0
        idle_human = 0
        idle_together = 0
        with open(action_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar="'")
            for row in reader:
                if action_header==[]:
                    action_header=row
                    continue
                if row[2:4] not in unique_agent_moves:
                    unique_agent_moves.append(row[2:4])
                if row[4:6] not in unique_human_moves:
                    unique_human_moves.append(row[4:6])
                res = {action_header[i]: row[i] for i in range(len(action_header))}
                action_contents.append(res)
        
        with open(message_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar="'")
            for row in reader:
                if message_header==[]:
                    message_header=row
                    continue
                res = {message_header[i]: row[i] for i in range(len(message_header))}
                message_contents.append(res)

        no_messages_human = message_contents[-1]['total_number_messages_human']
        no_messages_agent = message_contents[-1]['total_number_messages_agent']
        no_ticks = action_contents[-1]['tick_nr']
        score = action_contents[-1]['score']
        completeness = action_contents[-1]['completeness']

        print("Saving output...")
        with open(os.path.join(recent_dir,'world_1/output.csv'),mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['completeness','score','no_ticks','agent_moves','human_moves','agent_messages','human_messages'])
            csv_writer.writerow([completeness,score,no_ticks,len(unique_agent_moves),len(unique_human_moves),no_messages_agent,no_messages_human])

    builder.stop()
