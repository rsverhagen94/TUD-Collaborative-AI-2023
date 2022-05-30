from distutils.command.build import build
import os, requests
import sys
import csv
import glob
import pathlib
from SaR_gui import visualization_server
from worlds1.worldBuilder import create_builder
from typing import final, List, Dict, Final
from pathlib import Path

if __name__ == "__main__":
    print("\nEnter one of the environments 'trial' or 'experiment':")
    choice1=input()
    if choice1=='trial':
        builder = create_builder(exp_version='trial',condition='tutorial')
    else:
        print("\nEnter one of the robot adaptation styles 'baseline', 'trust', 'workload', or 'performance':")
        choice2=input()

        #PAY ATTENTION
        if choice2=='trust' or choice2=='workload' or choice2=='performance':
            print("\nMake sure to add your agents to the agents folder, starting baseline now..")
            print()
            print()
            print()
            builder = create_builder(exp_version=choice1,condition="baseline")
        else:
            builder = create_builder(exp_version=choice1,condition=choice2)

    # Start overarching MATRX scripts and threads, such as the api and/or visualizer if requested. Here we also link our
    # own media resource folder with MATRX.
    media_folder = pathlib.Path().resolve()
    #media_folder = os.path.dirname(os.path.join(os.path.realpath("/home/ruben/Documents/MATRX/MATRX"), "media"))
    builder.startup(media_folder=media_folder)
    print("Starting custom visualizer")
    vis_thread = visualization_server.run_matrx_visualizer(verbose=False, media_folder=media_folder)
    world = builder.get_world()
    print("Started world...")
    #for world in builder.worlds():
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
                if row[2] == "" and row[4]!='RemoveObjectTogether':
                    idle_agent+=1
                if row[4] == "":
                    idle_human+=1
                if row[2] == "" and row[4]=="":
                    idle_together+=1
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
        ignored = message_contents[-1]['ignored']

        print("Saving output...")
        with open(os.path.join(recent_dir,'world_1/output.csv'),mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['completeness','score','no_ticks','ignored_suggestions','agent_moves','human_moves','agent_messages','human_messages','agent_idle','human_idle','simultaneous_idle'])
            csv_writer.writerow([completeness,score,no_ticks,ignored,len(unique_agent_moves),len(unique_human_moves),no_messages_agent,no_messages_human,idle_agent,idle_human,idle_together])

    builder.stop()
