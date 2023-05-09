import os, requests
import sys
import csv
import glob
import pathlib

def output_logger(fld):
    recent_dir = max(glob.glob(os.path.join(fld, '*/')), key=os.path.getmtime)
    recent_dir = max(glob.glob(os.path.join(recent_dir, '*/')), key=os.path.getmtime)
    action_file = glob.glob(os.path.join(recent_dir,'world_1/action*'))[0]
    action_header = []
    action_contents=[]
    # Calculate the unique human and agent actions
    unique_agent_actions = []
    unique_human_actions = []
    joint_actions_1 = []
    joint_actions_2 = []
    joint_actions_3 = []
    joint_actions_4 = []
    joint_actions = []

    shelter1 = True
    shelter2 = True
    shelter3 = True
    area_tiles = ['(2, 2)', '(2, 3)', '(3, 2)', '(3, 3)', '(4, 2)', '(4, 3)', '(8, 2)', '(8, 3)', '(9, 2)', '(9, 3)', '(10, 2)', '(10, 3)', '(14, 2)', '(14, 3)', '(15, 2)', '(15, 3)', '(16, 2)', '(16, 3)', '(20, 2)', '(20, 3)',
                '(21, 2)', '(21, 3)', '(22, 2)', '(22, 3)', '(2, 8)', '(2, 9)', '(3, 8)', '(3, 9)', '(4, 8)', '(4, 9)', '(8, 8)', '(8, 9)', '(9, 8)', '(9, 9)', '(10, 8)', '(10, 9)', '(14, 8)', '(14, 9)', '(15, 8)', '(15, 9)',
                '(16, 8)', '(16, 9)', '(2, 14)', '(2, 15)', '(3, 14)', '(3, 15)', '(4, 14)', '(4, 15)', '(8, 14)', '(8, 15)', '(9, 14)', '(9, 15)', '(10, 14)', '(10, 15)', '(14, 14)', '(14, 15)', '(15, 14)', '(15, 15)', '(16, 14)',
                '(16, 15)', '(2, 20)', '(2, 21)', '(3, 20)', '(3, 21)', '(4, 20)', '(4, 21)', '(8, 20)', '(8, 21)', '(9, 20)', '(9, 21)', '(10, 20)', '(10, 21)', '(14, 20)', '(14, 21)', '(15, 20)', '(15, 21)', '(16, 20)', 
                '(16, 21)', '(20, 20)', '(20, 21)', '(21, 20)', '(21, 21)', '(22, 20)', '(22, 21)', '(23, 8)', '(23, 9)', '(23, 10)', '(23, 11)', '(23, 12)', '(23, 13)', '(23, 14)', '(23, 15)']
    with open(action_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar="'")
        for row in reader:
            if action_header==[]:
                action_header=row
                continue
            if row[2:4] not in unique_agent_actions and row[2]!="":
                unique_agent_actions.append(row[2:4])
            if row[4:6] not in unique_human_actions and row[4]!="":
                unique_human_actions.append(row[4:6])
            if row[4] == 'RemoveObjectTogether' or row[4] == 'CarryObjectTogether' or row[4] == 'DropObjectTogether':
                if row[4:6] not in joint_actions:
                    joint_actions.append(row[4:6])
                    if int(row[7]) <= 50:
                        joint_actions_1.append(row[4:6])
                    if 50 < int(row[7]) <= 1850:
                        joint_actions_2.append(row[4:6])
                    if 1850 < int(row[7]) <= 2750:
                        joint_actions_3.append(row[4:6])
                    if 2750 < int(row[7]):
                        joint_actions_4.append(row[4:6])

                if row[4:6] not in unique_agent_actions:
                    unique_agent_actions.append(row[4:6])
            if row[7] == '50' and row[5] not in area_tiles:
                shelter1 = False
            if row[7] == '1850' and row[5] not in area_tiles:
                shelter2 = False
            if row[7] == '2750' and row[5] not in area_tiles:
                shelter3 = False
            res = {action_header[i]: row[i] for i in range(len(action_header))}
            action_contents.append(res)

    # Retrieve the number of ticks to finish the task, score, and completeness
    no_ticks = action_contents[-1]['tick_nr']
    score = action_contents[-1]['score']
    completeness = action_contents[-1]['completeness']
    # Save the output as a csv file
    print("Saving output...")
    with open(os.path.join(recent_dir,'world_1/output.csv'),mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['completeness','score','no_ticks','agent_actions','human_actions','shelter1','shelter2','shelter3', 'joint actions', 'number_joint', 'number_joint_1', 'number_joint_2', 'number_joint_3', 'number_joint_4'])
        csv_writer.writerow([completeness,score,no_ticks,len(unique_agent_actions),len(unique_human_actions),shelter1,shelter2,shelter3, joint_actions, len(joint_actions), len(joint_actions_1), len(joint_actions_2), len(joint_actions_3), len(joint_actions_4)]
