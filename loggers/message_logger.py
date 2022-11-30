from matrx.logger.logger import GridWorldLogger
from matrx.grid_world import GridWorld
import copy
import json
import numpy as np


class MessageLogger(GridWorldLogger):
    """ Logs messages send and received by (all) agents """

    def __init__(self, save_path="", file_name_prefix="", file_extension=".csv", delimiter=";"):
        super().__init__(save_path=save_path, file_name=file_name_prefix, file_extension=file_extension,
                         delimiter=delimiter, log_strategy=1)

    def log(self, grid_world: GridWorld, agent_data: dict):

        log_data = {
            'total_number_messages_human': 0,
            'total_number_messages_agent': 0,
            'average_message_length_human': 0,
            'average_message_length_agent': 0,
            'ignored': 0,
            'trustPhase': 2,
            'trustValue': 0.25,
            'noIgnored': 0,
            'noSuggestions': 0,
            'workload': 0,
            'timePressure':0,
            'taskSeverity':0,
            'taskSwitch':0,
            'occupiedTime':0,
            'complySpeed':-1,
            'performanceGap':-1,
            'responseTime':-1,
            'searchTookLong':-1
        }

        gwmm = grid_world.message_manager
        t = grid_world.current_nr_ticks - 1
        tot_messages_human = 0
        tot_messages_agent = 0

        mssg_len_human = []
        mssg_len_agent = []
        for i in range(0, t):
            if i in gwmm.preprocessed_messages.keys():
                for mssg in gwmm.preprocessed_messages[i]:
                    if 'human' in mssg.from_id and 'score' not in mssg.content and 'ignored' not in mssg.content:
                        tot_messages_human += 1
                        mssg_len_human.append(len(mssg.content.split()))
                    if 'RescueBot' in mssg.from_id and 'score' not in mssg.content and 'ignored' not in mssg.content:
                        tot_messages_agent += 1
                        mssg_len_agent.append(len(mssg.content.split()))
                    if 'RescueBot' in mssg.from_id and 'ignored' in mssg.content:
                        log_data['ignored'] = mssg.content.split()[-1]
                        if "trustAgentVarPresent" in mssg.content:
                            log_data['noSuggestions'] = int(mssg.content.split()[-3])
                            log_data['noIgnored'] = int(mssg.content.split()[-4])
                            log_data['trustPhase'] = int(mssg.content.split()[-5])
                            log_data['trustValue'] = float(mssg.content.split()[-6])
                    if 'RescueBot' in mssg.from_id and 'workload' in mssg.content:
                        log_data['workload'] = mssg.content.split()[-1]
                        log_data['timePressure'] = mssg.content.split()[-2]
                        log_data['taskSeverity'] = mssg.content.split()[-3]
                        log_data['taskSwitch'] = mssg.content.split()[-4]
                        log_data['occupiedTime'] = mssg.content.split()[-5]
                    if 'RescueBot' in mssg.from_id and 'Your performance' in mssg.content:
                        if 'comply speed' in mssg.content:
                            log_data['complySpeed'] = mssg.content.split()[-1]
                        elif 'performance gap' in mssg.content:
                            log_data['performanceGap'] = mssg.content.split()[-1]
                        elif 'response time' in mssg.content:
                            log_data['responseTime'] = mssg.content.split()[-1]
                        elif 'search took long' in mssg.content:
                            log_data['searchTookLong'] = mssg.content.split()[-1]

        log_data['total_number_messages_human'] = tot_messages_human
        log_data['total_number_messages_agent'] = int(tot_messages_agent / 2)

        return log_data