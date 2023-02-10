from matrx.logger.logger import GridWorldLogger
from matrx.grid_world import GridWorld

class ActionLogger(GridWorldLogger):
    '''
    Logger for saving the actions of all agents during each tick of the task.
    '''
    def __init__(self, save_path="", file_name_prefix="", file_extension=".csv", delimiter=";"):
        super().__init__(save_path=save_path, file_name=file_name_prefix, file_extension=file_extension, delimiter=delimiter, log_strategy=1)

    def log(self, grid_world, agent_data):
        # Create a dictionary with the log data
        log_data = {}
        # We will log score and completeness of the task
        log_data['score'] = grid_world.simulation_goal.score(grid_world)
        log_data['completeness'] = grid_world.simulation_goal.progress(grid_world)
        # For both human and agent, log their action and location per tick
        for agent_id, agent_body in grid_world.registered_agents.items():
            log_data[agent_id + '_action'] = agent_body.current_action
            log_data[agent_id + '_location'] = agent_body.location
                
        return log_data