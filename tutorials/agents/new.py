from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker

def initialize(self):
    self.state_tracker = StateTracker(agent_id=self.agent_id)

    self.navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set,
                               algorithm=Navigator.A_STAR_ALGORITHM)
    
    self.goal_cycle = ["find_room", "open_door", "search_room", "grab_block", "to_dropoff", "drop_block", "done"]

    self.block_orders = ['yellow', 'green', 'blue', 'green', 'red']


def filter_observations(self, state):
    """
    Filtering the agent's observations.
    :param state:
    :return:
    """
    new_state = state.copy()
    closed_room_colors = []

    for k, obj in state.items():
        if 'door@' in k and obj.get('is_open') is False:
            color = k.split('_', 1)[0]
            closed_room_colors.append(color)
    for k, obj in state.items():
        for color in closed_room_colors:
            if (color in k) and ('doormat' not in k) and ('block' in k):
                new_state.pop(k)

    self.state_tracker.update(new_state)
    return new_state

def decide_on_action(self, state):
    # Determine current goal for this agent
    global cycle
    self.current_goal = self.goal_cycle[0]

    # Determine which block color is needed
    if len(self.block_orders) > 0:
        current_order = self.block_orders[0]
    else:
        return StandStill.__name__, {}

    # Gather the id of this agent and the other
    this_agent = self.agent_id
    for k, obj in state.items():
        if 'Bot' in k and this_agent not in k:
            other_agent = k

    # From all objects, gather the objects that are doors and save them 
    # Approach can be taken for any object
    objects = list(state.keys())
    doors = [obj for obj in objects
             if 'class_inheritance' in state[obj] and state[obj]['class_inheritance'][0] == "Door"]
    door_locations = []
    door_ids = []
    for door in doors:
        door_ids.append(door)
        door_location = state[door]['location']
        door_locations.append(door_location)

    return StandStill.__name__, {}

    # First, check if the other agent(s) have sent messages. (See GitHub for method implementation)
    self.check_for_update(current_order)

    # Navigate to a room
    if self.current_goal == "find_room":
        self.navigator.reset_full()
        # Setting location that is in front of a door
        for doormat in doormats:
            doormat_id = doormats[doormat]['doormat_id']
            if current_order in doormat_id:
                doormat_waypoint = doormats[doormat]['location']
                self.navigator.add_waypoint(doormat_waypoint)
                move_action = self.navigator.get_move_action(self.state_tracker)

                current_waypoint = doormat_waypoint
                # Update other agent(s) that this order is already executed
                if self.agent_properties['location'] == current_waypoint:
                    self.send_message(message_content={"id": doormat_id},
                                      to_id=other_agent)
                    self.goal_cycle.pop(0)
                    
    # Return the action from the Navigator
    return move_action, {}
    