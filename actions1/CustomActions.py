import numpy as np
from matrx.actions.action import Action, ActionResult
from matrx.objects.agent_body import AgentBody
from matrx.objects.standard_objects import AreaTile
from matrx.actions.object_actions import _is_drop_poss, _act_drop, _possible_drop, _find_drop_loc, GrabObject, GrabObjectResult, RemoveObject, RemoveObjectResult, DropObject
from matrx.objects import EnvObject
from matrx.utils import get_distance
import random

class AddObject(Action):
    """ An action that can add a product to the gridworld """

    def __init__(self, duration_in_ticks=0):
        super().__init__(duration_in_ticks)

    def is_possible(self, grid_world, agent_id, **kwargs):
        return AddObjectResult(AddObjectResult.ACTION_SUCCEEDED, True)

    def mutate(self, grid_world, agent_id, **kwargs):
        for i in range(len(kwargs['add_objects'])):
            obj_body_args = {
                "location": kwargs['add_objects'][i]['location'],
                "name": "water",
                "class_callable": EnvObject,
                "is_traversable": True,
                "is_movable": False,
                "visualize_size": 1,
                "img_name": kwargs['add_objects'][i]['img_name']
            }
        
            env_object = EnvObject(**obj_body_args)
            grid_world._register_env_object(env_object)


        return AddObjectResult(AddObjectResult.ACTION_SUCCEEDED, True)

class Idle(Action):
    """ Let's an agent be idle for a specified number of ticks."""
    def __init__(self, duration_in_ticks=1):
        super().__init__(duration_in_ticks)

    def is_possible(self, grid_world, agent_id, **kwargs):
        return IdleResult(IdleResult.RESULT_SUCCESS, True)

class RemoveObjectTogether(Action):
    """ Removes an object from the world"""

    def __init__(self, duration_in_ticks=0):
        super().__init__(duration_in_ticks)

    def mutate(self, grid_world, agent_id, world_state, **kwargs):
        """ Removes the specified object"""
        assert 'object_id' in kwargs.keys()  # assert if object_id is given.
        object_id = kwargs['object_id']  # assign
        remove_range = 1  # default remove range
        other_agent = world_state[{"name": "RescueBot"}]
        other_human = world_state[{"name": "human"}]
        condition = None if 'condition' not in kwargs else kwargs['condition']
        if 'remove_range' in kwargs.keys():  # if remove range is present
            assert isinstance(kwargs['remove_range'], int)  # should be of integer
            assert kwargs['remove_range'] >= 0  # should be equal or larger than 0
            remove_range = kwargs['remove_range']  # assign

        # get the current agent (exists, otherwise the is_possible failed)
        agent_avatar = grid_world.registered_agents[agent_id]
        agent_loc = agent_avatar.location  # current location

        # Get all objects in the remove_range
        objects_in_range = grid_world.get_objects_in_range(agent_loc, object_type="*", sense_range=remove_range)

        # You can't remove yourself
        objects_in_range.pop(agent_id)
        for obj in objects_in_range:  # loop through all objects in range
            # CURRENTLY FOR ROCK OR STONE BUT CAN BE EDITED
            if obj == object_id and get_distance(other_agent['location'], world_state[obj]['location'])<=remove_range and get_distance(other_human['location'], world_state[obj]['location'])<=remove_range and 'rock' in obj and condition!='baseline' or \
            obj == object_id and get_distance(other_agent['location'], world_state[obj]['location'])<=remove_range and get_distance(other_human['location'], world_state[obj]['location'])<=remove_range and 'stone' in obj and condition!='baseline':  # if object is in that list
                success = grid_world.remove_from_grid(object_id)  # remove it, success is whether GridWorld succeeded
                if success:  # if we succeeded in removal return the appropriate ActionResult
                    return RemoveObjectResult(RemoveObjectResult.OBJECT_REMOVED.replace('object_id'.upper(),
                                                                                        str(object_id)), True)
                else:  # else we return a failure due to the GridWorld removal failed
                    return RemoveObjectResult(RemoveObjectResult.REMOVAL_FAILED.replace('object_id'.upper(),
                                                                                        str(object_id)), False)

        # If the object was not in range, or no objects were in range we return that the object id was not in range
        return RemoveObjectResult(RemoveObjectResult.OBJECT_ID_NOT_WITHIN_RANGE
                                  .replace('remove_range'.upper(), str(remove_range))
                                  .replace('object_id'.upper(), str(object_id)), False)

    def is_possible(self, grid_world, agent_id, **kwargs):
        """ Checks if an object can be removed"""
        agent_avatar = grid_world.get_env_object(agent_id, obj_type=AgentBody)  # get ourselves
        assert agent_avatar is not None  # check if we actually exist
        agent_loc = agent_avatar.location  # get our location

        remove_range = np.inf  # we do not know the intended range, so assume infinite
        # get all objects within infinite range
        objects_in_range = grid_world.get_objects_in_range(agent_loc, object_type="*", sense_range=remove_range)

        # You can't remove yourself
        objects_in_range.pop(agent_avatar.obj_id)

        if len(objects_in_range) == 0:  # if there are no objects in infinite range besides ourselves, we return fail
            return RemoveObjectResult(RemoveObjectResult.NO_OBJECTS_IN_RANGE.replace('remove_range'.upper(),
                                                                                     str(remove_range)), False)
        # need an object id to remove an object
        if 'object_id' not in kwargs:
            return RemoveObjectResult(RemoveObjectResult.REMOVAL_FAILED.replace('object_id'.upper(),
                                                                                str(None)), False)
        # check if the object is actually within removal range
        object_id = kwargs['object_id']
        if object_id not in objects_in_range:
            return RemoveObjectResult(RemoveObjectResult.REMOVAL_FAILED.replace('object_id'.upper(),
                                                                                str(object_id)), False)

        # otherwise some instance of RemoveObject is possible, although we do not know yet IF the intended removal is
        # possible.
        return RemoveObjectResult(RemoveObjectResult.ACTION_SUCCEEDED, True)

class CarryObject(Action):
    """ Grab and hold objects."""

    def __init__(self, duration_in_ticks=0):
        super().__init__(duration_in_ticks)

    def is_possible(self, grid_world, agent_id, world_state, **kwargs):
        """ Checks if the object can be grabbed."""
        # Set default values check
        object_id = None if 'object_id' not in kwargs else kwargs['object_id']
        grab_range = np.inf if 'grab_range' not in kwargs else kwargs['grab_range']
        max_objects = np.inf if 'max_objects' not in kwargs else kwargs['max_objects']
        condition = None if 'condition' not in kwargs else kwargs['condition']
        # EDIT BELOW TO ACCOUNT FOR YOUR CONDITION
        if object_id and 'critical' in object_id and condition!='baseline':
            return GrabObjectResult(GrabObjectResult.RESULT_OBJECT_UNMOVABLE, False)
        if object_id and 'stone' in object_id and condition!='baseline' or object_id and 'rock' in object_id and condition!='baseline' or object_id and 'tree' in object_id and condition!='baseline':
            return GrabObjectResult(GrabObjectResult.RESULT_OBJECT_UNMOVABLE, False)
        else:
            return _is_possible_grab(grid_world, agent_id=agent_id, object_id=object_id, grab_range=grab_range,
                                    max_objects=max_objects) 

    def mutate(self, grid_world, agent_id, world_state, **kwargs):
        """ Grabs an object."""

        # Additional check
        assert 'object_id' in kwargs.keys()
        assert 'grab_range' in kwargs.keys()
        assert 'max_objects' in kwargs.keys()

        # if possible:
        object_id = kwargs['object_id']  # assign

        # Loading properties
        reg_ag = grid_world.registered_agents[agent_id]  # Registered Agent
        env_obj = grid_world.environment_objects[object_id]  # Environment object

        # Updating properties
        env_obj.carried_by.append(agent_id)
        reg_ag.is_carrying.append(env_obj)  # we add the entire object!

        if 'critical' in object_id and 'human' in agent_id:
            reg_ag.change_property("img_name", "/images/carry-critical-human.svg")
        if 'healthy' in object_id and 'human' in agent_id:
            reg_ag.change_property("img_name", "/images/carry-healthy-human.svg")
        if 'mild' in object_id and 'human' in agent_id:
            reg_ag.change_property("img_name", "/images/carry-mild-human.svg")
        if 'critical' in object_id and 'bot' in agent_id:
            reg_ag.change_property("img_name", "/images/carry-critical-robot.svg")
        if 'mild' in object_id and 'bot' in agent_id:
            reg_ag.change_property("img_name", "/images/carry-mild-robot.svg")

        # Remove it from the grid world (it is now stored in the is_carrying list of the AgentAvatar
        succeeded = grid_world.remove_from_grid(object_id=env_obj.obj_id, remove_from_carrier=False)
        if not succeeded:
            return GrabObjectResult(GrabObjectResult.FAILED_TO_REMOVE_OBJECT_FROM_WORLD.replace("{OBJECT_ID}",
                                                                                                env_obj.obj_id), False)

        # Updating Location (done after removing from grid, or the grid will search the object on the wrong location)
        env_obj.location = reg_ag.location

        return GrabObjectResult(GrabObjectResult.RESULT_SUCCESS, True)

class Drop(Action):
    """ Drops a carried object."""
    
    def __init__(self, duration_in_ticks=0):
        super().__init__(duration_in_ticks)

    def is_possible(self, grid_world, agent_id, world_state, **kwargs):
        """ Checks if the object can be dropped."""
        reg_ag = grid_world.registered_agents[agent_id]
        drop_range = 1 if 'drop_range' not in kwargs else kwargs['drop_range']
        condition = None if 'condition' not in kwargs else kwargs['condition']
        other_human = world_state[{"name": "human"}]
        other_agent_id = world_state[{"name": "RescueBot"}]['obj_id']
        other_agent = grid_world.registered_agents[other_agent_id]

        # If no object id is given, the last item is dropped
        if 'object_id' in kwargs:
            obj_id = kwargs['object_id']
        elif len(reg_ag.is_carrying) > 0:
            obj_id = reg_ag.is_carrying[-1].obj_id
        else:
            return DropObjectResult(DropObjectResult.RESULT_NO_OBJECT, False)

        # EDIT BELOW TO ACCOUNT FOR YOUR CONDITION
        if 'critical' in obj_id and condition!='baseline' or 'mild' in obj_id and other_agent.properties['visualization']['opacity']==0 and condition!='baseline':
            return DropObjectResult(DropObjectResult.RESULT_UNKNOWN_OBJECT_TYPE, False)            
        else:
            return _possible_drop(grid_world, agent_id=agent_id, obj_id=obj_id, drop_range=drop_range)

    def mutate(self, grid_world, agent_id, world_state, **kwargs):
        """ Drops the carried object."""
        reg_ag = grid_world.registered_agents[agent_id]
        if 'human' in agent_id and len(reg_ag.is_carrying)<2:
            reg_ag.change_property("img_name", "/images/rescue-man-final3.svg")
        if 'bot' in agent_id:
            reg_ag.change_property("img_name", "/images/robot-final4.svg")

        # fetch range from kwargs
        drop_range = 1 if 'drop_range' not in kwargs else kwargs['drop_range']

        # If no object id is given, the last item is dropped
        if 'object_id' in kwargs:
            obj_id = kwargs['object_id']
            env_obj = [obj for obj in reg_ag.is_carrying if obj.obj_id == obj_id][0]
        elif len(reg_ag.is_carrying) > 0:
            env_obj = reg_ag.is_carrying[-1]
        else:
            return DropObjectResult(DropObjectResult.RESULT_NO_OBJECT_CARRIED, False)

        # check that it is even possible to drop this object somewhere
        if not env_obj.is_traversable and not reg_ag.is_traversable and drop_range == 0:
            raise Exception(
                f"Intraversable agent {reg_ag.obj_id} can only drop the intraversable object {env_obj.obj_id} at its "
                f"own location (drop_range = 0), but this is impossible. Enlarge the drop_range for the DropAction to "
                f"atleast 1")

        # check if we can drop it at our current location
        curr_loc_drop_poss = _is_drop_poss(grid_world, env_obj, reg_ag.location, agent_id)

        # drop it on the agent location if possible
        if curr_loc_drop_poss:
            return _act_drop(grid_world, agent=reg_ag, env_obj=env_obj, drop_loc=reg_ag.location)

        # if the agent location was the only within range, return a negative action result
        elif not curr_loc_drop_poss and drop_range == 0:
            return DropObjectResult(DropObjectResult.RESULT_OBJECT, False)

        # Try finding other drop locations from close to further away around the agent
        drop_loc = _find_drop_loc(grid_world, reg_ag, env_obj, drop_range, reg_ag.location)

        # If we didn't find a valid drop location within range, return a negative action result
        if not drop_loc:
            return DropObjectResult(DropObjectResult.RESULT_OBJECT, False)

        return _act_drop(grid_world, agent=reg_ag, env_obj=env_obj, drop_loc=drop_loc)

class CarryObjectTogether(Action):
    """ Carries objects together."""

    def __init__(self, duration_in_ticks=0):
        super().__init__(duration_in_ticks)

    def is_possible(self, grid_world, agent_id, world_state, **kwargs):
        """ Checks if the object can be grabbed."""
        # Set default values check
        object_id = None if 'object_id' not in kwargs else kwargs['object_id']
        grab_range = np.inf if 'grab_range' not in kwargs else kwargs['grab_range']
        max_objects = np.inf if 'max_objects' not in kwargs else kwargs['max_objects']
        other_agent = world_state[{"name": "RescueBot"}]
        condition = None if 'condition' not in kwargs else kwargs['condition']

        # EDIT BELOW TO ACCOUNT FOR YOUR CONDITION
        if object_id and get_distance(other_agent['location'], world_state[object_id]['location']) > grab_range or condition=='baseline':
            return GrabObjectResult(GrabObjectResult.NOT_IN_RANGE, False)
        else:
            return _is_possible_grab(grid_world, agent_id=agent_id, object_id=object_id, grab_range=grab_range,
                                 max_objects=max_objects)

    def mutate(self, grid_world, agent_id, world_state, **kwargs):
        """ Grabs an object."""

        # Additional check
        assert 'object_id' in kwargs.keys()
        assert 'grab_range' in kwargs.keys()
        assert 'max_objects' in kwargs.keys()

        # if possible:
        object_id = kwargs['object_id']  # assign

        # Loading properties
        reg_ag = grid_world.registered_agents[agent_id]  # Registered Agent
        env_obj = grid_world.environment_objects[object_id]  # Environment object

        # Updating properties
        env_obj.carried_by.append(agent_id)
        reg_ag.is_carrying.append(env_obj)  # we add the entire object!

        other_agent_id = world_state[{"name": "RescueBot"}]['obj_id']

        # if we want to change objects, we need to change the grid_world object 
        other_agent = grid_world.registered_agents[other_agent_id]
        agent = grid_world.registered_agents[agent_id]

        # make the other agent invisible 
        other_agent.change_property("visualize_opacity", 0)

        # change our image 
        
        object_id = None if 'object_id' not in kwargs else kwargs['object_id']
        if 'critical' in object_id and 'human' in agent_id:
            # change our image 
            agent.change_property("img_name", "/images/carry-critical-final.svg")
        if 'mild' in object_id and 'human' in agent_id:
            agent.change_property("img_name", "/images/carry-mild-final.svg")

        # Remove it from the grid world (it is now stored in the is_carrying list of the AgentAvatar
        succeeded = grid_world.remove_from_grid(object_id=env_obj.obj_id, remove_from_carrier=False)
        if not succeeded:
            return GrabObjectResult(GrabObjectResult.FAILED_TO_REMOVE_OBJECT_FROM_WORLD.replace("{OBJECT_ID}",
                                                                                                env_obj.obj_id), False)

        # Updating Location (done after removing from grid, or the grid will search the object on the wrong location)
        env_obj.location = reg_ag.location

        return GrabObjectResult(GrabObjectResult.RESULT_SUCCESS, True)

class DropObjectTogether(Action):
    """ Drops a carried object."""
    
    def __init__(self, duration_in_ticks=0):
        super().__init__(duration_in_ticks)

    def is_possible(self, grid_world, agent_id, world_state, **kwargs):
        """ Checks if the object can be dropped."""
        reg_ag = grid_world.registered_agents[agent_id]
        drop_range = 1 if 'drop_range' not in kwargs else kwargs['drop_range']
        other_agent_id = world_state[{"name": "RescueBot"}]['obj_id']
        other_agent = grid_world.registered_agents[other_agent_id]
        condition = None if 'condition' not in kwargs else kwargs['condition']
        # If no object id is given, the last item is dropped
        if 'object_id' in kwargs:
            obj_id = kwargs['object_id']
        elif len(reg_ag.is_carrying) > 0:
            obj_id = reg_ag.is_carrying[-1].obj_id
        else:
            return DropObjectResult(DropObjectResult.RESULT_NO_OBJECT, False)

        # EDIT BELOW TO ACCOUNT FOR YOUR CONDITION
        if 'healthy' in obj_id and other_agent.properties['visualization']['opacity']!=0 or 'mild' in obj_id and other_agent.properties['visualization']['opacity']!=0 or 'critical' in obj_id and other_agent.properties['visualization']['opacity']!=0:
            return DropObjectResult(DropObjectResult.RESULT_UNKNOWN_OBJECT_TYPE, False)            
        else:
            return _possible_drop(grid_world, agent_id=agent_id, obj_id=obj_id, drop_range=drop_range)

    def mutate(self, grid_world, agent_id, world_state, **kwargs):
        """ Drops the carried object."""
        reg_ag = grid_world.registered_agents[agent_id]
        other_agent_id = world_state[{"name": "RescueBot"}]['obj_id']
        other_agent = grid_world.registered_agents[other_agent_id]
        agent = grid_world.registered_agents[agent_id]
        # fetch range from kwargs
        drop_range = 1 if 'drop_range' not in kwargs else kwargs['drop_range']

        # If no object id is given, the last item is dropped
        if 'object_id' in kwargs:
            obj_id = kwargs['object_id']
            env_obj = [obj for obj in reg_ag.is_carrying if obj.obj_id == obj_id][0]
        elif len(reg_ag.is_carrying) > 0:
            env_obj = reg_ag.is_carrying[-1]
        else:
            return DropObjectResult(DropObjectResult.RESULT_NO_OBJECT_CARRIED, False)
        
        other_agent.change_property("location", agent.properties['location'])

        # make the other agent visible again 
        other_agent.change_property("visualize_opacity", 1)

        # change the agent image back to default 
        agent.change_property("img_name", "/images/rescue-man-final3.svg")

        # check that it is even possible to drop this object somewhere
        if not env_obj.is_traversable and not reg_ag.is_traversable and drop_range == 0:
            raise Exception(
                f"Intraversable agent {reg_ag.obj_id} can only drop the intraversable object {env_obj.obj_id} at its "
                f"own location (drop_range = 0), but this is impossible. Enlarge the drop_range for the DropAction to "
                f"atleast 1")

        # check if we can drop it at our current location
        curr_loc_drop_poss = _is_drop_poss(grid_world, env_obj, reg_ag.location, agent_id)

        # drop it on the agent location if possible
        if curr_loc_drop_poss:
            return _act_drop(grid_world, agent=reg_ag, env_obj=env_obj, drop_loc=reg_ag.location)

        # if the agent location was the only within range, return a negative action result
        elif not curr_loc_drop_poss and drop_range == 0:
            return DropObjectResult(DropObjectResult.RESULT_OBJECT, False)

        # Try finding other drop locations from close to further away around the agent
        drop_loc = _find_drop_loc(grid_world, reg_ag, env_obj, drop_range, reg_ag.location)

        # If we didn't find a valid drop location within range, return a negative action result
        if not drop_loc:
            return DropObjectResult(DropObjectResult.RESULT_OBJECT, False)

        return _act_drop(grid_world, agent=reg_ag, env_obj=env_obj, drop_loc=drop_loc)

#------------------------------------------------------------------------------------------------------------------------#
class AddObjectResult(ActionResult):
    """ Result when assignment failed """
    # failed
    NO_AGENTBRAIN = "No object passed under the `agentbrain` key in kwargs"
    NO_AGENTBODY = "No object passed under the `agentbody` key in kwargs"
    # success
    ACTION_SUCCEEDED = "Object was succesfully added to the gridworld."

    def __init__(self, result, succeeded):
        super().__init__(result, succeeded)

class IdleResult(ActionResult):
    RESULT_SUCCESS = 'Idling action successful'
    RESULT_FAILED = 'Failed to idle'

    def __init__(self, result, succeeded):
        super().__init__(result, succeeded)

class RemoveObjectResult(ActionResult):
    """ActionResult for a RemoveObjectAction"""

    """ Result when the specified object is successfully removed. """
    OBJECT_REMOVED = "The object with id `OBJECT_ID` is removed."

    """ Result when no objects were within the specified range. """
    NO_OBJECTS_IN_RANGE = "No objects were in `REMOVE_RANGE`."

    """ Result when the specified object is not within the specified range. """
    OBJECT_ID_NOT_WITHIN_RANGE = "The object with id `OBJECT_ID` is not within the range of `REMOVE_RANGE`."

    """ Result when the world could not remove the object for some reason. """
    REMOVAL_FAILED = "The object with id `OBJECT_ID` failed to be removed by the environment for some reason."

    def __init__(self, result, succeeded):
        super().__init__(result, succeeded)

class GrabObjectResult(ActionResult):
    """ActionResult for a GrabObjectAction"""

    """ Result when the object can be successfully grabbed. """
    RESULT_SUCCESS = 'Grab action success'

    """ Result when the grabbed object cannot be removed from the 
    :class:`matrx.grid_world.GridWorld`. """
    FAILED_TO_REMOVE_OBJECT_FROM_WORLD = 'Grab action failed; could not remove object with id {OBJECT_ID} from grid.'

    """ Result when the specified object is not within range. """
    NOT_IN_RANGE = 'Object not in range'

    """ Result when the specified object is an agent. """
    RESULT_AGENT = 'This is an agent, cannot be picked up'

    """ Result when no object was specified. """
    RESULT_NO_OBJECT = 'No Object specified'

    """ Result when the agent is at its maximum carrying capacity. """
    RESULT_CARRIES_OBJECT = 'Agent already carries the maximum amount of objects'

    """ Result when the specified object is already carried by another agent. 
    """
    RESULT_OBJECT_CARRIED = 'Object is already carried by {AGENT_ID}'

    """ Result when the specified object does not exist in the 
    :class:`matrx.grid_world.GridWorld` """
    RESULT_UNKNOWN_OBJECT_TYPE = 'obj_id is no Agent and no Object, unknown what to do'

    """ Result when the specified object is not movable. """
    RESULT_OBJECT_UNMOVABLE = 'Object is not movable'

    def __init__(self, result, succeeded):
        super().__init__(result, succeeded)

class DropObjectResult(ActionResult):
    """ ActionResult for a DropObjectAction."""

    """ Result when dropping the object succeeded. """
    RESULT_SUCCESS = 'Drop action success'

    """ Result when there is not object in the agent's inventory. """
    RESULT_NO_OBJECT = 'The item is not carried'

    """ Result when the specified object is not in the agent's inventory. """
    RESULT_NONE_GIVEN = "'None' used as input id"

    """ Result when the specified object should be dropped on an agent. """
    RESULT_AGENT = 'Cannot drop item on an agent'

    """ Result when the specified object should be dropped on an intraversable 
    object."""
    RESULT_OBJECT = 'Cannot drop item on another intraversable object'

    """ Result when the specified object does not exist (anymore). """
    RESULT_UNKNOWN_OBJECT_TYPE = 'Cannot drop item on an unknown object'

    """ Result when the agent is not carrying anything. """
    RESULT_NO_OBJECT_CARRIED = 'Cannot drop object when none carried'

    def __init__(self, result, succeeded, obj_id=None):
        super().__init__(result, succeeded)
        self.obj_id = obj_id


def _is_possible_grab(grid_world, agent_id, object_id, grab_range, max_objects):
    """ Private MATRX method."""

    reg_ag = grid_world.registered_agents[agent_id]  # Registered Agent
    loc_agent = reg_ag.location  # Agent location

    if object_id is None:
        return GrabObjectResult(GrabObjectResult.RESULT_NO_OBJECT, False)

    # Already carries an object
    if len(reg_ag.is_carrying) >= max_objects:
        return GrabObjectResult(GrabObjectResult.RESULT_CARRIES_OBJECT, False)

    # Go through all objects at the desired locations
    objects_in_range = grid_world.get_objects_in_range(loc_agent, object_type="*", sense_range=grab_range)
    objects_in_range.pop(agent_id)

    # Set random object in range
    if not object_id:
        # Remove all non objects from the list
        for obj in list(objects_in_range.keys()):
            if obj not in grid_world.environment_objects.keys():
                objects_in_range.pop(obj)

        # Select a random object
        if objects_in_range:
            object_id = grid_world.rnd_gen.choice(list(objects_in_range.keys()))
        else:
            return GrabObjectResult(GrabObjectResult.NOT_IN_RANGE, False)

    # Check if object is in range
    if object_id not in objects_in_range:
        return GrabObjectResult(GrabObjectResult.NOT_IN_RANGE, False)

    # Check if object_id is the id of an agent
    if object_id in grid_world.registered_agents.keys():
        # If it is an agent at that location, grabbing is not possible
        return GrabObjectResult(GrabObjectResult.RESULT_AGENT, False)

    # Check if it is an object
    if object_id in grid_world.environment_objects.keys():
        env_obj = grid_world.environment_objects[object_id]  # Environment object
        # Check if the object is not carried by another agent
        if len(env_obj.carried_by) != 0:
            return GrabObjectResult(GrabObjectResult.RESULT_OBJECT_CARRIED.replace("{AGENT_ID}",
                                                                                   str(env_obj.carried_by)),
                                    False)
        elif not env_obj.properties["is_movable"]:
            return GrabObjectResult(GrabObjectResult.RESULT_OBJECT_UNMOVABLE, False)
        else:
            # Success
            return GrabObjectResult(GrabObjectResult.RESULT_SUCCESS, True)
    else:
        return GrabObjectResult(GrabObjectResult.RESULT_UNKNOWN_OBJECT_TYPE, False)
    
class GrabObjectResult(ActionResult):
    """ActionResult for a GrabObjectAction"""

    """ Result when the object can be successfully grabbed. """
    RESULT_SUCCESS = 'Grab action success'

    """ Result when the grabbed object cannot be removed from the 
    :class:`matrx.grid_world.GridWorld`. """
    FAILED_TO_REMOVE_OBJECT_FROM_WORLD = 'Grab action failed; could not remove object with id {OBJECT_ID} from grid.'

    """ Result when the specified object is not within range. """
    NOT_IN_RANGE = 'Object not in range'

    """ Result when the specified object is an agent. """
    RESULT_AGENT = 'This is an agent, cannot be picked up'

    """ Result when no object was specified. """
    RESULT_NO_OBJECT = 'No Object specified'

    """ Result when the agent is at its maximum carrying capacity. """
    RESULT_CARRIES_OBJECT = 'Agent already carries the maximum amount of objects'

    """ Result when the specified object is already carried by another agent. 
    """
    RESULT_OBJECT_CARRIED = 'Object is already carried by {AGENT_ID}'

    """ Result when the specified object does not exist in the 
    :class:`matrx.grid_world.GridWorld` """
    RESULT_UNKNOWN_OBJECT_TYPE = 'obj_id is no Agent and no Object, unknown what to do'

    """ Result when the specified object is not movable. """
    RESULT_OBJECT_UNMOVABLE = 'Object is not movable'

    def __init__(self, result, succeeded):
        super().__init__(result, succeeded)


def _act_drop(grid_world, agent, env_obj, drop_loc):
    """ Private MATRX method."""

    # Updating properties
    agent.is_carrying.remove(env_obj)
    env_obj.carried_by.remove(agent.obj_id)

    # We return the object to the grid location we are standing at without registering a new ID
    env_obj.location = drop_loc
    grid_world._register_env_object(env_obj, ensure_unique_id=False)

    return DropObjectResult(DropObjectResult.RESULT_SUCCESS, True)


def _is_drop_poss(grid_world, env_obj, drop_location, agent_id):
    """ Private MATRX method."""

    # Count the intraversable objects at the current location if we would drop the
    # object here
    objs_at_loc = grid_world.get_objects_in_range(drop_location, object_type="*", sense_range=0)

    # Remove area objects from the list
    for key in list(objs_at_loc.keys()):
        if AreaTile.__name__ in objs_at_loc[key].class_inheritance:
            objs_at_loc.pop(key)

    # Remove the agent who drops the object from the list (an agent can always drop the
    # traversable object its carrying at its feet, even if the agent is intraversable)
    if agent_id in objs_at_loc.keys():
        objs_at_loc.pop(agent_id)

    in_trav_objs_count = 1 if not env_obj.is_traversable else 0
    in_trav_objs_count += len([obj for obj in objs_at_loc if not objs_at_loc[obj].is_traversable])

    # check if we would have an in_traversable object and other objects in
    # the same location (which is impossible)
    if in_trav_objs_count >= 1 and (len(objs_at_loc) + 1) >= 2:
        return False
    else:
        return True


def _possible_drop(grid_world, agent_id, obj_id, drop_range):
    """ Private MATRX method."""
    reg_ag = grid_world.registered_agents[agent_id]  # Registered Agent
    loc_agent = reg_ag.location
    loc_obj_ids = grid_world.grid[loc_agent[1], loc_agent[0]]

    # No object given
    if not obj_id:
        return DropObjectResult(DropObjectResult.RESULT_NONE_GIVEN, False)

    # No object with that name
    if isinstance(obj_id, str) and not any([obj_id == obj.obj_id for obj in reg_ag.is_carrying]):
        return DropObjectResult(DropObjectResult.RESULT_NO_OBJECT, False)

    if len(loc_obj_ids) == 1:
        return DropObjectResult(DropObjectResult.RESULT_SUCCESS, True)

    # TODO: incorporate is_possible check from DropAction.mutate is_possible here

    return DropObjectResult(DropObjectResult.RESULT_SUCCESS, True)


def _find_drop_loc(grid_world, agent, env_obj, drop_range, start_loc):
    """ Private MATRX method."""
    queue = collections.deque([[start_loc]])
    seen = {start_loc}

    width = grid_world.shape[0]
    height = grid_world.shape[1]

    while queue:
        path = queue.popleft()
        x, y = path[-1]

        # check if we are still within drop_range
        if get_distance([x, y], start_loc) > drop_range:
            return False

        # check if we can drop at this location
        if _is_drop_poss(grid_world, env_obj, [x, y], agent.obj_id):
            return [x, y]

        # queue unseen neighbouring tiles
        for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= x2 < width and 0 <= y2 < height and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))
    return False