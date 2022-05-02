import numpy as np
from matrx.actions.action import Action, ActionResult
from matrx.objects.agent_body import AgentBody
from matrx.objects.standard_objects import AreaTile
from matrx.actions.object_actions import _is_drop_poss, _act_drop, _possible_drop, _find_drop_loc, GrabObject, GrabObjectResult, RemoveObject, RemoveObjectResult, DropObject
from matrx.utils import get_distance
import random

class Idle(Action):
    def __init__(self, duration_in_ticks=1):
        super().__init__(duration_in_ticks)

    def is_possible(self, grid_world, agent_id, **kwargs):
        # Maybe do a check to see if the empty location is really and still empty?
        return IdleResult(IdleResult.RESULT_SUCCESS, True)


class IdleResult(ActionResult):
    """ Result when falling succeeded. """
    RESULT_SUCCESS = 'Idling action successful'

    """ Result when the emptied space was not actually empty. """
    RESULT_FAILED = 'Failed to idle'

    def __init__(self, result, succeeded):
        super().__init__(result, succeeded)

#class RemoveObjectTogether(RemoveObject):
#    def __init__(self, duration_in_ticks=1):
#        super().__init__(duration_in_ticks)

#    def is_possible(self, grid_world, agent_id, world_state, **kwargs):
#        remove_range = np.inf if 'remove_range' not in kwargs else kwargs['remove_range']
#        other_agent = world_state[{"name": "RescueBot"}]
#        obj_to_remove = world_state[kwargs["object_id"]]

        # check if the collaborating agent is close enough to the object as well 
#        if get_distance(other_agent['location'], obj_to_remove['location']) > remove_range and 'bush' in obj_to_remove:
#            return RemoveTogetherResult(RemoveTogetherResult.OTHER_TOO_FAR, False)

#        return super().is_possible(grid_world, agent_id, world_state, **kwargs)

#class RemoveTogetherResult(RemoveObjectResult):
#    REMOVE_SUCCESS = 'Successfully removed object together'
#    OTHER_TOO_FAR = 'Failed to remove object. The other agent is too far from the object'

class RemoveObjectTogether(Action):
    """ Removes an object from the world.
    An action that permanently removes an
    :class:`matrx.objects.env_object.EnvObject` from the world, which can be
    any object except for the agent performing the action.
    Parameters
    ----------
    duration_in_ticks : int
        Optional. Default: ``1``. Should be zero or larger.
        The default duration of this action in ticks during which the
        :class:`matrx.grid_world.GridWorld` blocks the agent performing other
        actions. By default this is 1, meaning that all actions of this type will take
        both the tick in which it was decided upon and the subsequent tick.
        When the agent is blocked / busy with an action, only the
        :meth:`matrx.agents.agent_brain.AgentBrain.filter_observations` method is called for that agent, and the
        :meth:`matrx.agents.agent_brain.AgentBrain.decide_on_action` method is skipped.
        This means that agents that are busy with an action can only perceive the world but not decide on
        a new action untill the action has completed.
        An agent can overwrite the duration of an action by returning the ``action_duration`` in the ``action_kwargs``
        in the :meth:`matrx.agents.agent_brain.AgentBrain.decide_on_action` method, as so:
        ``return >action_name<, {'action_duration': >ticks<}``
    """

    def __init__(self, duration_in_ticks=0):
        super().__init__(duration_in_ticks)

    def mutate(self, grid_world, agent_id, world_state, **kwargs):
        """ Removes the specified object.
        Removes a specific :class:`matrx.objects.env_object.EnvObject` from
        the world. Can be any object except for the agent performing the
        action.
        Parameters
        ----------
        grid_world : GridWorld
            The ``matrx.grid_world.GridWorld`` instance in which the object is
            sought according to the `object_id` parameter.
        agent_id : str
            The string representing the unique identifier that represents the
            agent performing this action.
        world_state : State
            The State object representing the entire world. Can be used to
            simplify search of objects and properties when performing an
            action. Note that this is the State of the entire world, not
            that of the agent performing the action.
        object_id: str (Optional. Default: None)
            The string representing the unique identifier of the
            :class:`matrx.objects.env_object.EnvObject` that should be
            removed. If not given, the closest object is selected.
            removed.
        remove_range : int (Optional. Default: 1)
            The range in which the :class:`matrx.objects.env_object.EnvObject`
            should be in for it to be removed.
        Returns
        -------
        RemoveObjectResult
            Depicts the action's success or failure and reason for that result.
            See :class:`matrx.actions.object_actions.RemoveObjectResult` for
            the results it can contain.
        """
        assert 'object_id' in kwargs.keys()  # assert if object_id is given.
        object_id = kwargs['object_id']  # assign
        remove_range = 1  # default remove range
        other_agent = world_state[{"name": "RescueBot"}]
        other_human = world_state[{"name": "Human"}]
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
            if obj == object_id and get_distance(other_agent['location'], world_state[obj]['location'])<=remove_range and get_distance(other_human['location'], world_state[obj]['location'])<=remove_range and 'rocks' in obj:  # if object is in that list
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
        """ Checks if an object can be removed.
        Parameters
        ----------
        grid_world : GridWorld
            The :class:`matrx.grid_world.GridWorld` instance in which the
            object is sought according to the `object_id` parameter.
        agent_id: str
            The string representing the unique identified that represents the
            agent performing this action.
        world_state : State
            The State object representing the entire world. Can be used to
            simplify search of objects and properties when checking if an
            action can be performed. Note that this is the State of the
            entire world, not that of the agent performing the action.
        object_id: str (Optional. Default: None)
            The string representing the unique identifier of the
            :class:`matrx.objects.env_object.EnvObject` that should be
            removed. If not given, the closest object is selected.
        remove_range : int (Optional. Default: 1)
            The range in which the :class:`matrx.objects.env_object.EnvObject`
            should be in for it to be removed.
        Returns
        -------
        RemoveObjectResult
            The :class:`matrx.actions.action.ActionResult` depicting the
            action's expected success or failure and reason for that result.
            See :class:`matrx.actions.object_actions.RemoveObjectResult` for
            the results it can contain.
        """
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


class RemoveObjectResult(ActionResult):
    """ActionResult for a RemoveObjectAction
    The results uniquely for RemoveObjectAction are (as class constants):
    * OBJECT_REMOVED: If the object was successfully removed.
    * REMOVAL_FAILED: If the object could not be removed by the
      :class:`matrx.grid_world.GridWorld`.
    * OBJECT_ID_NOT_WITHIN_RANGE: If the object is not within specified range.
    * NO_OBJECTS_IN_RANGE: If no objects are within range.
    Parameters
    ----------
    result: str
        A string representing the reason for the (expected) success or fail of
        a :class:`matrx.actions.object_actions.RemoveObjectAction`.
    succeeded: bool
        A boolean representing the (expected) success or fail of a
        :class:`matrx.actions.object_actions.RemoveObjectAction`.
    See Also
    --------
    :class:`matrx.actions.object_actions.RemoveObjectAction`
    """

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

class CarryObject(GrabObject):
    def __init__(self, duration_in_ticks=1):
        super().__init__(duration_in_ticks)        

    def mutate(self, grid_world, agent_id, world_state, **kwargs):
        # Additional check
        assert 'object_id' in kwargs.keys()
        assert 'grab_range' in kwargs.keys()
        assert 'max_objects' in kwargs.keys()

        # if possible:
        object_id = kwargs['object_id']  # assign

        # Loading properties
        reg_ag = grid_world.registered_agents[agent_id]  # Registered Agent
        if 'critical' in object_id and 'human' in agent_id:
            # change our image 
            reg_ag.change_property("img_name", "/images/carry-critical-human.svg")
        if 'mild' in object_id and 'human' in agent_id:
            reg_ag.change_property("img_name", "/images/carry-mild-human.svg")
        if 'critical' in object_id and 'bot' in agent_id:
            # change our image 
            reg_ag.change_property("img_name", "/images/carry-critical-robot.svg")
        if 'mild' in object_id and 'bot' in agent_id:
            reg_ag.change_property("img_name", "/images/carry-mild-robot.svg")

        # pickup the object 
        return super().mutate(grid_world, agent_id, world_state, **kwargs)

class Drop(DropObject):
    def __init__(self, duration_in_ticks=0):
        super().__init__(duration_in_ticks)

    def mutate(self, grid_world, agent_id, world_state, **kwargs):
        agent = grid_world.registered_agents[agent_id]
        # change the agent image back to default 
        agent.change_property("img_name", "/images/rescue-man-final3.svg")

        # drop the actual object like we would do with a normal drop action 
        return super().mutate(grid_world, agent_id, world_state, **kwargs)

class CarryObjectTogether(GrabObject):
    def __init__(self, duration_in_ticks=1):
        super().__init__(duration_in_ticks)

    def is_possible(self, grid_world, agent_id, world_state, **kwargs):
        object_id = None if 'object_id' not in kwargs else kwargs['object_id']
        grab_range = np.inf if 'grab_range' not in kwargs else kwargs['grab_range']
        max_objects = np.inf if 'max_objects' not in kwargs else kwargs['max_objects']

        #grab_range = np.inf if 'grab_range' not in kwargs else kwargs['grab_range']
        other_agent = world_state[{"name": "Robot"}]
        if object_id in kwargs:
            obj_to_grab = world_state[kwargs["object_id"]]
        if object_id not in kwargs:
            obj_to_grab = None
            pass

        # check if the collaborating agent is close enough to the object as well 
        if obj_to_grab and get_distance(other_agent['location'], obj_to_grab['location']) > grab_range:
            return CarryTogetherResult(CarryTogetherResult.OTHER_TOO_FAR, False)
        
        # do the checks for grabbing a regular object as well
        return super().is_possible(grid_world, agent_id, world_state, **kwargs)
        
        

    def mutate(self, grid_world, agent_id, world_state, **kwargs):
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
            agent.change_property("img_name", "/images/carry-critical-human.svg")
        if 'mild' in object_id and 'human' in agent_id:
            agent.change_property("img_name", "/images/carry-mild-human.svg")

        # pickup the object 
        return super().mutate(grid_world, agent_id, world_state, **kwargs)


class DropObjectTogether(DropObject):
    def __init__(self, duration_in_ticks=0):
        super().__init__(duration_in_ticks)

    def mutate(self, grid_world, agent_id, world_state, **kwargs):
        other_agent_id = world_state[{"name": "RescueBot"}]['obj_id']

        # if we want to change objects, we need to change the grid_world object 
        other_agent = grid_world.registered_agents[other_agent_id]
        agent = grid_world.registered_agents[agent_id]

        # teleport the other agent to our current position 
        other_agent.change_property("location", agent.properties['location'])

        # make the other agent visible again 
        other_agent.change_property("visualize_opacity", 1)

        # change the agent image back to default 
        agent.change_property("img_name", "/images/rescue-man-final3.svg")

        # drop the actual object like we would do with a normal drop action 
        return super().mutate(grid_world, agent_id, world_state, **kwargs)


class CarryTogetherResult(ActionResult):
    PICKUP_SUCCESS = 'Successfully grabbed object together'
    OTHER_TOO_FAR = 'Failed to grab object. The other agent is too far from the object'