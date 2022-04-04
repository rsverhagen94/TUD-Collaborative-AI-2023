from abc import  ABC, abstractmethod
from matrx.agents.agent_utils.state import State
from matrx.agents import AgentBrain

class BW4TBrain(AgentBrain, ABC):
    """
    This class is the obligatory base class for BW4T agents.
    BW4T agents must implement decide_on_bw4t_action
    """
    

    def __init__(self, condition, slowdown:int):
        '''
        @param slowdown an integer. Basically this sets action_duration
        field to the given slowdown. 1 implies normal speed
        of 1 action per tick. 3 givs 1 allowed action every 3 ticks. etc
        Implementors of BW4TBrain are NOT ALLOWED TO CHANGE THIS VALUE.
        This is to ensure that agents run at the required speed.
        FIXME this is hacky. These parameters should really be private.
        '''
        self.__slowdown = slowdown
        super().__init__()
    
    #Override
    def decide_on_action(self, state:State):
        '''
        Final . Agents must override decide_on_bw4t_action instead
        '''
        act,params = self.decide_on_bw4t_action(state)
        params['grab_range']=1
        # door_range=1 does not work, doors don't open
       # params['door_range']=1
        params['max_objects']=1
        params['action_duration'] = self.__slowdown
        return act,params
    
    #Override 
    def filter_observations(self,state:State)->State:
        '''
        Final. Agents must override filter_bw4t_observations.
        This to ensure that agents can not by-pass imposed percept filtering.
        '''
        return self.filter_bw4t_observations(state)
        
    def filter_bw4t_observations(self,state)->State:
        """ 
        Filters the world state before deciding on an action.
        This function is called every tick, so use this for message processing.

        In this method you filter the received world state to only those
        properties and objects the agent is actually supposed to see.

        Currently the world returns ALL properties of ALL objects within a
        certain range(s), as specified by :
        class:`matrx.agents.capabilities.capability.SenseCapability`. But
        perhaps some objects are obscured because they are behind walls and
        this agent is not supposed to look through walls, or an agent is not
        able to see some properties of certain objects (e.g. colour).

        The adjusted world state that this function returns is directly fed to
        the agent's decide function. Furthermore, this returned world state is
        also fed through the MATRX API to any visualisations.

        Override this method when creating a new AgentBrain and you need to
        filter the world state further.

        Parameters
        ----------
        state: State
            A state description containing all perceived
            :class:`matrx.objects.env_object.EnvObject` and objects inheriting
            from this class within a certain range as defined by the
            :class:`matrx.agents.capabilities.capability.SenseCapability`.

            The keys are the unique identifiers, as values the properties of
            an object. See :class:`matrx.objects.env_object.EnvObject` for the
            kind of properties that are always included. It will also contain
            all properties for more specific objects that inherit from that
            class.

            Also includes a 'world' key that describes common information about
            the world (e.g. its size).

        Returns
        -------
        filtered_state : State
            A dictionary similar to `state` but describing the filtered state
            this agent perceives of the world.

        Notes
        -----
        A future version of MATRX will include handy utility function to make
        state filtering less of a hassle (e.g. to easily remove specific
        objects or properties, but also ray casting to remove objects behind
        other objects)

        """
        return state
    
    @abstractmethod
    def decide_on_bw4t_action(self, state:State):
        '''
        @param state
        A state description as given by the agent's
        :meth:`matrx.agents.agent_brain.AgentBrain.filter_observation` method.

        Contains the decision logic of the agent.
        @return tuple (action name:str,  action arguments:dict)
        
        action is a string of the class name of an action that is also in the
        `action_set` class attribute. To ensure backwards compatibility
        we advise to use Action.__name__ where Action is the intended
        action.
        
        action_args is a dictionary with keys any action arguments and as values the
        actual argument values. If a required argument is missing an
        exception is raised, if an argument that is not used by that
        action a warning is printed. 
        
        An argument that is always possible is that of action_duration, which
        denotes how many ticks this action should take and overrides the
        action duration set by the action implementation. The minimum of 1
        is used if you provide a value <1.
            
        Check the action documentation to determine possible arguments.
        BW4T agents can not use NOT_ALLOWED_PARAMS above,
        they are set fixed by the environment builder
        
        This function is called only when the agent can take an action.
        Actions that need execution every tick should be placed in
        filter_bw4t_observations.
        '''
        pass
    