from matrx.agents.agent_utils.state import State

def sameAppearance(appearance1:dict, appearance2:dict):
    """
    @return true if the given 2 appearances are the same.
    This means both dicts have the same value for "shape", "size" and "colour"
    FIXME this would be better an appearance object
    """
    return appearance1['shape']==appearance2['shape']  \
        and appearance1['size']==appearance2['size']   \
        and appearance1['colour']==appearance2['colour'] 
       
class BlockPositions:
    """
    This class tries to support the book-keeping of
    which blocks have been seen recently and where.
    It seems similar to the matrx state_tracker.py
    IMMUTABLE
    """
    def __init__(self, blocks={}):
        self._blocks=blocks
    
    def update(self, state:State):#->BlockPositions
        '''
        @param state the latest state.
        This will extract all blocks and update this accordingly
        '''
        blocks=self
        for id in state.keys():
            block=state[id]
            if 'class_inheritance' in block \
                and 'CollectableBlock' in block['class_inheritance'] \
                and block['is_movable']:
                blocks=blocks.updateInfo(block)
        return blocks

        
    def updateInfo(self, blockinfo:dict):
        '''
        call this when given blockid was last seen at loc
        @param details the loc where the block was seen.
        must contain 'obj_id' 'location' and 'visualization' elements
        '''
        if not ('obj_id' in blockinfo and 'location' in blockinfo and 'visualization' in blockinfo):
            raise ValueError("blockinfo must contain location, visualization and obj_id but got "+str(blockinfo))
        blocks=self._blocks.copy()
        blocks[blockinfo['obj_id']]=blockinfo
        return BlockPositions(blocks)

    def getBlocksAt(self, loc:tuple):
        '''
        @param loc the location you want the blocks at
        @return the list of details of the all blocks that are at loc
        '''
        return [info for info in self._blocks.values()
                if loc==info['location']]

    
    def getAppearance(self, appearance:dict)->list:
        '''
        @param appearance the appearance you need
        @return the list of details of the all blocks that have given appearance
        '''
        return [info for info in self._blocks.values()
                if sameAppearance(appearance,info['visualization'])]
    
    def getDifference(self, other)->dict:
        """
        @param other another :BlockPositions object
        @return the difference (list of changed objects) with another BlockPositions.
        Each dict is a full dict of the changed objects.
        Notice blocks can not disappear, only new blocks can appear.
        """
        changes=[]
        #workaround:  way to get set of all ids. set does not support '+'
        allids = set(list(self._blocks.keys()) + list(other._blocks.keys()))
        for id in allids:
            if not id in self._blocks:
                changes.append(other._blocks[id])
            elif not id in other._blocks:
                changes.append(self._blocks[id])
            elif not self._blocks[id] == other._blocks[id]:
                changes.append(self._blocks[id])
        return changes