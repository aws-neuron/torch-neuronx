"""
Process group variable tracking for torch.dynamo integration
"""

from torch._dynamo.variables.distributed import ProcessGroupVariable

from torch_neuronx.distributed.backend import ProcessGroupNeuron


class ProcessGroupNeuronVariable(ProcessGroupVariable):
    """Variable tracker for ProcessGroupNeuron objects in dynamo"""

    def __init__(self, value, **kwargs):
        super().__init__(value, **kwargs)

    @staticmethod
    def create(value, **kwargs):
        """Create ProcessGroupNeuronVariable for ProcessGroupNeuron objects"""
        if isinstance(value, ProcessGroupNeuron):
            return ProcessGroupNeuronVariable(value, **kwargs)
        return None


# Register ProcessGroupNeuron with dynamo's variable tracking system
def register_process_group_neuron():
    """Register ProcessGroupNeuron with dynamo variable tracking"""
    from torch._dynamo.variables.builder import VariableBuilder

    # Add ProcessGroupNeuron to the variable builder
    original_wrap = VariableBuilder.__call__

    def wrap_with_neuron_pg(self, value):
        if isinstance(value, ProcessGroupNeuron):
            return ProcessGroupNeuronVariable(value)
        return original_wrap(self, value)

    VariableBuilder.__call__ = wrap_with_neuron_pg


# Auto-register when module is imported
register_process_group_neuron()
