from minestudio.simulator.callbacks.callback import MinecraftCallback


class VoxelsCallback(MinecraftCallback):
    def __init__(self, voxels_ins=[-7, 7, -7, 7, -7, 7]):
        super().__init__()
        self.voxels_ins = voxels_ins

    def before_step(self, sim, action):
        action["voxels"] = self.voxels_ins
        return action
