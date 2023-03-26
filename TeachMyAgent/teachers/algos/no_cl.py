import numpy as np
from gym.spaces import Box
from TeachMyAgent.teachers.algos.AbstractTeacher import AbstractTeacher
from TeachMyAgent.teachers.utils.gaussian_torch_distribution import GaussianTorchDistribution

class NonCurriculum(AbstractTeacher):
    def __init__(self, mins, maxs, seed, env_reward_lb, env_reward_ub, target_dist=None):
        '''
            Directly train on the target distribution
        '''
        AbstractTeacher.__init__(self, mins, maxs, env_reward_lb, env_reward_ub, seed)

        target_mean, target_variance = self.get_or_create_dist(target_dist, mins, maxs, subspace=False) # Full task space if no intial dist
        flat_target_chol = GaussianTorchDistribution.flatten_matrix(target_variance, tril=False)
        self.target_dist = GaussianTorchDistribution(target_mean, flat_target_chol, use_cuda=False)
        self.context_bounds = (np.array(mins), np.array(maxs))

    def sample_task(self):
        sample = self.target_dist.sample().detach().numpy()
        return np.clip(sample, self.context_bounds[0], self.context_bounds[1], dtype=np.float32)

    def non_exploratory_task_sampling(self):
        return {"task": self.sample_task(),
                "infos": {
                    "bk_index": -1,
                    "task_infos": None}
                }