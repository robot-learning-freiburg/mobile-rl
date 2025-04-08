import copy
from typing import Optional

import gym
import gym.spaces
import torch
from ray.rllib import SampleBatch
from ray.rllib.agents.sac.sac_torch_model import SACTorchModel
from ray.rllib.models import ModelCatalog, ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from modulation.models.mapencoder import LocalMapCNN, LocalDoubleMapCNN
from modulation.myray.ray_wandb import get_ckpt_weights


class RayMapEncoder(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        original_space = obs_space.original_space if hasattr(obs_space, "original_space") else obs_space
        assert isinstance(original_space, gym.spaces.Tuple)
        if len(original_space) == 2:
            robot_state_space, map_space = original_space
            action_space = None
        elif len(original_space) == 3:
            robot_state_space, map_space, action_space = original_space
        else:
            raise NotImplementedError()

        # hardcoded values for now:
        OUT_CHANNELS = 10 if map_space.shape[2] <= 2 else 32

        enc = model_config.get("custom_model_config").get("map_encoder")
        if enc == 'local':
            cnn_fn = LocalMapCNN
        elif enc == "localdouble":
            cnn_fn = LocalDoubleMapCNN
        else:
            raise ValueError()
        self.cnn = cnn_fn(in_shape=map_space.shape, out_channels=OUT_CHANNELS, stride=1)
        self.cnn = self.cnn.to(memory_format=torch.channels_last)

        concat_size = robot_state_space.shape[0] + self.cnn.output_size
        if action_space is not None:
            concat_size += action_space.shape[0]

        # Optional post-concat FC-stack.
        layers = []
        in_size = concat_size
        for hidden in model_config.get("post_fcnet_hiddens", []):
            layers.append(SlimFC(in_size=in_size, out_size=hidden, activation_fn=model_config.get("post_fcnet_activation", "relu")))
            in_size = hidden
        self.post_fc_stack = nn.Sequential(*layers)

        # Actions and value heads.
        self.logits_layer = None

        if num_outputs:
            # Action-distribution head.
            self.logits_layer = SlimFC(in_size=model_config.get("post_fcnet_hiddens", [])[-1],
                                       out_size=num_outputs,
                                       activation_fn=None)
        else:
            self.num_outputs = concat_size

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict[SampleBatch.OBS]
        if len(obs) == 2:
            robot_state, local_map = obs
            non_img_obs = [robot_state]
        elif len(obs) == 3:
            robot_state, local_map, actions = obs
            non_img_obs = [robot_state, actions]
        else:
            raise ValueError()

        local_map = local_map.permute([0, 3, 1, 2]).contiguous(memory_format=torch.channels_last)
        map_features = self.cnn(local_map)

        out = torch.cat(non_img_obs + [map_features], dim=-1)

        # Push through (optional) FC-stack (this may be an empty stack).
        out = self.post_fc_stack(out)

        # No logits/value branches.
        if self.logits_layer is None:
            return out, []

        # Logits- and value branches.
        logits = self.logits_layer(out)
        return logits, []

    @override(ModelV2)
    def value_function(self):
        raise NotImplementedError()


class MapEncoderSACTorchModel(SACTorchModel):
    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: Optional[int],
                 model_config: ModelConfigDict,
                 name: str,
                 policy_model_config: ModelConfigDict = None,
                 q_model_config: ModelConfigDict = None,
                 twin_q: bool = False,
                 initial_alpha: float = 1.0,
                 target_entropy: Optional[float] = None):
        policy_model_config = copy.copy(model_config['custom_model_config'])
        q_model_config = copy.copy(model_config['custom_model_config'])

        super(MapEncoderSACTorchModel, self).__init__(obs_space,
                                                      action_space,
                                                      num_outputs,
                                                      model_config,
                                                      name,
                                                      policy_model_config,
                                                      q_model_config,
                                                      twin_q,
                                                      initial_alpha,
                                                      target_entropy)

        # ckpt_weights = policy_model_config.get('ckpt_weights', None)
        ckpt_path = policy_model_config.get('ckpt_path', None)
        if ckpt_path:
            ckpt_weights = get_ckpt_weights(ckpt_path)
            weights = convert_to_torch_tensor(ckpt_weights['default_policy'], device=self.variables()[0].device)
            missing_keys, unexpected_keys = self.load_state_dict(weights, strict=True)
            assert not missing_keys, missing_keys
            assert not unexpected_keys, unexpected_keys

    def get_policy_output(self, model_out: TensorType) -> TensorType:
        """Returns policy outputs, given the output of self.__call__().

        For continuous action spaces, these will be the mean/stddev
        distribution inputs for the (SquashedGaussian) action distribution.
        For discrete action spaces, these will be the logits for a categorical
        distribution.

        Args:
            model_out (TensorType): Feature outputs from the model layers
                (result of doing `self.__call__(obs)`).

        Returns:
            TensorType: Distribution inputs for sampling actions.
        """
        # Problem: SAC.forward() doesn't do anything but to return the input obs.
        # BUT: this already changes the flattened obs to the original obs shape
        # As a result self.action_model.forward() fails because it first wants to reshape a flat obs into the original shape as well
        # -> flatten it again first
        assert len(model_out) == 2
        model_out = torch.cat([model_out[0], model_out[1].flatten(1)], dim=-1)

        out, _ = self.action_model({"obs": model_out}, [], None)
        return out
