import torch
import torch.onnx
from sac import SAC
from arg_parser import args
from main import file_name, env_path

# 유니티 네이놈
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# Unity Environment
engine_configuration_channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name=env_path, side_channels=[engine_configuration_channel], seed=args.seed)
env.reset()

# Unity Brain
behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]
engine_configuration_channel.set_configuration_parameters(time_scale=12.0)

num_inputs = spec.observation_specs[0].shape[0]
action_spec = spec.action_spec
ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(file_name, "1")

# Initialize model
model = SAC(spec.observation_specs[0].shape[0], spec.action_spec, args)

# Load pre-trained weights 
model.load_checkpoint(ckpt_path, evaluate=True)

# Set the model to evaluation mode
model.train(is_training=False)  # Ensure model is in evaluation mode

# Dummy input
dummy_input = torch.randn(1, num_inputs)  # Assuming state input shape (1, num_inputs)

# Export model to ONNX format
output_path = f"{file_name}.onnx"
torch.onnx.export(model, dummy_input, output_path, export_params=True, opset_version=11)

print(f"Model exported to {output_path}")

