import datetime
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from arg_parser import args

# envs
file_name = "subway103"
env_path = f"./{file_name}"

# #############################################################################
# 임시
train_mode = True
max_episode_steps = 1000000

def env_reset(env, behavior_name):
    env.reset()
    dec, term = env.get_steps(behavior_name)
    done = len(term.agent_id) > 0
    state = term.obs[0] if done else dec.obs[0]
    return state

def env_next_step(env, behavior_name, action):
    action_tuple = ActionTuple()
    action_tuple.add_continuous(action)
    env.set_actions(behavior_name, action_tuple)
    env.step()
    
    dec, term = env.get_steps(behavior_name)
    done = len(term.agent_id) > 0
    reward = term.reward if done else dec.reward
    next_state = term.obs[0] if done else dec.obs[0]
    
    return next_state, reward, done

if __name__ == "__main__":
    # CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Unity Environment
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_path, side_channels=[engine_configuration_channel], seed=args.seed)
    env.reset()
    
    # Unity Brain
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
    
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Agent
    agent = SAC(spec.observation_specs[0].shape[0], spec.action_spec, args)
    
    #Tesnorboard
    writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name, args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)
    
    # Training Loop
    total_numsteps = 0
    updates = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env_reset(env, behavior_name)

        while not done:
            if args.start_steps > total_numsteps:
                action = spec.action_spec.random_action(1)._continuous  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1


            next_state, reward, done = env_next_step(env, behavior_name, action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state

        if total_numsteps > args.num_steps:
            break


        writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward[0], 2)))

        if i_episode % 10 == 0 and args.eval is True:
            avg_reward = 0.
            episodes = 10
            for _  in range(episodes):
                state = env_reset(env, behavior_name)
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done = env_next_step(env, behavior_name, action)
                    episode_reward += reward

                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes


            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward[0], 2)))
            print("----------------------------------------")
            

    env.close()

