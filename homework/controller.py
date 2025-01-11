import pystk

#using below stuff for PPO policy network
import torch 
import torch.nn as nn
import torch.distributions as D

import numpy as np


# Policy network and initialization from henanmemeda RL-Adventure-2/ppo on github
# https://github.com/henanmemeda/RL-Adventure-2/blob/master/3.ppo.ipynb
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class PPOPolicy(nn.Module):

    # Neural network architecture as defined by henanmemeda Rl-Adventure-2/ppo on github
    # Two networks, one actor and one critic
    def __init__(self, num_inputs, num_outputs, hidden_size =128, std=0.0):
        super(PPOPolicy, self).__init__()
    
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
    
    # Forward pass, also defined by henanmemeda RL-Adventure-2/ppo on github
    def forward(self, state):
        value = self.critic(state)
        mu    = self.actor(state)
        std   = self.log_std.exp()
        dist  = D.Normal(mu, std)

        #print("Potential NaN values?")
        #print("valueVALUE", value) #this is the output of critic NN, should only be 1 entry
        #print("muMU", mu) # this is the output of the actor NN, should be 5 entries
        #print("stdSTD", std) # this is log std, shouldnalso be 5 entries

        # Had to add this exception block because sometimes an error would be thrown
        # Usually it is because there are NaN values, not sure why
        try:
            return dist, value
        except:
            mean_list = [0, 0, 0, 0, 0]
            mean = torch.tensor(mean_list).float()
            std_list = [1, 1, 1, 1, 1]
            std = torch.tensor(std_list).float()
            dist = D.Normal(mean, std)        
        return dist, value

def sample_actions(dist):
    # given distribution, sample to get an action
    actions = dist.sample()
    
    # Clip actions within bounds defined by the game (i.e. steering is between -1 and 1)
    # I added this exception block because I was getting an error when the array had only one entry for each action 
    try:
        actions[:, 0] = torch.clamp(actions[:, 0], -1, 1)  # First entry
        actions[:, 1] = torch.clamp(actions[:, 1], 0, 1)   # Second entry
        actions[:, 2] = torch.clamp(actions[:, 2], 0, 1)  # Third entry
        actions[:, 3] = torch.clamp(actions[:, 3], 0, 1)   # Fourth entry
        actions[:, 4] = torch.clamp(actions[:, 4], 0, 1)  # Fifth entry
    except:
        actions[0] = torch.clamp(actions[0], -1, 1)  # First entry
        actions[1] = torch.clamp(actions[1], 0, 1)   # Second entry
        actions[2] = torch.clamp(actions[2], 0, 1)  # Third entry
        actions[3] = torch.clamp(actions[3], 0, 1)   # Fourth entry
        actions[4] = torch.clamp(actions[4], 0, 1)  # Fifth entry
    return actions


# GAE =  Generalized Advantage Estimate
# This function is copied from henanmemeda RL-Adventure-2/ppo on github
# Advantage estimate is used in the PPO objective function
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

# Mini batch implementation, also defined by henanmemeda RL-Adventure-2/ppo on github
# Should help train faster
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)

        # Small edit, need to convert to tensor with type long
        rand_ids = torch.tensor(rand_ids, dtype=torch.long)
        yield (
            states[rand_ids, :], 
            actions[rand_ids, :], 
            log_probs[rand_ids, :], 
            returns[rand_ids], 
            advantage[rand_ids]
        )

# PPO update implementation by henanmemeda RL-Adventure-2/ppo on github
def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, actions, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions)

            ratio = (new_log_probs - old_log_probs).exp()

            # first surrogate, had to add some transposes here
            surr1 = torch.transpose(ratio, 0, 1) * advantage
            surr1 = torch.transpose(surr1, 0, 1)

            # CLIP part of PPO
            input = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)

            # second surrogate, had to also add some transposes here
            surr2 = torch.transpose(input, 0, 1) * advantage
            surr2 = torch.transpose(surr2, 0, 1)

            # calculating loss for each NN (actor and critic)
            actor_loss  =  -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            # calculating overall loss
            loss = (0.5 * critic_loss + actor_loss - 0.001 * entropy)

            # Perform SGD on network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Return calculated Loss
            return loss.item()

# This is the updated controller function
# The goal of the controler function is to take actions (steer, accelerate, brake, drift, nitro) that will best follow the aim point
def control(aim_point, current_vel, policy_net, steer_gain=6, skid_thresh=0.2, target_vel=80):
    import numpy as np

    # Initialize pystk action
    action = pystk.Action()

    # Initializing state tensor which is defined by the aim point coordinates and the current velocity
    state_input = [aim_point[0], aim_point[1], current_vel]
    state_tensor = torch.tensor(state_input)

    with torch.no_grad():
        dist, value = policy_net(state_tensor)  # Get action distribution and value prediction
        actions = sample_actions(dist)

        # had to add an exception catcher for dimension issues
        try: 
            action.steer = (actions[0])[0]
            acceleration = (actions[0])[1]

            # from test experience, never want acceleration to be less than 0.5 
            if acceleration<0.5:
                action.acceleration = 0.5
            else:
                action.acceleration = acceleration
            
            # Because brake, drift, and nitro are booleans but we have continuous values, need to round values to 0 or 1 
            if (actions[0])[2]<0.5:
                action.brake = 0
            else:
                # tweaked with never breaking 
                action.brake = 1

            if (actions[0])[3]<0.5:
                action.drift = 0
            else:
                action.drift = 1

            if (actions[0])[4]<0.5:
                action.nitro = 0
            else:
                action.nitro = 1

        except:
            action.steer = actions[0].item()
            acceleration = actions[0].item()

            if acceleration<0.5:
                action.acceleration = 0.5
            else:
                action.acceleration = acceleration

            if actions[0].item()<0.5:
                action.brake = 0
            else:
                action.brake = 1

            if actions[0].item()<0.5:
                action.drift = 0
            else:
                action.drift = 1

            if actions[0].item()<0.5:
                action.nitro = 0
            else:
                action.nitro = 1

    return action, value, dist, actions[0]

# OLD CONTROLER 
def control_old(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=80):
    import numpy as np
    #this seems to initialize an object
    action = pystk.Action()

    # Trying to find the radian angle given by the aim point and scale that to the scale [-1, 1]
    if aim_point[0]<0:
        if aim_point[1]>0:
            action.drift = True
            action.steer = -0.9
        else:
            if aim_point[0]<-0.5:
                if aim_point[1]>-0.5:
                    action.drift = True
            else:
                action.drift = False
            
            if aim_point[0]>-0.03:
                action.steer = 0
                target_vel = 80
            elif aim_point[0]>-0.2:
                action.steer = -0.8
                target_vel = 80
            else:
                action.steer = -0.9
                target_vel = 30
    else:
        if aim_point[1]>0:
            action.drift = True
            action.steer = 0.9
        else:
            if aim_point[0]>0.5:
                if aim_point[1]>-0.5:
                    action.drift = True
            else:
                action.drift = False
            
            if aim_point[0]<0.03:
                action.steer = 0
                target_vel = 80
            elif aim_point[0]<0.2:
                action.steer = 0.8
                target_vel = 80
            else:
                action.steer = 0.9
                target_vel = 30

    # set acceleration and break to reach target velocity
    if current_vel < target_vel:
        action.brake = False
        if (current_vel/target_vel) < 0.5:
            if abs(aim_point[0])<0.1:
                action.nitro = False
        else:
            action.nitro = False
        action.acceleration = 1 - current_vel/target_vel
    elif current_vel > target_vel:
        action.brake= True

    return action

if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()

        use_cuda = torch.cuda.is_available()
        device   = torch.device("cuda" if use_cuda else "cpu")

        # initialize policy network
        PPO_policy = PPOPolicy(3,5).to(device)

        # initialize optimizer for training
        optimizer = torch.optim.SGD(PPO_policy.parameters(), lr=1e-4)

        # list to record losses
        losses = []

        # going through 50 epochs
        for t in args.track:
            for i in range(10):
                print("training epoch", i)
                steps, how_far, loss = pytux.rollout_train(t, control, PPO_policy, optimizer, 1, max_frames=1000, verbose=args.verbose)
                losses.append(loss)
                print(steps, how_far)
        pytux.close()

        # Graphing loss over epochs
        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("PPO Loss")
        plt.show()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
