import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BATCH_SIZE, TARGET_REPLACE_ITER, MEMORY_CAPACITY, E_GREEDY

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# DQN
class Net(nn.Module):
    def __init__(self, action_n, state_n):
        super(Net, self).__init__()
        # input (4, 160, 380)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4).to(device) # (32, 39, 94)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1).to(device) # (64, 36, 91)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=6, stride=5).to(device) # (64, 7, 18)
        self.fc1 = nn.Linear(64 * 7 * 18, 50).to(device)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, action_n).to(device)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self, action_n, state_n, env_shape, learning_rate=0.01, reward_decay=0.9):
        self.eval_net, self.target_net = Net(action_n=action_n, state_n=state_n), Net(action_n=action_n, state_n=state_n)

        self.action_n = action_n
        self.state_n = state_n # (160, 380)
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = E_GREEDY
        self.env_shape = env_shape

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, 4 * self.state_n[0] * self.state_n[1] * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x).to(device), 0)
        # input only one sample
        if np.random.uniform() < self.epsilon:  # greedy
            actions_value = self.eval_net.forward(x).cpu()
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.env_shape == 0 else action.reshape(self.env_shape)
        else:  # random
            action = np.random.randint(0, self.action_n)
            action = action if self.env_shape == 0 else action.reshape(self.env_shape)
        return action

    def store_transition(self, s, a, r, s_):
        s = np.reshape(s, -1) # (4, 160, 380) -> 4 * 160 * 380
        s_ = np.reshape(s_, -1)
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        state_idx = 4 * self.state_n[0] * self.state_n[1] # 4 * 160 * 380
        b_s = torch.FloatTensor(b_memory[:, :state_idx]).to(device)
        b_a = torch.LongTensor(b_memory[:, state_idx:state_idx+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, state_idx+1:state_idx+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -state_idx:]).to(device)
        # reshape (batch_size, 4, 160, 380)
        b_s = np.reshape(b_s, (BATCH_SIZE, 4, self.state_n[0], self.state_n[1])).to(device)
        b_s_ = np.reshape(b_s_, (BATCH_SIZE, 4, self.state_n[0], self.state_n[1])).to(device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model(self):
        torch.save(self.eval_net.state_dict(), 'model/DQN/eval_{}_{}_{}.pkl'.format(self.lr, self.epsilon, BATCH_SIZE))
        torch.save(self.target_net.state_dict(), 'model/DQN/target_{}_{}_{}.pkl'.format(self.lr, self.epsilon, BATCH_SIZE))

    def load_model(self, model_name):
        self.eval_net.load_state_dict(torch.load(model_name))