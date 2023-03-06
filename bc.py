import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.distributions import Independent, Normal
from torch.nn import Parameter, functional as F
import numpy as np
from torch import optim
from tqdm import tqdm
from collections import deque
from transformers import BartTokenizer
# from kmeans import kmeans_cluster_predict
from Model import Model

CLUSTER_ID = 'kmeans/bart/cluster_ids_x_60.pth'
CLUSTER_CENTER = 'kmeans/bart/cluster_centers_60.pth'
SAVE_TRAINING_EMB_PATH = 'kmeans/bart/sentence_embedding_0_20.pth'
TRAINING_DATA2 = 'data/training_data2.txt'
DEVICE = 'cuda:1'
A2CMODEL_PATH = './BC_A2C/net_dd_bart.pth'
NUM_CLUSTER = 60
ACTIVATION_FUNCTIONS = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}

class TransitionDataset(Dataset):
    def __init__(self, transitions):
        super().__init__()
        self.states, self.actions, self.terminals = transitions['states'], transitions['actions'].detach(), transitions['terminals']

    def __getitem__(self, idx):                                                      
        if isinstance(idx, str):
            if idx == 'states':
                return self.states
            elif idx == 'actions':
                return self.actions
            elif idx == 'terminals':
                return self.terminals
        else:
            return dict(states=self.states[idx], actions=self.actions[idx], next_states=self.states[idx + 1], terminals=self.terminals[idx])

    def __len__(self):
        return self.terminals.size(0) - 1

def _create_fcnn(input_size, hidden_size, output_size, activation_function, dropout=0, final_gain=1.0):
    assert activation_function in ACTIVATION_FUNCTIONS.keys()
    network_dims, layers = (input_size, hidden_size, hidden_size), []
    for l in range(len(network_dims) - 1):
        layer = nn.Linear(network_dims[l], network_dims[l + 1])
        nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain(activation_function))
        nn.init.constant_(layer.bias, 0)
        layers.append(layer)
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(ACTIVATION_FUNCTIONS[activation_function]())

    final_layer = nn.Linear(network_dims[-1], output_size)
    nn.init.orthogonal_(final_layer.weight, gain=final_gain)
    nn.init.constant_(final_layer.bias, 0)
    layers.append(final_layer)
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, activation_function='tanh', log_std_dev_init=-0.5, dropout=0):
        super().__init__()
        self.actor = _create_fcnn(state_size, hidden_size, output_size=action_size, activation_function=activation_function, dropout=dropout, final_gain=0.01)
        self.log_std_dev = Parameter(torch.full((action_size, ), log_std_dev_init, dtype=torch.float32))

    def forward(self, state):
        mean = self.actor(state)
        policy = Independent(Normal(mean, self.log_std_dev.exp()), 1)
        return policy

    def log_prob(self, state, action):
        return self.forward(state).log_prob(action)

    def _get_action_uncertainty(self, state, action):
        ensemble_policies = []
        for _ in range(5):  
            ensemble_policies.append(self.log_prob(state, action).exp())
        return torch.stack(ensemble_policies).var(dim=0)

    def set_uncertainty_threshold(self, expert_state, expert_action):
        self.q = torch.quantile(self._get_action_uncertainty(expert_state, expert_action), 0.98).item()

    def predict_reward(self, state, action):
        uncertainty_cost = self._get_action_uncertainty(state, action)
        neg_idxs = uncertainty_cost.less_equal(self.q)
        uncertainty_cost[neg_idxs] = -1
        uncertainty_cost[~neg_idxs] = 1
        return -uncertainty_cost

class Critic(nn.Module):
    def __init__(self, state_size, hidden_size, activation_function='tanh'):
        super().__init__()
        self.critic = _create_fcnn(state_size, hidden_size, output_size=1, activation_function=activation_function)

    def forward(self, state):
        value = self.critic(state).squeeze(dim=1)
        return value

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, activation_function='tanh', log_std_dev_init=-0.5, dropout=0):
        super().__init__()
        self.actor = Actor(state_size, action_size, hidden_size, activation_function=activation_function, log_std_dev_init=log_std_dev_init, dropout=dropout)
        self.critic = Critic(state_size, hidden_size, activation_function=activation_function)

    def forward(self, state):
        policy, value = self.actor(state), self.critic(state)
        return policy, value

    def get_greedy_action(self, state):
        return self.actor(state).mean

    def log_prob(self, state, action):
        return self.actor.log_prob(state, action)

def preprocess():
    cluster_id = torch.load(CLUSTER_ID)
    sen_emb = torch.load(SAVE_TRAINING_EMB_PATH)
    cluster_center = torch.load(CLUSTER_CENTER).to(DEVICE)
    center_id = {}
    for i in range(NUM_CLUSTER):
        center_id[str(i)] = cluster_center[i]
    expert_obs = sen_emb
    expert_action = torch.empty([0, 1024], dtype=torch.float32, device=DEVICE)
    for id in cluster_id:
        tensor = center_id[str(id.item())].unsqueeze(0)
        expert_action = torch.cat((expert_action, tensor), 0)
    with open (TRAINING_DATA2, 'r') as f:
        line = f.readlines()
    num_dialog = []
    expert_terminal = []
    count = 0
    for i in line:
        if i == 's\n':
            count += 1
            expert_terminal.append(0)
        elif i == '<eod>\n':
            expert_terminal.pop()
            expert_terminal.append(1)
            num_dialog.append(count)
            count = 0
    expert_terminal = np.array(expert_terminal)
    expert_terminal = torch.from_numpy(expert_terminal).to(DEVICE)
    transion = {'states':expert_obs, 'actions':expert_action, 'terminals':expert_terminal}
    expert_trajectories = TransitionDataset(transion)
    return expert_trajectories
    
def behavioural_cloning_update(agent, expert_trajectories, agent_optimiser, batch_size):
    expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True)
    for expert_transition in expert_dataloader:
        expert_state, expert_action = expert_transition['states'], expert_transition['actions']
        agent_optimiser.zero_grad(set_to_none=True)
        behavioural_cloning_loss = -agent.log_prob(expert_state, expert_action).mean()
        behavioural_cloning_loss.backward()
        agent_optimiser.step()

def find_nearest_id(policy, cluster_center):
    result = F.cosine_similarity(policy, cluster_center, dim=1)
    id = torch.argmax(result)
    return id

def similarity(l1, l2):
    assert l1.shape[0] == l2.shape[0]
    sum = 0
    for i in range(l1.shape[0]):
        if int(l1[i]) == int(l2[i]):
            sum += 1
    print(sum/l1.shape[0])

def train():
    state_size = 1024
    action_size = 1024
    expert_trajectories = preprocess()
    # for i in expert_trajectories:
    #     print(i)
    agent = ActorCritic(state_size, action_size, 256, log_std_dev_init=-2).to(DEVICE)
    agent_optimizer = optim.RMSprop(agent.parameters(), lr=0.0003, alpha=0.9)
    metrics = dict(train_steps=[], train_returns=[], test_steps=[], test_returns=[])
    recent_returns = deque(maxlen=5)
    # state, terminal, train_return, trajectories = env.reset(), False, 0, []
    for step in tqdm(range(1, 50)):
      if step == 1:
        for _ in tqdm(range(20), leave=False):
            behavioural_cloning_update(agent, expert_trajectories, agent_optimizer, 16)
    torch.save({
        'net': agent.state_dict(),
        'optimizer': agent_optimizer.state_dict(),
    }, A2CMODEL_PATH)

def evaluation(training_path, eva, modelpth):
    cluster_center = torch.load(CLUSTER_CENTER).to(DEVICE)
    state_size = 1024
    action_size = 1024
    torch.set_printoptions(profile="full")
    model = ActorCritic(state_size, action_size, 256, log_std_dev_init=-2).to(DEVICE)
    checkpoint = torch.load(A2CMODEL_PATH)
    model.load_state_dict(checkpoint['net'])

    if eva:
        test = torch.load('kmeans/bart/sentence_embedding_0_20.pth')
        # test = torch.load('kmeans/0.2_cluster_40_50/test_sentence_embedding_40_50.pth')
        label = torch.load('kmeans/bart/cluster_ids_x_60.pth')
        # label = torch.load('kmeans/0.2_cluster_40_50/test_cluster_ids_60.pth')
        print(label[1100:1200])
        test1 = test[1100:1200, :]
        test1 = test1.unsqueeze(dim=0)
        policy = model.get_greedy_action(test1.to(DEVICE))
        policy = policy.squeeze()
        id = []
        for i in policy:
            id.append(find_nearest_id(i, cluster_center).item())
        print(torch.tensor(id))
        similarity(label[1100:1200], torch.tensor(id))
    else:
        device='cuda:1'
        # CHECKPOINT_PATH = 'model/0/best_model_20.pth'
        encoder = Model().to(device)
        encoder = nn.DataParallel(encoder, device_ids=[1, 2])
        encoder = encoder.to(device)
        # checkpoint = torch.load(CHECKPOINT_PATH)
        # encoder.load_state_dict(checkpoint['model_state_dict'])
        
        encoder.eval()
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

        is_train = True
        if not is_train:
            with torch.no_grad():
                with open(training_path, 'r') as f:
                    sentence = f.readlines()
                    dim = 1024
                    # size = len(sentence)
                    data = torch.empty([0, dim], dtype=torch.float32, device=device)
                    i=0
                    for sen in tqdm(sentence):
                        i += 1
                        encode_input = tokenizer(sen, return_tensors='pt')
                        encode_output = encoder(**encode_input)
                        data = torch.cat((data, encode_output), 0)
            torch.save(data, 'dd_eva_bart_valid.pth')
            test = data
        test = torch.load(modelpth)
        test = test.unsqueeze(dim=0)
        policy = model.get_greedy_action(test)
        policy = policy.squeeze()
        id = []
        for i in policy:
            id.append(find_nearest_id(i, cluster_center).item())
        # print(torch.tensor(id[:100]))
        # print('------------------------------------------')
        # label = torch.load("kmeans/0.2_cluster_40_50/cluster_ids_x_60.pth")
        # print(label[1:201:2])
        # similarity(label[1:201:2], torch.tensor(id[:100]))
        return id

def inference_cluster(sentence, order):
    device='cuda:1'
    CHECKPOINT_PATH = 'model/0/best_model_50.pth'
    state_size = 1024
    action_size = 1024

    encoder = Model().to(device)
    encoder = nn.DataParallel(encoder, device_ids=[1, 2])
    encoder = encoder.to(device)
    checkpoint = torch.load(CHECKPOINT_PATH)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    # input = tokenizer(sentence, return_tensors='pt')
    sen_emb = encoder(**sentence)
    # id = kmeans_cluster_predict(sen_emb)

    model = ActorCritic(state_size, action_size, 256, log_std_dev_init=-2).to(DEVICE)
    checkpoint = torch.load(A2CMODEL_PATH)
    model.load_state_dict(checkpoint['net'])
    cluster_center = torch.load(CLUSTER_CENTER).to(DEVICE)
    policy = model.get_greedy_action(sen_emb)
    if order == "action":
        return policy.squeeze()
    elif order == "center":
        id = find_nearest_id(policy, cluster_center).item()
        return cluster_center[id]
