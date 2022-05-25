import numpy as np
from torch.nn import BatchNorm1d, Identity
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations


def setup_seed(seed):
    if seed == 100:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)


def mix_up(x_mix, mix_ratio):
    n = x_mix.shape[0]
    ratio = torch.ones(size=(n, 1)) * mix_ratio
    ratio_ = torch.ones(size=(n, 1)) * (1 - mix_ratio)
    ratio = torch.cat([ratio, ratio_], dim=1).to(x_mix.device)
    tmp_soft = x_mix * ratio.view(n, 2, 1)
    tmp_soft = torch.sum(tmp_soft, dim=1)
    return tmp_soft


def cal_stu_reward(model, h, y_soft):
    out = model.predict(h)
    loss = F.kl_div(out.log_softmax(dim=-1), y_soft, log_target=False, reduction='none').sum(dim=-1, keepdim=True)




class BaseSampler:
    def __init__(self, device, args):
        raise NotImplementedError

    def learn(self, model, data, y_soft, train_mask, warmup=None):
        pass


class OriginalSampler(BaseSampler):
    def __init__(self, device, args):
        self.device = device

    def sample(self, x):
        num_node, _ = x.shape
        reliable_soft = torch.ones(num_node, dtype=torch.float).view(-1, 1).to(self.device)
        return reliable_soft


class RandomSampler(BaseSampler):
    def __init__(self, device, args):
        self.device = device

    def sample(self, x):

        # Random
        num_node, _ = x.shape
        reliable_soft = torch.randint(2, (num_node, 1)).to(torch.float).to(self.device)
        # num_node.. 0,1

        return reliable_soft


class SinglePolicyGradientSampler(BaseSampler):
    def __init__(self, device, args):
        # Hyperparameters
        self.learning_rate = 0.005
        # self.start_entropy = 0.05 # Larger -> uniform
        self.start_entropy = 0.2 # Larger -> uniform
        self.end_entropy = 0.01 # End entropy
        self.decay = 30 # Linear
        self.num_updates = 5 # Iterations
        self.clip_norm = 30
        self.epochs = 20
        self.mix_ratio = args.mix_ratio
        self.add_soft = args.add_soft

        self.pos_pair, self.pos_soft, self.neg_pair, self.neg_soft = None, None, None, None

        self.entropy_coefficient = self.start_entropy
        if args.generate:
            self.generate = True
        else:
            self.generate = False
        self.pos_sample = args.pos_sample
        self.neg_sample = args.neg_sample

        self.device = device
        input_feat = args.hidden_mlp
        if self.add_soft:
            input_feat += args.num_class
        self.net = SinglePolicyNet(
            input_feat,
            2
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
        )

    def generate_data(self, val_mask, y_mask, y_soft_label, y_soft):
        if self.generate:
            self.pos_pair, self.pos_soft = generate_positive(val_mask, y_mask, y_soft_label, y_soft, self.pos_sample, self.mix_ratio)
            self.neg_pair, self.neg_soft = generate_negative(val_mask, y_mask, y_soft_label, y_soft, self.neg_sample, self.mix_ratio)
        else:
            if self.pos_pair == None:
                self.pos_pair, self.pos_soft = generate_positive(val_mask, y_mask, y_soft_label, y_soft, self.pos_sample, self.mix_ratio)
                self.neg_pair, self.neg_soft = generate_negative(val_mask, y_mask, y_soft_label, y_soft, self.neg_sample, self.mix_ratio)

    def sample(self, x, training=True, return_logits=False):
        num_node, dim = x.shape
        if training:
            self.net.train()
        else:
            self.net.eval()
        logits = self.net.forward(x)
        policy_logits = logits["policy"]
        if training:
            action = torch.multinomial(F.softmax(policy_logits, dim=-1), num_samples=1).squeeze()
        else:
            action = torch.argmax(policy_logits, dim=1).squeeze()

        if return_logits:
            return action, logits
        else:
            return action

    def learn(self, model, data, y_soft, train_mask, warmup=None):
        model.eval()
        y_soft_train = y_soft[train_mask]
        y_truth = data.y[train_mask]
        y_soft_label = torch.argmax(y_soft, dim=1)
        y_mask = (y_soft_label == data.y).to(torch.int64)

        y_pred = torch.argmax(y_soft_train, dim=-1)
        pred_mask = (y_pred == y_truth).to(torch.int64)

        with torch.no_grad():
            # h = model.encode(data.x[train_mask])
            h = model.encode(data.x)
            # h = torch.cat([h, y_soft_train], dim=1)
        h_x = h[train_mask]
        dim = h.shape[-1]

        self.generate_data(train_mask, y_mask, y_soft_label, y_soft)
        h_pos = h[self.pos_pair.view(-1)].view(-1, 2, dim)
        h_pos = mix_up(h_pos, self.mix_ratio)

        num_ori = h_x.shape[0]
        pos_num = self.pos_pair.shape[0]
        neg_num = self.neg_pair.shape[0]

        h_neg = h[self.neg_pair.view(-1)].view(-1, 2, dim)
        h_neg = mix_up(h_neg, self.mix_ratio)

        h_mix = torch.cat([h_x, h_pos, h_neg], dim=0)
        mix_soft = torch.cat((y_soft_train, self.pos_soft, self.neg_soft), dim=0)
        mix_mask = torch.cat([torch.ones(size=(pos_num, 1)), torch.zeros(size=(neg_num, 1))], dim=0).view(-1).to(y_soft.device)
        # cal_stu_reward(model, h_mix, mix_soft)
        if self.add_soft:
            h_mix = torch.cat([h_mix, mix_soft], dim=1)

        count = 0
        stats = {
            "policy_loss": [],
            "entropy_loss": [],
            "total_loss": [],
        }
        for epoch in range(self.epochs):
            action, logits = self.sample(h_mix, return_logits=True)
            reward_x = compute_reward(action[0:num_ori], pred_mask)
            reward_mix = compute_rewardmix(action[num_ori:], mix_mask)
            reward = torch.cat([reward_x, reward_mix])
            ones = torch.nonzero(reward).view(-1)
            # ones = torch.nonzero(reward + 1).view(-1)
            reward_ratio = ones.shape[0]/reward.shape[0]

            policy_logits = logits["policy"]

            # Normalize rewards
            reward = (reward - reward.mean()) / (reward.std() + 10 - 8)
            # Policy gradient
            policy_loss = compute_policy_loss(policy_logits, action, reward)

            # Entropy
            entropy_loss = compute_entropy_loss(policy_logits)

            if warmup is not None:
                total_loss = entropy_loss
            else:
                total_loss = policy_loss + entropy_loss * self.entropy_coefficient

            self.optimizer.zero_grad()
            total_loss.backward()
            # nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_norm)
            self.optimizer.step()

            print('### Epoch: {} reward_ratio: {} action: {} polich_loss: {} entropy_loss: {}'.format(epoch, reward_ratio, action.sum()/action.shape[0], policy_loss.item(), entropy_loss.item()))

            stats["policy_loss"].append(policy_loss.item())
            stats["entropy_loss"].append(entropy_loss.item())
            stats["total_loss"].append(total_loss.item())

            count += 1
            if warmup is None and count >= self.num_updates:
                break
            elif warmup is not None and count >= warmup:
                break

        stats = {key: np.mean(stats[key]) for key in stats}
        print(stats)

        if warmup is None:
            self.entropy_coefficient = max(self.entropy_coefficient - (self.start_entropy - self.end_entropy) / self.decay, self.end_entropy)


class SingleActorCriticeSampler(BaseSampler):
    def __init__(self, device, args):
        # Hyperparameters
        self.learning_rate = 0.01
        self.start_entropy = 0.05 # Larger -> uniform
        self.end_entropy = 0.01 # End entropy
        self.decay = 30 # Linear
        self.num_updates = 10 # Iterations
        self.clip_norm = 30
        self.value_coeficient = 0.5

        self.entropy_coefficient = self.start_entropy

        self.device = device
        self.net = SinglePolicyValueNet(
            args.hidden_channels,
            args.num_layers
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
        )

    def sample(self, h_edge_0, h_edge_1, training=True, return_logits=False):
        num_edges, max_hop, _ = h_edge_0.shape
        edge_features = torch.cat((h_edge_0[:,0].detach(), h_edge_1[:,0].detach()), dim=1)
        logits = self.net.forward(edge_features)
        policy_logits = logits["policy"]
        if training:
            action = torch.multinomial(F.softmax(policy_logits, dim=-1), num_samples=1).squeeze()
        else:
            action = torch.argmax(policy_logits, dim=1).squeeze()

        if return_logits:
            return action, logits
        else:
            return action

    def learn(self, model, data, pos_valid_edge, valid_index_loader, warmup=None):
        model.eval()
        with torch.no_grad():
            h = model(data.x, data.edge_index)

        count = 0
        stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "total_loss": [],
        }
        for perm in valid_index_loader:
            valid_edge = pos_valid_edge[perm].t()
            valid_edge_neg = generate_neg_sample(pos_valid_edge, data.num_nodes, data.x.device, perm.shape[0])

            pos_num, neg_num = valid_edge.shape[1], valid_edge_neg.shape[1]
            out, action, logits = model.compute_pred_and_logits(
                h,
                torch.cat((valid_edge, valid_edge_neg), dim=1),
                self,
            )
            policy_logits = logits["policy"]
            value_logits = logits["value"]
            out = out.squeeze()
            pos_out, neg_out = out[:pos_num], out[-neg_num:]
            pos_reward = torch.log(pos_out + 1e-15)
            neg_reward = torch.log(1 - neg_out + 1e-15)
            reward = torch.cat((pos_reward, neg_reward))
            advantage = reward - value_logits.detach().squeeze()

            # Normalize advantage
            advantage = (advantage - advantage.mean()) / (advantage.std() + 10-8)

            # Policy gradient
            policy_loss = compute_policy_loss(policy_logits, action, advantage)

            # Value loss
            value_loss = compute_value_loss(reward - value_logits) * self.value_coeficient

            # Entropy
            entropy_loss = compute_entropy_loss(policy_logits)

            if warmup is not None:
                total_loss = entropy_loss
            else:
                total_loss = policy_loss + value_loss + entropy_loss * self.entropy_coefficient

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_norm)
            self.optimizer.step()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy_loss"].append(entropy_loss.item())
            stats["total_loss"].append(total_loss.item())

            count += 1
            if warmup is None and count >= self.num_updates:
                break
            elif warmup is not None and count >= warmup:
                break

        stats = {key: np.mean(stats[key]) for key in stats}
        print(stats)

        if warmup is None:
            self.entropy_coefficient = max(self.entropy_coefficient - (self.start_entropy - self.end_entropy) / self.decay, self.end_entropy)


samplers = {
    "original": OriginalSampler,
    "random": RandomSampler,
    "single_policy_gradient": SinglePolicyGradientSampler,
    "single_actor_critic": SingleActorCriticeSampler
}


def get_sampler(name):
    if name not in samplers:
        return ValueError("Sampler not supported. Choices: {}".format(samplers.keys()))
    return samplers[name]


class SinglePolicyValueNet(nn.Module):
    def __init__(self, dim, num_layers):
        super(SinglePolicyValueNet, self).__init__()

        self.fc1 = nn.Linear(dim*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.policy_head = nn.Linear(32, num_layers)
        self.value_head = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        policy_logits = self.policy_head(x)
        value_logits = self.value_head(x)
        return {"policy": policy_logits, "value": value_logits}


class SinglePolicyNet(nn.Module):
    def __init__(self, dim, num_layers, dropout=0.5):
        super(SinglePolicyNet, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(dim, 128)
        self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(64, num_layers)
        self.norms = torch.nn.ModuleList()
        batch_norm_kwargs = {}

        for hidden_channels in [128, 64, 32]:
            norm = BatchNorm1d(hidden_channels, **batch_norm_kwargs)
            self.norms.append(norm)

    def forward(self, x):
        x = F.relu(self.norms[0](self.fc1(x)))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.norms[1](self.fc2(x)))
        # x = F.relu(self.norms[1](self.fc3(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc4(x)
        return {"policy": x}


def compute_policy_loss(policy_logits, action_id, reward):

    cross_entropy = F.nll_loss(
        F.log_softmax(policy_logits, dim=-1),
        target=action_id,
        reduction='none')

    loss = cross_entropy * reward
    loss = torch.mean(loss)

    return loss


def compute_reward(action, pred_mask):
    reward = (action == pred_mask).to(torch.float)
    # pos_reward = torch.nonzero(reward).view(-1)
    zero_index = reward - torch.ones_like(reward)
    # score = torch.stack([y_soft[i, y_pred[i]] for i in pos_reward])
    # reward[pos_reward] = score
    return reward
    # return reward + zero_index


def compute_rewardmix(action, pred_mask):
    reward = (action == pred_mask).to(torch.float)
    # pos_reward = torch.nonzero(reward).view(-1)
    zero_index = reward - torch.ones_like(reward)
    # score = torch.stack([y_soft[i, y_pred[i]] for i in pos_reward])
    # reward[pos_reward] = score
    return reward
    # return reward + zero_index

def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.mean(policy * log_policy)


def compute_value_loss(advantages):
    return 0.5 * torch.mean(advantages ** 2)


def generate_positive(train_mask, y_mask,  y_soft_label, y_soft, pos_sample, mix_ratio):
    x_index = torch.where(train_mask == True)[0]
    x_mask = y_mask[x_index]
    true_index = torch.where(x_mask == 1)[0]
    x_index = x_index[true_index]

    c = y_soft.shape[1]  # number of labels
    mixup = []
    y_train_soft = y_soft_label[x_index]
    for i in range(c):
        c_index = (y_train_soft == i).int()
        c_index = torch.nonzero(c_index).view(-1)
        cand_index = x_index[c_index].cpu().data.numpy()
        # total_mix = np.array([[i, j] for i in cand_index for j in cand_index if i != j])
        total_mix = np.array(list(combinations(cand_index, 2)))
        mixup_index = total_mix[np.random.choice(total_mix.shape[0], pos_sample, replace=False), :]
        mixup_index = torch.from_numpy(mixup_index)
        mixup.append(mixup_index)

    mixup = torch.cat(mixup, dim=0)
    n = mixup.shape[0]
    tmp_soft = y_soft[mixup.view(-1), :]
    tmp_soft = tmp_soft.view(-1, 2, c)
    ratio = torch.ones(size=(n, 1)) * mix_ratio
    ratio_ = torch.ones(size=(n, 1)) * (1 - mix_ratio)
    ratio = torch.cat([ratio, ratio_], dim=1).to(y_soft.device)
    tmp_soft = tmp_soft * ratio.view(n, 2, 1)
    tmp_soft = torch.sum(tmp_soft, dim=1)
    return mixup, tmp_soft


def generate_negative(train_mask, y_mask,  y_soft_label, y_soft, neg_sample, mix_ratio):
    x_index = torch.where(train_mask == True)[0]
    val_n = x_index.shape[0]
    x_mask = y_mask[x_index]
    false_index = torch.where(x_mask == 0)[0]
    x_index = x_index[false_index]
    c = y_soft.shape[1]  # number of labels
    cand_index = x_index.cpu().data.numpy()
    # total_mix = np.array([[i, j] for i in cand_index for j in cand_index if i != j])
    total_mix = np.array(list(combinations(cand_index, 2)))
    if total_mix.shape[0] > val_n:
        total_mix = total_mix[np.random.choice(total_mix.shape[0], neg_sample, replace=False), :]
    mixup = torch.from_numpy(total_mix)

    n = mixup.shape[0]
    tmp_soft = y_soft[mixup.view(-1), :]
    tmp_soft = tmp_soft.view(-1, 2, c)
    ratio = torch.ones(size=(n, 1)) * mix_ratio
    ratio_ = torch.ones(size=(n, 1)) * (1 - mix_ratio)
    ratio = torch.cat([ratio, ratio_], dim=1).to(y_soft.device)
    tmp_soft = tmp_soft * ratio.view(n, 2, 1)
    tmp_soft = torch.sum(tmp_soft, dim=1)
    return mixup, tmp_soft
