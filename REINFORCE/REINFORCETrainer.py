import numpy as np
import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils import AverageMeter
from SchedulingModel import build_model_input


# REINFORCE Trainer
class Trainer:
    def __init__(self, model, optimizer_params, training_params, device=None):
        self.training_params = training_params
        self.optimizer_params = optimizer_params
        self.model = model
        self.device = device or next(model.parameters()).device
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        self.logger = lambda x: print(x)

    def train(self, env, first_epoch=False):
        score_am, loss_am = AverageMeter(), AverageMeter()

        self.model.train()
        self.model.set_decode_type('sampling')
        loop_cnt = 0

        episode = 0
        while episode < self.training_params['episode']:
            remaining = self.training_params['episode'] - episode
            bs = min(self.training_params['batch_size'], remaining)
            env.generate_data(bs)

            # get dynamic data
            state = env.reset_state()

            state_list, action_list = [], []
            reward_list, scaler_list = [], []
            done = False
            self.model.eval()
            self.model.set_decode_type('sampling')
            with torch.no_grad():
                while not done:
                    state, scaler = build_model_input(state, env)
                    state = [s.to(self.device) if isinstance(s, torch.Tensor) else s for s in state]
                    action, prob = self.model(state)

                    state_list.append([s.cpu() if isinstance(s, torch.Tensor) else s for s in state])
                    action_list.append((action[0].cpu(), action[1].cpu()))

                    state, reward, done = env.step(action)

                    reward_list.append(reward)
                    scaler_list.append(scaler)

            returns = self.get_return(reward_list)
            returns = np.array(returns) / np.array(scaler_list)
            returns = torch.tensor(returns, dtype=torch.float, device=self.device)

            # shape: (num_steps, batch)
            advantage = returns - returns.mean(dim=1, keepdim=True)
            advantage_t = advantage.T

            # Recompute step-by-step with gradients (Gradient Checkpointing)
            self.model.train()
            self.model.set_decode_type('teacher_forcing')
            self.optimizer.zero_grad()
            total_steps = len(state_list)
            batch_loss = 0.0

            for step in range(total_steps):
                st = [s.to(self.device) if isinstance(s, torch.Tensor) else s for s in state_list[step]]
                act = (action_list[step][0].to(self.device), action_list[step][1].to(self.device))
                _, prob = self.model(st, action=act)
                log_prob = prob.log()
                adv_step = advantage_t[:, step]
                loss_step = - torch.sum(adv_step * log_prob) / bs
                loss_step.backward()
                batch_loss += loss_step.item()

            self.optimizer.step()
            loss_am.update(batch_loss, bs)

            # Score
            ###############################################
            score_mean = state.finish_time.max(axis=1).max(axis=1).mean()
            score_am.update(score_mean.item(), bs)

            episode += bs

            if first_epoch and (loop_cnt <= 10):
                self.logger('Epoch 1: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                            .format(episode, self.training_params['episode'],
                                    100. * episode / self.training_params['episode'],
                                    score_am.avg, loss_am.avg))
            loop_cnt += 1
        self.scheduler.step()

        return score_am.avg, loss_am.avg, self.model.state_dict()

    def get_return(self, reward_list):
        discount_factor = self.training_params['discount_factor']
        return_list = []
        g = 0
        for reward in reversed(reward_list):
            g = g * discount_factor + reward
            return_list.insert(0, g)

        return return_list

    def set_logger(self, logger):
        self.logger = logger

    def load_checkpoint(self, optimizer_params, scheduler_params, path):
        try:
            self.optimizer.load_state_dict(optimizer_params)
            self.logger('Optimizer loaded from {}'.format(path))
        except KeyError:
            self.logger('Saved Optimizer can not be used and new Optimizer is used instead')

        try:
            self.scheduler.load_state_dict(scheduler_params)
            self.logger('Scheduler loaded from {}'.format(path))
        except KeyError:
            self.logger('Saved Scheduler can not be used and new Scheduler is used instead')
