import torch
import torch.nn as nn
import torch.nn.functional as F

from pfunet.modules import (
    build_pf_obs,
    rasterize_obs,
    readout,
    UNet2D,
    MachineAttention,
    MLP,
    Obs,
)

class SchedulingModel(nn.Module):
    """
    Priority-Field U-Net (PF-UNet) Policy & Value Network.
    Fully drop-in compatible with ReSched's SchedulingModel interface.
    """
    def __init__(self,
                 canvas_hw=(64, 64),
                 c_in: int = 8,
                 unet_channels=(32, 64, 128, 256),
                 blocks_per_level: int = 2,
                 gn_groups: int = 8,
                 attn_layers: int = 2,
                 attn_heads: int = 4,
                 attn_ff: int = 512,
                 head_hidden=(128, 64),
                 critic_hidden: int = 128,
                 raw_dim: int = 7,
                 rl_algo: str = 'ppo',
                 **kwargs):
        super().__init__()
        self.H, self.W = canvas_hw
        self.rl_algo = rl_algo.lower()
        D = unet_channels[-1]
        
        self.unet = UNet2D(c_in, unet_channels, blocks_per_level, gn_groups)
        self.mach_attn = MachineAttention(D, attn_heads, attn_layers, attn_ff)
        self.film = nn.Linear(D, 2 * D)
        
        # Readout dimension: d0 channels + d1 channels + raw_dim + D
        rd = unet_channels[0] + unet_channels[1] + raw_dim + D  # 32 + 64 + 7 + 256 = 359
        self.head = MLP([rd, *head_hidden, 1])
        if self.rl_algo == 'reinforce':
            self.critic = None
        else:
            self.critic = MLP([D, critic_hidden, 1])
        
        self.decode_type = None

    def set_decode_type(self, decode_type: str):
        self.decode_type = decode_type

    def core(self, obs: Obs):
        B, M, J = obs.mask_pair.shape
        # 1. Splat to continuous canvas [B*M, 8, H, W]
        canvas = rasterize_obs(obs, self.H, self.W)
        
        # 2. U-Net encode to bottleneck skips and h [B*M, 256, H/8, W/8]
        skips, h = self.unet.encode(canvas)
        
        # 3. Machine Attention on pooled bottleneck [B, M, D]
        g = self.mach_attn(h.mean((2, 3)).view(B, M, -1))
        
        # 4. FiLM conditioning from machine representation g onto bottleneck h
        gam, bet = self.film(g).chunk(2, -1)
        h = h * (1.0 + gam.reshape(B * M, -1, 1, 1)) + bet.reshape(B * M, -1, 1, 1)
        
        # 5. U-Net decode back to multi-scale feature maps
        d0, d1 = self.unet.decode(h, skips)[:2]  # d0: [B*M, 32, H, W], d1: [B*M, 64, H/2, W/2]
        
        # 6. Readout at candidate coordinates (x, y)
        f0 = readout(d0, obs.xy, B, M)  # [B, M, J, 32]
        f1 = readout(d1, obs.xy, B, M)  # [B, M, J, 64]
        gk = g[:, :, None, :].expand(B, M, J, -1)  # [B, M, J, D]
        
        # 7. Action logits and critic value
        features = torch.cat([f0, f1, obs.raw, gk], dim=-1)  # [B, M, J, 359]
        score = self.head(features).squeeze(-1)  # [B, M, J]
        logits = score.masked_fill(~obs.mask_pair, float('-inf'))
        value = self.critic(g.mean(1)).squeeze(-1) if self.critic is not None else None
        
        return logits, value

    def forward(self, state, action=None, return_all=None):
        """
        Forward pass matching ReSched SchedulingModel interface.
        Supports both PPO (4 outputs) and REINFORCE (2 outputs).
        """
        dev = next(self.parameters()).device
        state = [s.to(dev) if isinstance(s, torch.Tensor) else s for s in state]
        if action is not None:
            action = [a.to(dev) if isinstance(a, torch.Tensor) else a for a in action]

        (min_dur, dur, o_at, m_at, dep_o, dep_m, act_mask, rev_pos) = state
        B, N, M = dur.shape
        
        obs = build_pf_obs(state)
        logits, value = self.core(obs)  # [B, M, J], [B] or None
        J = logits.shape[-1]
        
        # Map logits [B, M, J] -> probability distribution in env space [B, N_rem, M]
        flat = logits.permute(0, 2, 1).reshape(B, J * M)  # order (j, k)
        p_jm = torch.softmax(flat, dim=-1).view(B, J, M) * obs.has_head[..., None]
        
        prob = dur.new_zeros(B, N, M)
        prob.scatter_add_(1, obs.head_idx.clamp_min(0)[..., None].expand(B, J, M), p_jm)
        
        # Exact action selection logic matching ReSched SchedulingModel
        if self.decode_type == 'sampling':
            samples = torch.multinomial(input=prob.view(B, -1), num_samples=1)
            seq_id = (samples // prob.shape[-1]).squeeze(-1)
            machine_id = (samples % prob.shape[2]).squeeze(-1)
        elif self.decode_type == 'greedy':
            samples = torch.argmax(prob.view(B, -1), dim=-1, keepdim=True)
            seq_id = (samples // prob.shape[-1]).squeeze(-1)
            machine_id = (samples % prob.shape[2]).squeeze(-1)
        elif self.decode_type == 'teacher_forcing':
            assert action is not None, "action must be provided for teacher_forcing"
            seq_id, machine_id = action
            if isinstance(seq_id, int):
                seq_id = torch.tensor([seq_id], device=dur.device)
            if isinstance(machine_id, int):
                machine_id = torch.tensor([machine_id], device=dur.device)
        else:
            raise ValueError(f"Unknown decode type: {self.decode_type}")
            
        prob_selected = prob[torch.arange(B, device=dur.device), seq_id, machine_id]
        
        if return_all is None:
            return_all = (self.rl_algo == 'ppo')
            
        if return_all:
            return [seq_id, machine_id], prob_selected, value, prob
        else:
            return [seq_id, machine_id], prob_selected
