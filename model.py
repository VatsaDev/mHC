import math 
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect

config = {
    "n_embd": 128,         
    "n_head": 2,  # keep a minimum of two heads, but always aim for head_size = 128 at scale        
    "n_layer": 5,         
    "dropout": 0.05,         
    "vocab_size": 50257,
    "ctx_len": 1024, 
    "bias": False,
    "v_res": True,
    "streams": 4, # number of streams for the hypernetwork
}

class RoPE(nn.Module):

    def __init__(self, d_head): # rope changes qk after it is split by attn heads
        super().__init__()

        self.dim = d_head 
        self.ctx = config['ctx_len']

        # pre-compute the theta values and store them

        theta = 10000.0 ** (-2.0 * torch.arange(0, self.dim, 2).float() / self.dim)
        t = torch.arange(self.ctx, dtype=torch.float)

        # shapes t -> (ctx_len, 1), theta -> (1, dim/2) broadcast (ctx_len, dim/2)
        freqs = t.unsqueeze(1) * theta.unsqueeze(0)

        # complex number trick, (cos th + i * sin th)
        freq_cis = torch.polar(torch.ones_like(freqs), freqs)

        self.register_buffer('freq_cis', freq_cis)

    def forward(self, x):

        B, nh, T, hs = x.shape # input is B, nh, T, hs

        x_complex = torch.view_as_complex(x.float().reshape(B, nh, T, hs//2, 2)) # split into 2 groups 

        freq_cis = self.freq_cis[:T].view(1, 1, T, -1)

        x_rot_complex = x_complex * freq_cis

        x_rot = torch.view_as_real(x_rot_complex)
        x_out = x_rot.reshape(B, nh, T, hs)

        return x_out.type_as(x)

class MHA(nn.Module):

    def __init__(self):
        super().__init__()

        assert config['n_embd'] % config['n_head'] == 0

        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'], bias=config.get('bias', False))
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'], bias=config.get('bias', False))

        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])

        self.n_embd = config['n_embd']
        self.n_head = config['n_head']
        self.dropout = config['dropout']
        self.block_size = config['ctx_len']

        # qk norm and rope share the same size

        self.q_norm = nn.RMSNorm(self.n_embd//self.n_head) 
        self.k_norm = nn.RMSNorm(self.n_embd//self.n_head)

        # rope

        self.rope = RoPE(self.n_embd//self.n_head)

        # value residual (based off the resformer paper, referenced code from @cloneofsimo diffusion speedrun)

        if config["v_res"]:
            self.lamb1 = nn.Parameter(torch.tensor(0.5))
        else:
            self.lamb1 = 1.0

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(self.block_size, self.block_size))
                                        .view(1, 1, self.block_size, self.block_size))

    def forward(self, x, v1=None):
        B, T, C = x.size() # bs, ctx_len, n_embd

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # qk norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # qk rope
        q = self.rope(q) 
        k = self.rope(k)

        if v1 is None:
            v1 = v

        # value res
        lamb1 = torch.sigmoid(self.lamb1)
        v = lamb1 * v + (1-lamb1) * v1.view_as(v) 

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # flash_attn
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True, scale=(1.0 / k.size(-1))) # MuP 1/d scale, not 1/root(d)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / k.size(-1)) # MuP scaling
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y, v1

# Swiglu replaces MLP 

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        n = int((8/3) * config['n_embd'])
        appr = (n + 63) & ~(63) # make it a multiple of 64

        # combine gate and value
        self.gate_value_proj = nn.Linear(config['n_embd'], 2 * appr, bias=False) # Llama uses no bias
        self.linear_out = nn.Linear(appr, config['n_embd'], bias=False)
        self.silu = nn.SiLU()

        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        
        # project input to 2 * appr, split the tensor in half, gate and val
        gate_value = self.gate_value_proj(x)
        gate, value = torch.chunk(gate_value, 2, dim=-1)

        x = self.silu(gate) * value
        x = self.linear_out(x)
        x = self.dropout(x)

        return x

class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config['n_embd'])
        self.attn = MHA()
        self.ln_2 = nn.RMSNorm(config['n_embd'])
        self.mlp = MLP()

        self.branch_scale = 1.0 / math.sqrt(config['n_layer']) # MuP residual rule a/root(L), a = 1.0 here

    def forward(self, x, v1=None):
        
        attn_out, v1 = self.attn(self.ln_1(x), v1)

        x = x + self.branch_scale * attn_out
        x = x + self.branch_scale * self.mlp(self.ln_2(x))
        return x, v1

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.block_size = config['ctx_len']

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['n_embd']), # tok embd
            wpe = nn.Embedding(self.block_size, config['n_embd']), # pos embd
            drop = nn.Dropout(config['dropout']),
            h = nn.ModuleList([Block() for _ in range(config['n_layer'])]) 
        ))

        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        self.h_scale = 1/config['n_embd'] # muP
        
        # weight-tying
        self.lm_head.weight = self.transformer.wte.weight

        # init all weights
        self.apply(self._init_weights)
        
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config['n_layer']))

        print(f"Model initialized with {sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters")


    def _init_weights(self, module):
    
        if isinstance(module, nn.Linear):
            if module.out_features == config['vocab_size']: # LM_head check
                torch.nn.init.zeros_(module.weight) # MuP at all-zero LM head better stability
            else:
                # MuP std = 1 / sqrt(fan_in)
                fan_in = module.weight.size(1)
                std = 1.0 / math.sqrt(fan_in)
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)

        if isinstance(module, nn.Embedding):
            # MuP for Embeddings, Constant variance std=0.02 is standard
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        
        device = idx.device
        b, t = idx.size()

        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        
        x = self.transformer.drop(tok_emb + pos_emb)
        
        v1 = None
        for block in self.transformer.h:
            x, v1 = block(x, v1)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward final layer norm and lm_head on the very last position
            # Note: This optimization is not needed if using generate method below
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        logits = logits * self.h_scale 
        logits = 30.0 * torch.tanh(logits / 30.0) # softcap logits at 30, gemma style

        return logits, loss

    @torch.no_grad() 
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] 
            logits = logits * self.h_scale 
            logits = 30.0 * torch.tanh(logits / 30.0) # gemma softcap
            logits = logits / temperature
            
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):

        # MuP, kept constant for Muon, didnt show much diff, maybe useful if you are extremely narrow
        # Also used as 1/(layer_mult) in AdamW, but the rule is only for RMSnorm in my code, so its basically useless
        # could try later, mostly not worth it

        depth_scale = 1.0 # EDIT later, micro model is 16 deep, l_mult = depth_big/depth_small 

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        muon_groups = []
        adam_groups = []

        for name, p in param_dict.items():
            if p.dim() < 2:
                
                # Norms/Biases
                is_global = any(k in name for k in ["ln_f", "embed", "head"])
                p_lr = learning_rate if is_global else (learning_rate * depth_scale)
            
                adam_groups.append({
                    'params': [p], 
                    'lr': p_lr, 
                    'weight_decay': 0.0
                })

            elif any(k in name for k in ["embed", "token", "wte", "head", "output"]):
            
                #  MuP rule constant LR
                adam_groups.append({
                    'params': [p], 
                    'lr': learning_rate, 
                    'weight_decay': weight_decay
                })

            else:
            
                # MuP and Muon Spectral Scaling
                dout, din = p.shape[0], p.shape[1]
                spectral_scaling = math.sqrt(max(1, dout / din))
            
                # Combine Muon base LR (5*LR) * spectral * depth
                p_lr = (5 * learning_rate) * spectral_scaling * depth_scale
            
                muon_groups.append({
                    'params': [p], 
                    'lr': p_lr, 
                    'weight_decay': weight_decay
                })

        # Prepare extra args (fused AdamW, etc)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type.startswith('cuda')
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer_adam = torch.optim.AdamW(adam_groups, betas=betas, **extra_args)
        optimizer_muon = torch.optim.Muon(muon_groups)

        return [optimizer_muon, optimizer_adam]
    
    def estimate_mfu(self, fwdbwd_per_iter, dt): # completely need to change this function, its too old
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters()) # Use actual parameter count
        cfg = self.transformer.h[0].attn # Get config from an attention block instance if needed
                                         # Or better, access directly if stored on self or use model.config
                                         # For now, let's assume model.config is globally updated
        L, H, Q, T = config['n_layer'], config['n_head'], config['n_embd']//config['n_head'], self.block_size

        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak FLOPS
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        # Adjust peak flops based on hardware if necessary
        # A100 = 312 TFLOPS (BF16), H100 = 989 TFLOPS (FP16) / 495 TFLOPS (TF32)
        # T4 = 65 TFLOPS (FP16)
        flops_promised = 312e12 # A100 BF16 peak flops
        if torch.cuda.is_available():
             dev_prop = torch.cuda.get_device_properties(torch.cuda.current_device())
             if dev_prop.major >= 8: # Ampere or newer
                 # Use BF16 peak for A100
                 flops_promised = 312e12
                 if dev_prop.major >= 9: # Hopper
                      # Using FP16 peak for H100, adjust if using TF32
                      flops_promised = 989e12
             elif dev_prop.major == 7: # Volta/Turing (like T4)
                 flops_promised = 65e12 # T4 FP16 peak

        mfu = flops_achieved / flops_promised
        return mfu

