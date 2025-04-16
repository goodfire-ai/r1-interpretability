import torch
from math import prod


class BatchTopKTiedSAE(torch.nn.Module):
    def __init__(
        self,
        d_in,
        d_hidden,
        k,
        device,
        dtype,
        normalization_constant: int = 1,
        tiebreaker_epsilon: float = 1e-6,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = k
        self.norm_constant = normalization_constant
        W_mat = torch.randn((d_in, d_hidden))
        W_mat = 0.1 * W_mat / torch.linalg.norm(W_mat, dim=0, ord=2, keepdim=True)
        self.W = torch.nn.Parameter(W_mat)
        self.b_enc = torch.nn.Parameter(torch.zeros(self.d_hidden))
        self.b_dec = torch.nn.Parameter(torch.zeros(self.d_in))
        self.device = device
        self.dtype = dtype
        self.tiebreaker_epsilon = tiebreaker_epsilon
        self.tiebreaker = torch.linspace(0, tiebreaker_epsilon, d_hidden)
        self.to(self.device, self.dtype)

    def encoder_pre(self, x):
        return x @ self.W + self.b_enc

    def encode(self, x, tiebreak=False):
        f = torch.nn.functional.relu(self.encoder_pre(x))
        return self._batch_topk(f, self.k, tiebreak=tiebreak)

    def _batch_topk(self, f, k, tiebreak=False):
        if tiebreak:  # break ties in feature order for determinism
            f += self.tiebreaker.broadcast_to(f)
        *input_shape, _ = (
            f.shape
        )  # handle higher-dim tensors (e.g. from sequence input)
        numel = k * prod(input_shape)
        f_topk = torch.topk(f.flatten(), numel, dim=-1)
        f_topk = (
            torch.zeros_like(f.flatten())
            .scatter(-1, f_topk.indices, f_topk.values)
            .reshape(f.shape)
        )
        return f_topk

    def decode(self, f):
        return f @ self.W.T + self.b_dec

    def forward(self, x):
        x = x * self.norm_constant
        f = self.encode(x)
        return self.decode(f), f


def load_r1_sae(file_path, device: str = "cpu", k: int = 128, norm: float = 1.0):
    state_dict = torch.load(file_path, weights_only=True, map_location=device)
    sae = BatchTopKTiedSAE(d_in=7168, d_hidden=7168 * 4, dtype=torch.bfloat16, k=k, normalization_constant=norm)
    sae.load_state_dict(state_dict)
    return sae

def load_math_sae(file_path, device: str = "cpu"):
    return load_r1_sae(file_path, device, k=128, norm=1.0)

def load_logic_sae(file_path, device: str = "cpu"):
    return load_r1_sae(file_path, device, k=64, norm=13.081755638122559)
