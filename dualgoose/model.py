"""Tiny denoiser stack that is ready for toy training."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .mixers import BidirectionalRWKVMixer


def timestep_features(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(0.0, math.log(1000.0), half, device=t.device, dtype=t.dtype)
    )
    angles = t[:, None] * freqs[None, :]
    features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if features.shape[-1] < dim:
        features = torch.cat([features, t[:, None]], dim=-1)
    return features


class TinyDualGooseBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        direction: str = "bidirectional",
        merge: str = "film",
        use_rank: bool = True,
        scan_backend: str = "reference",
        recurrence: str = "rwkv7",
        short_conv: int = 0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = BidirectionalRWKVMixer(
            d_model,
            direction=direction,
            merge=merge,
            use_rank=use_rank,
            scan_backend=scan_backend,
            recurrence=recurrence,
            short_conv=short_conv,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x), t)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyDualGooseDenoiser(nn.Module):
    """Small MDLM denoiser used for synthetic validation."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        *,
        d_model: int = 64,
        n_layers: int = 2,
        direction: str = "bidirectional",
        merge: str = "film",
        use_rank: bool = True,
        scan_backend: str = "reference",
        recurrence: str = "rwkv7",
        short_conv: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.scan_backend = scan_backend
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_length, d_model)
        self.time_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            [
                TinyDualGooseBlock(
                    d_model,
                    direction=direction,
                    merge=merge,
                    use_rank=use_rank,
                    scan_backend=scan_backend,
                    recurrence=recurrence,
                    short_conv=short_conv,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.max_length:
            raise ValueError("sequence length exceeds max_length")
        if t.ndim == 0:
            t = t.expand(tokens.shape[0])
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)[None, :, :]
        x = x + self.time_proj(timestep_features(t.to(x.dtype), x.shape[-1]))[:, None, :]
        for block in self.blocks:
            x = block(x, t)
        return self.lm_head(self.norm(x))


def _rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary position embedding on (B, H, L, D) with D even (rotate-half convention)."""
    d = x.shape[-1]
    half = d // 2
    inv = base ** (-torch.arange(half, device=x.device, dtype=x.dtype) / half)
    ang = torch.outer(torch.arange(x.shape[-2], device=x.device, dtype=x.dtype), inv)  # (L, half)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class TinyAttentionBlock(nn.Module):
    """Small bidirectional attention block for DiT-style toy controls."""

    def __init__(self, d_model: int, *, n_heads: int = 4, rope: bool = False):
        super().__init__()
        self.rope = rope
        if rope and (d_model // n_heads) % 2 != 0:
            raise ValueError("rope requires an even head dimension")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # adaLN-Zero: norms carry no affine; per-block shift/scale/gate come from the
        # timestep embedding (this is what makes it a real DiT block — the previous
        # version discarded t entirely and collapsed to the frequency baseline).
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.adaln[-1].weight)  # zero-init → block is identity at start
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1[:, None, :]) + shift1[:, None, :]
        batch, length, width = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        if self.rope:
            # rotary positions are relative -> position-generalizes across shifted layouts
            # (the failure mode of learned absolute pos_emb under the moving-table curriculum)
            q, k = _rope(q), _rope(k)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        x = x + gate1[:, None, :] * self.out(y)
        h = self.norm2(x) * (1 + scale2[:, None, :]) + shift2[:, None, :]
        x = x + gate2[:, None, :] * self.ffn(h)
        return x


class TinyAttentionDenoiser(nn.Module):
    """Small bidirectional attention denoiser used as a DiT-style baseline."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        *,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        rope: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.rope = rope
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # with rotary positions the learned absolute table is dropped entirely
        self.pos_emb = None if rope else nn.Embedding(max_length, d_model)
        self.time_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            [TinyAttentionBlock(d_model, n_heads=n_heads, rope=rope) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.max_length:
            raise ValueError("sequence length exceeds max_length")
        if t.ndim == 0:
            t = t.expand(tokens.shape[0])
        x = self.token_emb(tokens)
        if self.pos_emb is not None:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            x = x + self.pos_emb(positions)[None, :, :]
        cond = self.time_proj(timestep_features(t.to(x.dtype), x.shape[-1]))
        for block in self.blocks:
            x = block(x, cond)
        return self.lm_head(self.norm(x))


class TinySelectiveSSM(nn.Module):
    """Small diagonal selective SSM used as a Mamba-style toy mixer."""

    def __init__(self, d_model: int, *, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
        )
        self.param_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u_conv = F.pad(u.transpose(1, 2), (self.kernel_size - 1, 0))
        u = F.silu(self.conv(u_conv).transpose(1, 2))
        delta, b, c = self.param_proj(u).chunk(3, dim=-1)
        delta = torch.sigmoid(delta)
        b = torch.tanh(b)
        c = torch.tanh(c)

        state = torch.zeros(x.shape[0], x.shape[-1], device=x.device, dtype=x.dtype)
        outputs = []
        for idx in range(x.shape[1]):
            state = (1.0 - delta[:, idx]) * state + delta[:, idx] * b[:, idx] * u[:, idx]
            outputs.append(c[:, idx] * state)
        y = torch.stack(outputs, dim=1) * torch.sigmoid(z)
        return self.out_proj(y)


class TinyMambaBlock(nn.Module):
    """Bidirectional Mamba-style block for toy diffusion baselines."""

    def __init__(self, d_model: int, *, kernel_size: int = 3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.fwd = TinySelectiveSSM(d_model, kernel_size=kernel_size)
        self.bwd = TinySelectiveSSM(d_model, kernel_size=kernel_size)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        del t
        h = self.norm1(x)
        y_fwd = self.fwd(h)
        y_bwd = self.bwd(h.flip(1)).flip(1)
        x = x + 0.5 * (y_fwd + y_bwd)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyMambaDenoiser(nn.Module):
    """Small bidirectional Mamba-style denoiser used as a toy baseline.

    Pure-PyTorch diagonal selective SSM (no mamba_ssm/CUDA) — the controllable stand-in for
    the official Mamba-2 arm on GPUs where mamba_ssm won't build, and the fallback vehicle for
    the conv-decomposition (kernel_size=1 neutralizes the short causal conv). See
    docs/armed_gdn_experiment.md."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        *,
        d_model: int = 64,
        n_layers: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_length, d_model)
        self.time_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            [TinyMambaBlock(d_model, kernel_size=kernel_size) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.max_length:
            raise ValueError("sequence length exceeds max_length")
        if t.ndim == 0:
            t = t.expand(tokens.shape[0])
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)[None, :, :]
        x = x + self.time_proj(timestep_features(t.to(x.dtype), x.shape[-1]))[:, None, :]
        for block in self.blocks:
            x = block(x, t)
        return self.lm_head(self.norm(x))


class TinyLocalAttention(nn.Module):
    """Sliding-window bidirectional attention for sparse hybrid controls."""

    def __init__(
        self,
        d_model: int,
        *,
        n_heads: int = 4,
        window_size: int = 64,
    ):
        super().__init__()
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if window_size < 0:
            raise ValueError("window_size must be non-negative")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        radius = min(self.window_size, max(length - 1, 0))
        if radius >= length - 1:
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            window = 2 * radius + 1
            k_padded = F.pad(k, (0, 0, radius, radius))
            v_padded = F.pad(v, (0, 0, radius, radius))
            k_windows = k_padded.unfold(2, window, 1).movedim(-1, -2)
            v_windows = v_padded.unfold(2, window, 1).movedim(-1, -2)
            scores = (q.unsqueeze(-2) * k_windows).sum(dim=-1)
            scores = scores * (self.head_dim**-0.5)

            offsets = torch.arange(-radius, radius + 1, device=x.device)
            positions = torch.arange(length, device=x.device)[:, None] + offsets[None, :]
            valid = (positions >= 0) & (positions < length)
            scores = scores.masked_fill(
                ~valid[None, None, :, :],
                torch.finfo(scores.dtype).min,
            )
            weights = torch.softmax(scores, dim=-1)
            y = torch.matmul(weights.unsqueeze(-2), v_windows).squeeze(-2)

        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.out(y)


class TinyDiffuMambaHBlock(nn.Module):
    """Mamba-style block plus sparse local attention for hybrid controls."""

    def __init__(
        self,
        d_model: int,
        *,
        n_heads: int = 4,
        attention_window: int = 64,
    ):
        super().__init__()
        self.norm_ssm = nn.LayerNorm(d_model)
        self.fwd = TinySelectiveSSM(d_model)
        self.bwd = TinySelectiveSSM(d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = TinyLocalAttention(
            d_model,
            n_heads=n_heads,
            window_size=attention_window,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        del t
        h = self.norm_ssm(x)
        y_fwd = self.fwd(h)
        y_bwd = self.bwd(h.flip(1)).flip(1)
        x = x + 0.5 * (y_fwd + y_bwd)
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x


class TinyDiffuMambaHDenoiser(nn.Module):
    """Small DiffuMamba-H-style hybrid denoiser for toy training."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        *,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        attention_window: int = 64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_length, d_model)
        self.time_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            [
                TinyDiffuMambaHBlock(
                    d_model,
                    n_heads=n_heads,
                    attention_window=attention_window,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.max_length:
            raise ValueError("sequence length exceeds max_length")
        if t.ndim == 0:
            t = t.expand(tokens.shape[0])
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)[None, :, :]
        x = x + self.time_proj(timestep_features(t.to(x.dtype), x.shape[-1]))[:, None, :]
        for block in self.blocks:
            x = block(x, t)
        return self.lm_head(self.norm(x))


def _load_mamba2_class():
    try:
        from mamba_ssm import Mamba2
    except ImportError as exc:
        raise RuntimeError(
            "mamba_ssm is required for TinyMamba2Denoiser; install the optional "
            "Mamba dependencies before selecting the mamba2 baseline"
        ) from exc
    return Mamba2


def mamba2_default_headdim(d_model: int, expand: int = 2) -> int:
    d_inner = expand * d_model
    if d_inner < 8:
        raise ValueError("expand * d_model must be at least 8 for Mamba2")
    if d_inner % 8 != 0:
        raise ValueError("expand * d_model must be divisible by 8 for Mamba2")
    return d_inner // 8


class TinyMamba2Block(nn.Module):
    """Bidirectional wrapper around the official mamba_ssm Mamba2 module."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int | None = None,
        chunk_size: int = 64,
    ):
        super().__init__()
        if d_state % 4 != 0:
            raise ValueError("d_state must be divisible by 4 for causal-conv strides")
        headdim = headdim or mamba2_default_headdim(d_model, expand)
        d_inner = expand * d_model
        if d_inner % headdim != 0:
            raise ValueError("headdim must divide expand * d_model")
        if (d_inner // headdim) % 8 != 0:
            raise ValueError("Mamba2 head count must be divisible by 8 on this path")
        mamba2_cls = _load_mamba2_class()
        kwargs = {
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "headdim": headdim,
            "chunk_size": chunk_size,
            "use_mem_eff_path": False,
        }
        self.norm1 = nn.LayerNorm(d_model)
        self.fwd = mamba2_cls(d_model, **kwargs)
        self.bwd = mamba2_cls(d_model, **kwargs)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        del t
        h = self.norm1(x)
        y_fwd = self.fwd(h)
        y_bwd = self.bwd(h.flip(1)).flip(1)
        x = x + 0.5 * (y_fwd + y_bwd)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyMamba2Denoiser(nn.Module):
    """Small denoiser backed by official Mamba2 blocks for baseline probes."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        *,
        d_model: int = 64,
        n_layers: int = 2,
        d_state: int = 64,
        chunk_size: int = 64,
        d_conv: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_length, d_model)
        self.time_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            [
                TinyMamba2Block(
                    d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    chunk_size=chunk_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.max_length:
            raise ValueError("sequence length exceeds max_length")
        if t.ndim == 0:
            t = t.expand(tokens.shape[0])
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)[None, :, :]
        x = x + self.time_proj(timestep_features(t.to(x.dtype), x.shape[-1]))[:, None, :]
        for block in self.blocks:
            x = block(x, t)
        return self.lm_head(self.norm(x))


class TinyMamba2RefSSM(nn.Module):
    """Pure-PyTorch reference of the Mamba-2 SSD recurrence with an explicit d_state dimension.

    A DIAGONAL (per-head scalar) selective SSM whose state h ∈ R^{nheads x headdim x d_state} has
    d_inner*d_state elements — so it can be STATE-MATCHED to the rank-1 cells (d_state=16 at
    d_model=32 => 64*16 = 1024, matching a d_key=32 delta cell's 32^2). No Triton / mamba_ssm, so it
    runs on any GPU incl. Pascal (GTX 1070, sm_61) and CPU — the state-matched Mamba-2 comparator for
    the fair-fight P1 where the fused official kernel can't run. NOT rank-1: this is the diagonal
    control. `kernel_size` toggles the short causal conv (1 = neutralized, for the conv decomposition).
    See docs/armed_gdn_experiment.md.
    """

    def __init__(self, d_model, *, expand=2, headdim=None, d_state=16, kernel_size=4):
        super().__init__()
        d_inner = expand * d_model
        headdim = headdim or mamba2_default_headdim(d_model, expand)
        if d_inner % headdim != 0:
            raise ValueError("headdim must divide expand * d_model")
        self.d_inner, self.headdim = d_inner, headdim
        self.nheads = d_inner // headdim
        self.d_state, self.kernel_size = d_state, kernel_size
        conv_ch = d_inner + 2 * d_state  # conv mixes (x, B, C), as in Mamba-2
        self.in_proj = nn.Linear(d_model, 2 * d_inner + 2 * d_state + self.nheads)  # z, x, B, C, dt
        self.conv = nn.Conv1d(conv_ch, conv_ch, kernel_size=kernel_size, groups=conv_ch)
        self.A_log = nn.Parameter(torch.zeros(self.nheads))   # A = -exp(A_log) < 0 -> decay in (0,1)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))        # skip connection
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        batch, length, _ = u.shape
        z, xbc, dt = torch.split(
            self.in_proj(u), [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim=-1
        )
        xbc = xbc.transpose(1, 2)
        xbc = self.conv(F.pad(xbc, (self.kernel_size - 1, 0)))  # left-pad => strictly causal
        xbc = F.silu(xbc.transpose(1, 2))
        x, b_mat, c_mat = torch.split(xbc, [self.d_inner, self.d_state, self.d_state], dim=-1)
        out_dtype = u.dtype
        # recurrence in fp32 (the state is precision-sensitive), like the other reference cells
        with torch.autocast(device_type=u.device.type, enabled=False):
            x = x.float().view(batch, length, self.nheads, self.headdim)
            b_mat, c_mat = b_mat.float(), c_mat.float()
            dt = F.softplus(dt.float() + self.dt_bias.float())            # (B, L, H) selective
            dA = torch.exp(dt * (-torch.exp(self.A_log.float())))         # (B, L, H) in (0, 1)
            h = torch.zeros(batch, self.nheads, self.headdim, self.d_state, device=u.device)
            outs = []
            for ti in range(length):
                dBx = (dt[:, ti].unsqueeze(-1) * x[:, ti]).unsqueeze(-1) * b_mat[:, ti].view(
                    batch, 1, 1, self.d_state
                )
                h = dA[:, ti].view(batch, self.nheads, 1, 1) * h + dBx    # diagonal decay + input
                outs.append((h * c_mat[:, ti].view(batch, 1, 1, self.d_state)).sum(-1))
            y = torch.stack(outs, dim=1)                                  # (B, L, H, headdim)
            y = y + self.D.float().view(1, 1, self.nheads, 1) * x
            y = y.reshape(batch, length, self.d_inner)
        return self.out_proj(y.to(out_dtype) * F.silu(z))


class TinyMamba2RefBlock(nn.Module):
    """Bidirectional wrapper around the pure-PyTorch SSD reference cell (mirrors TinyMamba2Block)."""

    def __init__(self, d_model: int, *, d_state: int = 16, kernel_size: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.fwd = TinyMamba2RefSSM(d_model, d_state=d_state, kernel_size=kernel_size)
        self.bwd = TinyMamba2RefSSM(d_model, d_state=d_state, kernel_size=kernel_size)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        del t
        h = self.norm1(x)
        y_fwd = self.fwd(h)
        y_bwd = self.bwd(h.flip(1)).flip(1)
        x = x + 0.5 * (y_fwd + y_bwd)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyMamba2RefDenoiser(nn.Module):
    """Denoiser backed by the pure-PyTorch state-matched SSD reference (no mamba_ssm/Triton).

    The state-matched diagonal-SSM comparator on GPUs where the official fused Mamba-2 can't run
    (Pascal sm_61). Set d_state to match the rank-1 cells (16 at d_model=32 => 1024 state elements)
    and kernel_size to toggle the short conv. See docs/armed_gdn_experiment.md."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        *,
        d_model: int = 64,
        n_layers: int = 2,
        d_state: int = 16,
        kernel_size: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_length, d_model)
        self.time_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            [
                TinyMamba2RefBlock(d_model, d_state=d_state, kernel_size=kernel_size)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.max_length:
            raise ValueError("sequence length exceeds max_length")
        if t.ndim == 0:
            t = t.expand(tokens.shape[0])
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)[None, :, :]
        x = x + self.time_proj(timestep_features(t.to(x.dtype), x.shape[-1]))[:, None, :]
        for block in self.blocks:
            x = block(x, t)
        return self.lm_head(self.norm(x))


def build_tiny_denoiser(
    vocab_size: int,
    max_length: int,
    *,
    model_type: str = "dualgoose",
    d_model: int = 64,
    n_layers: int = 2,
    direction: str = "bidirectional",
    merge: str = "film_warm",  # warm-started FiLM is the validated default (C10, docs/film_ablation_rented.md)
    use_rank: bool = True,
    scan_backend: str = "reference",
    short_conv: int = 0,
    n_heads: int = 4,
    attention_window: int = 64,
    mamba_kernel_size: int = 3,
    mamba2_d_state: int = 64,
    mamba2_chunk_size: int = 64,
    mamba2_d_conv: int = 4,
) -> nn.Module:
    """Build a tiny denoiser from checkpoint/train metadata."""

    if model_type in {"dualgoose", "deltanet", "deltanet_linq", "gated_deltanet"}:
        # non-dualgoose values are controlled recurrence swaps (same denoiser stack: bidirectional
        # + merge + FFN + depth), only the recurrence cell differs — to test whether DualGoose's
        # rank-1 advantage is RWKV-7-specific or general to the rank-1 class (and, for
        # gated_deltanet, whether RWKV-7's diagonal decay term is what helps state-tracking).
        # short_conv>0 arms the recurrence with a Mamba-2/GDN-style short causal conv (state-free)
        # — the "armed" cell for the fair-fight-vs-Mamba-2 experiment.
        return TinyDualGooseDenoiser(
            vocab_size,
            max_length,
            d_model=d_model,
            n_layers=n_layers,
            direction=direction,
            merge=merge,
            use_rank=use_rank,
            scan_backend=scan_backend,
            recurrence="rwkv7" if model_type == "dualgoose" else model_type,
            short_conv=short_conv,
        )
    if model_type == "attention":
        return TinyAttentionDenoiser(
            vocab_size,
            max_length,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
        )
    if model_type == "attention_rope":
        # position-generalizing attention control (rotary/relative positions, no learned
        # absolute table) — the valid ceiling for shifting-layout curriculum cells
        return TinyAttentionDenoiser(
            vocab_size,
            max_length,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            rope=True,
        )
    if model_type == "mamba":
        return TinyMambaDenoiser(
            vocab_size,
            max_length,
            d_model=d_model,
            n_layers=n_layers,
            kernel_size=mamba_kernel_size,
        )
    if model_type == "diffumamba_h":
        return TinyDiffuMambaHDenoiser(
            vocab_size,
            max_length,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            attention_window=attention_window,
        )
    if model_type == "mamba2":
        return TinyMamba2Denoiser(
            vocab_size,
            max_length,
            d_model=d_model,
            n_layers=n_layers,
            d_state=mamba2_d_state,
            chunk_size=mamba2_chunk_size,
            d_conv=mamba2_d_conv,
        )
    if model_type == "mamba2_ref":
        # pure-PyTorch state-matched Mamba-2 (no Triton/mamba_ssm) — runs on Pascal/CPU.
        # d_state matches the rank-1 cells; mamba2_d_conv toggles the short conv (1 = no conv).
        return TinyMamba2RefDenoiser(
            vocab_size,
            max_length,
            d_model=d_model,
            n_layers=n_layers,
            d_state=mamba2_d_state,
            kernel_size=mamba2_d_conv,
        )
    raise ValueError(
        "model_type must be dualgoose, attention, attention_rope, mamba, diffumamba_h, "
        "mamba2, or mamba2_ref"
    )
