"""Toy recurrent mixers used to validate DualGoose mechanics."""

from __future__ import annotations

import torch
from torch import nn

from .rwkv7_backends import make_scan_fn


class RWKV7RefLayer(nn.Module):
    """Small projection wrapper around the sequential reference scan."""

    def __init__(
        self,
        d_model: int,
        d_key: int | None = None,
        use_rank: bool = True,
        scan_backend: str = "reference",
        *,
        short_conv: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key or d_model
        self.use_rank = use_rank
        self.scan_backend = scan_backend
        self.scan_scale = self.d_key**-0.5
        self.scan_fn = make_scan_fn(scan_backend)
        self.proj = nn.Linear(d_model, d_model + 5 * self.d_key)
        self.out = nn.Linear(d_model, d_model)
        # Optional Mamba-2/GDN-style short causal depthwise conv on the interaction tensors
        # (v, k, r) — a STATE-FREE local mixing primitive the bare recurrence lacks. Off by
        # default (short_conv=0) so existing behaviour is unchanged; short_conv=k enables a
        # depthwise causal conv of width k (see the armed-cell experiment).
        self.short_conv = short_conv
        if short_conv:
            conv_ch = self.d_model + 2 * self.d_key  # channels for concat(v, k, r)
            self.vkr_conv = nn.Conv1d(conv_ch, conv_ch, kernel_size=short_conv, groups=conv_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v, k, r, w, kappa, a = torch.split(
            self.proj(x),
            [self.d_model, self.d_key, self.d_key, self.d_key, self.d_key, self.d_key],
            dim=-1,
        )
        if self.short_conv:
            # applied pre-activation, like Mamba-2 / fla GatedDeltaNet; left-pad => strictly causal
            vkr = torch.cat([v, k, r], dim=-1).transpose(1, 2)  # (B, d_model + 2 d_key, L)
            vkr = self.vkr_conv(torch.nn.functional.pad(vkr, (self.short_conv - 1, 0)))
            vkr = torch.nn.functional.silu(vkr.transpose(1, 2))
            v, k, r = torch.split(vkr, [self.d_model, self.d_key, self.d_key], dim=-1)
        w = torch.sigmoid(w)
        a = torch.sigmoid(a) * w
        k = torch.tanh(k) * self.scan_scale
        r = torch.tanh(r) * self.scan_scale
        kappa = torch.tanh(kappa) * self.scan_scale
        # The recurrent scan runs in fp32 regardless of autocast: the Triton
        # backward uses atomic adds, which do not support bf16/fp16, and the
        # recurrent state is the part most sensitive to reduced precision.
        out_dtype = v.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            scanned = self.scan_fn(
                v.float(),
                k.float(),
                r.float(),
                w.float(),
                kappa.float(),
                a.float(),
                use_rank=self.use_rank,
            )
        return self.out(scanned.to(out_dtype))


class DeltaNetRefLayer(nn.Module):
    """DeltaNet recurrence (delta rule): the other beyond-TC^0 rank-1 recurrence, used as a
    baseline against RWKV-7 to test whether DualGoose's rank-1 advantage is RWKV-7-specific or
    general to the rank-1 / delta-rule class.

    State update (per step):  S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T   (rank-1 delta rule;
    equivalently S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T), output  o_t = S_t q_t. Keys are
    L2-normalized and beta_t in (0,1) is a learned write strength. Reference sequential scan (no fast
    kernel) — fine for the short-sequence MQAR / state-tracking probes; slow at long context."""

    def __init__(
        self,
        d_model: int,
        d_key: int | None = None,
        *,
        q_act: str = "tanh",
        gated: bool = False,
        short_conv: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key or d_model
        self.scan_scale = self.d_key**-0.5
        self.q_act = q_act          # "tanh" (squashed) or "linear" (cleaner readout) — tuning knob
        self.gated = gated          # Gated DeltaNet: add a learned diagonal decay (like RWKV-7's w)
        extra = self.d_key if gated else 0
        self.proj = nn.Linear(d_model, 3 * self.d_key + 1 + extra)  # q, k, v, beta (, decay)
        self.out = nn.Linear(self.d_key, d_model)
        # Optional Mamba-2/GDN-style short causal depthwise conv on (q, k, v). STATE-FREE local
        # mixing — the recall primitive bare DeltaNet lacks vs Mamba-2. Off by default; the
        # "armed_gdn" arm sets gated=True + short_conv>0 to match Mamba-2's conv at equal state.
        self.short_conv = short_conv
        if short_conv:
            self.qkv_conv = nn.Conv1d(
                3 * self.d_key, 3 * self.d_key, kernel_size=short_conv, groups=3 * self.d_key
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sizes = [self.d_key, self.d_key, self.d_key, 1] + ([self.d_key] if self.gated else [])
        parts = torch.split(self.proj(x), sizes, dim=-1)
        q, k, v, beta = parts[0], parts[1], parts[2], parts[3]
        if self.short_conv:
            # applied pre-activation, like Mamba-2 / fla GatedDeltaNet; left-pad => strictly causal
            qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)  # (B, 3 d_key, L)
            qkv = self.qkv_conv(torch.nn.functional.pad(qkv, (self.short_conv - 1, 0)))
            qkv = torch.nn.functional.silu(qkv.transpose(1, 2))
            q, k, v = torch.split(qkv, self.d_key, dim=-1)
        w = torch.sigmoid(parts[4]) if self.gated else None  # diagonal decay in (0,1)
        q = (torch.tanh(q) if self.q_act == "tanh" else q) * self.scan_scale
        k = torch.nn.functional.normalize(k, dim=-1)  # L2-normalized keys (DeltaNet standard)
        beta = torch.sigmoid(beta)                    # write strength in (0,1)
        out_dtype = v.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v, beta = q.float(), k.float(), v.float(), beta.float()
            w = w.float() if self.gated else None
            batch, length, d_k = k.shape
            state = torch.zeros(batch, d_k, d_k, device=x.device, dtype=torch.float32)
            outs = []
            for ti in range(length):
                k_t, v_t, q_t = k[:, ti], v[:, ti], q[:, ti]          # (B, d_k)
                b_t = beta[:, ti]                                     # (B, 1)
                if self.gated:
                    state = state * w[:, ti].unsqueeze(1)             # diagonal decay on key axis
                s_k = torch.bmm(state, k_t.unsqueeze(-1)).squeeze(-1)  # S k_t
                delta = b_t * (v_t - s_k)                             # error * write strength
                state = state + delta.unsqueeze(-1) * k_t.unsqueeze(1)  # rank-1 delta update
                outs.append(torch.bmm(state, q_t.unsqueeze(-1)).squeeze(-1))  # o_t = S q_t
            scanned = torch.stack(outs, dim=1)
        return self.out(scanned.to(out_dtype))


class StaticMerge(nn.Module):
    """Timestep-independent concatenation merge."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            eye = torch.eye(d_model)
            self.proj.weight[:, :d_model].copy_(0.5 * eye)
            self.proj.weight[:, d_model:].copy_(0.5 * eye)

    def forward(
        self, u_fwd: torch.Tensor, u_bwd: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        del t
        return self.proj(torch.cat([u_fwd, u_bwd], dim=-1))


class TimestepFiLMMerge(nn.Module):
    """Gated FiLM merge with near-zero initial residual scale."""

    def __init__(
        self,
        d_model: int,
        initial_scale: float = 1e-3,
        gate_bias_init: float = -6.0,
        warm_start: bool = False,
    ):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.cond = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3 * d_model),
        )
        self.gate = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)
        if warm_start:
            # Warm-start so the merge begins as the static 0.5/0.5 average (passed
            # through the norm) rather than near-zero: proj averages the two scans,
            # the gate opens (bias +), and the residual scale is unity. The timestep
            # conditioning (gamma/beta/gate modulation) then learns on top of a useful
            # merge, removing the under-training confound flagged for C10.
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            with torch.no_grad():
                eye = torch.eye(d_model)
                self.proj.weight[:, :d_model].copy_(0.5 * eye)
                self.proj.weight[:, d_model:].copy_(0.5 * eye)
            self.residual_scale = nn.Parameter(torch.full((1,), 1.0))
            nn.init.constant_(self.gate.bias, 6.0)
        else:
            self.residual_scale = nn.Parameter(torch.full((1,), initial_scale))
            nn.init.constant_(self.gate.bias, gate_bias_init)

    def forward(
        self, u_fwd: torch.Tensor, u_bwd: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        if t is None:
            raise ValueError("timestep-conditioned merge requires t")
        if t.ndim == 0:
            t = t.expand(u_fwd.shape[0])
        cond = self.cond(t.to(dtype=u_fwd.dtype, device=u_fwd.device)[:, None])
        gamma, beta, gate_bias = cond.chunk(3, dim=-1)
        merged = self.norm(self.proj(torch.cat([u_fwd, u_bwd], dim=-1)))
        merged = (1.0 + gamma[:, None, :]) * merged + beta[:, None, :]
        gate = torch.sigmoid(self.gate(merged) + gate_bias[:, None, :])
        return self.residual_scale * gate * merged


class BidirectionalRWKVMixer(nn.Module):
    """Forward, backward, or bidirectional RWKV-style mixer."""

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
        if direction not in {"forward", "backward", "bidirectional"}:
            raise ValueError("direction must be forward, backward, or bidirectional")
        self.direction = direction
        self.scan_backend = scan_backend
        self.recurrence = recurrence
        if recurrence == "rwkv7":
            self.fwd = RWKV7RefLayer(
                d_model, use_rank=use_rank, scan_backend=scan_backend, short_conv=short_conv
            )
            self.bwd = RWKV7RefLayer(
                d_model, use_rank=use_rank, scan_backend=scan_backend, short_conv=short_conv
            )
        elif recurrence in {"deltanet", "deltanet_linq", "gated_deltanet"}:
            kw = {
                "deltanet": {},
                "deltanet_linq": {"q_act": "linear"},
                "gated_deltanet": {"gated": True},
            }[recurrence]
            self.fwd = DeltaNetRefLayer(d_model, short_conv=short_conv, **kw)
            self.bwd = DeltaNetRefLayer(d_model, short_conv=short_conv, **kw)
        else:
            raise ValueError("recurrence must be rwkv7, deltanet, deltanet_linq, or gated_deltanet")
        if merge == "film":
            self.merge = TimestepFiLMMerge(d_model)
        elif merge == "film_warm":
            self.merge = TimestepFiLMMerge(d_model, warm_start=True)
        elif merge == "film_raw":
            # Ungated, full-strength conditioning from init (no near-zero damping):
            # the ablation control for C11 (does the gated/zero init stabilize training?).
            self.merge = TimestepFiLMMerge(d_model, initial_scale=1.0, gate_bias_init=0.0)
        elif merge == "static":
            self.merge = StaticMerge(d_model)
        else:
            raise ValueError("merge must be film, film_warm, film_raw, or static")

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if self.direction == "forward":
            return self.fwd(x)
        if self.direction == "backward":
            return self.bwd(x.flip(1)).flip(1)
        u_fwd = self.fwd(x)
        u_bwd = self.bwd(x.flip(1)).flip(1)
        return self.merge(u_fwd, u_bwd, t)
