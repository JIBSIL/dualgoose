import torch

from dualgoose.model import TinyDualGooseDenoiser


def test_checkpoint_save_load_preserves_logits(tmp_path):
    torch.manual_seed(0)
    model_args = {
        "vocab_size": 24,
        "max_length": 8,
        "d_model": 12,
        "n_layers": 1,
        "direction": "bidirectional",
        "merge": "film",
        "use_rank": True,
    }
    model = TinyDualGooseDenoiser(**model_args)
    tokens = torch.randint(0, 24, (2, 8))
    t = torch.tensor([0.25, 0.75])
    logits = model(tokens, t)
    path = tmp_path / "model.pt"
    torch.save({"model_args": model_args, "model_state": model.state_dict()}, path)

    loaded_blob = torch.load(path, map_location="cpu", weights_only=True)
    loaded = TinyDualGooseDenoiser(**loaded_blob["model_args"])
    loaded.load_state_dict(loaded_blob["model_state"])
    loaded_logits = loaded(tokens, t)
    assert torch.allclose(logits, loaded_logits)
