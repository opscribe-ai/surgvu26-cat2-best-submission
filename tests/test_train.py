import numpy as np
import pytest
import torch

from surgvu.train import (
    load_checkpoint, prepare_batch, save_checkpoint, seed_everything,
)


def test_prepare_batch_is_channel_first_and_unit_scaled():
    frames = np.full((2, 8, 8, 3), 255, dtype=np.uint8)
    batch = prepare_batch(frames, device="cpu")
    assert batch.shape == (2, 3, 8, 8)
    assert batch.dtype == torch.float32
    assert float(batch.max()) == 1.0


def test_prepare_batch_resizes_when_asked():
    """Shards are 512; the backbone was pretrained at 384. If training and
    inference resize differently the model silently loses accuracy, so the
    resize lives in one function used by both."""
    frames = np.zeros((2, 512, 512, 3), dtype=np.uint8)
    assert prepare_batch(frames, device="cpu", image_size=384).shape == (2, 3, 384, 384)
    assert prepare_batch(frames, device="cpu").shape == (2, 3, 512, 512)


def test_prepare_batch_converts_bgr_to_rgb():
    """Shards hold OpenCV BGR. ImageNet weights expect RGB. Swapping the
    channels silently costs accuracy that looks like an architecture problem."""
    frame = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    frame[..., 0] = 255                      # blue in BGR
    batch = prepare_batch(frame, device="cpu")
    assert float(batch[0, 2].max()) == 1.0   # lands in the RED channel
    assert float(batch[0, 0].max()) == 0.0


def test_seed_everything_makes_two_runs_identical():
    seed_everything(7)
    a = torch.randn(4)
    seed_everything(7)
    assert torch.equal(a, torch.randn(4))


# ------------------------------------------------------- cuDNN determinism
# Seeding python/numpy/torch is not enough on CUDA: cuDNN picks convolution
# algorithms that are themselves nondeterministic. Two runs of the task model
# at the same config and seed returned val_acc 0.6175 and 0.6293, and epoch-3
# macro-F1 0.8695 vs 0.8657. Every number we report has to say which side of
# this switch produced it, so the switch has to exist and has to be explicit.

@pytest.fixture(autouse=True)
def _restore_cudnn_flags():
    """These are process-global. A test that flips one must not leak it into
    the next, or the leak looks exactly like the bug under test."""
    before = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    yield
    (torch.backends.cudnn.deterministic,
     torch.backends.cudnn.benchmark) = before


def test_determinism_is_opt_in_not_the_default():
    """Deterministic cuDNN costs throughput, and the numbers currently in
    config/perception.json were produced WITHOUT it. Silently flipping the
    default would make them unreproducible in the other direction."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    seed_everything(7)

    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is False


def test_deterministic_seeding_pins_both_cudnn_flags():
    """`deterministic` alone is not enough: benchmark=True re-picks the
    algorithm per input shape, so it has to be turned off in the same call or
    the guarantee is only sometimes true."""
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    seed_everything(7, deterministic=True)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_deterministic_seeding_still_seeds():
    """The flags are an addition to the seeding, not a replacement for it."""
    seed_everything(7, deterministic=True)
    a = torch.randn(4)
    seed_everything(7, deterministic=True)

    assert torch.equal(a, torch.randn(4))


def _training_script(name):
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module(name)


@pytest.mark.parametrize("script", ["train_tools", "train_task"])
def test_the_training_scripts_expose_the_switch_and_default_it_off(script):
    """A knob only reachable from the library is a knob no reported run can
    say it used. Both entrypoints take --deterministic, and both default to
    the setting that produced the checkpoints we ship."""
    parser = _training_script(script).build_parser()

    assert parser.parse_args(["--out", "x"]).deterministic is False
    assert parser.parse_args(["--out", "x", "--deterministic"]).deterministic is True


@pytest.mark.parametrize("script", ["train_tools", "train_task"])
def test_the_training_scripts_actually_apply_the_switch(script, monkeypatch):
    """Parsing the flag and honouring it are different things. This runs the
    real main() as far as the first thing that needs data, which is past the
    seeding, and reads the process-global flag back."""
    import sys

    module = _training_script(script)

    class Stop(Exception):
        """The loader is the first thing main() does that needs a filesystem."""

    def stop(*args, **kwargs):
        raise Stop()

    monkeypatch.setattr(module, "shard_paths_for_split", stop)
    if hasattr(module, "load_corpus"):
        monkeypatch.setattr(module, "load_corpus", stop)
    monkeypatch.setattr(sys, "argv", [script, "--out", "x", "--deterministic"])
    torch.backends.cudnn.deterministic = False

    with pytest.raises(Stop):
        module.main()

    assert torch.backends.cudnn.deterministic is True


def test_run_epoch_uses_the_target_fn_it_was_given(tmp_path):
    """All three models share one loader but consume different labels. If the
    target were inferred from the loss class, the action model would silently
    train against the 12-way tool vector and still report a falling loss."""
    from surgvu.train import run_epoch

    frames = torch.zeros(2, 4, 4, 3, dtype=torch.uint8)
    tools = torch.zeros(2, 12)
    task = torch.tensor([1, 0])
    loader = [(frames, tools, task)]
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 8))

    stats = run_epoch(model, loader, torch.nn.CrossEntropyLoss(),
                      target_fn=lambda tools, task: task, device="cpu")

    assert stats["targets"].tolist() == [1, 0]


def test_run_epoch_rejects_an_empty_loader():
    """An empty loader is the same silent-nothing failure as an empty split
    or an empty ShardFrames: training would run to completion, converge on
    zero batches, and report a meaningless loss of 0.0 as if nothing were
    wrong. run_epoch must refuse rather than return that."""
    from surgvu.train import run_epoch

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 8))

    with pytest.raises(ValueError, match="zero batches"):
        run_epoch(model, [], torch.nn.CrossEntropyLoss(),
                  target_fn=lambda tools, task: task, device="cpu")


def test_run_epoch_forwards_image_size_to_prepare_batch():
    """Shards are 512x512; the backbone was pretrained at 384. run_epoch must
    forward image_size through to prepare_batch, which is where the resize
    actually happens (see test_prepare_batch_resizes_when_asked). Training at
    one resolution while serving at another is a silent accuracy loss that
    reads as a bad architecture choice, not a wiring bug -- so this has to be
    caught here, not just in prepare_batch's own unit tests."""
    from surgvu.train import run_epoch

    class RecordingModel(torch.nn.Module):
        """Stub that records the spatial shape it was actually called with."""

        def __init__(self):
            super().__init__()
            self.seen_shape = None
            self.linear = torch.nn.Linear(3, 2)

        def forward(self, x):
            self.seen_shape = tuple(x.shape)
            return self.linear(x.mean(dim=(2, 3)))

    frames = torch.zeros(2, 16, 16, 3, dtype=torch.uint8)
    tools = torch.zeros(2, 12)
    task = torch.tensor([0, 1])
    loader = [(frames, tools, task)]
    model = RecordingModel()

    run_epoch(model, loader, torch.nn.CrossEntropyLoss(),
              target_fn=lambda tools, task: task, device="cpu", image_size=8)

    assert model.seen_shape == (2, 3, 8, 8)


def test_checkpoint_roundtrips_weights_and_metadata(tmp_path):
    model = torch.nn.Linear(4, 2)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, {"macro_f1": 0.5, "thresholds": [0.3] * 2})

    restored = torch.nn.Linear(4, 2)
    meta = load_checkpoint(path, restored)

    assert meta["macro_f1"] == 0.5
    assert torch.equal(model.weight, restored.weight)
