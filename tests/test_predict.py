"""Frame predictions -> one clip prediction.

The graded unit is a 30-second clip, not a frame, so everything here is about
what happens between a model's per-frame logits and the single per-class
answer that gets submitted for the window.
"""
import numpy as np
import pytest
import torch

from surgvu.predict import aggregate_window, predict_window


class ConstantLogits(torch.nn.Module):
    """Emits identical logits for every frame, and records the mode it ran in.

    With the model's output held constant, aggregation is an identity and the
    activation is the only thing the test can be measuring.

    Deliberately parameter-free. A stub with trainable parameters would make
    `logits.requires_grad` true outside `torch.no_grad()`, and predict_window
    would then blow up inside `.numpy()` -- which kills a no_grad mutation by
    accident, leaving the assertion below never actually exercised. With no
    parameters the mutant runs to completion and the assertion is what has to
    catch it.
    """

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor([logits], dtype=torch.float32))
        self.seen_training = None
        self.seen_grad_enabled = None

    def forward(self, x):
        self.seen_training = self.training
        self.seen_grad_enabled = torch.is_grad_enabled()
        return self.logits.expand(x.shape[0], -1)


class RecordingModel(torch.nn.Module):
    """Stub that records what it was actually handed.

    Mirrors the stub in
    tests/test_train.py::test_run_epoch_forwards_image_size_to_prepare_batch --
    the only way to prove a preprocessing argument was forwarded is to look at
    the tensor that arrives at the forward pass.
    """

    def __init__(self, num_outputs=2):
        super().__init__()
        self.seen_shape = None
        self.seen_batch = None
        self.linear = torch.nn.Linear(3, num_outputs)

    def forward(self, x):
        self.seen_shape = tuple(x.shape)
        self.seen_batch = x.detach().clone()
        return self.linear(x.mean(dim=(2, 3)))


def test_aggregate_window_averages_over_frames():
    frame_probs = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32)
    assert aggregate_window(frame_probs).tolist() == [
        pytest.approx(0.3), pytest.approx(0.7)]


def test_aggregate_window_suppresses_a_single_outlier_frame():
    """One frame where the tool is fully occluded must not flip the clip.
    Occlusion is the known blind spot; averaging is the cheap defence."""
    frame_probs = np.vstack([np.full((29, 1), 0.9, dtype=np.float32),
                             np.zeros((1, 1), dtype=np.float32)])
    assert float(aggregate_window(frame_probs)[0]) > 0.85


def test_aggregate_window_rejects_an_empty_stack():
    with pytest.raises(ValueError, match="no frames"):
        aggregate_window(np.zeros((0, 12), dtype=np.float32))


def test_aggregate_window_rejects_an_already_aggregated_vector():
    """A (C,) input is the shape of this function's own OUTPUT, so passing one
    back in is the easy mistake -- and without a guard it does not raise: it
    averages across the CLASS axis and returns a single scalar, quietly
    collapsing 12 tool probabilities into one meaningless number. The frame
    axis must be present, not merely non-empty."""
    with pytest.raises(ValueError, match="2-D"):
        aggregate_window(np.full((12,), 0.5, dtype=np.float32))


def test_predict_window_sigmoid_is_genuinely_multi_label():
    """The tool head is multi-label: several tools are installed at once, so
    the per-class probabilities must not compete for a fixed budget of 1.0.
    Softmax here would produce well-formed numbers that are simply wrong --
    two confidently-present tools would each be reported at 0.5."""
    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    model = ConstantLogits([2.0, 2.0])

    probs = predict_window(model, frames, device="cpu", image_size=None, activation="sigmoid")

    assert probs.shape == (2,)
    assert all(0.0 < float(p) < 1.0 for p in probs)
    assert float(probs[0]) == pytest.approx(0.8807971, rel=1e-5)   # sigmoid(2)
    assert float(probs.sum()) > 1.5                                # NOT a distribution


def test_predict_window_softmax_sums_to_one():
    """The task head is multi-class: exactly one task is underway, so the
    classes must share one unit of probability mass."""
    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    model = ConstantLogits([2.0, 0.0, -1.0])

    probs = predict_window(model, frames, device="cpu", image_size=None, activation="softmax")

    assert probs.shape == (3,)
    assert float(probs.sum()) == pytest.approx(1.0, rel=1e-6)
    assert float(probs[0]) > float(probs[1]) > float(probs[2])


def test_predict_window_rejects_an_unknown_activation():
    """Serving the wrong squashing is silent: both branches return numbers in
    (0, 1). A typo must be an error, never a default."""
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    model = ConstantLogits([1.0, 1.0])

    with pytest.raises(ValueError, match="unknown activation"):
        predict_window(model, frames, device="cpu", image_size=None, activation="softmaxx")


def test_predict_window_forwards_image_size_to_prepare_batch():
    """Shards hold 512x512; both experts are trained at 384 and record that
    in their checkpoint meta. predict_window is the serving path, so if it
    dropped image_size the model would be served at a resolution it was never
    trained at -- a silent accuracy loss that reads as a mediocre model, not
    as a wiring bug. The resize itself lives in prepare_batch; what is pinned
    here is that predict_window actually forwards the argument."""
    frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)

    resized = RecordingModel()
    predict_window(resized, frames, device="cpu", image_size=8)
    assert resized.seen_shape == (3, 3, 8, 8)

    native = RecordingModel()
    predict_window(native, frames, device="cpu", image_size=None)
    assert native.seen_shape == (3, 3, 16, 16)


def test_predict_window_requires_an_explicit_image_size():
    """`image_size` must be stated, never defaulted.

    Shards hold 512x512 and both experts train at 384, recording it in
    meta["image_size"]. A default of None means a caller who simply forgets
    serves the model at 512 -- no error, no warning, just a model evaluated at
    a resolution it never saw. Making the argument required converts that
    silent miscalibration into a TypeError at the call site. "No resize" stays
    expressible, but only by typing None on purpose.
    """
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    model = ConstantLogits([1.0, 1.0])

    with pytest.raises(TypeError):
        predict_window(model, frames, device="cpu")


def test_predict_window_runs_in_eval_mode_without_a_grad_graph():
    """Inference-time correctness, not just speed: BatchNorm and dropout in a
    model left in train() mode make the answer depend on the other frames in
    the window, and a retained grad graph over 30 512x512 frames is a memory
    leak inside the 16 GiB container budget."""
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    model = ConstantLogits([1.0, 1.0])
    model.train()

    predict_window(model, frames, device="cpu", image_size=None)

    assert model.seen_training is False
    assert model.seen_grad_enabled is False


def test_predict_window_aggregates_across_the_windows_frames():
    """End-to-end proof that predict_window returns the WINDOW's answer, not
    the first frame's: one confidently-negative frame out of four must move
    the clip probability, and must not decide it."""

    class PerFrameLogits(torch.nn.Module):
        """+6 for a bright frame, -6 for a black one."""

        def forward(self, x):
            bright = x.mean(dim=(1, 2, 3)) > 0.5
            return torch.where(bright, 6.0, -6.0).reshape(-1, 1)

    frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
    frames[:3] = 255

    probs = predict_window(PerFrameLogits(), frames, device="cpu", image_size=None)

    assert float(probs[0]) == pytest.approx(0.75, abs=0.01)


def test_predict_window_takes_bgr_uint8_frames_unconverted():
    """Pins the `frames` contract: a numpy uint8 (N, H, W, 3) BGR array,
    handed to prepare_batch as-is.

    train.run_epoch calls prepare_batch(frames.numpy(), ...) because its
    frames arrive from a DataLoader as torch tensors. predict_window's do
    not -- at serving they come straight off a shard as an OpenCV array. If
    predict_window copied that `.numpy()` it would raise AttributeError on
    the only input a serving caller has. The two call sites are in different
    files, so the difference is pinned here rather than assumed.
    """
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    frames[..., 0] = 255                      # blue, in OpenCV BGR order
    model = RecordingModel()

    predict_window(model, frames, device="cpu", image_size=None)

    assert model.seen_shape == (1, 3, 2, 2)
    assert model.seen_batch.dtype == torch.float32
    assert float(model.seen_batch[0, 2].max()) == 1.0   # lands in RED
    assert float(model.seen_batch[0, 0].max()) == 0.0
