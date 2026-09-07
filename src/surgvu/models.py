"""Backbones for the perception experts.

EfficientNetV2-S is the default because the 2025 winner reached 97% macro-F1
with it on this exact data. That is a strong empirical prior, not a principled
choice, so `backbone` stays a parameter and the ablation includes ResNet-50.

Weights come from torchvision, which is BSD-3. This matters: the repo goes
public at submission, and an AGPL detector dependency would force AGPL on the
whole released work.
"""
import torch.nn as nn
import torchvision

# WHY THIS LIST GREW, AND HOW IT WAS CHOSEN
#
# The v2 overnight run measured that ensemble gains come from ARCHITECTURAL
# DIVERSITY and not from averaging. Two EfficientNets at different seeds
# ensemble to 0.6707, BELOW a single EfficientNet at 0.6847; EfficientNet plus
# ResNet-50 reaches 0.7171. So the useful axis is "a model that is wrong in
# different places", and the way to buy more of that is more design families,
# not more seeds and not more frames (8 frames beat 30) and not test-time
# augmentation (every arm lost).
#
# These four are picked to be far apart in design space rather than to be
# individually strong:
#
#   convnext_small  a CNN redesigned along transformer lines -- large kernels,
#                   LayerNorm, inverted bottlenecks. Convolutional like the
#                   incumbents but with very different inductive biases.
#   swin_s          a hierarchical windowed TRANSFORMER. No convolution at
#                   all, so its errors have the least reason to correlate with
#                   the two CNNs already in the ensemble.
#   densenet121     dense connectivity, an older and genuinely different
#                   topology. Small and cheap, which matters because the whole
#                   point is running several.
#   regnet_y_8gf    a design-space-search network with squeeze-excite; close
#                   enough to ResNet to be a useful control on how much
#                   "different family" really has to mean.
#
# All torchvision, all BSD-3. That is not incidental: the repo goes public at
# submission and an AGPL detector dependency would force AGPL on the whole
# released work, which is why Ultralytics YOLO was ruled out.
_BACKBONES = {
    "efficientnet_v2_s": (
        torchvision.models.efficientnet_v2_s,
        torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
    ),
    "resnet50": (
        torchvision.models.resnet50,
        torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
    ),
    "convnext_small": (
        torchvision.models.convnext_small,
        torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
    ),
    "swin_s": (
        torchvision.models.swin_s,
        torchvision.models.Swin_S_Weights.IMAGENET1K_V1,
    ),
    "densenet121": (
        torchvision.models.densenet121,
        torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
    ),
    "regnet_y_8gf": (
        torchvision.models.regnet_y_8gf,
        torchvision.models.RegNet_Y_8GF_Weights.IMAGENET1K_V2,
    ),
}


# THE 3D BACKBONES, and why these two.
#
# Every 2D model here treats the 16 frames of a clip as 16 unrelated images
# and averages their predictions, so the ordering is discarded entirely --
# shuffle the frames and the answer is bit-identical. That is the standing
# architectural gap: "is tissue being cut" is a question about motion, and a
# scissors resting on tissue and a scissors closing on it are nearly identical
# in any single still.
#
#   r2plus1d_18   PRIMARY. Factorises each 3D convolution into a spatial 2D
#                 convolution followed by a temporal 1D one. More non-linearity
#                 for the same parameter count, easier to optimise than full
#                 3D, and the strongest standard baseline of this family.
#   r3d_18        CONTROL. True 3D convolutions -- the literal (x, y, t) kernel.
#                 Carried so that "factorisation matters" is measured rather
#                 than assumed.
#
# Both Kinetics-400 pretrained, both BSD-3 via torchvision. The licence is not
# incidental: the repo goes public at submission, which is what ruled out
# AGPL Ultralytics and the restrictive DINOv3 terms -- and, tonight, the
# CC-BY-NC-ND SurgVISTA weights, which would have been the better domain match.
#
# TWO CONFOUNDS TO HOLD IN MIND WHEN READING ANY RESULT FROM THESE:
#   * Kinetics-400 is human action video. It is far smaller than ImageNet and
#     much further from surgery, and these are EIGHTEEN-layer networks against
#     a ResNet-50. A loss here may be capacity and pretraining, not temporality.
#   * They expect 112x112. Our 2D path runs 384, and instrument tips are small.
# Hence the sequence-model control over existing 2D features: it separates
# "temporal structure carries signal" from "this backbone is any good".
_VIDEO_BACKBONES = {
    "r2plus1d_18": (
        torchvision.models.video.r2plus1d_18,
        torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1,
    ),
    "r3d_18": (
        torchvision.models.video.r3d_18,
        torchvision.models.video.R3D_18_Weights.KINETICS400_V1,
    ),
}

#: Kinetics normalisation, from the torchvision video weights' own transforms.
#: NOT ImageNet, and not the [0, 1]-only convention `prepare_batch` uses for
#: the 2D path. Getting this wrong is the EndoViT trap in a new costume: a
#: quiet accuracy loss that reads as "the architecture is mediocre".
VIDEO_MEAN = (0.43216, 0.394666, 0.37645)
VIDEO_STD = (0.22803, 0.22145, 0.216989)


def is_video_backbone(backbone):
    return backbone in _VIDEO_BACKBONES


def build_video_model(num_outputs, backbone="r2plus1d_18", pretrained=True):
    """A Kinetics-pretrained 3D CNN with its classifier resized.

    Takes (B, 3, T, H, W) and returns RAW LOGITS, matching `build_model`, so
    the same loss functions and the same training loop apply unchanged.
    """
    if backbone not in _VIDEO_BACKBONES:
        raise ValueError("unknown video backbone %r; expected one of %s"
                         % (backbone, sorted(_VIDEO_BACKBONES)))
    factory, weights = _VIDEO_BACKBONES[backbone]
    model = factory(weights=weights if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_outputs)
    return model


def build_model(num_outputs, backbone="efficientnet_v2_s", pretrained=True):
    """A backbone with its classifier replaced by a `num_outputs` head.

    Returns RAW LOGITS. BCEWithLogitsLoss and CrossEntropyLoss both apply
    their own squashing; a model that pre-applied it would be squashed twice.
    """
    if backbone not in _BACKBONES:
        raise ValueError("unknown backbone %r; expected one of %s"
                         % (backbone, sorted(_BACKBONES)))
    factory, weights = _BACKBONES[backbone]
    model = factory(weights=weights if pretrained else None)

    # Each family hides its classifier somewhere different, and getting this
    # wrong is SILENT: replacing the wrong module leaves a 1000-way ImageNet
    # head in the graph and trains something that looks like a bad model
    # rather than a broken one. Handled by inspection instead of by name so a
    # future backbone works without another branch here.
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_outputs)     # resnet, regnet
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_outputs)  # swin
    elif hasattr(model, "heads"):                                    # vit
        model.heads.head = nn.Linear(model.heads.head.in_features, num_outputs)
    elif isinstance(getattr(model, "classifier", None), nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features,
                                     num_outputs)                    # densenet
    elif hasattr(model, "classifier"):                               # effnet, convnext
        for index in range(len(model.classifier) - 1, -1, -1):
            layer = model.classifier[index]
            if isinstance(layer, nn.Linear):
                model.classifier[index] = nn.Linear(layer.in_features,
                                                    num_outputs)
                break
        else:
            raise ValueError("%s has a classifier with no Linear layer to "
                             "replace" % (backbone,))
    else:
        raise ValueError("do not know where %s keeps its classifier" % (backbone,))
    return model
