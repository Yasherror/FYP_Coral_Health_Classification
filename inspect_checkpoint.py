"""
inspect_checkpoint.py
Standalone diagnostic -- run this directly (NOT through Streamlit) to see
exactly what's inside your .pth file and confirm it lines up with the
custom classifier head.

Usage:
    python inspect_checkpoint.py
    (edit MODEL_PATH below, or pass a path as the first CLI argument)
"""

import sys
import torch
import torch.nn as nn
from torchvision import models

MODEL_PATH = r"C:\Users\Yashreen\Downloads\coral-health-classification\src\components\data\efficientnetb3_tuned_best.pth"


def load_raw_state_dict(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    print(f"torch.load() returned type: {type(checkpoint)}")

    if not isinstance(checkpoint, dict):
        print(
            "\n*** This is NOT a plain state_dict. ***\n"
            "It looks like the full model/module object was saved with "
            "torch.save(obj) instead of torch.save(obj.state_dict()).\n"
        )
        if hasattr(checkpoint, "state_dict"):
            return checkpoint.state_dict()
        sys.exit(1)

    state_dict = checkpoint
    for key in ("model_state_dict", "state_dict", "net", "model", "classifier_state_dict"):
        if key in checkpoint:
            print(f"Found nested state dict under checkpoint['{key}']")
            state_dict = checkpoint[key]
            break
    else:
        print("Checkpoint is a dict but no known nesting key found -- "
              "treating its top-level keys as the state_dict directly.")

    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def build_custom_head(in_features=1536, num_classes=3):
    return nn.Sequential(
        nn.Dropout(p=0.32, inplace=True),
        nn.Linear(in_features, 256, bias=True),
        nn.SiLU(),
        nn.Dropout(p=0.24, inplace=True),
        nn.Linear(256, num_classes, bias=True),
    )


def main(path):
    print(f"Inspecting: {path}\n{'='*70}")
    state_dict = load_raw_state_dict(path)

    keys = list(state_dict.keys())
    print(f"\nTotal tensors in checkpoint: {len(keys)}")
    print("All keys:")
    for k in keys:
        print(f"  {k}  -> shape {tuple(state_dict[k].shape)}")

    head = build_custom_head()
    head_state = head.state_dict()
    head_keys = list(head_state.keys())
    print(f"\nExpected classifier-head keys ({len(head_keys)} total):")
    for k in head_keys:
        print(f"  {k}  -> shape {tuple(head_state[k].shape)}")

    print("\n" + "=" * 70)
    # Case A: checkpoint keys already match the head's own state_dict keys directly
    direct_matched = [k for k, v in head_state.items()
                       if k in state_dict and state_dict[k].shape == v.shape]
    print(f"Direct match (checkpoint keys == head.state_dict() keys): "
          f"{len(direct_matched)}/{len(head_state)}")

    # Case B: checkpoint keys are prefixed with 'classifier.'
    stripped = {k[len("classifier."):]: v for k, v in state_dict.items() if k.startswith("classifier.")}
    prefixed_matched = [k for k, v in head_state.items()
                         if k in stripped and stripped[k].shape == v.shape]
    print(f"Prefixed match (checkpoint keys == 'classifier.<key>'):      "
          f"{len(prefixed_matched)}/{len(head_state)}")

    if len(direct_matched) == len(head_state):
        print("\n✅ This checkpoint IS the classifier head's state_dict directly. "
              "Use: model.classifier.load_state_dict(checkpoint)")
    elif len(prefixed_matched) == len(head_state):
        print("\n✅ This checkpoint has classifier keys prefixed with 'classifier.'. "
              "Strip that prefix, then: model.classifier.load_state_dict(stripped)")
    else:
        print("\n⚠️ Neither pattern matched fully. Compare the two key lists "
              "printed above by hand -- the Linear layer sizes (1536->256->3) "
              "or dropout/module indices may differ from what's assumed here.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    main(path)