import importlib
import os
from os import path as osp

import torch
from litsr import create_model, load_model
from litsr.utils.registry import ModelRegistry


def forward_self_ensemble(model, lr, out_size):
    def _transform(v, op):
        v = v.float()
        v2np = v.data.cpu().numpy()
        if op == "v":
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == "h":
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == "t":
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.tensor(tfnp, dtype=torch.float, requires_grad=False, device="cuda")
        return ret

    lr_list = [lr]
    for tf in "v", "h", "t":
        lr_list.extend([_transform(t, tf) for t in lr_list])

    sr_list = []
    for i, aug in enumerate(lr_list):
        if i > 3:
            _out_size = (out_size[1], out_size[0])
        else:
            _out_size = out_size
        sr = model.forward(aug, _out_size).detach()
        sr_list.append(sr)

    for i in range(len(sr_list)):
        if i > 3:
            sr_list[i] = _transform(sr_list[i], "t")
        if i % 4 > 1:
            sr_list[i] = _transform(sr_list[i], "h")
        if (i % 4) % 2 == 1:
            sr_list[i] = _transform(sr_list[i], "v")

    output_cat = torch.cat(sr_list, dim=0)
    output = output_cat.mean(dim=0, keepdim=True)
    return output


# Import all models

__all__ = ["create_model", "load_model", "ModelRegistry"]

model_folder = osp.dirname(osp.abspath(__file__))
for root, dirs, files in os.walk(model_folder):
    for f in files:
        if f.endswith("_model.py"):
            importlib.import_module(f"models.{osp.splitext(f)[0]}")
