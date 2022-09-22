import argparse
import os
import numpy as np
import torch
from litsr.data import DownsampledDataset
from litsr.utils import mkdir, read_yaml
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from tqdm import tqdm
from models import load_model

seed_everything(123)


def test_pipeline(args):

    # setup scales and datasets
    test_datasets = (
        [_ for _ in args.datasets.split(",")]
        if args.datasets
        else ["sr-geo-15", "Google-15"]
    )

    # config ckpt path
    exp_path = os.path.dirname(os.path.dirname(args.checkpoint))
    ckpt_path = args.checkpoint

    # read config
    config = read_yaml(os.path.join(exp_path, "hparams.yaml"))

    # create model
    model = load_model(config, ckpt_path, strict=False)
    model.eval()

    # set gpu
    if args.gpus:
        model.cuda()

    scales = args.scales.split(",") if args.scales else [2, 3, 4]
    scales = [float(s) for s in scales]

    for dataset_name in test_datasets:
        for scale in scales:
            # config result path
            rslt_path = os.path.join(
                exp_path,
                "results",
                dataset_name,
                "x" + str(scale),
            )
            mkdir(rslt_path)

            print(
                "==== Dataset {}, Scale Factor x{:.2f} ====".format(dataset_name, scale)
            )

            dataset = DownsampledDataset(
                datapath="load/benchmark/{0}/HR".format(dataset_name),
                scale=scale,
                is_train=False,
                cache="bin",
                rgb_range=config.data_module.args.rgb_range,
                mean=config.data_module.args.get("mean"),
                std=config.data_module.args.get("std"),
                return_img_name=True,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False
            )

            psnrs, ssims, run_times = [], [], []
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataset)):
                if args.gpus:
                    lr, hr, name = batch
                    batch = (lr.cuda(), hr.cuda(), name)
                with torch.no_grad():
                    rslt = model.test_step(
                        batch,
                        batch_idx,
                        patch_sz=48,
                        overlap=8,
                        thresholds=None,
                        self_ensemble=False,
                    )

                file_path = os.path.join(rslt_path, rslt["name"])

                if "log_img" in rslt.keys():
                    plt.imsave(file_path, rslt["log_img"])
                if "log_img_sr" in rslt.keys():
                    plt.imsave(file_path, rslt["log_img_sr"])
                if "log_img_smap" in rslt.keys():
                    plt.imsave(
                        file_path.replace(".png", "_smap.png"),
                        rslt["log_img_smap"][:, :, 0],
                        cmap="gray",
                    )

                psnrs.append(rslt["val_psnr"])
                ssims.append(rslt["val_ssim"])
                run_times.append(rslt["time"])

            mean_psnr = np.array(psnrs).mean()
            mean_ssim = np.array(ssims).mean()
            mean_runtime = np.array(run_times[1:]).mean()

            print("- PSNR:      {:.4f}".format(mean_psnr))
            print("- SSIM:      {:.4f}".format(mean_ssim))
            print("- Runtime :  {:.4f}".format(mean_runtime))
            print("=" * 42)


def getTestParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, help="checkpoint index")
    parser.add_argument(
        "-g",
        "--gpus",
        default="0",
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    parser.add_argument("--datasets", default="", type=str, help="dataset names")
    parser.add_argument("--scales", default="", type=str, help="scale factors")
    return parser


test_parser = getTestParser()

if __name__ == "__main__":
    args = test_parser.parse_args()
    test_pipeline(args)
