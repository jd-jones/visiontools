import argparse
import os

import yaml
import torch
import numpy as np
from matplotlib import pyplot as plt

from seqtools import utils
from visiontools import render
from blocks.core import labels


def main(out_dir=None):
    out_dir = os.path.expanduser(out_dir)

    intrinsic_matrix = torch.tensor(render.intrinsic_matrix, dtype=torch.float).cuda()
    camera_pose = torch.tensor(render.camera_pose, dtype=torch.float).cuda()
    colors = torch.tensor(render.object_colors, dtype=torch.float).cuda()

    nr = render.NeuralRenderer(
        intrinsic_matrix=intrinsic_matrix,
        camera_pose=camera_pose,
        colors=colors,
        light_intensity_ambient=1
    )

    background_plane = None
    assembly = labels.constructGoalState(4)
    component_poses = (
        (np.eye(3), np.zeros(3)),
    )

    rendered = nr.renderScene(background_plane, assembly, component_poses, render_background=False)

    plt.figure()
    plt.imshow(rendered[0])
    plt.savefig(os.path.join(out_dir, f"rendered_rgb.png"))
    plt.close()

    plt.figure()
    plt.imshow(rendered[1])
    plt.savefig(os.path.join(out_dir, f"rendered_depth.png"))
    plt.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

    # Load config file and override with any provided command line args
    config_file_path = args.pop('config_file', None)
    if config_file_path is None:
        file_basename = utils.stripExtension(__file__)
        config_fn = f"{file_basename}.yaml"
        config_file_path = os.path.expanduser(
            os.path.join(
                '~', 'repo', 'visiontools', 'tests', config_fn
            )
        )
    with open(config_file_path, 'rt') as config_file:
        config = yaml.safe_load(config_file)
    config.update(args)

    # Create output directory, instantiate log file and write config options
    out_dir = os.path.expanduser(config['out_dir'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))
    with open(os.path.join(out_dir, config_fn), 'w') as outfile:
        yaml.dump(config, outfile)
    utils.copyFile(__file__, out_dir)

    utils.autoreload_ipython()

    main(**config)
