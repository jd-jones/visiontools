import argparse
import os

import yaml
import torch
import numpy as np
import tqdm
import imageio

from mathtools import utils
from visiontools import render
from blocks.core import labels


def main(out_dir=None):
    out_dir = os.path.expanduser(out_dir)

    intrinsic_matrix = torch.tensor(render.intrinsic_matrix, dtype=torch.float).cuda()
    camera_pose = torch.tensor(render.camera_pose, dtype=torch.float).cuda()
    colors = torch.tensor(render.object_colors, dtype=torch.float).cuda()

    nr = render.TorchSceneRenderer(
        intrinsic_matrix=intrinsic_matrix,
        camera_pose=camera_pose,
        colors=colors,
        light_intensity_ambient=1
    )

    background_plane = None
    assembly = labels.constructGoalState(4)

    R_init = np.eye(3)
    t_init = np.zeros(3)
    radius = 250
    num_frames = 90
    loop = tqdm.tqdm(range(num_frames))
    writer = imageio.get_writer(os.path.join(out_dir, "rendered_rgb.gif"), mode='I')
    for num in loop:
        loop.set_description('Drawing')

        angle = num * (2 * np.pi / num_frames)
        R = R_init
        t = t_init + radius * np.array([np.cos(angle), np.sin(angle), 0])

        component_poses = ((R, t),)
        rgb_image, depth_image = nr.renderScene(
            background_plane, assembly, component_poses,
            render_background=False, as_numpy=True
        )
        writer.append_data((255 * rgb_image).astype(np.uint8))
    writer.close()


def batch_main(out_dir=None):
    out_dir = os.path.expanduser(out_dir)

    intrinsic_matrix = torch.tensor(render.intrinsic_matrix, dtype=torch.float).cuda()
    camera_pose = torch.tensor(render.camera_pose, dtype=torch.float).cuda()
    colors = torch.tensor(render.object_colors, dtype=torch.float).cuda()

    nr = render.TorchSceneRenderer(
        intrinsic_matrix=intrinsic_matrix,
        camera_pose=camera_pose,
        colors=colors,
        light_intensity_ambient=1
    )

    background_plane = None
    assembly = labels.constructGoalState(4)

    R_init = np.eye(3)
    t_init = np.zeros(3)
    radius = 250
    num_frames = 90
    loop = tqdm.tqdm(range(num_frames))
    writer = imageio.get_writer(os.path.join(out_dir, "rendered_rgb.gif"), mode='I')
    for num in loop:
        loop.set_description('Drawing')

        angle = num * (2 * np.pi / num_frames)
        R = R_init
        t = t_init + radius * np.array([np.cos(angle), np.sin(angle), 0])

        component_poses = ((R, t),)
        rgb_image, depth_image = nr.renderScene(
            background_plane, assembly, component_poses,
            render_background=False, as_numpy=True
        )
        writer.append_data((255 * rgb_image).astype(np.uint8))
    writer.close()


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

    batch_main(**config)
