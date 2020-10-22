import argparse
import os

import yaml
import torch
import numpy as np
import tqdm
import imageio
from scipy.spatial.transform import Rotation

from mathtools import utils
from visiontools import render
from blocks.core import labels


def renderTemplates(renderer, assembly, t, R):
    if R.shape[-1] != t.shape[-1]:
        err_str = f"R shape {R.shape} doesn't match t shape {t.shape}"
        raise AssertionError(err_str)

    num_templates = R.shape[-1]

    component_poses = ((np.eye(3), np.zeros(3)),)
    assembly = assembly.setPose(component_poses, in_place=False)

    init_vertices = render.makeBatch(assembly.vertices, dtype=torch.float).cuda()
    faces = render.makeBatch(assembly.faces, dtype=torch.int).cuda()
    textures = render.makeBatch(assembly.textures, dtype=torch.float).cuda()

    vertices = torch.einsum('nvj,jit->nvit', [init_vertices, R]) + t
    vertices = vertices.permute(-1, 0, 1, 2)

    faces = faces.expand(num_templates, *faces.shape)
    textures = textures.expand(num_templates, *textures.shape)

    rgb_images_obj, depth_images_obj = renderer.render(
        torch.reshape(vertices, (-1, *vertices.shape[2:])),
        torch.reshape(faces, (-1, *faces.shape[2:])),
        torch.reshape(textures, (-1, *textures.shape[2:]))
    )
    rgb_images_scene, depth_images_scene, label_images_scene = render.reduceByDepth(
        torch.reshape(rgb_images_obj, vertices.shape[:2] + rgb_images_obj.shape[1:]),
        torch.reshape(depth_images_obj, vertices.shape[:2] + depth_images_obj.shape[1:]),
    )

    return rgb_images_scene, depth_images_scene


def main(out_dir=None, shrink_by=3):
    out_dir = os.path.expanduser(out_dir)

    intrinsic_matrix = torch.tensor(render.intrinsic_matrix, dtype=torch.float).cuda()
    camera_pose = torch.tensor(render.camera_pose, dtype=torch.float).cuda()
    colors = torch.tensor(render.object_colors, dtype=torch.float).cuda()

    intrinsic_matrix[:2, 2] /= shrink_by
    nr = render.TorchSceneRenderer(
        intrinsic_matrix=intrinsic_matrix,
        camera_pose=camera_pose,
        colors=colors,
        light_intensity_ambient=1,
        image_size=render.IMAGE_WIDTH // shrink_by,
        orig_size=render.IMAGE_WIDTH // shrink_by
    )

    assembly = labels.constructGoalState(4)

    num_frames = 36
    angles = torch.arange(num_frames).float() * (2 * np.pi / num_frames)
    rotations = Rotation.from_euler('Z', angles)
    t = torch.stack((torch.zeros_like(angles),) * 3, dim=0).float().cuda()
    R = torch.tensor(rotations.as_dcm()).permute(1, 2, 0).float().cuda()

    rgb_templates, depth_templates = renderTemplates(nr, assembly, t, R)

    loop = tqdm.tqdm(range(num_frames))
    writer = imageio.get_writer(os.path.join(out_dir, "rendered_rgb.gif"), mode='I')
    for num in loop:
        loop.set_description('Saving')
        rgb_image = rgb_templates[num, ...].detach().cpu().numpy()
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

    main(**config)
