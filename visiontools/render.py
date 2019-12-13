import logging
import json
import os

import numpy as np
from scipy.spatial.qhull import QhullError
import torch

import mathtools as m
from mathtools import utils

from . import geometry


logger = logging.getLogger(__name__)


""" Functions and attributes for rendering images.

Attributes
----------
intrinsic_matrix : numpy array of float, shape (3, 3)
    The intrinsic matrix estimated during camera calibration. This array is
    loaded from ``~/repo/blocks/blocks/assets/camera_params.json``. For more
    information about how these parameters were estimated, see
    ``README_camera_params.md`` in the same directory. Layout of the intrinsic
    matrix is as follows:
    ..math:
        K = \left[ \begin{matrix}
            \alpha_x &     0    & o_x \\
                0    & \alpha_y & o_y \\
                0    &     0    &  1  \\
        \end{matrix} \right]
    where :math:`\alpha_x = f s_x` is the size of unit length in horizontal
    pixels and :math:`o_x` is the horizontal coordinate of the principal point,
    in pixels. :math:`\alpha_y` and :math:`o_y` are defined the same way, but
    are vertical measurements.
camera_pose : numpy array of float, shape (4, 4)
object_colors : numpy array of float, shape (num_blocks + 1, 3)
"""

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320


def loadCameraParams(
        assets_dir=None, camera_params_fn=None, camera_pose_fn=None,
        object_colors_fn=None, as_dict=False):
    """ Load camera parameters from external files.

    Parameters
    ----------
    assets_dir : str, optional
    camera_params_fn : str, optional
    camera_pose_fn : str, optional
    object_colors_fn : str, optional
    as_dict : bool, optional
        If True, the parameters are returned as a dictionary instead of a tuple,
        with format
        {
            'intrinsic_matrix': intrinsic_matrix,
            'camera_pose': camera_pose,
            'object_colors': object_colors
        }

    Returns
    -------
    intrinsic_matrix : numpy array of float, shape (3, 3)
        The intrinsic matrix estimated during camera calibration. Layout is as
        follows:
        ..math:
            K = \left[ \begin{matrix}
                \alpha_x &     0    & o_x \\
                    0    & \alpha_y & o_y \\
                    0    &     0    &  1  \\
            \end{matrix} \right]
        where :math:`\alpha_x = f s_x` is the size of unit length in horizontal
        pixels and :math:`o_x` is the horizontal coordinate of the principal point,
        in pixels. :math:`\alpha_y` and :math:`o_y` are defined the same way, but
        are vertical measurements.
    camera_pose : numpy array of float, shape (4, 4)
    object_colors : numpy array of float, shape (num_blocks + 1, 3)
    """

    if assets_dir is None:
        assets_dir = os.path.expanduser(os.path.join('~', 'repo', 'blocks', 'blocks', 'assets'))

    if camera_params_fn is None:
        camera_params_fn = 'camera_params.json'

    if camera_pose_fn is None:
        camera_pose_fn = 'camera_pose.json'

    if object_colors_fn is None:
        object_colors_fn = 'object_colors.csv'

    # Load camera params
    with open(os.path.join(assets_dir, camera_params_fn), 'rt') as f:
        json_obj = json.load(f)['camera_intrinsics']['intrinsic_matrix']
        intrinsic_matrix = m.np.transpose(m.np.array(json_obj))

    # Load camera pose
    with open(os.path.join(assets_dir, camera_pose_fn), 'rt') as f:
        camera_pose_dict = json.load(f)['camera_pose']

    R_camera = geometry.rotationMatrix(**camera_pose_dict['orientation'])
    t_camera = m.np.array(camera_pose_dict['position'])
    camera_pose = geometry.homogeneousMatrix(R_camera, t_camera, range_space_homogeneous=True)

    # Load object colors (ie rudimentary appearance model)
    object_colors = m.np.loadtxt(
        os.path.join(assets_dir, object_colors_fn),
        delimiter=',', skiprows=1
    )

    if as_dict:
        return {
            'intrinsic_matrix': intrinsic_matrix,
            'camera_pose': camera_pose,
            'colors': object_colors
        }

    return intrinsic_matrix, camera_pose, object_colors


intrinsic_matrix, camera_pose, object_colors = loadCameraParams()


# -=( HELPER FUNCTIONS FOR PYTORCH RENDERER )==--------------------------------
def reduceByDepth(rgb_images, depth_images):
    """ For each pixel in a scene, select the object closest to the camera.

    Parameters
    ----------
    rgb_images : torch.tensor of float, shape (batch_size, img_height, img_width)
    depth_images : torch.tensor of float, shape (batch_size, img_height, img_width)

    Returns
    -------
    rgb_image : torch.tensor of float, shape (img_height, img_width)
    depth_image : torch.tensor of float, shape (img_height, img_width)
    label_image : torch.tensor of int, shape (img_height, img_width)
    """

    label_image = depth_images.argmin(0)
    num_rows, num_cols = label_image.shape
    r, c = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
    i_min = label_image.contiguous().view(-1)
    r = r.contiguous().view(-1)
    c = c.contiguous().view(-1)

    depth_image = depth_images[i_min, r, c].view(num_rows, num_cols)
    rgb_image = rgb_images[i_min, r, c, :].view(num_rows, num_cols, -1)

    return rgb_image, depth_image, label_image


def planeVertices(plane, intrinsic_matrix, camera_pose, image_shape=None):
    if image_shape is None:
        image_shape = (IMAGE_HEIGHT, IMAGE_WIDTH)

    image_shape = tuple(float(x) for x in image_shape)

    face_coords = m.np.array([
        [0, 0],
        [0, image_shape[0]],
        [image_shape[1], 0],
        [image_shape[1], image_shape[0]]
    ])

    plane_faces = m.np.array([
        [0, 1, 2],
        [3, 2, 1]
    ])

    # Consruct face_coords_camera in a way that allows
    # geometry.slopeIntercept to compute the plane parameters it needs.
    face_coords_camera = m.np.zeros((3, 3))
    face_coords_camera[0, :] = plane._t
    face_coords_camera[1, :] = plane._t + plane._U[:, 0]
    face_coords_camera[2, :] = plane._t + plane._U[:, 0] + plane._U[:, 1]

    # Backproject each pixel in the face to its location in camera coordinates
    n, b = geometry.slopeIntercept(face_coords_camera)
    metric_coords_camera = geometry.backprojectIntoPlane(face_coords, n, b, intrinsic_matrix)
    vertices = geometry.homogeneousVector(metric_coords_camera) @ m.np.transpose(camera_pose)

    return vertices, plane_faces


def makeTextures(faces, texture_size=2, uniform_color=None):
    """

    Parameters
    ----------
    faces : array of int, shape (num_faces, 3)
    texture_size : int, optional
    uniform_color : [int, int, int], optional

    Returns
    -------
    textures : array of float, shape (num_faces, texture_size, texture_size, texture_size, 3)
    """

    # create texture [num_faces, texture_size, texture_size, texture_size, RGB]
    textures = m.np.zeros(
        faces.shape[0], texture_size, texture_size, texture_size, 3,
        dtype=torch.float32
    )

    if uniform_color is not None:
        textures[..., :] = torch.tensor(uniform_color)

    return textures


# -=( HELPER FUNCTIONS FOR LEGACY RENDERER )==---------------------------------
def findCentroid(img):
    if not img.any():
        raise ValueError

    if len(img.shape) > 2:
        img = img.sum(axis=2)
    rows, cols = np.nonzero(img)

    cent_r = np.rint(rows.mean()).astype(int)
    cent_c = np.rint(cols.mean()).astype(int)
    centroid = (cent_r, cent_c)

    len_r = rows.max() - rows.min()
    len_c = cols.max() - cols.min()
    nonzero_shape = (len_r, len_c)

    return centroid, nonzero_shape


def centerBoundingBox(img_centroid, img_shape):
    r_centroid, c_centroid = img_centroid
    r_len, c_len = img_shape[0:2]

    r_max_centered = r_centroid + r_len // 2
    if r_len % 2:
        r_max_centered += 1
    r_min_centered = r_centroid - r_len // 2
    r_extent = (r_min_centered, r_max_centered)

    c_max_centered = c_centroid + c_len // 2
    if c_len % 2:
        c_max_centered += 1
    c_min_centered = c_centroid - c_len // 2
    c_extent = (c_min_centered, c_max_centered)

    return r_extent, c_extent


def cropImage(img, shape=None):
    img_centroid, nonzero_shape = findCentroid(img)

    # By default, shape is a bounding square for the nonzero image elements
    if shape is None:
        r_len, c_len = nonzero_shape
        max_len = (r_len ** 2 + c_len ** 2) ** 0.5
        max_len = np.ceil(max_len).astype(int)
        shape = (max_len, max_len)

    (r_min, r_max), (c_min, c_max) = centerBoundingBox(img_centroid, shape)
    cropped = img[r_min:r_max, c_min:c_max].copy()
    return cropped


def renderScene(
        background_plane, assembly, component_poses,
        camera_pose=None, camera_params=None, object_appearances=None):
    """ Render a scene consisting of a spatial assembly and a background plane.

    Parameters
    ----------

    Returns
    -------
    """

    # Start by rendering the background
    rgb_image, range_image, label_image = renderPlane(
        background_plane, camera_pose, camera_params,
        plane_appearance=object_appearances[0, :],
    )

    # Then render each foreground object one-by-one
    for comp_idx, comp_key in enumerate(assembly.connected_components.keys()):
        comp_pose = component_poses[comp_idx]
        _ = renderComponent(
            assembly, comp_key, component_pose=comp_pose,
            camera_pose=camera_pose, camera_params=camera_params,
            block_colors=object_appearances,
            range_image=range_image, label_image=label_image, rgb_image=rgb_image,
            crop_rendered=False
        )

    return rgb_image, range_image, label_image


def renderPlane(
        plane, camera_pose=None, camera_params=None, plane_appearance=None,
        range_image=None, label_image=None, rgb_image=None):
    """ Render a component of the state.

    Parameters
    ----------
    plane : geometry.Plane
        Plane that should be rendered.
    camera_pose : numpy array of float, shape (4, 4)
        The camera's pose with respect to the world coordinate frame. This is
        a rigid motion (R, t), represented as a homogeneous matrix.
    camera_params : numpy array of float, shape (3, 4)
        The camera's intrinsic parameters.
    plane_appearance : numpy array, shape (3,)
        The color of the plane.
    range_image : numpy array of float, shape (img_height, img_width), optional
        Pre-existing Z_buffer. Each pixel value is the distance from the camera
        in mm.
    label_image : numpy array of int, shape (img_height, img_width), optional
        Label image corresponding to the Z-buffer. Each pixel value is the label
        of the object that was projected onto the pixel.
    rgb_image : numpy array of float, shape (img_height, img_width, 3), optional
        Pre-existing RGB image.

    Returns
    -------
    rgb_image : numpy array of float, shape (img_height, img_width, 3)
        Color image in RGB format.
    range_image : numpy array of float, shape (img_height, img_width)
        The Z-buffer. Each pixel value is the distance from the camera in mm.
    label_image : numpy array of int, shape (img_height, img_width)
        Label image corresponding to the Z-buffer. Each pixel value is the label
        of the object that was projected onto the pixel.
    """

    if range_image is None:
        img_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
        range_image = np.full(img_size, np.inf)

    if label_image is None:
        img_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
        label_image = np.zeros(img_size, dtype=int)

    face_coords = np.array([
        [0, 0],
        [0, range_image.shape[0]],
        [range_image.shape[1], 0],
        [range_image.shape[1], range_image.shape[0]]
    ])

    zBufferConvexPolygon(
        range_image, label_image, camera_pose, camera_params,
        face_coords_image=face_coords, plane=plane,
        face_label=0
    )

    # Render plane appearance
    if rgb_image is None:
        rgb_shape = label_image.shape + plane_appearance.shape
        rgb_image = np.zeros(rgb_shape)
    rgb_image[:,:] = plane_appearance

    return rgb_image, range_image, label_image


def renderComponent(
        state, component_idx, component_pose=None, img_type=None,
        camera_pose=None, camera_params=None, block_colors=None,
        range_image=None, label_image=None, rgb_image=None,
        crop_rendered=False, in_place=True):
    """ Render a component of the state.

    Parameters
    ----------
    state : blockassembly.BlockAssembly
        Spatial assembly that should be rendered.
    component_index : int
        Index of the sub-component of the spatial assembly that should be
        rendered.
    component_pose : tuple(numpy array of shape (3,3), numpy array of shape (3,))
        This component's pose with respect to the canonical retinal coordinate
        frame, represented as a rotation matrix and translation vector :math:`(R, t)`.
        Units are expressed in millimeters.
    img_type : {'rgb', 'depth', 'label', None}
        If None, this function returns all three images. Otherwise it returns
        the specified image only.
    camera_pose : numpy array of float, shape (4, 4)
        The camera's pose with respect to the world coordinate frame. This is
        a rigid motion (R, t), represented as a homogeneous matrix.
    camera_params : numpy array of float, shape (3, 4)
        The camera's intrinsic parameters.
    block_colors : numpy array, shape (num_blocks + 1, 3)
        Each row is the color of a block. Note that the first row corresponds
        to the background.
    crop_rendered : bool, optional
        If True, the rendered image is cropped to a bounding box around the
        nonzero portion. Default is True.
    range_image : numpy array of float, shape (img_height, img_width), optional
        Pre-existing Z_buffer. Each pixel value is the distance from the camera
        in mm.
    label_image : numpy array of int, shape (img_height, img_width), optional
        Label image corresponding to the Z-buffer. Each pixel value is the label
        of the object that was projected onto the pixel.
    rgb_image : numpy array of float, shape (img_height, img_width, 3), optional
        Pre-existing RGB image.
    in_place : bool, optional
        If True, this function modifies the pre-existing images when rendering.
        Otherwise it makes a local copy.

    Returns
    -------
    rgb_image : numpy array of float, shape (img_height, img_width, 3)
        Color image in RGB format.
    range_image : numpy array of float, shape (img_height, img_width)
        The Z_buffer. Each pixel value is the distance from the camera in mm.
    label_image : numpy array of int, shape (img_height, img_width)
        Label image corresponding to the Z-buffer. Each pixel value is the label
        of the object that was projected onto the pixel.
    """

    if not in_place:
        if rgb_image is not None:
            rgb_image = rgb_image.copy()
        if range_image is not None:
            range_image = range_image.copy()
        if label_image is not None:
            label_image = label_image.copy()

    if component_pose is None:
        R = np.eye(3)
        t = np.zeros(3)
    else:
        R, t = component_pose

    if state.blocks:
        s = state.copy()
        s.centerComponent(component_idx, zero_at='centroid')
        s.centerComponent(component_idx, zero_at='smallest_z')
        s.setComponentPose(component_idx, R, t)

        range_image, label_image = zBufferComponent(
            s, component_idx, camera_pose, camera_params,
            range_image=range_image, label_image=label_image
        )

        if crop_rendered:
            range_image = cropImage(range_image)
            label_image = cropImage(label_image)
    else:
        range_image = np.zeros((1, 1), dtype=float)
        label_image = np.zeros((1, 1), dtype=int)

    # Render block appearances using the label image and block colors
    if rgb_image is None:
        rgb_shape = label_image.shape + block_colors.shape[1:2]
        rgb_image = np.zeros(rgb_shape, dtype=block_colors.dtype)
    if label_image.any():
        for i in range(1, label_image.max() + 1):
            obj_patch = label_image == i
            rgb_image[obj_patch, :] = block_colors[i, :]

    if img_type == 'rgb':
        return rgb_image
    elif img_type == 'depth':
        return range_image
    elif img_type == 'label':
        return label_image
    return rgb_image, range_image, label_image


def zBufferComponent(
        state, component_index, camera_pose, camera_params,
        range_image=None, label_image=None):
    """ Render depth and label images of a component of a spatial assembly.

    Parameters
    ----------
    state : blockassembly.BlockAssembly
        Spatial assembly that should be rendered.
    component_index : int
        Index of the sub-component of the spatial assembly that should be
        rendered.
    camera_pose : numpy array of float, shape (4, 4)
        The camera's pose with respect to the world coordinate frame. This is
        a rigid motion (R, t), represented as a homogeneous matrix.
    camera_params : numpy array of float, shape (3, 4)
        The camera's intrinsic parameters.
    range_image : numpy array of float, shape (img_height, img_width), optional
        Pre-existing Z_buffer. Each pixel value is the distance from the camera
        in mm.
    label_image : numpy array of int, shape (img_height, img_width), optional
        Label image corresponding to the Z-buffer. Each pixel value is the label
        of the object that was projected onto the pixel.

    Returns
    -------
    range_image : numpy array of float, shape (img_height, img_width)
        The Z-buffer. Each pixel value is the distance from the camera in mm.
    label_image : numpy array of int, shape (img_height, img_width)
        Label image corresponding to the Z-buffer. Each pixel value is the label
        of the object that was projected onto the pixel.
    """

    if range_image is None:
        img_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
        range_image = np.full(img_size, np.inf)

    if label_image is None:
        img_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
        label_image = np.zeros(img_size, dtype=int)

    component = state.connected_components[component_index]
    for index in component:
        block = state.getBlock(index)
        zBufferBlock(block, range_image, label_image, camera_pose, camera_params)

    range_image[np.isinf(range_image)] = 0  # np.nan

    return range_image, label_image


def zBufferBlock(block, range_image, label_image, camera_pose, camera_params):
    """ Draw a block to the Z_buffer.

    Parameters
    ----------
    block : blockassembly.Block
        The block to render.
    range_image : numpy array of float, shape (img_height, img_width)
        The Z_buffer. Each pixel value is the distance from the camera in mm.
    label_image : numpy array of int, shape (img_height, img_width)
        Label image corresponding to the Z-buffer. Each pixel value is the label
        of the object that was projected onto the pixel.
    camera_pose : numpy array of float, shape (4, 4)
        The camera's pose with respect to the world coordinate frame. This is
        a rigid motion (R, t), represented as a homogeneous matrix.
    camera_params : numpy array of float, shape (3, 4)
        The camera's intrinsic parameters.
    """

    img_h, img_w = range_image.shape

    block_label = block.index + 1

    vertex_coords = block.metric_vertices
    for i, face_coords in enumerate(makeFaces(vertex_coords)):
        zBufferConvexPolygon(
            range_image, label_image, camera_pose, camera_params,
            face_coords_world=face_coords, face_label=block_label
        )


def makeFaces(vertex_coords):
    """ Construct a cube's faces from its vertices.

    Parameters
    ----------
    vertex_coords : numpy array of float, shape (num_vertices, 3)
        Vertex coordinates in the world frame.

    Returns
    -------
    faces : generator(numpy array of float, shape (4, 3)
        Coordinates of each face. For each face, coordinates are arranged in
        conter-clockwise order.
    """

    vertex_indices = (
        [0, 1, 2, 3],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [3, 2, 6, 7],
        [4, 5, 6, 7],
        [0, 3, 7, 4]
    )

    faces = (vertex_coords[idxs, :] for idxs in vertex_indices)
    return faces


def zBufferConvexPolygon(
        range_image, label_image, camera_pose, camera_params,
        face_coords_world=None, face_coords_image=None, plane=None,
        face_label=0):
    """ Draw a convex polygon to the Z-buffer.

    Parameters
    ----------
    range_image : numpy array of float, shape (img_height, img_width)
        The Z_buffer. Each pixel value is the distance from the camera in mm.
    label_image : numpy array of int, shape (img_height, img_width)
        Label image corresponding to the Z-buffer. Each pixel value is the label
        of the object that was projected onto the pixel.
    camera_pose : numpy array of float, shape (4, 4)
        The camera's pose with respect to the world coordinate frame. This is
        a rigid motion (R, t), represented as a homogeneous matrix.
    camera_params : numpy array of float, shape (3, 4)
        The camera's intrinsic parameters.
    face_coords_world : numpy array of float, shape (num_face_points, 3), optional
        Coordinates of the vertices of this face, in the world reference frame.
    face_coords_image : numpy array of float, shape (num_face_points, 2), optional
        Coordinates of the vertices of this face, in the image reference frame.
    plane : geometry.Plane, optional
        A Plane object whose parameters are expressed in the camera reference
        frame.
    face_label : int, optional
        The integer label associated with this face.
    """

    if face_coords_image is None:
        # Project face vertices from world coordinates to pixel coordinates
        proj = geometry.homogeneousMatrix(np.eye(3), np.zeros(3))
        face_coords_camera, _ = geometry.projectHomogeneous(
            geometry.homogeneousVector(face_coords_world) @ m.np.transpose(camera_pose)
        )
        face_coords_image, _ = geometry.projectHomogeneous(
            geometry.homogeneousVector(face_coords_camera) @ m.np.transpose(camera_params @ proj)
        )
        face_coords_image = utils.roundToInt(face_coords_image)
    elif face_coords_world is None:
        # Consruct face_coords_camera in a way that allows
        # geometry.slopeIntercept to compute the plane parameters it needs.
        face_coords_camera = np.zeros((3, 3))
        face_coords_camera[0, :] = plane._t
        face_coords_camera[1, :] = plane._t + plane._U[:, 0]
        face_coords_camera[2, :] = plane._t + plane._U[:, 0] + plane._U[:, 1]
    else:
        err_str = f"This function requires either face_coords_image or face_coords_world"
        raise ValueError(err_str)

    bounding_box = geometry.boundingBox(face_coords_image, range_image.shape)

    try:
        pixel_in_hull = geometry.in_hull(bounding_box, face_coords_image)
    except QhullError:
        return

    image_pixels = bounding_box[pixel_in_hull,:]
    if image_pixels.shape[0] == 1:
        # msg_str = 'Only one pixel in object image: {}'.format(image_pixels)
        # logger.warn(msg_str)
        return

    # Backproject each pixel in the face to its location in camera coordinates
    n, b = geometry.slopeIntercept(face_coords_camera)
    metric_coords_camera = geometry.backprojectIntoPlane(image_pixels, n, b, camera_params)

    # Remove any points that are occluded by another face
    z_computed = metric_coords_camera[:,2]
    rows = image_pixels[:,1]
    cols = image_pixels[:,0]
    computed_is_nearer = z_computed < range_image[rows, cols]
    rows = rows[computed_is_nearer]
    cols = cols[computed_is_nearer]
    z_computed = z_computed[computed_is_nearer]

    # Write to the Z-buffer and label image
    range_image[rows, cols] = z_computed
    label_image[rows, cols] = face_label
