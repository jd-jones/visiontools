import logging
import operator
import functools
import warnings
import math

from matplotlib import pyplot as plt
from skimage import color, feature, morphology, segmentation
from skimage import transform, draw, filters, img_as_float
from skimage.future import graph
from sklearn import cluster
import numpy as np

from mathtools import utils
# FIXME: Remove dependency on definitions
from blocks.core import definitions as defn
from visiontools import geometry


logger = logging.getLogger(__name__)


# -=( PIXEL ACCESS )==---------------------------------------------------------
def imageFromForegroundPixels(
        foreground_pixels, foreground_labels, image_transform=None, background_class_index=None):
    """ Construct an image from its foreground pixels.

    Parameters
    ----------
    foreground_pixels : numpy array, shape (num_foreground_pixels, num_channels)
        The foreground pixels in `image`, as identified by `foreground_labels`.
        If a value was passed for `image_transform`, the number of columns in
        the output could be different.
    foreground_labels : numpy array of bool or int, shape (img_height, img_width)
        Pixel-level label image. See argument `background_class_index`.
    image_transform : funtion, optional
        If provided, this function is applied to `image` after it is constructed
        from `foreground pixels`.
    background_class_index : int, optional
        If this argument is `None`, `foreground_labels` is treated as a boolean
        array whose (i, j)-th element is True if the (i, j)-th pixel of `image`
        is in the foreground, and False if it is in the background.
        If this argument is an integer, `foreground_labels` is treated as an
        array of class labels. This can be convenient if you have a label image
        that assigns labels >= 1 to foreground and 0 to background, for example.
        Default is None.

    Returns
    -------
    image : numpy array, shape (img_height, img_width, num_channels)
        Each pixel takes the value zero if in the background, and takes a value
        from `foreground_pixels` if in the foreground. If a value was passed
        for `image_transform`, the number of channels in the output could be
        different from `num_channels`.

    See Also
    --------
    features.foregroundPixels
    """

    if not foreground_pixels.any():
        return np.zeros(foreground_labels.shape, dtype=foreground_pixels.dtype)

    if background_class_index is None:
        is_foreground = foreground_labels
    else:
        is_foreground = ~(foreground_labels == background_class_index)

    image_dims = foreground_labels.shape
    if len(foreground_pixels.shape) > 1:
        num_channels = foreground_pixels.shape[1]
        image_dims += (num_channels,)

    image = np.zeros(image_dims, dtype=foreground_pixels.dtype)
    image[is_foreground] = foreground_pixels

    if image_transform is not None:
        # Colorspace conversion transforms can throw a divide-by-zero warning
        # for images with zero-valued pixels, but we don't want to see them.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "divide by zero")
            image = image_transform(image)

    return image


def foregroundPixels(image, foreground_labels, image_transform=None, background_class_index=None):
    """ Extract foreground pixels from an image.

    Parameters
    ----------
    image : numpy array, shape (img_height, img_width, num_channels)
        Image the foreground pixels will be extracted from.
    foreground_labels : numpy array of bool or int, shape (img_height, img_width)
        Pixel-level label image. See argument `background_class_index`.
    image_transform : funtion, optional
        If provided, this function is applied to `image` before extracting
        foreground pixels.
    background_class_index : int, optional
        If this argument is `None`, `foreground_labels` is treated as a boolean
        array whose (i, j)-th element is True if the (i, j)-th pixel of `image`
        is in the foreground, and False if it is in the background.
        If this argument is an integer, `foreground_labels` is treated as an
        array of class labels. This can be convenient if you have a label image
        that assigns labels >= 1 to foreground and 0 to background, for example.
        Default is None.

    Returns
    -------
    foreground_pixels : numpy array, shape (num_foreground_pixels, num_channels)
        The foreground pixels in `image`, as identified by `foreground_labels`.
        If a value was passed for `image_transform`, the number of columns in
        the output could be different from `num_channels`.

    See Also
    --------
    features.imageFromForegroundPixels
    """

    if background_class_index is None:
        is_foreground = foreground_labels
    else:
        is_foreground = ~(foreground_labels == background_class_index)

    if image_transform is not None:
        # Colorspace conversion transforms can throw a divide-by-zero warning
        # for images with zero-valued pixels, but we don't want to see them.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "divide by zero")
            image = image_transform(image)

    foreground_pixels = image[is_foreground]
    return foreground_pixels


def matchingPixels(reference_image, match_value=None):
    """ Find the coordinates of pixels matching a particular value.

    Parameters
    ----------
    reference_image : numpy array, shape (img_height, img_width, num_channels)
        This image will be matched against `match_value`.
    match_value : int, optional
        If a value is passed for this argument, the function returns the
        coordinates of pixels matching this value in the input. If not, the
        function returns the coordinates of all nonzero pixels in the input.

    Returns
    -------
    nonzero_rows :
    nonzero_cols :
    """

    # Reduce an RGB image along its channel dimension
    if len(reference_image.shape) > 2:
        reference_image = reference_image.sum(2)

    if match_value is not None:
        reference_image = reference_image == match_value

    nonzero_rows, nonzero_cols = np.nonzero(reference_image)
    return nonzero_rows, nonzero_cols


# -=( VISUALIZATION )==--------------------------------------------------------
def displayImages(*images, num_rows=1, num_cols=None, figsize=None, file_path=None):
    """ Display images in a horizontally-oriented array  using matplotlib.

    Parameters
    ----------
    *images : numpy array, shape (img_height, img_width)
        The images to display.
    num_rows : int, optional
        Number of subplot rows.
    figsize : tuple(int, int), optional
        Figure size. If `figsize` is None, the default value is (5, 3 * num_images).
    file_path : string, optional
        Location where the figure should be saved. If `None`, the figure is
        displayed to the iPython console instead.
    """

    num_images = len(images)
    if num_cols is None:
        num_cols = math.ceil(num_images / num_rows)
    if num_rows is None:
        num_rows = math.ceil(num_images / num_cols)

    if figsize is None:
        # figsize = (5 * num_cols, 3 * num_rows)
        figsize = (3 * num_cols, 3 * num_rows)

    if num_images == 0:
        warn_str = 'No image arguments provided!'
        logger.warning(warn_str)
        return

    if num_images == 1:
        displayImage(images[0], figsize=figsize, file_path=file_path)
        return

    f, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    for axis, img in zip(axes.ravel(), images):
        axis.imshow(img)
        axis.axis('off')
    plt.tight_layout()

    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()


def displayImage(image, figsize=None, file_path=None):
    if figsize is None:
        figsize = (10, 6)

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()

    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path)
        plt.close()


def getPixelCoords(label_image, label_index):
    matches_label = label_image == label_index
    rows_and_cols = np.nonzero(matches_label)
    return np.column_stack(rows_and_cols)


# -=( COLOR SPACE PROCESSING )==-----------------------------------------------
def shiftHue(hue_img):
    hue_rows, hue_cols = hue_img.shape
    should_shift = hue_img > 0.5
    shifted_hue = hue_img.ravel() - 1

    shifted_hue[~should_shift.ravel()] = hue_img.ravel()[~should_shift.ravel()]
    shifted_hue = shifted_hue.reshape(hue_rows, hue_cols)
    return shifted_hue


def invShiftHue(hue_img):
    hue_rows, hue_cols = hue_img.shape
    should_shift = hue_img < 0
    shifted_hue = hue_img.ravel() + 1

    shifted_hue[~should_shift.ravel()] = hue_img.ravel()[~should_shift.ravel()]
    shifted_hue = shifted_hue.reshape(hue_rows, hue_cols)
    return shifted_hue


def quantizeHue(superpixel_image, hsv_image, sat_thresh=0.4, val_thresh=0.25):
    hue_image = hsv_image[:,:,0].copy()
    sat_image = hsv_image[:,:,1].copy()
    val_image = hsv_image[:,:,2].copy()

    hue_image = shiftHue(hue_image)

    num_superpixels = superpixel_image.max() + 1
    hue_labels = np.zeros_like(superpixel_image)
    quantized_hue = np.zeros_like(hue_image)

    for superpixel_index in range(1, num_superpixels):
        superpixel_mask = superpixel_image == superpixel_index

        hue_patch = hue_image[superpixel_mask]
        sat_patch = sat_image[superpixel_mask]
        val_patch = val_image[superpixel_mask]

        if np.median(sat_patch) < sat_thresh:
            continue

        if np.median(val_patch) < val_thresh:
            continue

        closest_hue_name = closestHue(hue_patch, defn.hue_name_dict)
        if closest_hue_name in defn.nuisance_hue_names:
            continue

        closest_hue_label = defn.hue_label_dict[closest_hue_name] + 1
        closest_hue = defn.hue_name_dict[closest_hue_name]
        hue_labels[superpixel_mask] = closest_hue_label
        quantized_hue[superpixel_mask] = closest_hue

    quantized_hue = invShiftHue(quantized_hue)
    return quantized_hue, hue_labels


def closestHue(hue_patch, hue_dict):
    med_hue = np.median(hue_patch)

    best_hue_name = None
    best_hue_dist = np.inf
    for hue_name, hue in hue_dict.items():
        hue_dist = abs(med_hue - hue)
        if hue_dist < best_hue_dist:
            best_hue_name = hue_name
            best_hue_dist = hue_dist

    return best_hue_name


def quantizeImage(model, rgb_image, foreground_label_image):
    # Convert image to HSV space because cluster centers are in HSV space
    hsv_image = color.rgb2hsv(rgb_image)

    # Make a background mask
    in_foreground = foreground_label_image != 0
    foreground_pixels = hsv_image[in_foreground, :]

    # Assign each foreground pixel to the nearest cluster center
    cluster_labels = model.predict(foreground_pixels)
    quantized_hsv = model.cluster_centers_[cluster_labels]

    # Make an image displaying the segmentation
    cluster_label_image = np.zeros(foreground_label_image.shape, dtype=int)
    cluster_label_image[in_foreground] = cluster_labels + 1

    # Make a quantized image
    quantized_hsv_image = np.zeros_like(hsv_image)
    quantized_hsv_image[in_foreground] = quantized_hsv

    # Convert from HSV to RGB
    quantized_rgb_image = color.hsv2rgb(quantized_hsv_image)

    return quantized_rgb_image, cluster_label_image


def saturateImage(rgb_image, background_mask=None, to_float=True):
    """ Convert to HSV, set saturation and value to max, and convert back to RGB.

    If the input image is in integer format (values in [0, 255]), it is first
    converted to float format (values in [0, 1]).

    Parameters
    ----------
    rgb_image : numpy array of float or int, shape (img_height, img_width, num_channels)
    background_mask : numpy array of bool, shape (img_height, img_width)
        If provided, this function only saturates the foreground of the image.

    Returns
    -------
    rgb_saturated : numpy array of float, shape (img_height, img_width, num_channels)
        Saturated copy of the input, in float format.
    """

    if to_float:
        rgb_image = img_as_float(rgb_image)

    # Convert to HSV space
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero")
        hsv_image = color.rgb2hsv(rgb_image)

    # Saturate in HSV space
    hue = hsv_image[:, :, 0]
    ONE = np.ones_like(hue)
    hsv_saturated = np.dstack((hue, ONE, ONE))

    # Convert back to RGB space
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero")
        rgb_saturated = color.hsv2rgb(hsv_saturated)

    if background_mask is not None:
        rgb_saturated[background_mask] = rgb_image[background_mask]

    return rgb_saturated


# -=( GRADIENT )==-------------------------------------------------------------
def imgGradient(img, sigma=None):
    """ Return image gradients in the row and column directions. """

    # gradient in X direction (cols)
    aug_img = np.hstack((img[:,0:1], img))
    grad_c = np.diff(aug_img, 1)

    # gradient in Y direction (rows)
    aug_img = np.vstack((img[0:1,:], img))
    grad_r = np.diff(aug_img, 0)

    if sigma is not None:
        grad_c = filters.gaussian(grad_c, sigma=sigma, multichannel=True)
        grad_r = filters.gaussian(grad_r, sigma=sigma, multichannel=True)

    return grad_r, grad_c


def templateGradient(T, V, U, theta, viz=False):
    Ur, Uc = utils.splitColumns(U)

    Tr, Tc = imgGradient(T)

    if viz:
        logger.info('U: {} {}'.format(U.any(), U))
        tr_img = np.abs(Tr)
        displayImages(tr_img, np.abs(Tc))
        tr_img[Ur, Uc] = [1, 0, 1]
        displayImages(tr_img)

    if len(T.shape) > 2:
        Tr = Tr.sum(2)
        Tc = Tc.sum(2)

    dT_dtr = Tr[Ur, Uc]
    dT_dtc = Tc[Ur, Uc]
    dT_dt = np.column_stack((dT_dtr, dT_dtc))

    dR_dtheta = geometry.rDot(theta)
    dT_dtheta = np.sum(dT_dt * (V @ np.transpose(dR_dtheta)), 1)

    return dT_dtr, dT_dtc, dT_dtheta


# -=( LOSS FUNCTIONS )==-------------------------------------------------------
def sse(observed, predicted, is_img=False):
    """
    Compute the sum of squared errors (SSE) measure for predicting `predicted`
    when we really see `observed`.

    FIXME: Because it calls np.linalg.norm, this function actually computes the
        square root of the SSE metric.

    Parameters
    ----------
    observed : np array of float, shape variable
        The observed image.
        If `is_img`, shape is (NUM_ROWS, NUM_COLS, NUM_CHANNELS)
        If not `is_img`, shape is (NUM_FOREGROUND_PIXELS * NUM_CHANNELS,)
    predicted : np array of float, shape variable
        The predicted (rendered) image.
        If `is_img`, shape is (NUM_ROWS, NUM_COLS, NUM_CHANNELS)
        If not `is_img`, shape is (NUM_FOREGROUND_PIXELS * NUM_CHANNELS,)
    is_img : bool, optional
        [DEPRECATED---WILL BE REMOVED]
        Information about the shape of the input arrays. If `True`, the inputs
        are standard multichannel image arrays. If `False`, the inputs have
        been flattened into vectors.

    Returns
    -------
    sse : float

    Raises
    ------
    ValueError
        When `observed.shape != predicted.shape`
    """

    obsv_shape = observed.shape
    pred_shape = predicted.shape

    if observed.shape != predicted.shape:
        err_str = 'Observation shape ({}) differs from predicted shape ({})!'
        raise ValueError(err_str.format(obsv_shape, pred_shape))

    residual = observed - predicted

    if len(residual.shape) > 2:
        sse = 0
        for i in range(residual.shape[2]):
            sse += np.linalg.norm(residual[:,:,i], ord='fro')
    else:
        sse = np.linalg.norm(residual)

    return sse


def mse(observed, predicted, is_img=False):
    """
    Compute the mean squared error (MSE) measure for predicting `predicted`
    when we really see `observed`.

    Parameters
    ----------
    observed : np array of float, shape variable
        The observed image.
        If `is_img`, shape is (NUM_ROWS, NUM_COLS, NUM_CHANNELS)
        If not `is_img`, shape is (NUM_FOREGROUND_PIXELS * NUM_CHANNELS,)
    predicted : np array of float, shape variable
        The predicted (rendered) image.
        If `is_img`, shape is (NUM_ROWS, NUM_COLS, NUM_CHANNELS)
        If not `is_img`, shape is (NUM_FOREGROUND_PIXELS * NUM_CHANNELS,)
    is_img : bool, optional
        Information about the shape of the input arrays. If `True`, the inputs
        are standard multichannel image arrays. If `False`, the inputs have
        been flattened into vectors.

    Returns
    -------
    mse : float

    Raises
    ------
    ValueError
        When `observed.shape != predicted.shape`
    """

    if is_img:
        num_elem = functools.reduce(operator.mul, observed.shape)
    else:
        # FIXME: This doesn't count the number of channels!!!
        num_elem = observed.shape[0]

    return sse(observed, predicted, is_img=is_img) / num_elem


def overlap(observed, predicted, tol=0.001):
    """ Compute the number of matching pixels (up to tolerance `tol`)

    Parameters
    ----------
    observed : np array of float, shape (NUM_ROWS, NUM_COLS, NUM_CHANNELS)
        The observed image.
    predicted : np array of float, shape (NUM_ROWS, NUM_COLS, NUM_CHANNELS)
        The predicted (rendered) image.
    tol : float, optional
        Pixels in `observed` and `predicted` are considered equivalent if the
        sum of the errors for each individual channel is less than this value.
        FIXME: measure square or absolute error

    Returns
    -------
    overlap : float

    Raises
    ------
    ValueError
        When `observed.shape != predicted.shape`
    """

    obsv_shape = observed.shape
    tmpl_shape = predicted.shape

    if observed.shape != predicted.shape:
        err_str = 'Observed shape ({}) differs from predicted shape ({})!'
        raise ValueError(err_str.format(obsv_shape, tmpl_shape))

    innovation = (observed - predicted).sum(2)
    return (innovation > tol).sum()


# -=( 3D GEOMETRY )==-----------------------------------------------------
def estimateSegmentPose(
        camera_params, camera_pose, depth_image, segment_mask,
        estimate_orientation=True):
    """ Estimate the 3D pose of an image segment.

    Parameters
    ----------
    camera_params : numpy array of float, shape (3, 3)
        The intrinsic parameters of the camera that captured this image.
    camera_pose : numpy array of float, shape (4, 4)
        Homogeneous matrix represing the rigid transformation :math:`y = R x + t`.
    estimate_orientation: bool, optional
        If False, this function returns the identity matrix instead of
        estimating the orientation using SVD. Can be useful when there are
        many points, because the current implementation is inefficient.

    Returns
    -------
    R : numpy array of float, shape (3, 3)
        Rotation matrix representing the segment's estimated 3D orientation (in
        mm).
    t : numpy array of float, shape (3,)
        Vector reptresenting the segment's estimated 3D position (in mm)
    """

    pixel_coords = np.column_stack(np.nonzero(segment_mask))
    depth = depth_image[pixel_coords[:,0], pixel_coords[:,1]]

    world_coords = backprojectPixels(
        camera_params, camera_pose, np.flip(pixel_coords, axis=1), depth
    )

    R, t = geometry.estimatePose(
        world_coords, xy_only=True, estimate_orientation=estimate_orientation
    )

    return R, t


def backprojectPixels(camera_params, camera_pose, pixel_coords, depths, in_camera_coords=False):
    """ Backproject multiple pixels using a pinhole camera model.

    This function basically inverts the image formation equation
    :math..
        x = \frac{f}{z} X + o
    to obtain
    :math..
        X = \frac{z}{f} (x - o)

    Parameters
    ----------
    camera_params : numpy array of float, shape (3, 3)
        The intrinsic parameters of the camera that captured this image.
    camera_pose : numpy array of float, shape (4, 4)
        Homogeneous matrix represing the rigid transformation :math:`y = R x + t`.
    pixel_coords : numpy array of int, shape (num_pixels, 2)
        Each pixel's image coordinate, expressed as (column, row) ie (X, Y)---
        NOT the standard (row, column) format.
    depths : numpy array of float, shape (num_pixels,)
        Each pixel's distance from the camera.

    Returns
    -------
    world_coords : numpy array of float, shape (num_pixels, 3)
        Each pixel's metric coordinates with respect to the world reference
        frame. Units are expressed in millimeters.
    """

    num_pixels, num_dims = pixel_coords.shape
    if num_dims != 2:
        err_str = 'pixel_coords is shape ({}, {}), but should be (num_pixels, 2)'
        raise ValueError(err_str.format(num_pixels, num_dims))

    # Estimate pixel coordinates in the canonical retinal frame
    backproject = geometry.invertHomogeneous(
        camera_params, A_property='diag', range_space_homogeneous=True
    )

    camera_coords_scaled = geometry.homogeneousVector(pixel_coords) @ np.transpose(backproject)
    camera_coords = camera_coords_scaled * np.column_stack((depths, depths, depths))

    if in_camera_coords:
        return camera_coords

    # Convert pixel coordinates to the world frame
    inv_camera_pose = geometry.invertHomogeneous(camera_pose, A_property='ortho')
    world_coords = geometry.homogeneousVector(camera_coords) @ np.transpose(inv_camera_pose)

    if len(world_coords.shape) > 1:
        world_coords = world_coords[:, 0:3].squeeze()
    else:
        world_coords = world_coords[0:3]

    return world_coords


def backprojectSegment(camera_params, depth_image, label_image, label_index):
    pixel_coords = getPixelCoords(label_image, label_index)
    z_coords = depth_image[pixel_coords[:,0], pixel_coords[:,1]]
    metric_coords = backprojectPixels(
        camera_params, np.flip(pixel_coords, axis=1), z_coords
    )
    return metric_coords


# -=( 2D GEOMETRY )==----------------------------------------------------------
def imageIntervals(image_shape):
    intervals = tuple((0, size - 1) for size in image_shape)
    return intervals


def imageMidpoint(image_shape, integer_divide=False):
    if integer_divide:
        return np.array([dim_size // 2 for dim_size in image_shape])
    return np.array([dim_size / 2 for dim_size in image_shape])


def projectIntoImage(pixel_coords, image_shape):
    intervals = imageIntervals(image_shape)
    return geometry.projectIntoVolume(pixel_coords, *intervals)


def rectangle_perimeter(r_bounds, c_bounds, shape=None, clip=False):
    rr = [r_bounds[0], r_bounds[1], r_bounds[1], r_bounds[0]]
    cc = [c_bounds[0], c_bounds[0], c_bounds[1], c_bounds[1]]
    return draw.polygon_perimeter(rr, cc, shape=shape, clip=clip)


def nonzeroRange(collection):
    """ return min nonzero index and max nonzero index """
    if not any(collection):
        return tuple()

    min_idx = collection.min()
    max_idx = collection.max()

    return min_idx, max_idx


# -=( MASKING )==--------------------------------------------------------------
def majorityVote(labels, min_snr=None, min_signal_count=None):
    if min_snr is not None and min_signal_count is not None:
        err_str = 'Only one of min_snr and min_signal_count can be passed!'
        raise ValueError(err_str)

    # background_count = np.sum(labels == 0)
    noise_count = np.sum(labels == 1) + np.sum(labels == 2)
    signal_count = np.sum(labels == 3)

    if not signal_count and not noise_count:
        return 0

    if not noise_count:
        return 3

    if not signal_count:
        return 2

    if signal_count > noise_count:
        return 3

    if min_snr is not None:
        snr = signal_count / noise_count
        if snr > min_snr:
            return 1
        return 2

    if min_signal_count is not None:
        if signal_count > min_signal_count:
            return 1
        return 2

    return 1


def maskDepthArtifacts(depth_image, lower_bound=0, upper_bound=750):
    """ Identify artifact pixels in a depth image.

    Parameters
    ----------
    depth_image : numpy array of float, shape (img_height, img_width)
    lower_bound : float, optional
        Pixels with depth values below this threshold will be masked. Usually
        set to 0, to mask registration artifacts.
    upper_bound : float, optional
        Pixels with depth values above this threshold will be basked. Usually
        set to some margin above the maximum depth value possible for the scene.

    Returns
    -------
    is_depth_artifact : numpy array of bool, shape (img_height, img_width)
        A mask array. A pixel's value is True if its corresponding depth pixel
        falls outside the range ``(lower_bound, upper_bound)``.
    """

    # Find zero-depth pixels. These are registration artifacts.
    depth_too_low = depth_image <= lower_bound

    # Find artifact pixels (far enough away from the camera that they can't
    # have been in the real scene)
    depth_too_high = depth_image >= upper_bound

    is_depth_artifact = depth_too_low + depth_too_high
    return is_depth_artifact


def maskOutsideBuildArea(label_image, mask_left_side=0.3, mask_bottom=0.4):
    """ Segment the frame into three rectangles. """

    rows, cols = label_image.shape
    row_bound = int(mask_bottom * rows)
    col_bound = int(mask_left_side * cols)

    if mask_left_side:
        label_image[:, :col_bound] = 1

    if mask_bottom:
        label_image[row_bound:, col_bound:] = 1

    return label_image


def maskBackground(rgb_image, depth_image):
    hsv = color.rgb2hsv(rgb_image)
    hue_as_rgb = hueImgAsRgb(hsv)

    background_mask = makeBackgroundMask(depth_image)
    background_mask = maskOutsideBuildArea(background_mask)

    rgb_foreground = hue_as_rgb.copy()
    rgb_foreground[background_mask] = 0

    return img_as_float(rgb_foreground)


# -=( SUPERPIXELS)==-----------------------------------------------------------
def makeSuperpixelImage(
        superpixels, hsv_image, quantized_hue=None, as_hsv=False):
    hue = shiftHue(hsv_image[:,:,0])
    # sat = hsv_image[:,:,1]
    # val = hsv_image[:,:,2]

    if quantized_hue is not None:
        quantized_hue = shiftHue(quantized_hue)

    num_superpixels = superpixels.max() + 1
    superpixel_image = np.zeros_like(hsv_image)

    for superpixel_index in range(1, num_superpixels):
        superpixel_mask = superpixels == superpixel_index

        hue_patch = hue[superpixel_mask]
        # sat_patch = sat[superpixel_mask]
        # val_patch = val[superpixel_mask]

        if quantized_hue is None:
            # superpixel_image[:,:,0][superpixel_mask] = np.median(hue_patch)
            superpixel_image[:,:,0][superpixel_mask] = hue_patch
        else:
            superpixel_image[:,:,0][superpixel_mask] = quantized_hue[superpixel_mask]
        # superpixel_image[:,:,1][superpixel_mask] = np.median(sat_patch)
        superpixel_image[:,:,1][superpixel_mask] = 1
        # superpixel_image[:,:,2][superpixel_mask] = np.median(val_patch)
        superpixel_image[:,:,2][superpixel_mask] = 1

    superpixel_image[:,:,0] = invShiftHue(superpixel_image[:,:,0])

    if as_hsv:
        return superpixel_image

    return color.hsv2rgb(superpixel_image)


def mergeSuperpixels(
        superpixels, rgb_frame, sat_frame, depth_frame,
        rgb_thresh=55, sat_thresh=0.20, depth_thresh=25):

    if rgb_thresh >= 0:
        rgb_rag = graph.rag_mean_color(rgb_frame, superpixels)
        sp_merged_rgb = graph.cut_threshold(superpixels, rgb_rag, rgb_thresh)
        sp_joined = sp_merged_rgb

    if sat_thresh >= 0:
        sat_rag = graph.rag_mean_color(sat_frame, superpixels)
        sp_merged_sat = graph.cut_threshold(superpixels, sat_rag, sat_thresh)
        if rgb_thresh >= 0:
            sp_joined = segmentation.join_segmentations(
                sp_joined, sp_merged_sat)

    if depth_thresh >= 0:
        depth_rag = graph.rag_mean_color(depth_frame, superpixels)
        sp_merged_depth = graph.cut_threshold(
            superpixels, depth_rag, depth_thresh)
        if sat_thresh >= 0:
            sp_joined = segmentation.join_segmentations(
                sp_joined, sp_merged_depth)

    return sp_joined


# -=( SEGMENT PROCESSING )==---------------------------------------------------
def getImageSegments(rgb_image, label_image):
    num_labels = label_image.max() + 1
    f = functools.partial(getImageSegment, rgb_image, label_image)
    return tuple(f(i) for i in range(1, num_labels))


def getImageSegment(image, label_image, label_index):
    pixel_coords = getPixelCoords(label_image, label_index)
    pix_vals = image[pixel_coords[:,0], pixel_coords[:,1]]
    return pix_vals


def removeLargeObjects(label_image, max_size=750):
    relabeled = label_image.copy()

    num_labels = relabeled.max() + 1
    for label_index in range(1, num_labels):
        matching_pixels = label_image == label_index
        num_matching = matching_pixels.sum()
        if num_matching > max_size:
            relabeled[matching_pixels] = 0

    relabeled, __, __ = segmentation.relabel_sequential(relabeled)
    return relabeled


def segmentBoundingBox(label_img, label_index):
    """

    Parameters
    ----------
    label_img :
    label_index :
    """

    if not label_img.any():
        return tuple()

    nonzero_rows, nonzero_cols = matchingPixels(label_img, label_index)
    row_range = nonzeroRange(nonzero_rows)
    col_range = nonzeroRange(nonzero_cols)

    return row_range, col_range


def segmentBoundingBoxes(rgb_img, label_img=None):
    # Make a label image if one wasn't passed
    if label_img is None:
        label_img, num_labels = labelsFromRgb(rgb_img)
    else:
        num_labels = label_img.max() + 1

    bb = functools.partial(segmentBoundingBox, label_img)
    bboxes = tuple(bb(i) for i in range(1, num_labels))
    return tuple(filter(None, bboxes))


def labelsFromRgb(rgb_img):
    channel_sum = rgb_img.sum(2)
    label_img, num_labels = morphology.label(channel_sum != 0, return_num=True)

    return label_img, num_labels


# -=( DEPRECATED )==-----------------------------------------------------------
def featureFilter(num_pix, avg_depth, std_depth, avg_hue, avg_sat, med_sat):
    if med_sat < 0.1:
        return True

    if med_sat > 0.4 and med_sat < 0.6:
        return True

    if num_pix > 1500 and std_depth > 0.7:
        return True

    return False


def sizeFilter(num_pix, avg_depth, std_depth, avg_hue, avg_sat, med_sat):
    if num_pix > 750:
        return True

    return False


def hueImgAsRgb(hsv_image):
    hue = hsv_image[:,:,0]
    ONE = np.ones_like(hue)

    new_hsv = np.zeros_like(hsv_image)
    new_hsv[:,:,0] = hue
    new_hsv[:,:,1] = ONE
    new_hsv[:,:,2] = ONE

    return color.hsv2rgb(new_hsv)


def shift(img, interval_length=1.0):
    """ shift a range of data from [0, interval_length] to
    [-interval_length/2, interval_length/2]
    """

    midpoint = interval_length / 2

    shifted = img.copy()
    shifted[shifted > midpoint] -= interval_length
    shifted[shifted < -midpoint] += interval_length

    return shifted


def makeObjectLabels(
        depth_image, rgb_image,
        num_seg_pixels=200, depth_coeff=1, hue_coeff=100,
        rgb_thresh=55, sat_thresh=0.20, depth_thresh=25):

    hsv_image = color.rgb2hsv(rgb_image)
    hue_image = hsv_image[:,:,0]
    sat_image = hsv_image[:,:,1]
    # val_image = hsv_image[:,:,2]

    background_mask = makeBackgroundMask(depth_image)
    background_mask = maskOutsideBuildArea(background_mask)

    label_mask = labelObjects(background_mask)

    superpixels = segmentObjects(
        label_mask, num_seg_pixels,
        depth_coeff * depth_image,
        hue_coeff * shift(hue_image))

    any_thresh = rgb_thresh >= 0 or sat_thresh >= 0 or depth_thresh >= 0
    if np.any(superpixels) and any_thresh:
        rgb_masked = rgb_image.copy()
        rgb_masked[label_mask == 0] = 0
        sat_masked = sat_image.copy()
        sat_masked[label_mask == 0] = 0
        depth_masked = depth_image.copy()
        depth_masked[label_mask == 0] = 0
        superpixels = mergeSuperpixels(
            superpixels, rgb_masked, sat_masked, depth_masked,
            rgb_thresh=rgb_thresh, sat_thresh=sat_thresh,
            depth_thresh=depth_thresh)

    return superpixels


def maskDepth(is_nuisance, min_snr=None):
    num_nuisance = is_nuisance.sum()
    if not num_nuisance:
        return False

    num_signal = (~is_nuisance).sum()
    if not num_signal:
        return True

    if min_snr is None:
        return False

    depth_snr = num_signal / num_nuisance

    # if depth_snr < min_snr:
    #     avg_depth = depth_pix.mean()
    #     logger.info(f'Mean depth: {avg_depth}')

    return depth_snr < min_snr


def maskNuisance(labels, min_snr=None):
    num_signal = np.sum(labels == 3)
    num_noise = np.sum(labels == 2) + np.sum(labels == 1)

    if not num_noise:
        return False

    if not num_signal:
        return True

    if min_snr is None:
        return False

    snr = num_signal / num_noise
    return snr < min_snr


def maskLargeSegments(labels, max_num_pixels=3000):
    num_pixels = len(labels)
    if num_pixels > max_num_pixels:
        if num_pixels > 65000:
            return False
        return True
    return False


def filterObjects(label_img, img, filter_func, num_labels):
    new_img = np.zeros_like(img)
    for i in range(1, num_labels + 1):
        filterObject(label_img, img, filter_func, new_img, i)

    return new_img


def filterObject(label_img, img, filter_func, new_img, label_idx):
    pixel_matches = label_img == label_idx
    matching_pixels = img[pixel_matches]

    filter_response = filter_func(matching_pixels)
    new_img[pixel_matches] = filter_response

    return new_img


def labelObjects(
        background_mask,
        neighbors=None, background=None, return_num=False, connectivity=None,
        min_size=200, filter_connectivity=1):

    label_mask = morphology.label(
        ~background_mask,
        neighbors=neighbors,
        background=background,
        return_num=return_num,
        connectivity=connectivity)

    label_mask = morphology.remove_small_objects(
        label_mask,
        min_size=min_size,
        connectivity=filter_connectivity)

    return segmentation.relabel_sequential(label_mask)[0]


def segmentObjects(label_mask, num_seg_pixels=200, *images):
    if not images:
        err_str = 'At least one image must be supplied!'
        raise ValueError(err_str)

    segmented_labels = np.zeros_like(label_mask)
    num_objects = label_mask.max() + 1
    offset = 1

    for object_index in range(offset, num_objects):
        object_mask = label_mask == object_index

        num_segments = int(object_mask.sum() / num_seg_pixels)
        num_segments = max(num_segments, 1)

        pixel_indices = np.argwhere(object_mask)
        points = [pixel_indices]
        for image in images:
            img_points = image[object_mask].reshape(-1, 1)
            points.append(img_points)
        points = np.hstack(tuple(points))

        segments = segmentObject(points, num_segments)

        segmented_labels[object_mask] = segments + offset
        offset += num_segments

    return segmented_labels


def segmentObject(points, num_segments):
    kmeans = cluster.KMeans(n_clusters=num_segments)
    kmeans.fit(points)  # .reshape(-1,1))
    return kmeans.labels_


vote = functools.partial(majorityVote, min_snr=0.25)
mask_depth = functools.partial(maskDepth, min_snr=1)
strict_vote = functools.partial(majorityVote, min_snr=2.0)
