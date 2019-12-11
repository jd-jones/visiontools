import logging
import functools
import itertools
import operator

from scipy import spatial

import mathtools
from mathtools import utils


logger = logging.getLogger(__name__)


# -=( PLANES AND SUBSPACES )==-------------------------------------------------
class AffineSubspace(object):
    """ k-dimensional affine subspace in R^d """

    def __init__(self, k, d):
        # The columns if this matrix span the affine subspace
        self._V = None

        # This translation maps the zero element to its location in the affine
        # subspace
        self._t = None

        if k > d:
            err_str = (
                'k must be <= d, since this object represents '
                'a k-dimensional subspace of R^d'
            )
            raise AssertionError(err_str)

        self._k = k
        self._d = d

    def estimate(self, data):
        """ Fit an affine subspace to data.

        Parameters
        ----------
        data : numpy array of float, shape (num_points, d)

        Returns
        -------
        success : bool
            True if the estimated subspace has rank == k.
        """

        # Fit translation of an affine subspace
        data_mean = data.mean(axis=0)
        centered = data - data_mean

        # Fit subspace to centered data by projecting
        u, s, __ = mathtools.np.linalg.svd(centered.T, full_matrices=False, compute_uv=True)

        # If the data have at least rank k (ie the SVD has at least k singular
        # values), then we can fit a k-dimensional subspace. Otherwise the
        # subspace is degenerate---it has dimensionality less than k.
        success = s.shape[0] >= self._k

        if success:
            self._S = s[:self._k]
            self._U = u[:, :self._k]
            self._t = data_mean

        return success

    def residuals(self, data):
        """ Shortest distance from each point in data to the plane

        Distances are obtained by measuring the point's magnitude orthogonal to
        the plane.

        Parameters
        ----------
        data : numpy array of float, shape (num_points, d)

        Returns
        -------
        residuals : numpy array of float, shape (num_points)
        """

        self._checkFit()

        projected = self._project(data)
        error = data - projected

        return mathtools.np.linalg.norm(error, ord=2, axis=1)

    def _project(self, data):
        """ Orthogonally project data onto the affine subspace.

        Parameters
        ----------
        data : numpy array of float, shape (num_points, d)

        Returns
        -------
        projected : numpy array of float, shape (num_points, k)
        """

        self._checkFit()

        # Shift data to match the affine subspace
        centered = data - self._t

        # Project into affine subspace
        # Transpose multiplication because data is composed of row vectors
        # instead of column vectors
        projector_T = self._U @ self._U.T
        projected = centered @ projector_T

        return projected + self._t

    def _checkFit(self):
        if self._U is None or self._t is None:
            err_str = 'Fit the model using estimate() before calling this method.'
            raise ValueError(err_str)

    @property
    def min_samples(self):
        """ Return the minimum number of samples sufficient to estimate an
        affine subspace of dimension k.
        """

        return self._k + 1


class Plane(AffineSubspace):
    """ 2-dimensional affine subspace in R^3 """

    def __init__(self):
        super().__init__(2, 3)


def standardBasis(dim, i, expand_axis=None):
    """ Construct the i-th standard basis vector e_i.

    Parameters
    ----------
    dim : int
        Dimensionality of the vector space.
    i : int
        Index of the axis along which the basis vector is aligned.
    expand_axis : {0, 1}, optional
        If 0, the basis vector is returned as a row vector with shape (1, dim).
        If 1, the basis vector is returned as a column vector with shape (dim, 1).
        If this argument is omitted, the basis vector is returned with shape (dim,).

    Returns
    -------
    e_i : numpy array of float, shape (dim,) or (1, dim) or (dim, 1)
        A vector of all zeroes with a single one at index `i`. In other words,
        the i-th row or column of the identity matrix.
    """

    e_i = mathtools.np.zeros(dim)
    e_i[i] = 1

    if expand_axis is not None:
        e_i = mathtools.np.expand_dims(e_i, axis=expand_axis)

    return e_i


def projectHomogeneous(x):
    """ Project a set of points in homogeneous coordinates to Euclidean coordinates.

    Each point ``x`` is projected as ``x_p = x[:-1] / x[-1]``

    Parameters
    ----------
    x : numpy array of float, shape (num_points, num_dims + 1) or (num_dims + 1,)
        A collection of points in homogeneous coordinates.

    Returns
    -------
    x_projected : numpy array of float, shape (num_points, num_dims) or (num_dims,)
        Homogeneous points projected into Euclidean coordinates.
    scale : numpy array of float, shape (num_points,) or (,)
        The homogeneous coordinate of each point.
    """

    if len(x.shape) < 2:
        x = mathtools.np.expand_dims(x, axis=0)

    scale = x[:, -1]

    if mathtools.np.any(scale == 0):
        warn_str = (
            f"Tried to project {mathtools.np.sum(scale == 0)} points with homogeneous coord == 0"
        )
        logger.warning(warn_str)
        return x[:-1], scale

    num_dims = x.shape[1] - 1
    x_projected = x[:, :-1] / mathtools.np.column_stack((scale,) * num_dims)

    return x_projected.squeeze(), scale.squeeze()


def homogeneousVector(x):
    """ Return a vector's representation in homogeneous coordinates.

    For an input ``x = [x_1, ..., x_n]``, the output is ``[x_1, ..., x_n, 1]``.

    Parameters
    ----------
    x : numpy array, shape (num_dims,) or (num_points, num_dims)

    Returns
    -------
    x_homogeneous : numpy array, shape (num_dims + 1) or (num_points, num_dims + 1)
    """

    if len(x.shape) < 2:
        x = mathtools.np.expand_dims(x, axis=0)

    ONES = mathtools.np.ones((x.shape[0], 1))
    x_homogeneous = mathtools.np.hstack((x, ONES))

    return x_homogeneous.squeeze()


def homogeneousMatrix(A, b, range_space_homogeneous=False):
    """ Collect affine transformation parameters into a matrix in homogeneous coordinates.

    A homogenous matrix :math:`M` represents the transformation :math:`y = A x + b`
    in homogeneous coordinates. More precisely,
    ..math:
        M \tilde{x} = \left[ \begin{matrix}
            A & b \\
        \end{matrix} \right]
        \left[ \begin{matrix}
            x \\
            1
        \end{matrix} \right]

    Parameters
    ----------
    A : numpy array of float, shape (num_dims,num_dims)
        An arbitrary square matrix.
    b : numpy array of float, shape (num_dims,)
        Translation vector.
    range_space_homogeneous : bool, optional
        If True, the output has an extra row :math:`[ 0 1 ]` appended to the bottom
        so that its range space is also expressed in homogeneous coordinates.

    Returns
    -------
    M : numpy array of float, shape (num_dims, num_dims + 1) or (num_dims + 1, num_dims + 1)
        Affine transformation represented in homogeneous coordinates.
    """

    b = mathtools.np.expand_dims(b, axis=1)
    M = mathtools.np.hstack((A, b))

    if range_space_homogeneous:
        homogeneous_row = standardBasis(M.shape[1], -1, expand_axis=0)
        M = mathtools.np.vstack((M, homogeneous_row))

    return M


def fromHomogeneous(M):
    """ Return the linear and affine parts of a homogenous matrix :math:`M = [A, b]`.

    Parameters
    ----------
    M : numpy array of float, shape (num_dims, num_dims + 1) or (num_dims + 1, num_dims + 1)
        Matrix representing an affine transformation in homogeneous coordinates.
        if ``M.shape == (num_dims + 1, num_dims + 1)``, its last row is :math:`[0 1]`
        so that its output is also in homogeneous coordinates.

    Returns
    -------
    A : numpy array of float, shape (num_dims, num_dims)
    b : numpy array of float, shape (num_dims,)
    """

    num_rows, num_cols = M.shape

    if num_rows == num_cols:
        row_bound = -1
    elif num_cols == num_rows + 1:
        row_bound = None
    else:
        err_str = f"Input shape {M.shape} does not correspond to a homogeneous matrix"
        raise ValueError(err_str)

    A = M[:row_bound, :-1]
    b = M[:row_bound, -1]

    return A, b


def invertHomogeneous(M, range_space_homogeneous=False, A_property=None):
    """ Return the inverse transformation of a homogeneous matrix.

    A homogenous matrix :math:`M` represents the transformation :math:`y = A x + b`
    in homogeneous coordinates. More precisely,
    ..math:
        M \tilde{x} = \left[ \begin{matrix}
            A & b \\
        \end{matrix} \right]
        \left[ \begin{matrix}
            x \\
            1
        \end{matrix} \right]

    Its inverse is the homogeneous matrix that represents the transformation
    :math:`x = A^{-1} ( y - b )`.

    Parameters
    ----------
    M : numpy array of float, shape (num_dims, num_dims + 1) or (num_dims + 1, num_dims + 1)
        Matrix representing an affine transformation in homogeneous coordinates.
        if ``M.shape == (num_dims + 1, num_dims + 1)``, its last row is :math:`[0 1]`
        so that its output is also in homogeneous coordinates.
    range_space_homogeneous : bool, optional
        If True, the output has an extra row :math:`[ 0 1 ]` appended to the bottom
        so that its range space is also expressed in homogeneous coordinates.
    A_property : {'diag', 'ortho'}, optional
        Special property of the submatrix `A` that could make inversion easier.
        If no argument is given, this function just calls `mathtools.np.linalg.pinv`.

    Returns
    -------
    M_inverse : numpy array of float, shape (num_dims, num_dims + 1) or (num_dims + 1, num_dims + 1)
        Inverse transformation corresponding to input `M`.
    """

    if A_property is None:
        invert = mathtools.np.pinv
    elif A_property == 'diag':
        def invert(x):
            return mathtools.np.diag(1 / mathtools.np.diag(A))
    elif A_property == 'ortho':
        invert = mathtools.np.transpose
    else:
        err_str = f"Can't parse keyword argument 'A_property={A_property}'"
        raise ValueError(err_str)

    A, b = fromHomogeneous(M)

    A_inverse = invert(A)
    b_inverse = -A_inverse @ b

    M_inverse = homogeneousMatrix(
        A_inverse, b_inverse,
        range_space_homogeneous=range_space_homogeneous
    )

    return M_inverse


def axisAligned(angle, tol=None, axis=None):
    """ Determine if a line (represented by its angle) is aligned with an axis.

    Parameters
    ----------
    angle : float
        The line's angle of inclination (in radians)
    tol : float
        Maximum distance from `axis` for which `angle` is still considered to
        be aligned.
    axis : {'horizontal', 'vertical'}
        The reference axis.

    Returns
    -------
    is_aligned : bool
        True if `angle` is within `tol` radians of `axis`.
    """

    if axis == 'horizontal':
        target_angle = 1.57     # about pi / 2
    elif axis == 'vertical':
        target_angle = 0.0

    distance = abs(target_angle - abs(angle))
    is_aligned = distance < tol

    return is_aligned


def solveLine(angle, offset, x=None, y=None):
    """ Solve a linear equation for the X or Y coordinate.

    The arguments specify a linear equation in standard form:

    :math..
        y \sin(angle) + x \cos(angle) - offset = 0

    Parameters
    ----------
    angle : float
        Orientation of the line's normal vector (in radians).
    offset : float
        The line's distance from the origin.
    x : float, optional
        Coordinate along the horizontal axis. If a value is provided for `x`,
        this function solves for `y`.
    y : float, optional
        Coordinate along the vertical axis. If a value is provided for `y`,
        this function solves for `x`.

    Returns
    -------
    solution : float
        The solution to this linear equation. See parameters `x` and `y`.
    """

    if x is not None:
        solution = (offset - x * mathtools.np.cos(angle)) / mathtools.np.sin(angle)
        return solution

    if y is not None:
        solution = (offset - y * mathtools.np.sin(angle)) / mathtools.np.cos(angle)
        return solution


def rc2xy(Ur, Uc, image_dims):
    """ Convert an input from pixel to Cartesian coordinates.

    Parameters
    ----------
    Ur : numpy array of float, shape (num_points,)
        Pixel row coordinates.
    Uc : numpy array of float, shape (num_points,)
        Pixel column coordinates.
    image_dims :

    Returns
    -------
    Ux : numpy array of float, shape (num_points,)
        Cartesian coordinates along the X axis.
    Uy : numpy array of float, shape (num_points,)
        Cartesian coordinates along the Y axis.
    """

    img_height, img_width = image_dims.shape[0:2]
    Ux = Uc - img_width // 2
    Uy = -Ur + img_height // 2

    if len(image_dims) > 2:
        num_channels = image_dims[2]
        Ux = mathtools.np.repeat(Ux[:, :, None], num_channels, axis=2)
        Uy = mathtools.np.repeat(Uy[:, :, None], num_channels, axis=2)

    return Ux, Uy


def rDotField(X, Y):
    """ Make a vector field representing an infinitesimal rotation. """

    THETA = mathtools.np.arctan2(Y, X)
    XY = mathtools.np.dstack((X, Y))

    U = mathtools.np.zeros_like(X)
    V = mathtools.np.zeros_like(Y)

    theta_rows, theta_cols = THETA.shape
    for i in range(theta_rows):
        for j in range(theta_cols):
            xy = XY[i,j]
            theta = THETA[i,j]
            r_dot = rDot(theta)
            uv = r_dot @ xy
            U[i,j] = uv[0]
            V[i,j] = uv[1]

    return U, V


def wrapTheta(theta):
    """ Wrap an angle from the real line onto (-180, 180].

    Parameters
    ----------
    theta : float
        Angle, in degrees.

    Returns
    -------
    theta_wrapped : float
        The input, projected into (-180, 180].
    """
    return (theta + 90) % 360 - 90


def backprojectIntoPlane(px_coords, n, b, camera_params):
    """ Backproject a pixel to a location on a plane in metric space.

    FIXME: I think this is a weird way of implementing a homography.

    Parameters
    ----------
    px_coord : numpy array of int, shape (num_pixels,) or (num_pixels, 2)
        Pixel coordinates of a point (or points) in an image.
    n : numpy array of float, shape (3,)
        Normal vector of a plane.
    b : float
        Translation of a plane.
    camera_params : numpy array of float, shape (3, 3)
        Matrix of intrinsic camera parameters

    Returns
    -------
    camera_coords : numpy array of float, shape (num_pixels, 3))
        The backprojection of `px_coord` from the image frame to the canonical
        retinal (ie camera) frame. This point lies in the plane defined by
        :math:`n^T x + b = 0`.
    """

    if len(px_coords.shape) < 2:
        px_coords = mathtools.np.expand_dims(px_coords, axis=0)

    # maps [x, y, 1] --> [X / Z, Y / Z, 1]
    backproject = invertHomogeneous(
        camera_params, A_property='diag', range_space_homogeneous=True
    )

    # metric_coord = [X / Z, Y / Z, 1] ==> Z * metric_coord = [X, Y, Z]
    camera_coords = homogeneousVector(px_coords) @ backproject.T
    # <n, Z * metric_coord> + b = 0 ==> Z = - b / <n, metric_coord>
    z_coords = - b / (camera_coords @ n)
    camera_coords[:, -1] = z_coords

    return camera_coords.squeeze()


def projectNullspace(x, A):
    """ Project vector(s) into the null space of a matrix.

    Since the columns of A span a subspace, the null space is the orthogonal
    complement of that subspace.

    WARNING: This has not been tested

    Parameters
    ----------
    x : numpy array of float, shape (num_points, d)
    A : numpy array of float, shape (d, k < d)
        The columns of A should span a subspace of its row space---in other
        words, A is a tall and thin matrix.
    """

    d, k = A.shape
    Identity = mathtools.np.eye(d)
    A_d = mathtools.np.vstack(A, Identity[:, k + 1:])

    # Project x into the range space of A, then subtract off that component
    x_perp = x - x @ A_d.T

    return x_perp


def slopeIntercept(points):
    """ Compute the parameters of a plane in slope-intercept form.

    This function returns parameters of a plane in the form
    ..math:
        n^T x + c = 0

    Parameters
    ----------
    points : numpy array of float, shape (3, 3) or (2, 3)
        Three non-collinear points on the plane.

    Returns
    -------
    n : numpy array of float, shape (3,)
        The normal vector to the plane.
    c : float
        The plane's offset.
    """

    plane_vectors = mathtools.np.diff(points, axis=0)

    n = mathtools.np.cross(plane_vectors[0,:], plane_vectors[1,:])
    c = - mathtools.np.dot(n, points[0,:])

    return n, c


def check3d(x):
    """ Verify that the input is a vector in R^3 or a matrix in R^3x3. """

    shape = x.shape

    if not shape == (3,) or shape == (3, 3):
        err_str = f'Input is not a 3D vector or matrix -- has shape {shape}'
        raise ValueError(err_str)


def rotationMatrix(z_angle=None, y_angle=None, x_angle=None, in_degrees=True):
    """ Construct a rotation matrix from sequential rotations about the Z, Y, and X axes.

    Parameters
    ----------
    z_angle : float, optional
        Angle of rotation about the Z-axis (ie axis 2), in radians.
    y_angle : float, optional
        Angle of rotation about the Y-axis (ie axis 1), in radians.
    x_angle : float, optional
        Angle of rotation about the X-axis (ie axis 0), in radians.
    in_degrees : bool, optional
        If True, the input is assumed to have units of degrees instead of
        radians.

    Returns
    -------
    R : numpy array of float, shape (2, 2) or (3, 3)
        A matrix representing the effect of rotating by `theta_xy` about the
        X-Y axis, then `theta_yz` about the Y-Z axis.
    """

    z_only = False
    if z_angle is not None and (y_angle is None and x_angle is None):
        z_only = True

    if z_angle is None:
        z_angle = 0
    if y_angle is None:
        y_angle = 0
    if x_angle is None:
        x_angle = 0

    if in_degrees:
        z_angle = mathtools.np.radians(z_angle)
        y_angle = mathtools.np.radians(y_angle)
        x_angle = mathtools.np.radians(x_angle)

    # Return a 2D array if we didn't get rotations about X or Y
    if z_only:
        R = mathtools.np.array(
            [[mathtools.np.cos(z_angle), -mathtools.np.sin(z_angle)],
             [mathtools.np.sin(z_angle), mathtools.np.cos(z_angle)]]
        )
        return R

    # Build up the complete rotation one axis at a time
    R = mathtools.np.eye(3)
    for i, angle in enumerate((x_angle, y_angle, z_angle), start=1):
        indices = mathtools.np.arange(i, i + 2) % 3
        rows, cols = zip(*itertools.product(indices, indices))
        R_axis = mathtools.np.eye(3)
        R_axis[rows, cols] = rotationMatrix(z_angle=angle, in_degrees=False).ravel()
        R = R @ R_axis

    return R


def reflectionMatrix(axis):
    """ Construct a matrix that reflects about the specified axis.

    Parameters
    ----------
    axis : int
        The returned matrix will reflect points about this axis.

    Returns
    -------
    R : numpy array of float, shape (3, 3)
        Matrix that reflects points about `axis`.
    """

    R = mathtools.np.eye(3)
    R[axis, axis] = -1

    return R


def skewSymmetricMatrix(omega):
    """ Construct a skew-symmetric matrix from a vector.

    A skew-symmetric matrix :math:`W` has the property :math:`W^T = -W`.

    Every skew-symmetric matrix can be interpreted as
        1. An infintesimal rotation about a particular vector.
        2. The cross-product with a particular vector.
    That vector is sufficient to define the matrix.

    Parameters
    ----------
    omega : numpy array of float, shape (3,)
        The vector that defines the skew-symmetric matrix.

    Returns
    -------
    W : numpy array of float, shape (3, 3)
        The skew-symmetric matrix generated from `omega`.
    """

    check3d(omega)

    W = mathtools.np.zeros((3, 3))
    for i, w in enumerate(omega):
        row = (i + 1) % 3
        col = (i + 2) % 3

        W[row, col] = w
        W[col, row] = -w

    return W


def exponentialMap(omega, small_angle=False, in_degrees=True):
    """ Compute the exponential map of an infinitesimal rotation. """

    omega_norm = mathtools.np.linalg.norm(omega)
    omega_unit = omega / omega_norm

    if in_degrees:
        omega_norm = mathtools.np.radians(omega_norm)

    skew_sym = skewSymmetricMatrix(omega_unit)

    if small_angle:
        exp_map = mathtools.np.eye(3) + skew_sym
        return exp_map

    exp_map = (
        mathtools.np.eye(3)
        + mathtools.np.sin(omega_norm) * skew_sym
        + (1 - mathtools.np.cos(omega_norm)) * (skew_sym @ skew_sym)
    )

    return exp_map


def estimateRotation(centered_points):
    """ Estimate a rotation matrix from a zero-mean set of points.

    NOTE: R is only accurate up to a global sign.

    Parameters
    ----------
    centered_points : numpy array of float, shape (num_points, 3)

    Returns
    -------
    R : numpy array of float, shape (3, 3)
        Rotation matrix representing the principal directions of variation in
        the input.
    """

    U, S, Vt = mathtools.np.linalg.svd(centered_points)
    R = Vt.T

    # Handle reflections / project from O(d) into SO(d)
    det_R = mathtools.np.linalg.det(R)
    if det_R < 0:
        R[:,-1] *= det_R

    return R


def estimatePose(points, xy_only=False, estimate_orientation=True):
    """ Estimate position and orientation from a zet of points.

    Parameters
    ----------
    points : numpy array of float, shape (num_points, 3)
    xy_only : bool, optional
        If True, the pose is only estimated along the first two dimensions of
        the input. The third dimension of the translation output is zero, and
        the third dimension of the rotation output is the identity.
    estimate_orientation: bool, optional
        If False, this function returns the identity matrix instead of
        estimating the orientation using SVD. Can be useful when there are
        many points, because the current implementation is inefficient.

    Returns
    -------
    t : numpy array of float, shape (3,)
        Translation vector representing the mean of the input.
    R : numpy array of float, shape (3, 3)
        Rotation matrix representing the principal directions of variation in
        the input.
    """

    if xy_only:
        points_xy = points[:,0:2]
        R_xy, t_xy = estimatePose(points_xy, estimate_orientation=estimate_orientation)
        t = mathtools.np.zeros(3)
        R = mathtools.np.eye(3)
        t[0:2] = t_xy
        R[0:2, 0:2] = R_xy
        return R, t

    t = points.mean(axis=0)
    if estimate_orientation:
        R = estimateRotation(points - t)
    else:
        num_dims = points.shape[1]
        R = mathtools.np.eye(num_dims)

    return R, t


def formDelaunayTriangulations(vertices):
    delaunays = {
        i: spatial.Delaunay(vtxs)
        for i, vtxs in vertices.items()}
    return delaunays


def formConvexHulls(vertices):
    hulls = {
        i: spatial.ConvexHull(vtxs)
        for i, vtxs in vertices.items()}
    return hulls


def getExtremePoints(convex_hull):
    vertex_indices = convex_hull.vertices
    vertices = convex_hull.points[vertex_indices,:]
    return vertices


def getVertices(state, centered):
    f = functools.partial(state.getComponentVertices, state)
    gen = (f(i, centered=centered) for i in state.connected_components.keys())
    return tuple(gen)


def matchingPolygon(points, polygons, z_vertices):
    num_points = points.shape[0]
    num_poly = len(polygons)
    num_z = len(z_vertices)
    if num_z != num_poly:
        err_str = 'number of polygons ({}) should match number of z coords ({})'
        raise ValueError(err_str.format(num_poly, num_z))

    poly_indices = mathtools.np.array(list(polygons.keys()))

    matching_poly_zvals = mathtools.np.zeros((num_points, num_poly))
    for i, poly_index in enumerate(polygons.keys()):
        poly = polygons[poly_index]
        vz = z_vertices[poly_index]
        z_max = vz.max()
        matching_poly_zvals[:,i] = (poly.find_simplex(points) >= 0) * z_max

    matching_polys = matching_poly_zvals.argmax(axis=1)
    matching_polys = poly_indices[matching_polys]
    matching_polys += 1

    matches_background = mathtools.np.logical_not(matching_poly_zvals.any(axis=1))
    matching_polys[matches_background] = 0

    return matching_polys


def toPixelCoord(metric_coord, pixel_mapping):
    metric_proj = toProjectiveCoords(metric_coord)
    pixel_proj = pixel_mapping @ metric_proj
    pixel_coord, scale = fromProjectiveCoords(pixel_proj)

    return pixel_coord


def backProject(camera_params, pixel_coords, depths):
    """ Backproject multiple pixels.
    NOTE: pixel_coords should be of size [num_pixels x 2]
    """
    origin = camera_params[0:2, 2]
    focal_params = camera_params[0:2, 0:2]

    num_pixels, num_dims = pixel_coords.shape

    if num_dims != 2:
        err_str = 'pixel_coords is shape ({}, {}), but should be (num_pixels, 2)'
        raise ValueError(err_str.format(num_pixels, num_dims))

    metric_coord = mathtools.np.zeros((num_pixels, 3))
    metric_coord[:,2] = depths

    inv_focal = mathtools.np.linalg.pinv(focal_params)
    metric_coord[:,0:2] = depths * (pixel_coords - origin) @ inv_focal.T

    metric_coord[:,1] *= -1

    return metric_coord


# FIXME: same as homogeneousVector
def toProjectiveCoords(coord):
    return mathtools.np.append(coord, 1)


def fromProjectiveCoords(coord):
    scale = coord[-1]
    if scale == 0:
        warn_str = 'Object lies on the image plane! (Z = 0)'
        logger.warning(warn_str)
        return coord[:-1], scale
    return coord[:-1] / scale, scale


def boundingBox(points, image_shape):
    """ Compute the coordinates of a bounding box around a set of points.

    Parameters
    ----------
    points : numpy array of int, shape (num_points, 2)
        The set of points contained by the bounding box.
    image_shape : (int, int)
        Max value for Y and X axes (respectively). Any bounding box points with
        coordinates outside these values are clipped.

    Returns
    -------
    bounding_box_coords : numpt array of int, shape (num_bounding_points, 2)
        The X-Y coordinates of the bounding box and its interior points.
    """

    x_pts = points[:,0]
    x_min = max(x_pts.min(), 0)
    x_max = min(x_pts.max() + 1, image_shape[1])

    y_pts = points[:,1]
    y_min = max(y_pts.min(), 0)
    y_max = min(y_pts.max() + 1, image_shape[0])

    x = mathtools.np.arange(x_min, x_max)
    y = mathtools.np.arange(y_min, y_max)
    bounding_box_coords = mathtools.np.dstack(mathtools.np.meshgrid(x, y)).reshape(-1, 2)
    return bounding_box_coords


def in_hull(p, hull):
    """ Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed

    (shamelessly copied from https://stackoverflow.com/a/16898636)
    """

    if not isinstance(hull, spatial.Delaunay):
        hull = spatial.Delaunay(hull)

    return hull.find_simplex(p) >= 0


def extremePoints(*intervals):
    """ NOTE: The name is a little misleading.

    Let R be the hyper-rectangle created by taking the Cartesian product over
    all intervals.

        R = { x : x is in every interval }

    Let V be the set of vertices on the hyper-rectangle R. These are the true
    extreme points of R in the sense that they are maximally far away from each
    other.

    This function only returns two vertices:
      min_vtx, the vertex whose coordinate is smallest along every dimension.
      max_vtx, the vertex whose coordinate is largest along every dimension.
    """

    min_coords, max_coords = zip(*intervals)

    min_vtx = mathtools.np.array(list(min_coords))
    max_vtx = mathtools.np.array(list(max_coords))

    return min_vtx, max_vtx


def midpoint(*intervals, cast_to_int=True):
    minpt, maxpt = extremePoints(*intervals)
    midpt = (minpt + maxpt) / 2
    if cast_to_int:
        midpt = utils.castToInt(midpt)

    return midpt


def sampleLiesInInterval(x, interval):
    lower, upper = interval
    in_interval = lower <= x and x <= upper

    return in_interval


def sampleLiesInVolume(x, *intervals):
    if len(x) != len(intervals):
        fmt_str = 'sample dims {} != interval dims {}'
        err_str = fmt_str.format(len(x), len(intervals))
        raise ValueError(err_str)

    in_intervals = tuple(
        sampleLiesInInterval(xi, inter)
        for xi, inter in zip(x, intervals))

    return functools.reduce(operator.and_, in_intervals)


def projectIntoInterval(x, interval, modify_original=False):
    if not modify_original:
        x = x.copy()

    lower, upper = interval

    beyond_lower = x < lower
    x[beyond_lower] = lower

    beyond_upper = x > upper
    x[beyond_upper] = upper

    return x


def projectIntoVolume(x, *intervals, modify_original=False):
    if not modify_original:
        x = x.copy()

    for interval in intervals:
        projectIntoInterval(x, interval, modify_original=True)

    return x


def checkIntervalBounds(interval):
    lower, upper = interval
    if upper < lower:
        warn_str = 'Reversing non-increasing interval {}'.format(interval)
        logger.warning(warn_str)
        interval = reversed(interval)

    return interval


def sampleIntervalUniform(interval, sample_period=1):
    lower, upper = interval
    sample_gen = range(lower, upper + 1, sample_period)

    return tuple(sample_gen)


def sampleInteriorUniform(*intervals, sample_period=1):
    interval_samples = tuple(sampleIntervalUniform(i) for i in intervals)
    interior_samples = itertools.product(*interval_samples)

    return mathtools.np.array([list(tup) for tup in interior_samples])


def sampleIntervalRandom(interval, integer_sample=True):
    interval = checkIntervalBounds(interval)
    lower, upper = interval

    if integer_sample:
        lower_int = mathtools.np.floor(lower)
        upper_int = mathtools.np.ceil(upper)
        return mathtools.np.random.randint(lower_int, upper_int)

    err_str = 'Non-integer samples are not supported yet.'
    raise NotImplementedError(err_str)


def sampleInteriorRandom(*intervals):
    samples = [sampleIntervalRandom(i) for i in intervals]
    return mathtools.np.array(samples)


def rDot(theta):
    dR_dtheta = mathtools.np.array([
        [-mathtools.np.sin(theta), -mathtools.np.cos(theta)],
        [mathtools.np.cos(theta), -mathtools.np.sin(theta)]
    ])

    return dR_dtheta


def computeImage(U, R, t, cast_to_int=True):
    """ Compute the image of U under rigid motion (R, t)
    v = R u + t, if u and v are column vectors """

    V = U @ R.T + t
    if cast_to_int:
        V = utils.castToInt(V)

    return V


def computePreimage(V, R, t, cast_to_int=True):
    """ Compute the preimage of V under rigid motion (R, t)
    u = R_inv (v - t), if u and v are column vectors """

    U = (V - t) @ R
    if cast_to_int:
        U = utils.castToInt(U)

    return U
