import torch

def generate_point_on_a_sphere(shape):
    """Generates a 3D point on a sphere using random angles."""
    theta = torch.pi / 3 * (2. * torch.rand(*shape) - 1.)
    phi = 2 * torch.pi * torch.rand(*shape)
    return torch.stack([
        phi.cos() * theta.cos(),
        theta.sin(),
        phi.sin() * theta.cos()
    ], -1), theta, phi

def get_camera(batch_size, target, r, eye=None, device='cuda:0'):
    """
        Creates and returns a Camera object positioned around a target with a given radius; 
        optionally accepts a predefined eye position.
    """
    if eye is None:
        eye, theta, phi = generate_point_on_a_sphere([batch_size,])
        eye = r * eye
    else:
        theta, phi = None, None

    eye = eye.to(device)
    target = target.to(device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    fov = torch.full_like(eye[..., 0], 30 * torch.pi / 180, device=device)
    
    return Camera(eye=eye, at=target, up=up, fov=fov, width=512, height=512), theta, phi

class CameraExtrinsics:
    def __init__(self, eye, at, up):
        """Prepares extrinsic rotation and translation matrices from camera eye, target, and up vector."""
        self.num_cameras = len(eye)
        if eye.ndim == 1:
            eye = eye.unsqueeze(0)
        if at.ndim == 1:
            at = at.unsqueeze(0)
        if up.ndim == 1:
            up = up.unsqueeze(0)
        backward = at - eye
        backward = torch.nn.functional.normalize(backward, dim=-1)
        right = torch.cross(backward, up, dim=-1)
        right = torch.nn.functional.normalize(right, dim=-1)
        up = torch.cross(right, backward, dim=-1)
        self.R = torch.stack((right, up, -backward), dim=1)
        self.t = -self.R @ eye.unsqueeze(-1)

    def transform(self, points):
        """Transforms 3D points using the camera extrinsic matrices."""
        num_points = points.shape[-2]
        points = points.expand(self.num_cameras, num_points, 3)[..., None]
        R = self.R[:, None].expand(self.num_cameras, num_points, 3, 3)
        t = self.t[:, None].expand(self.num_cameras, num_points, 3, 1)
        return (R @ points + t).squeeze(-1)

    def inv_transform_rays(self, ray_orig, ray_dir):
        """Inverts transform of rays from camera space back to world space."""
        num_rays = ray_dir.shape[-2]
        d = ray_dir.expand(self.num_cameras, num_rays, 3)[..., None]
        o = ray_orig.expand(self.num_cameras, num_rays, 3)[..., None]
        R = self.R[:, None].expand(self.num_cameras, num_rays, 3, 3)
        R_T = R.transpose(2, 3)
        t = self.t[:, None].expand(self.num_cameras, num_rays, 3, 1)
        transformed_dir = R_T @ d
        transformed_orig = R_T @ (o - t)
        return transformed_orig.squeeze(-1), transformed_dir.squeeze(-1)

class CameraIntrinsics:
    def __init__(self, fov, height, width, x0, y0):
        """Initializes camera intrinsic parameters from focal lengths, principal point, and near/far planes."""
        tanHalfAngle = torch.tan(fov / 2.0)
        aspect = height / 2.0
        self.height = height
        self.width = width
        self.focal_x = width / (2 * tanHalfAngle)
        self.focal_y = height / (2 * tanHalfAngle)
        self.near = 1e-2
        self.far = 1e2
        self.x0 = x0
        self.y0 = y0
        self.num_cameras = len(fov)

        self.device = fov.device
        self.dtype = fov.dtype

    def perspective_matrix(self,):
        """Builds the perspective projection matrix from intrinsics."""
        zero = torch.zeros_like(self.focal_x)
        one = torch.ones_like(self.focal_x)
        rows = [
            torch.stack([self.focal_x, zero,           -self.x0,    zero],       dim=-1),
            torch.stack([zero,         self.focal_y,   -self.y0,    zero],       dim=-1),
            torch.stack([zero,         zero,            zero,       one],        dim=-1),
            torch.stack([zero,         zero,            one,        zero],       dim=-1)
        ]
        persp_mat = torch.stack(rows, dim=1)
        return persp_mat

    def ndc_matrix(self,):
        """Builds the normalized device coordinate matrix."""
        top = self.height / 2
        bottom = -top
        right = self.width / 2
        left = -right

        tx = -(right + left) / (right - left)
        ty = -(top + bottom) / (top - bottom)

        U = -2.0 * self.near * self.far / (self.far - self.near)
        V = -(self.far + self.near) / (self.far - self.near)
        ndc_mat = torch.tensor([
            [2.0 / (right - left),  0.0,                   0.0,            -tx ],
            [0.0,                   2.0 / (top - bottom),  0.0,            -ty ],
            [0.0,                   0.0,                   U,               V  ],
            [0.0,                   0.0,                   0.0,            -1.0]
        ], dtype=self.dtype, device=self.device)
        return ndc_mat.unsqueeze(0)

    def projection_matrix(self,):
        """Combines perspective and NDC matrices into a full projection."""
        perspective_matrix = self.perspective_matrix()
        ndc = self.ndc_matrix()
        return ndc @ perspective_matrix

    def get_homogeneous_coordinates(self, points):
        """Appends homogeneous coordinate to points if missing."""
        if points.shape[-1] == 4:
            return points
        return torch.cat([points, torch.ones_like(points[..., 0:1])], dim=-1)

    def project(self, points):
        """Projects 3D points into the camera projection space using the full projection pipeline."""
        # points shape can be ([num_cameras], num_points, 4), ([num_cameras], num_points, 3)
        num_points = points.shape[-2]
        homogeneous_points = self.get_homogeneous_coordinates(points)
        homogeneous_points = homogeneous_points.expand(self.num_cameras, num_points, 4)[..., None]
        
        proj = self.projection_matrix()
        proj = proj[:, None].expand(self.num_cameras, num_points, 4, 4)
        return (proj @ homogeneous_points).squeeze(-1)

class Camera:
    def __init__(self, eye, at, up, fov, width, height,):
        """
            Initializes a Camera instance with eye, target, up, fov, and image size; 
            sets up intrinsic and extrinsic components.
        """
        self.fov = fov
        self.tan_half_fov = torch.tan(self.fov / 2.0)
        self.height = height
        self.width = width
        self.x0 = torch.zeros_like(eye[..., 0])
        self.y0 = torch.zeros_like(eye[..., 0])
        self.intrinsics = CameraIntrinsics(
            self.fov,
            self.height,
            self.width,
            self.x0,
            self.y0,
        )
        self.extrinsics = CameraExtrinsics(eye, at, up)

    def __len__(self):
        """Returns number of cameras based on eye positions."""
        return len(self.x0)
