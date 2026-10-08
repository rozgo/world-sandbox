"""A small software renderer for coloured point clouds, camera frustums and paths.

Coordinates are AMB3R-SLAM's: the world is the first camera's frame, x right, y down, z forward;
camera poses are camera-to-world 4x4 matrices. Points are splatted into a z-buffer (the nearest
fragment wins), with a splat size that grows for near points.
"""

import numpy as np
from PIL import Image, ImageDraw

UP = np.array([0.0, -1.0, 0.0])


def look_at(eye, target, up=UP):
    """World-to-camera rotation and the eye, for a camera at ``eye`` looking at ``target``."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)                                   # right, for y pointing down
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z]), np.asarray(eye, float)


class View:
    def __init__(self, R, eye, size=(1280, 720), f=None):
        self.R, self.eye = R, eye
        self.W, self.H = size
        self.f = f or 0.55 * self.W

    def project(self, X):
        """(N, 3) world -> (N, 2) pixels and (N,) depth."""
        C = (np.asarray(X, float) - self.eye) @ self.R.T
        z = C[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.stack([self.f * C[:, 0] / z + self.W / 2, self.f * C[:, 1] / z + self.H / 2], 1)
        return uv, z


def splat(view, P, C, bg=(15, 17, 19), size_k=None, max_size=4):
    """Render points P (N, 3) with colours C (N, 3) uint8; returns an RGB uint8 image."""
    uv, z = view.project(P)
    ok = z > 1e-3
    uv, z, col = uv[ok], z[ok], C[ok]
    k = size_k if size_k is not None else 0.004 * view.f
    s = np.clip(np.round(k / z), 1, max_size).astype(int)
    us, vs, zs, cs = [], [], [], []
    for size in range(1, max_size + 1):
        m = s == size
        if not m.any():
            continue
        u0 = np.floor(uv[m, 0]).astype(int) - size // 2
        v0 = np.floor(uv[m, 1]).astype(int) - size // 2
        for du in range(size):
            for dv in range(size):
                us.append(u0 + du); vs.append(v0 + dv); zs.append(z[m]); cs.append(col[m])
    img = np.empty((view.H, view.W, 3), np.uint8)
    img[:] = bg
    if not us:
        return img
    u, v, zz, cc = (np.concatenate(a) for a in (us, vs, zs, cs))
    inside = (u >= 0) & (u < view.W) & (v >= 0) & (v < view.H)
    u, v, zz, cc = u[inside], v[inside], zz[inside], cc[inside]
    order = np.argsort(zz, kind="stable")
    pix = (v * view.W + u)[order]
    _, first = np.unique(pix, return_index=True)
    sel = order[first]
    img.reshape(-1, 3)[v[sel] * view.W + u[sel]] = cc[sel]
    return img


def frustum_corners(pose, K, wh, depth):
    """World positions of the apex and the four image corners at ``depth`` (camera z)."""
    W, H = wh
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pix = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
    cam = np.stack([(pix[:, 0] - cx) / fx * depth, (pix[:, 1] - cy) / fy * depth, np.full(4, depth)], 1)
    world = cam @ pose[:3, :3].T + pose[:3, 3]
    return pose[:3, 3], world


def draw_path(draw, view, X, fill, width=3):
    uv, z = view.project(X)
    pts = [tuple(p) for p, d in zip(uv, z) if d > 1e-3 and np.all(np.abs(p) < 1e5)]
    if len(pts) > 1:
        draw.line(pts, fill=fill, width=width, joint="curve")


def draw_frustum(img, view, pose, K, wh, depth, colour, width=2, texture=None):
    """Draw a camera frustum; with ``texture`` (a PIL image), paint it on the far plane."""
    apex, corners = frustum_corners(pose, K, wh, depth)
    uv, z = view.project(np.vstack([apex, corners]))
    if np.any(z <= 1e-3):
        return img
    if texture is not None:
        quad = [tuple(p) for p in uv[1:]]
        coeffs = _perspective_coeffs(quad, texture.size)
        warped = texture.convert("RGBA").transform(img.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
        alpha = Image.new("L", img.size, 0)
        ImageDraw.Draw(alpha).polygon(quad, fill=225)
        img.paste(warped, (0, 0), alpha)
    d = ImageDraw.Draw(img)
    a = tuple(uv[0])
    c = [tuple(p) for p in uv[1:]]
    for p in c:
        d.line([a, p], fill=colour, width=width)
    d.line(c + [c[0]], fill=colour, width=width)
    return img


def _perspective_coeffs(quad, size):
    """PIL PERSPECTIVE coefficients mapping output pixels in ``quad`` to the texture's corners."""
    w, h = size
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    A, b = [], []
    for (x, y), (u, v) in zip(quad, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    return np.linalg.solve(np.array(A, float), np.array(b, float)).tolist()


def smooth_poses(poses, half=10):
    """Moving-average camera positions and forward directions, for a steady chase camera."""
    n = len(poses)
    pos = np.array([p[:3, 3] for p in poses])
    fwd = np.array([p[:3, 2] for p in poses])
    out_p, out_f = np.empty_like(pos), np.empty_like(fwd)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        out_p[i] = pos[a:b].mean(0)
        f = fwd[a:b].mean(0)
        out_f[i] = f / np.linalg.norm(f)
    return out_p, out_f


def chase_view(pos, fwd, size, back=2.6, up=1.1, ahead=3.0):
    """A camera behind and above ``pos``, looking ahead along ``fwd``."""
    flat = fwd - UP * np.dot(fwd, UP)
    flat /= np.linalg.norm(flat)
    eye = pos - back * flat + up * UP
    return View(*look_at(eye, pos + ahead * flat - 0.15 * UP), size=size)
