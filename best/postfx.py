# -*- coding: utf-8 -*-
"""
后期处理：把"渲染图"变成"照片"
================================================================

MuJoCo 的渲染器只负责把场景画对，不负责好看。真实的产品图和影视镜头之间，
差的不是模型精度，而是这几步：

    景深    焦点之外的东西模糊掉，画面立刻有"镜头"的感觉
    色调映射 把线性 HDR 压到显示器范围，高光不会一坨死白
    暗角    四角压暗，视线收到中心
    颗粒    一点点噪点，去掉"计算机生成"的塑料感
    锐化    补偿上一步模糊带来的软化

这里面没有任何一步需要更好的模型 —— 它们全都能在你现在的渲染结果上做。
"""

import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------- 色调映射
def aces(x):
    """
    ACES filmic 曲线。比简单的 clip 或 gamma 好在：高光会"滚"到白色而不是硬切，
    中间调保持对比。这是电影调色的默认起点。
    """
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def tone_map(img, exposure=1.0, contrast=1.06):
    """线性 RGB → 显示用 sRGB。exposure 相当于相机曝光补偿。"""
    x = img.astype(np.float32) / 255.0 * exposure
    x = np.clip(x, 0, None)
    x = np.where(x > 0.8, 0.8 + (1 - np.exp(-(x - 0.8) * 4)) * 0.2, x)  # 高光肩部
    x = aces(x * 1.15)
    # 中间调对比
    x = np.clip((x - 0.5) * contrast + 0.5, 0, 1)
    return (x * 255).astype(np.uint8)


# ---------------------------------------------------------------- 景深
def depth_of_field(img, depth, focus=0.55, aperture=3.2, max_sigma=9.0):
    """
    按深度做分层模糊，再按弥散圆混合。

    depth 是 MuJoCo 的归一化深度（0=近，1=远）。focus 是合焦位置。
    分层做而不是逐像素变 sigma，是因为逐像素高斯在 numpy 里慢到不可用；
    分 4 层再按 coc 插值，视觉上分辨不出来。
    """
    if depth is None:
        return img

    d = depth.astype(np.float32)
    if d.max() <= 1.0:
        d = d / max(float(d.max()), 1e-6)

    coc = np.clip(np.abs(d - focus) * aperture, 0.0, 1.0)   # 弥散圆 0~1
    base = Image.fromarray(img)

    levels = [0.0, 0.33, 0.66, 1.0]
    blurs = [base.filter(ImageFilter.GaussianBlur(max_sigma * lv)) for lv in levels]

    out = np.array(base, dtype=np.float32)
    acc = np.zeros_like(out)
    for i in range(len(levels) - 1):
        lo, hi = levels[i], levels[i + 1]
        m = ((coc >= lo) & (coc < hi)).astype(np.float32)[..., None]
        if m.max() == 0:
            continue
        t = np.clip((coc[..., None] - lo) / (hi - lo), 0, 1)
        a = np.array(blurs[i], dtype=np.float32)
        b = np.array(blurs[i + 1], dtype=np.float32)
        acc += (a * (1 - t) + b * t) * m

    m_far = (coc >= levels[-1]).astype(np.float32)[..., None]
    if m_far.max() > 0:
        acc += np.array(blurs[-1], dtype=np.float32) * m_far

    m_near = (coc < levels[0]).astype(np.float32)[..., None]
    acc += out * m_near
    return np.clip(acc, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 其他
def vignette(img, strength=0.42, power=2.4):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / np.sqrt(2)
    v = 1.0 - strength * np.power(np.clip(r, 0, 1), power)
    return np.clip(img.astype(np.float32) * v[..., None], 0, 255).astype(np.uint8)


def grain(img, amount=3.0, seed=0):
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(img.shape[:2])[..., None] * amount
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def sharpen(img, radius=1.2, percent=55):
    im = Image.fromarray(img)
    im = im.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=2))
    return np.array(im)


def chromatic(img, shift=1):
    """轻微色散：边缘 R/B 各错开一个像素。镜头感来源之一，过量会脏。"""
    if shift <= 0:
        return img
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], shift, axis=1)
    out[..., 2] = np.roll(img[..., 2], -shift, axis=1)
    return out


# ---------------------------------------------------------------- 管线
def saturate(img, scale=1.0):
    """饱和度。ACES 会把暖色推得很艳，桌面容易发橘，压一点点更耐看。"""
    g = img.astype(np.float32)
    lum = (g[..., 0] * 0.299 + g[..., 1] * 0.587 + g[..., 2] * 0.114)[..., None]
    return np.clip(lum + (g - lum) * scale, 0, 255).astype(np.uint8)


def apply(img, depth=None, exposure=0.92, dof=True, vig=0.34,
          dof_focus=0.55, dof_aperture=0.85, grain_amount=2.0):
    """完整后处理。顺序有讲究：先在"干净"的图像上做景深，最后才锐化。"""
    out = img
    if dof:
        out = depth_of_field(out, depth, focus=dof_focus, aperture=dof_aperture)
    out = tone_map(out, exposure=exposure)
    out = saturate(out, 0.94)
    out = chromatic(out, 1)
    out = vignette(out, strength=vig)
    out = grain(out, amount=grain_amount)
    out = sharpen(out, radius=1.1, percent=45)
    return out
