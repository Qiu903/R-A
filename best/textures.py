# -*- coding: utf-8 -*-
"""
程序化纹理
================================================================

全部用 numpy 现场算出来，不依赖任何图片素材。好处是仓库里不用塞二进制文件，
而且想调颜色的"脏"一点、木纹的疏密，改两个系数就行。

MuJoCo 的纹理就是一张裸的 RGB 字节流，所以这里每个函数都返回
(height, width, 3) 的 uint8 数组，调用方自己 tobytes()。
"""

import mujoco
import numpy as np


def _normalize(img):
    """夹到 0~255，避免溢出回卷成花斑。"""
    return np.clip(img, 0, 255).astype(np.uint8)


def fdm_grain(size=384, base=(236, 234, 229), line_strength=10.0,
              line_period=7.0, noise=3.0, seed=0):
    """
    3D 打印的层纹：沿一个方向等距的细线 + 轻微颗粒。

    为什么值得做：真机是 FDM 打印件，表面永远有一圈圈纹路。
    纯色材质一眼就是"渲染图"，加上层纹之后近距离看才有实物感。
    """
    rng = np.random.default_rng(seed)
    rows = np.arange(size)[:, None]
    # 层纹沿横向走，用正弦造出一条条明暗
    stripe = np.sin(rows * 2.0 * np.pi / line_period) * (line_strength * 0.5)
    grain = rng.standard_normal((size, size, 1)) * noise
    img = np.array(base, dtype=float)[None, None, :] + stripe[:, :, None] + grain
    return _normalize(img)


def matte_plastic(size=256, base=(238, 238, 240), noise=2.2, seed=1):
    """细磨砂：只有非常轻的颗粒，几乎没有结构。用于光洁表面。"""
    rng = np.random.default_rng(seed)
    grain = rng.standard_normal((size, size, 1)) * noise
    img = np.array(base, dtype=float)[None, None, :] + grain
    return _normalize(img)


def wood(size=512, dark=(126, 86, 54), light=(214, 172, 130), seed=2):
    """
    木纹桌面。

    单个正弦出来的是"斑马线"，一眼假。真实木纹是三层叠加：
        大尺度色差（板材不同部位深浅不一）
        年轮（沿一个方向弯曲的条纹）
        细纤维（高频、很弱）
    相位用 x 调制，条纹才是弯的；不调制就是等距平行线。
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(float) / size

    # 大尺度色差：低频正弦，很弱
    blob = 0.12 * np.sin(2 * np.pi * (x * 1.3 + 0.4)) * np.cos(2 * np.pi * y * 0.9)

    # 年轮：相位随 x 弯曲，越靠边弯得越明显
    phase = 2 * np.pi * (y * 19.0 + 0.55 * np.sin(2 * np.pi * x * 1.8) +
                         0.22 * np.sin(2 * np.pi * x * 2.7))
    rings = 0.5 + 0.5 * np.sin(phase)

    # 细纤维：沿纹理方向拉长的高频，振幅小
    fiber = rng.standard_normal((size, size))
    fiber = 0.5 * fiber + 0.25 * np.roll(fiber, 1, axis=0) + 0.25 * np.roll(fiber, 2, axis=0)

    t = 0.62 * rings + blob + 0.06 * fiber
    t = np.clip(t, 0, 1)[:, :, None]

    img = np.array(dark, float)[None, None, :] * (1 - t) + \
        np.array(light, float)[None, None, :] * t
    return _normalize(img)


def brushed_metal(size=256, base=(176, 179, 184), seed=3):
    """拉丝金属：沿一个方向拉长的噪声，就是拉丝效果。"""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((size, size))
    # 竖直方向做单像素宽度的高频、水平方向做低频，形成横向拉丝
    kernel_y = np.ones((1, 9)) / 9.0
    smeared = np.copy(noise)
    for shift in range(1, 9):
        smeared += np.roll(noise, shift - 4, axis=1)
    smeared /= 9.0
    img = np.array(base, float)[None, None, :] + smeared[:, :, None] * 9.0
    return _normalize(img)


def checker(size=256, a=(30, 32, 36), b=(120, 124, 130), cells=8):
    """
    棋盘格。这个不是为了好看，是调相机时的参照 ——
    方格贴在地面上，一眼就能看出透视和比例对不对。
    """
    idx = (np.arange(size) * cells // size)
    grid = (idx[:, None] + idx[None, :]) % 2
    img = np.where(grid[:, :, None] == 0,
                   np.array(a, float)[None, None, :],
                   np.array(b, float)[None, None, :])
    return _normalize(img)


def gradient(size=256, top=(24, 26, 30), bottom=(58, 62, 70)):
    """竖直渐变，用来当背景幕布，比纯黑有层次。"""
    t = np.linspace(0, 1, size)[:, None, None]
    img = np.array(top, float)[None, None, :] * (1 - t) + \
        np.array(bottom, float)[None, None, :] * t
    return _normalize(np.repeat(img, size, axis=1))


def register(spec, name, img, nchannel=3):
    """
    把 numpy 图注册成 MuJoCo 纹理。

    踩过的坑：
    1. MjsTexture.data 只收 bytes，传 ndarray 会直接 TypeError。
    2. material.textures 必须给满 10 个槽位（第一个是 RGB）。
    3. mjtTexture 的枚举是 CUBE=1 / SKYBOX=2 / 2D=0，**2D 是 0 不是 2**。
       按直觉把普通贴图写成 2，会全部被当成天空盒 —— 现象是背景变成贴图、
       物体表面反而没有纹理。
    """
    tex = spec.add_texture()
    tex.name = name
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.width = img.shape[1]
    tex.height = img.shape[0]
    tex.nchannel = nchannel
    tex.data = img.tobytes()
    # 关键：不设 embed，MuJoCo 会当作"外部文件纹理"处理，data 被忽略，
    # 编译时找不到文件就退回默认白纹理 —— 现象就是所有材质看起来都是纯色。
    tex.embed = True
    return tex


def material(spec, name, texture=None, rgba=(1, 1, 1, 1), roughness=0.7,
             metallic=0.0, specular=0.3, shininess=0.3, emission=0.0,
             texrepeat=(3.0, 3.0)):
    """建一个 PBR 材质；texture 为空则用纯色。"""
    mat = spec.add_material()
    mat.name = name
    mat.rgba = list(rgba)
    mat.roughness = roughness
    mat.metallic = metallic
    mat.specular = specular
    mat.shininess = shininess
    mat.emission = emission
    if texture is not None:
        # 关键：颜色纹理要放在第 1 个槽位，不是第 0 个。
        # MJCF 里写 <material texture="t"> 也是绑定到 slot 1；
        # 塞到 slot 0 的话编译不报错、texid 看着也对，但渲染时材质是纯色 ——
        # 这个坑排查了很久，记在这里。
        slots = [""] * 10
        slots[1] = texture
        mat.textures = slots
        mat.texrepeat = list(texrepeat)
    return mat
