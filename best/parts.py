# -*- coding: utf-8 -*-
"""
零件库：把机械臂拆成"能一个个装起来"的部件
================================================================

一根圆角长条看起来像积木，不像机械臂。真机上有这些东西：

    舵机        方形外壳 + 安装耳 + 输出盘
    紧固件      内六角螺栓 + 垫圈
    连接件      法兰、转台、轴承环
    走线        捆在一起的线缆
    加工痕迹    减重孔、倒角

这里每个函数往 body 下挂一组 geom。网格按"型号"缓存复用 ——
一颗 M3 螺栓只生成一次网格，四个安装孔共享同一份顶点。

尺寸按 SO-101 常用的 STS3215 舵机（45.2 × 24.7 × 35 mm）量级设计，
但为了配合本项目简化后的连杆长度，做了等比收敛。
"""

import numpy as np
import mujoco

import meshgen

# 网格缓存：同一个 spec 里，同型号零件只生成一次
_REG = {}


def reset_cache():
    _REG.clear()


def _mesh_once(spec, key, builder):
    if key not in _REG:
        builder()
        _REG[key] = True
    return key


def _geom(body, mesh_name, mat, pos=(0, 0, 0), quat=None, rgba=None):
    g = body.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_MESH
    g.meshname = mesh_name
    g.material = mat
    g.pos = list(pos)
    if quat is not None:
        g.quat = list(quat)
    if rgba is not None:
        g.rgba = list(rgba)
    g.contype = 0
    g.conaffinity = 0
    return g


# ---------------------------------------------------------------- 舵机
def servo(spec, body, scale=1.0, pos=(0, 0, 0), quat=(1, 0, 0, 0),
          accent="m_accent", shell="m_dark", metal="m_metal"):
    """
    一个完整的舵机总成：外壳 + 上盖 + 输出盘 + 两侧安装耳。

    scale 缩放整套尺寸。腕部用 scale≈0.8 会显得比例更合理 ——
    真机腕部舵机和肩部是同款，但全用大号会让末端笨重。
    """
    s = scale
    hx, hy, hz = 0.044 * s, 0.024 * s, 0.032 * s

    mb = f"m_servo_body_{int(s * 100)}"
    _mesh_once(spec, mb, lambda: meshgen.add_rounded_box(
        spec, mb, (hx / 2, hy / 2, hz / 2), roundness=6.0, nu=40, nv=20))
    _geom(body, mb, shell, pos=pos, quat=quat)

    # 上盖：比本体小一圈，颜色区分，看起来像可以拧开的面板
    mc = f"m_servo_cap_{int(s * 100)}"
    _mesh_once(spec, mc, lambda: meshgen.add_rounded_box(
        spec, mc, (hx * 0.36, hy * 0.36, 0.0016 * s), roundness=8.0, nu=28, nv=14))
    _geom(body, mc, accent, pos=(pos[0], pos[1], pos[2] + hz / 2 + 0.0012 * s),
          quat=quat)

    # 输出盘 + 中心轴
    mp = f"m_servo_horn_{int(s * 100)}"
    _mesh_once(spec, mp, lambda: meshgen.add_cylinder_mesh(
        spec, mp, 0.0085 * s, 0.0022 * s, nu=32))
    _geom(body, mp, metal, pos=(pos[0], pos[1], pos[2] + hz / 2 + 0.0042 * s),
          quat=quat)

    ma = f"m_servo_axle_{int(s * 100)}"
    _mesh_once(spec, ma, lambda: meshgen.add_cylinder_mesh(
        spec, ma, 0.0026 * s, 0.006 * s, nu=16))
    _geom(body, ma, metal, pos=(pos[0], pos[1], pos[2] + hz / 2 + 0.001 * s),
          quat=quat)

    # 安装耳：两端各伸出一片，带螺栓孔位
    mt = f"m_servo_tab_{int(s * 100)}"
    _mesh_once(spec, mt, lambda: meshgen.add_rounded_box(
        spec, mt, (0.005 * s, hy / 2 * 0.92, 0.0025 * s), roundness=8.0,
        nu=24, nv=12))
    for sx in (-1, 1):
        _geom(body, mt, shell, pos=(pos[0] + sx * (hx / 2 + 0.004 * s),
                                    pos[1], pos[2] - hz * 0.22), quat=quat)


# ---------------------------------------------------------------- 紧固件
def bolt(spec, body, pos, quat=(1, 0, 0, 0), r=0.0028, head=0.0024,
         metal="m_metal", with_washer=True):
    """M3 级内六角螺栓。六棱柱头 + 可选垫圈，四个角一摆就有"装配感"。"""
    mb = f"m_bolt_{int(r * 10000)}"
    _mesh_once(spec, mb, lambda: meshgen.add_prism(spec, mb, 6, r, head / 2))
    _geom(body, mb, metal, pos=pos, quat=quat)

    if with_washer:
        mw = f"m_washer_{int(r * 10000)}"
        _mesh_once(spec, mw, lambda: meshgen.add_torus(
            spec, mw, r * 1.15, r * 0.28, nu=20, nv=10))
        _geom(body, mw, metal,
              pos=(pos[0], pos[1], pos[2] - head * 0.55), quat=quat)


def bolt_ring(spec, body, radius, z, count=4, start=0.0, **kw):
    """沿圆周摆一圈螺栓。底座、法兰用得上。"""
    for i in range(count):
        a = start + 2 * np.pi * i / count
        bolt(spec, body, (radius * np.cos(a), radius * np.sin(a), z), **kw)


# ---------------------------------------------------------------- 结构件
def flange(spec, body, radius, thickness, pos=(0, 0, 0), mat="m_accent",
           bolts=4, bolt_radius=None, metal="m_metal"):
    """法兰盘 / 转台。中间厚圆盘，边缘一圈螺栓孔位。"""
    key = f"m_flange_{int(radius * 1000)}_{int(thickness * 1000)}"
    _mesh_once(spec, key, lambda: meshgen.add_cylinder_mesh(
        spec, key, radius, thickness / 2, nu=48))
    _geom(body, key, mat, pos=pos)

    if bolts:
        br = bolt_radius if bolt_radius is not None else radius * 0.72
        for i in range(bolts):
            a = 2 * np.pi * i / bolts + np.pi / 4
            bolt(spec, body,
                 (pos[0] + br * np.cos(a), pos[1] + br * np.sin(a),
                  pos[2] + thickness / 2),
                 r=radius * 0.09, head=radius * 0.08, metal=metal,
                 with_washer=False)


def ring(spec, body, R, r, pos=(0, 0, 0), mat="m_metal"):
    """轴承环 / O 圈，套在关节轴上。近景能看出"有转动副"。"""
    key = f"m_ring_{int(R * 10000)}_{int(r * 10000)}"
    _mesh_once(spec, key, lambda: meshgen.add_torus(spec, key, R, r, nu=36, nv=14))
    _geom(body, key, mat, pos=pos)


def plate(spec, body, half, pos=(0, 0, 0), quat=(1, 0, 0, 0), mat="m_shell",
          roundness=8.0):
    """圆角板件。连杆侧板、安装板都用它。"""
    key = f"m_plate_{int(half[0] * 1000)}_{int(half[1] * 1000)}_{int(half[2] * 1000)}"
    _mesh_once(spec, key, lambda: meshgen.add_rounded_box(
        spec, key, tuple(half), roundness=roundness, nu=44, nv=20))
    _geom(body, key, mat, pos=pos, quat=quat)


def beam(spec, body, length, width, thick, pos=(0, 0, 0), quat=(1, 0, 0, 0),
         mat="m_shell"):
    """
    镂空连杆：不是一整条实心，而是上下两条边梁 + 两端竖条，
    中间留空。真机为了减重都是这么做的，而且镂空后能看见里面的走线。
    """
    x0 = pos[0] - length / 2
    rail_h = thick * 1.1
    for sz in (-1, 1):
        plate(spec, body,
              (length / 2, width / 2, rail_h / 2),
              pos=(pos[0], pos[1], pos[2] + sz * (thick * 1.6)),
              quat=quat, mat=mat)
    # 两端竖条
    for sx in (-1, 1):
        plate(spec, body,
              (thick * 1.2, width / 2, thick * 2.2),
              pos=(x0 + (0 if sx < 0 else length), pos[1], pos[2]),
              quat=quat, mat=mat)


def cable(spec, body, p0, p1, radius=0.0026, mat="m_rubber", tag="a"):
    """
    线缆：两点之间一根圆柱。方向由两点决定，用四元数把 z 轴转到该方向。

    每段长度不同，而 MuJoCo 的 mesh 无法在 geom 上做非均匀缩放，
    所以按"长度取整到毫米"生成独立网格。段数很少，代价可以忽略。

    真实走线是贴着结构的，这里简化为直线；分几段挂在各自 body 上，
    关节转动时看着就像线束跟着走。
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None

    key = f"m_cable_{tag}_{int(round(L * 1000))}_{int(radius * 10000)}"
    _mesh_once(spec, key, lambda: meshgen.add_cylinder_mesh(
        spec, key, radius, L / 2, nu=14, nv=2))

    z = np.array([0, 0, 1.0])
    u = d / L
    axis = np.cross(z, u)
    s = np.linalg.norm(axis)
    if s < 1e-8:
        quat = [1, 0, 0, 0]
    else:
        axis = axis / s
        ang = np.arccos(float(np.clip(np.dot(z, u), -1, 1)))
        half = ang / 2
        quat = [float(np.cos(half)), float(axis[0] * np.sin(half)),
                float(axis[1] * np.sin(half)), float(axis[2] * np.sin(half))]

    return _geom(body, key, mat, pos=tuple((p0 + p1) / 2), quat=quat)
