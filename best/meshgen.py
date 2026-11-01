# -*- coding: utf-8 -*-
"""
程序化网格生成
================================================================

MuJoCo 自带 make_supersphere(e, n)，但这两个参数的语义和教科书上的超椭球指数
对不上：照着 n=4 生成出来的是尖刺状菱形，不是圆角盒。与其猜参数，
不如自己按定义生成顶点和面 —— 形状完全可控，还能顺手控制细分密度。

超椭球：|x/a|^n + |y/b|^n + |z/c|^n = 1
    n=2   → 椭球
    n=4~8 → 圆角盒（注塑件那种圆润边角）
    n→∞   → 立方体（棱角）

做法是按球面方向采样，再沿方向找到落在超椭球面上的半径，
这样极点、接缝都不会出问题。
"""

import numpy as np


def superellipsoid(n=4.0, half=(1.0, 1.0, 1.0), nu=56, nv=28):
    """
    返回 (verts, faces)：verts 是 (N,3)，faces 是展平的三角索引。

    nu 是绕纵轴的经度分段，nv 是纬度分段。连杆这种细长件 nu 给 40 就够，
    球状件要给高一点，否则轮廓线会看出多边形。
    """
    a, b, c = half
    verts = np.zeros(((nv + 1) * nu, 3))

    for j in range(nv + 1):
        v = np.pi * j / nv
        sv, cv = np.sin(v), np.cos(v)
        for i in range(nu):
            u = 2.0 * np.pi * i / nu
            d = np.array([sv * np.cos(u), sv * np.sin(u), cv])
            # 沿方向 d 求超椭球面的交点：|t*d/a|^n + ... = 1
            s = ((abs(d[0]) / a) ** n + (abs(d[1]) / b) ** n +
                 (abs(d[2]) / c) ** n) ** (1.0 / n)
            verts[j * nu + i] = d / s

    faces = []
    for j in range(nv):
        for i in range(nu):
            i2 = (i + 1) % nu
            p0 = j * nu + i
            p1 = j * nu + i2
            p2 = (j + 1) * nu + i2
            p3 = (j + 1) * nu + i
            # 极点那一圈所有顶点重合，跳过退化三角形
            if j > 0:
                faces += [p0, p1, p2]
            if j < nv - 1:
                faces += [p0, p2, p3]

    return verts, np.array(faces, dtype=np.int32)


def add_rounded_box(spec, name, half, roundness=4.0, nu=48, nv=24,
                    smooth=True):
    """
    往 spec 里注册一个圆角盒网格。

    smoothnormal=True 让 MuJoCo 按顶点邻接算法线，圆角处才不会出现面片感；
    关掉的话每个三角形一个硬法线，看起来像低模。
    """
    verts, faces = superellipsoid(roundness, half, nu=nu, nv=nv)
    mesh = spec.add_mesh()
    mesh.name = name
    mesh.uservert = verts.reshape(-1).tolist()
    mesh.userface = faces.tolist()
    mesh.smoothnormal = smooth
    # 顶点多了以后凸包上限要放大，否则圆角会被削平
    mesh.maxhullvert = 4096
    return mesh


def add_cylinder_mesh(spec, name, radius, half_height, nu=40, nv=2):
    """
    真正的圆柱（不是超椭球近似的）。nv=2 表示侧面只分两段，圆盘用扇形补。
    这里用 n=64 的超椭球近似也行，但侧壁会略鼓，做轴类零件不好看。
    """
    verts = []
    faces = []

    # 侧壁
    for j in range(nv + 1):
        z = -half_height + 2 * half_height * j / nv
        for i in range(nu):
            u = 2 * np.pi * i / nu
            verts.append([radius * np.cos(u), radius * np.sin(u), z])

    for j in range(nv):
        for i in range(nu):
            i2 = (i + 1) % nu
            p0 = j * nu + i
            p1 = j * nu + i2
            p2 = (j + 1) * nu + i2
            p3 = (j + 1) * nu + i
            faces += [p0, p1, p2, p0, p2, p3]

    # 上下端面
    for sign, z in ((1, half_height), (-1, -half_height)):
        center = len(verts)
        verts.append([0.0, 0.0, z])
        ring_start = len(verts)
        for i in range(nu):
            u = 2 * np.pi * i / nu
            verts.append([radius * np.cos(u), radius * np.sin(u), z])
        for i in range(nu):
            i2 = (i + 1) % nu
            if sign > 0:
                faces += [center, ring_start + i, ring_start + i2]
            else:
                faces += [center, ring_start + i2, ring_start + i]

    mesh = spec.add_mesh()
    mesh.name = name
    mesh.uservert = np.array(verts).reshape(-1).tolist()
    mesh.userface = faces
    mesh.smoothnormal = True
    mesh.maxhullvert = 4096
    return mesh
