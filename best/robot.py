# -*- coding: utf-8 -*-
"""
SO-101 外观重建
================================================================

运动学和 models/so101.xml 完全一致（关节轴、连杆长度、body 层级都不动），
只把每个 body 里那个"方块贴图"换成程序化生成的圆角网格 + PBR 材质。

为什么这样做：
    前面几级里的 box 够算物理，但一眼就是示意图。真机是 FDM 打印的塑料件，
    有圆角、有层纹、有不同配色的关节盖。把这些补上之后，渲染出来才像"设备"
    而不是"几何体演示"。

尺寸全部照抄 so101.xml，改一处两边都要动：
    底座圆柱 r30 h95 | 肩壳 52³ | 大臂 116×30×24 | 小臂 135×26×20
    腕 28×24×24 | 夹爪座 60×24×24 | 指尖 50×8×24

两类几何各用各的做法：
    方块 → 超椭球网格（圆角注塑件的感觉，box 做不到）
    圆柱 → MuJoCo 原生 cylinder（MjsMesh.make_cylinder 的 spacing 语义容易踩坑，
            而且圆盘这种简单形状没必要走网格）
"""

import math

import mujoco

import meshgen

# ---------------------------------------------------------------- 调色
COL_SHELL = (0.93, 0.93, 0.90, 1.0)     # 主体：打印白，略偏暖
COL_ACCENT = (0.94, 0.70, 0.16, 1.0)    # 关节盖：琥珀黄
COL_DARK = (0.26, 0.27, 0.30, 1.0)      # 深色件：石墨灰
COL_METAL = (0.72, 0.74, 0.79, 1.0)     # 螺丝 / 轴
COL_RUBBER = (0.13, 0.13, 0.14, 1.0)    # 指尖橡胶

QUAT_Y = (0.70710678, 0.70710678, 0.0, 0.0)   # 绕 x 转 90° → 圆柱轴朝 y
QUAT_X = (0.70710678, 0.0, 0.70710678, 0.0)   # 绕 y 转 90° → 圆柱轴朝 x


# ---------------------------------------------------------------- 网格
def rounded_box(spec, name, half, roundness=4.0, res=16):
    """
    圆角盒。走自己写的超椭球生成器，不用 MuJoCo 的 make_supersphere ——
    那个函数的 e/n 语义和教科书的指数对不上，n=4 会生成尖刺状菱形。
    """
    return meshgen.add_rounded_box(spec, name, half, roundness=roundness,
                                   nu=max(24, int(res * 3)),
                                   nv=max(12, int(res * 1.5)))


# ---------------------------------------------------------------- 材质
def _mat(spec, name, texture, rgba, roughness, metallic=0.0,
         specular=0.35, shininess=0.35, texrepeat=(2.0, 2.0)):
    mat = spec.add_material()
    mat.name = name
    mat.rgba = list(rgba)
    mat.roughness = roughness
    mat.metallic = metallic
    mat.specular = specular
    mat.shininess = shininess
    if texture:
        # 颜色纹理必须放在 slot 1（slot 0 是保留角色，塞那里渲染出来是纯色）
        slots = [""] * 10
        slots[1] = texture
        mat.textures = slots
        mat.texrepeat = list(texrepeat)
    return mat


# ---------------------------------------------------------------- geom 工具
def _add_mesh(body, mesh_name, mat, pos=(0, 0, 0), quat=(1, 0, 0, 0),
              contype=0, conaffinity=0, friction=None):
    g = body.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_MESH
    g.meshname = mesh_name
    g.material = mat
    g.pos = list(pos)
    g.quat = list(quat)
    g.contype = contype
    g.conaffinity = conaffinity
    if friction is not None:
        g.friction = list(friction)
    return g


def _add_cyl(body, radius, half_h, mat, pos=(0, 0, 0), quat=(1, 0, 0, 0),
             contype=0, conaffinity=0):
    g = body.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.size = [radius, half_h, 0.0]
    g.material = mat
    g.pos = list(pos)
    g.quat = list(quat)
    g.contype = contype
    g.conaffinity = conaffinity
    return g


def _clear(spec, body):
    """
    删掉这个 body 原有的 geom（原模型每个 body 只有一个 box/cylinder）。

    注意：不能写 `del body.geoms[0]` —— MjGeomVec 是只读视图，这么写不报错
    但也不会删掉任何东西，最后得到"新旧几何叠在一起"的怪样子。
    正确做法是 spec.delete(element)。
    """
    while len(body.geoms):
        spec.delete(body.geoms[0])


# ---------------------------------------------------------------- 主体
def build_visuals(spec):
    """把所有 body 的外观换掉，返回材质名，供场景里物体/桌面复用配色。"""
    M = {
        "shell": _mat(spec, "m_shell", "t_fdm_white", COL_SHELL, 0.62,
                      texrepeat=(2.5, 2.5)),
        "accent": _mat(spec, "m_accent", "t_fdm_amber", COL_ACCENT, 0.58,
                       texrepeat=(2.5, 2.5)),
        "dark": _mat(spec, "m_dark", "t_matte_dark", COL_DARK, 0.72),
        "metal": _mat(spec, "m_metal", "t_metal", COL_METAL, 0.28, metallic=0.85,
                      specular=0.6, shininess=0.7, texrepeat=(4, 4)),
        "rubber": _mat(spec, "m_rubber", None, COL_RUBBER, 0.95, specular=0.08),
    }

    # ---------------- 底座 ----------------
    base = spec.body("base")
    _clear(spec, base)
    _add_cyl(base, 0.030, 0.0475, "m_shell", pos=(0, 0, 0.0475))
    _add_cyl(base, 0.043, 0.006, "m_dark", pos=(0, 0, 0.006))
    _add_cyl(base, 0.033, 0.005, "m_accent", pos=(0, 0, 0.093))

    # 四颗螺丝：纯视觉，让底座像"上过螺丝"而不是一根光柱
    for angle in (0.785, 2.356, 3.927, 5.498):
        _add_cyl(base, 0.0035, 0.0012, "m_metal",
                 pos=(0.036 * math.cos(angle), 0.036 * math.sin(angle), 0.0125))

    # ---------------- 肩部（J1 回转）----------------
    sh = spec.body("shoulder")
    _clear(spec, sh)
    rounded_box(spec, "mesh_shoulder", (0.026, 0.026, 0.026), roundness=3.4, res=18)
    _add_mesh(sh, "mesh_shoulder", "m_shell")
    # 两侧舵机端盖
    _add_cyl(sh, 0.024, 0.008, "m_accent", pos=(0, 0.030, 0), quat=QUAT_Y)
    _add_cyl(sh, 0.024, 0.008, "m_accent", pos=(0, -0.030, 0), quat=QUAT_Y)

    # ---------------- 大臂 ----------------
    ua = spec.body("upper_arm")
    _clear(spec, ua)
    rounded_box(spec, "mesh_upper", (0.058, 0.015, 0.012), roundness=4.0, res=20)
    rounded_box(spec, "mesh_upper_rib", (0.042, 0.006, 0.004), roundness=5.0, res=14)
    _add_mesh(ua, "mesh_upper", "m_shell", pos=(0.058, 0, 0))
    _add_mesh(ua, "mesh_upper_rib", "m_accent", pos=(0.058, 0, 0.014))

    # ---------------- 小臂 ----------------
    la = spec.body("lower_arm")
    _clear(spec, la)
    rounded_box(spec, "mesh_lower", (0.0675, 0.013, 0.010), roundness=4.0, res=20)
    rounded_box(spec, "mesh_lower_rib", (0.048, 0.005, 0.0035), roundness=5.0, res=14)
    _add_mesh(la, "mesh_lower", "m_shell", pos=(0.0675, 0, 0))
    _add_mesh(la, "mesh_lower_rib", "m_accent", pos=(0.0675, 0, 0.012))

    # ---------------- 腕 ----------------
    wr = spec.body("wrist")
    _clear(spec, wr)
    rounded_box(spec, "mesh_wrist", (0.014, 0.012, 0.012), roundness=3.2, res=16)
    _add_mesh(wr, "mesh_wrist", "m_dark")

    # ---------------- 夹爪座 ----------------
    tool = spec.body("tool")
    _clear(spec, tool)
    rounded_box(spec, "mesh_gripper", (0.030, 0.012, 0.012), roundness=3.6, res=18)
    _add_mesh(tool, "mesh_gripper", "m_dark", pos=(0.035, 0, 0))

    # ---------------- 指尖（唯一参与接触的几何）----------------
    for side, sign in (("l", 1.0), ("r", -1.0)):
        body = spec.body(f"finger_{side}")
        _clear(spec, body)

        # 指尖本来就是一块橡胶平板，视觉和碰撞都用 box，不折腾网格。
        #
        # 之前的做法是圆角网格负责外观、内部藏一个小 box 负责碰撞，结果是
        # 两者法线/尺寸对不齐，画面上看着像两块错开的板；更要命的是圆角网格
        # 直接参与接触时，接触点落在圆角上、法线是斜的，物体一抬就被挤着往外滑。
        # 平板接触法向水平，摩擦才真正用来抗重力。
        g = body.add_geom()
        g.name = f"pad_{side}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.025, 0.004, 0.012]
        g.pos = [0.0, sign * 0.004, 0.0]
        g.material = "m_rubber"
        g.rgba = COL_RUBBER      # 双保险：geom 自带 rgba 在某些情况下会盖过材质
        g.contype = 1
        g.conaffinity = 2
        g.friction = [2.0, 0.02, 0.002]

    return M
