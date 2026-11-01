# -*- coding: utf-8 -*-
"""
SO-101 结构重建
================================================================

运动学和 models/so101.xml 完全一致（关节轴、连杆长度、body 层级都不动），
只把每个 body 原来那个"一个方块"的几何，换成由零件组装出来的结构：

    底座    安装底盘 + 螺栓 + 立柱 + 装饰环 + 转台法兰
    肩部    J1 舵机 + 转盘 + 侧支板 + 线缆
    大臂    J2/J3 舵机 + 镂空边梁（上下两条 + 两端立筋）+ 走线
    小臂    J3/J4 舵机 + 同样结构的短梁
    腕     J4 小舵机 + 法兰 + 连接座
    夹爪    J5 舵机 + 齿轮箱 + 两根橡胶指尖

为什么这么拆：
    一根圆角长条看起来像积木，因为它没有"装配痕迹" —— 没有紧固件、没有分缝、
    没有加强筋、没有走线。补上这些之后，即使每件都是简单的盒和柱，
    整体也会读作"一台设备"。这是工业渲染里最省力的一条路：
    细节靠零件层次，不靠高模。

尺寸仍然照抄 so101.xml，改一处两边都要动：
    底座到肩轴 95 | 肩侧偏 30 | 大臂 116 | 小臂 135 | 腕→TCP 110
"""

import mujoco

import meshgen
import parts

# ---------------------------------------------------------------- 调色
COL_SHELL = (0.93, 0.93, 0.90, 1.0)     # 主体：打印白，略偏暖
COL_ACCENT = (0.94, 0.70, 0.16, 1.0)    # 关节盖：琥珀黄
COL_DARK = (0.20, 0.21, 0.24, 1.0)      # 深色件：石墨灰
COL_METAL = (0.74, 0.76, 0.80, 1.0)     # 螺丝 / 轴
COL_RUBBER = (0.17, 0.17, 0.18, 1.0)    # 指尖橡胶 / 线缆
COL_GEAR = (0.30, 0.28, 0.26, 1.0)      # 齿轮箱

# 舵机朝向：默认输出轴沿 +z，转到对应关节轴
QUAT_AXIS_Y = (0.70710678, 0.70710678, 0.0, 0.0)   # z → y
QUAT_AXIS_X = (0.70710678, 0.0, 0.70710678, 0.0)   # z → x


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


def _clear(spec, body):
    """
    删掉这个 body 原有的 geom（原模型每个 body 只有一个 box/cylinder）。

    注意：不能写 `del body.geoms[0]` —— MjGeomVec 是只读视图，这么写不报错
    但也不会删掉任何东西，最后得到"新旧几何叠在一起"的怪样子。
    正确做法是 spec.delete(element)。
    """
    while len(body.geoms):
        spec.delete(body.geoms[0])


def _add_mesh(body, mesh_name, mat, pos=(0, 0, 0), quat=(1, 0, 0, 0)):
    g = body.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_MESH
    g.meshname = mesh_name
    g.material = mat
    g.pos = list(pos)
    g.quat = list(quat)
    g.contype = 0
    g.conaffinity = 0
    return g


# ---------------------------------------------------------------- 装配
def build_visuals(spec):
    """逐段把机械臂"装"起来，返回材质字典供场景复用配色。"""
    parts.reset_cache()

    M = {
        "shell": _mat(spec, "m_shell", "t_fdm_white", COL_SHELL, 0.62,
                      texrepeat=(2.5, 2.5)),
        "accent": _mat(spec, "m_accent", "t_fdm_amber", COL_ACCENT, 0.58,
                       texrepeat=(2.5, 2.5)),
        "dark": _mat(spec, "m_dark", "t_matte_dark", COL_DARK, 0.72),
        "metal": _mat(spec, "m_metal", "t_metal", COL_METAL, 0.26, metallic=0.88,
                      specular=0.62, shininess=0.75, texrepeat=(4, 4)),
        "rubber": _mat(spec, "m_rubber", None, COL_RUBBER, 0.95, specular=0.06),
        "gear": _mat(spec, "m_gear", "t_metal", COL_GEAR, 0.42, metallic=0.55,
                     specular=0.45, shininess=0.5, texrepeat=(6, 6)),
    }

    # ================= 底座 =================
    base = spec.body("base")
    _clear(spec, base)

    # 安装底盘
    parts.flange(spec, base, radius=0.052, thickness=0.008, pos=(0, 0, 0.005),
                 mat="m_dark", bolts=4, bolt_radius=0.040, metal="m_metal")
    parts.ring(spec, base, 0.0495, 0.0022, pos=(0, 0, 0.0102), mat="m_metal")

    # 立柱：两段，中间一道装饰环，看起来是分段打印后拧起来的
    meshgen.add_cylinder_mesh(spec, "m_base_col_lo", 0.0256, 0.019)
    _add_mesh(base, "m_base_col_lo", "m_shell", pos=(0, 0, 0.028))
    meshgen.add_cylinder_mesh(spec, "m_base_col_hi", 0.0242, 0.021)
    _add_mesh(base, "m_base_col_hi", "m_shell", pos=(0, 0, 0.069))
    parts.ring(spec, base, 0.0252, 0.0028, pos=(0, 0, 0.049), mat="m_metal")

    # 顶部转台
    parts.flange(spec, base, radius=0.0300, thickness=0.009, pos=(0, 0, 0.0895),
                 mat="m_accent", bolts=4, bolt_radius=0.0235, metal="m_metal")

    # 电源线：从底盘侧面引上来，贴着立柱走
    parts.cable(spec, base, (-0.026, -0.021, 0.012), (-0.026, -0.021, 0.080),
                radius=0.0030, tag="base")
    parts.cable(spec, base, (-0.052, -0.021, 0.004), (-0.026, -0.021, 0.012),
                radius=0.0030, tag="base2")

    # ================= 肩部（J1）=================
    sh = spec.body("shoulder")
    _clear(spec, sh)

    # J1 舵机竖直放置，藏在转台下面
    parts.servo(spec, sh, scale=0.92, pos=(0, 0, -0.021), quat=(1, 0, 0, 0))
    # 转盘
    parts.flange(spec, sh, radius=0.0275, thickness=0.007, pos=(0, 0, -0.001),
                 mat="m_accent", bolts=0)
    # 两侧支板把舵机和大臂连接起来
    parts.plate(spec, sh, (0.011, 0.0035, 0.017), pos=(0.006, 0.0165, -0.008),
                mat="m_shell")
    parts.plate(spec, sh, (0.011, 0.0035, 0.017), pos=(0.006, -0.0165, -0.008),
                mat="m_shell")
    parts.bolt(spec, sh, (0.013, 0.0165, -0.008), quat=QUAT_AXIS_Y, r=0.0026)
    parts.bolt(spec, sh, (0.013, -0.0165, -0.008), quat=QUAT_AXIS_Y, r=0.0026)
    # 肩部走线
    parts.cable(spec, sh, (-0.004, -0.020, -0.020), (-0.004, -0.020, 0.004),
                radius=0.0026, tag="sh")

    # ================= 大臂（J2 → J3）=================
    ua = spec.body("upper_arm")
    _clear(spec, ua)

    # J2 舵机：输出轴沿 y，正对着肩部
    parts.servo(spec, ua, scale=1.0, pos=(0.0, 0, 0), quat=QUAT_AXIS_Y,
                accent="m_accent", shell="m_dark")
    # 镂空边梁：上下两条 + 两端立筋
    parts.beam(spec, ua, length=0.104, width=0.0142, thick=0.0036,
               pos=(0.062, 0.0158, 0), mat="m_shell")
    parts.beam(spec, ua, length=0.104, width=0.0142, thick=0.0036,
               pos=(0.062, -0.0158, 0), mat="m_shell")
    # J3 舵机（本段末端，下一段的起点）
    parts.servo(spec, ua, scale=0.95, pos=(0.116, 0, 0), quat=QUAT_AXIS_Y,
                accent="m_accent", shell="m_dark")
    # 走线：贴着侧板上缘
    parts.cable(spec, ua, (0.016, 0.0185, 0.011), (0.108, 0.0185, 0.011),
                radius=0.0026, tag="ua")

    # ================= 小臂（J3 → J4）=================
    la = spec.body("lower_arm")
    _clear(spec, la)

    parts.servo(spec, la, scale=0.92, pos=(0.0, 0, 0), quat=QUAT_AXIS_Y,
                accent="m_accent", shell="m_dark")
    parts.beam(spec, la, length=0.120, width=0.0128, thick=0.0032,
               pos=(0.072, 0.0142, 0), mat="m_shell")
    parts.beam(spec, la, length=0.120, width=0.0128, thick=0.0032,
               pos=(0.072, -0.0142, 0), mat="m_shell")
    parts.servo(spec, la, scale=0.80, pos=(0.135, 0, 0), quat=QUAT_AXIS_Y,
                accent="m_accent", shell="m_dark")
    parts.cable(spec, la, (0.014, 0.0165, 0.010), (0.126, 0.0165, 0.010),
                radius=0.0024, tag="la")

    # ================= 腕（J4 → J5）=================
    wr = spec.body("wrist")
    _clear(spec, wr)

    parts.servo(spec, wr, scale=0.72, pos=(0.0, 0, 0), quat=QUAT_AXIS_Y,
                accent="m_accent", shell="m_dark")
    # 腕部连接座：把 J4 和 J5 接起来
    parts.plate(spec, wr, (0.012, 0.011, 0.010), pos=(0.014, 0, 0),
                mat="m_shell")
    parts.ring(spec, wr, 0.0105, 0.0018, pos=(0.006, 0, 0), mat="m_metal")
    parts.cable(spec, wr, (0.004, 0.0125, 0.006), (0.022, 0.0125, 0.006),
                radius=0.0022, tag="wr")

    # ================= 夹爪（J5）=================
    tool = spec.body("tool")
    _clear(spec, tool)

    # J5 舵机：输出轴沿前臂方向
    parts.servo(spec, tool, scale=0.52, pos=(0.004, 0, 0), quat=QUAT_AXIS_X,
                accent="m_accent", shell="m_dark")
    # 齿轮箱外壳
    meshgen.add_rounded_box(spec, "m_gear_case", (0.0125, 0.0115, 0.0105),
                            roundness=6.0, nu=40, nv=20)
    _add_mesh(tool, "m_gear_case", "m_gear", pos=(0.030, 0, 0))
    # 齿轮盘（露在外面的一截，暗示里面有减速齿轮）
    meshgen.add_cylinder_mesh(spec, "m_gear_disc", 0.0088, 0.0022, nu=28)
    _add_mesh(tool, "m_gear_disc", "m_metal", pos=(0.030, 0.0148, 0),
              quat=QUAT_AXIS_Y)
    # 指的根部导轨
    parts.plate(spec, tool, (0.008, 0.017, 0.004), pos=(0.046, 0, 0),
                mat="m_metal")

    # ================= 指尖（唯一参与接触）=================
    for side, sign in (("l", 1.0), ("r", -1.0)):
        body = spec.body(f"finger_{side}")
        _clear(spec, body)

        # 指尖是一块橡胶平板，视觉和碰撞都用 box，不折腾网格。
        #
        # 之前试过"圆角网格做外观 + 内部藏小 box 做碰撞"，结果两者对不齐，
        # 画面上像两块错开的板；而且圆角网格直接参与接触时，接触点落在圆角上、
        # 法向是斜的，物件一抬就被挤着往外滑。平板法向水平，摩擦才真正抗重力。
        g = body.add_geom()
        g.name = f"pad_{side}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.019, 0.004, 0.011]
        g.pos = [0.0, sign * 0.004, 0.0]
        g.material = "m_rubber"
        g.rgba = COL_RUBBER
        g.contype = 1
        g.conaffinity = 2
        g.friction = [2.0, 0.02, 0.002]

    return M
