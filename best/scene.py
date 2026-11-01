# -*- coding: utf-8 -*-
"""
场景搭建：纹理、桌面、背景、灯光、相机
================================================================

一个"像样"的仿真场景，光有几何不够，还差三件事：
    1. 材质要有微观结构（纯色一眼假）
    2. 光要有主次（单个头灯会让所有面一个亮度，没有体积感）
    3. 要有可读的背景和地面（否则机械臂悬在虚空里）

这里全部程序化生成，不依赖外部贴图文件。
"""

import numpy as np
import mujoco

import grasp_scene
import robot
import textures as tex


# ---------------------------------------------------------------- 纹理
def register_textures(spec):
    """现场生成所有纹理。返回名字字典，方便后面引用。"""
    names = {}

    pairs = {
        "t_fdm_white": tex.fdm_grain(base=(234, 233, 228), line_strength=9.0,
                                     line_period=6.0, seed=11),
        "t_fdm_amber": tex.fdm_grain(base=(238, 176, 44), line_strength=8.0,
                                     line_period=6.0, seed=12),
        "t_matte_dark": tex.matte_plastic(base=(42, 44, 48), noise=2.0, seed=13),
        "t_metal": tex.brushed_metal(base=(178, 181, 186), seed=14),
        "t_wood": tex.wood(seed=15),
        "t_backdrop": tex.gradient(size=256, top=(26, 28, 33), bottom=(64, 68, 76)),
    }
    for name, img in pairs.items():
        tex.register(spec, name, img)
        names[name] = name
    return names


# ---------------------------------------------------------------- 地面/背景
def add_table(spec, half=(0.42, 0.34, 0.02)):
    """
    桌面。顶面正好在 z=0，所以机械臂底座和物体都坐在它上面。
    用 box 而不是 plane：plane 是无限大的，透视里看不出"桌子边缘"，
    少了一条判断尺度的线索。
    """
    body = spec.worldbody.add_body()
    body.name = "table"
    # 桌面就是 z=0 这个平面。之前用 box 时要注意"中心 vs 顶面"，
    # 换成 plane 之后平面本身就在 pos 处，所以这里不能再减 half[2]，
    # 否则物体会直接穿过桌面掉到 z=-0.02 以下。
    body.pos = (0.10, 0.0, 0.0)

    g = body.add_geom()
    # 用 plane 而不是 box：box 每个面的 UV 是独立的，六个面接缝处纹理对不上，
    # 桌面上会出现一条明显的错位线。plane 是无限大的，UV 按世界坐标连续，
    # 顺带把背景从"纯黑虚空"变成有地面延伸，画面不容易显得悬空。
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [6.0, 6.0, 0.02]
    g.material = "m_wood"
    # 桌面属于"物体组"，只和被抓物、指尖碰，不被机械臂本体穿模
    g.contype = grasp_scene.OBJECT_CONTYPE
    g.conaffinity = grasp_scene.OBJECT_CONAFFINITY
    g.friction = [1.0, 0.01, 0.001]


def add_backdrop(spec, dist=0.62, height=0.75):
    """
    背景幕布：一块带渐变的竖板，用自发光材质，不受灯光影响。
    比纯黑背景有层次，也不会像 skybox 那样需要六张图。
    """
    body = spec.worldbody.add_body()
    body.name = "backdrop"
    body.pos = (0.19, dist, height * 0.5)

    g = body.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [3.4, 0.004, height * 0.5]
    g.material = "m_backdrop"
    g.contype = 0
    g.conaffinity = 0


# ---------------------------------------------------------------- 灯光
def add_lights(spec):
    """
    三点布光。主光负责立体感并投影，补光压暗部，轮廓光把机械臂从背景里剥出来。

    色温故意不一致：主光偏暖、补光偏冷，这样白色外壳上会出现微妙的冷暖过渡，
    比纯白光更像照片。
    """
    key = spec.worldbody.add_light()
    key.name = "key"
    key.pos = [0.42, -0.55, 0.78]
    key.dir = [-0.45, 0.6, -0.75]
    key.diffuse = [1.15, 1.09, 1.00]
    key.specular = [0.22, 0.21, 0.20]
    key.castshadow = True
    key.intensity = 1.0

    fill = spec.worldbody.add_light()
    fill.name = "fill"
    fill.pos = [0.05, -0.9, 0.35]
    fill.dir = [0.0, 1.0, -0.35]
    fill.diffuse = [0.22, 0.25, 0.32]
    fill.specular = [0.0, 0.0, 0.0]
    fill.castshadow = False

    rim = spec.worldbody.add_light()
    rim.name = "rim"
    rim.pos = [-0.45, 0.55, 0.60]
    rim.dir = [0.5, -0.55, -0.6]
    rim.diffuse = [0.50, 0.50, 0.54]
    rim.specular = [0.15, 0.15, 0.15]
    rim.castshadow = False


# ---------------------------------------------------------------- 被抓物
def add_cube(spec, pos=(0.20, 0.0, 0.015), half=0.015, mass=0.030,
             friction=1.0):
    """和 eval/sim 里同一个物体，只是换了材质，方便两边的数字能对上。"""
    body = spec.worldbody.add_body()
    body.name = "cube"
    body.pos = list(pos)
    body.add_freejoint()

    g = body.add_geom()
    g.name = "cube_geom"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [half, half, half]
    g.material = "m_cube"
    g.contype = grasp_scene.OBJECT_CONTYPE
    g.conaffinity = grasp_scene.OBJECT_CONAFFINITY
    g.friction = [friction, 0.01, 0.001]
    g.mass = mass
    g.solref = [0.004, 0.9]
    return body


# ---------------------------------------------------------------- 组装
def build(with_cube=True, object_pos=(0.20, 0.0, 0.015), mass=0.030,
          friction=1.0):
    """
    完整场景。顺序有讲究：
    先结构（夹爪要有 body 才能被换外观），再纹理，再外观，最后场景道具。
    """
    spec = mujoco.MjSpec.from_file(grasp_scene.MODEL_PATH)
    spec.option.gravity = [0.0, 0.0, -9.81]

    grasp_scene.add_gripper(spec)      # 夹爪的关节/执行器/equality
    register_textures(spec)            # 纹理要先注册，材质才能引用

    # 场景道具的材质
    tex.material(spec, "m_wood", "t_wood", rgba=(0.88, 0.86, 0.84, 1.0),
                 roughness=0.5, specular=0.30, texrepeat=(2.4, 2.4))
    # 背景板：不能自发光。emission 一高，整块变成纯白，把场景全冲掉
    tex.material(spec, "m_backdrop", "t_backdrop", rgba=(0.30, 0.31, 0.35, 1.0),
                 roughness=1.0, specular=0.0, texrepeat=(1, 1))
    # 方块不挂纹理：深色纹理乘上去会把蓝色压成黑块
    tex.material(spec, "m_cube", None, rgba=(0.20, 0.62, 0.86, 1.0),
                 roughness=0.38, specular=0.5, shininess=0.6)

    robot.build_visuals(spec)          # 换掉机械臂所有 geom

    # 收紧内部伺服。control.ServoModel 会在指令侧注入延迟和背隙，
    # 如果内部还是原来那套软参数（kp 6~25），两层不完美叠加会一起把姿态拖垮。
    # 真舵机的内部位置环其实很硬，软的部分主要体现在间隙和延迟上。
    for name, kp, kv in (("a_j1", 320, 26), ("a_j2", 320, 26),
                         ("a_j3", 260, 20), ("a_j4", 160, 12),
                         ("a_j5", 90, 6)):
        spec.actuator(name).set_to_position(kp=kp, kv=kv)

    # TCP 是测量用的 site，调试期看着像个红球，成品里要藏掉
    tcp = spec.site("tcp")
    tcp.rgba = [0.0, 0.0, 0.0, 0.0]
    tcp.size = [0.0001, 0.0001, 0.0001]

    add_table(spec)
    add_lights(spec)
    if with_cube:
        add_cube(spec, pos=object_pos, mass=mass, friction=friction)

    return spec


def compile_scene(**kwargs):
    spec = build(**kwargs)
    model = spec.compile()
    data = mujoco.MjData(model)

    # 离屏渲染要更大的缓冲，顺便开抗锯齿（默认 4 倍，这里明确写一遍）
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1920)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 1080)
    model.vis.quality.offsamples = 8
    model.vis.quality.shadowsize = 4096
    return model, data
