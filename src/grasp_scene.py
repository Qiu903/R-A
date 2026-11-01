# -*- coding: utf-8 -*-
"""
第 3 级场景构建：在 so101.xml 之上，用代码长出夹爪
================================================================

为什么不直接改 models/so101.xml：
    那个文件是第 1、2 级的基准（关重力、无接触、5 关节）。往里加夹爪会让
    第 2 级脚本的 ctrl 维度从 5 变 6，已经验证过的数字全部作废。
    而夹爪又必须挂在 tool 这个嵌套 body 下面，XML 的 include 只能插入
    整个顶层结构、没法往已有 body 里追加子元素。

    所以采用 MjSpec：读同一份 XML，在内存里追加夹爪，不改磁盘上的几何。
    "三级共用同一组 DH 参数和同一个臂的几何"这条原则因此仍然成立。

夹爪怎么设计的（简化成平行两指）：
    真实 SO-101 是齿轮联动的两指。这里用两个 slide 关节，各带一根手指，
    再用一个 joint equality 把它们锁成镜像：左指前进多少，右指就后退多少。
    只用一个执行器驱动左指，右指靠约束跟上 —— 这就是"联动"。

接触的分组方式（关键，别乱改）：
    contype / conaffinity 是位掩码，两者按位与非零才会产生接触。
        指尖    contype=1 conaffinity=2
        被抓物  contype=2 conaffinity=3      （下一步加）
    于是：指尖↔物体  2&2 ≠ 0  会碰
          指尖↔指尖  2&1 = 0  不碰（否则两指合拢时会被自己的接触力顶开）
          指尖↔连杆  2&0 = 0  不碰
"""

import os

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(HERE), "models", "so101.xml")

ARM_JOINTS = ["j1_pan", "j2_lift", "j3_elbow", "j4_wrist_flex", "j5_wrist_roll"]
ARM_ACTUATORS = ["a_j1", "a_j2", "a_j3", "a_j4", "a_j5"]

FINGER_OPEN = 0.020        # 单指行程：0 = 完全合拢，0.020 = 两指内表面相距 40 mm
FINGER_KP = 300.0          # 位置伺服刚度。太小的话"夹住"其实只是"贴着"，一抬就掉
FINGER_KV = 15.0
# 单指最大夹持力（N）。真舵机堵转力矩有限，不设上限位置伺服会一直加力，
# 轻物体直接被弹飞。这个值是抓取能力的主要瓶颈，见 eval_grasp 的归因实验。
GRIPPER_FORCE = 8.0

# 接触位掩码。被抓物用物体那一组，见模块开头的说明。
GRIPPER_CONTYPE = 1
GRIPPER_CONAFFINITY = 2
OBJECT_CONTYPE = 2
OBJECT_CONAFFINITY = 3


def add_gripper(spec):
    """往 tool body 下面挂两根手指，并把它们用 equality 锁成镜像。"""
    tool = spec.body("tool")

    for side, sign in (("l", 1.0), ("r", -1.0)):
        finger = tool.add_body()
        finger.name = f"finger_{side}"
        # 指根放在 85 mm 处、几何半长 25 mm，正好覆盖 60~110 mm，与 tcp site 对齐
        finger.pos = [0.085, 0.0, 0.0]

        joint = finger.add_joint()
        joint.name = f"j6_finger_{side}"
        joint.type = mujoco.mjtJoint.mjJNT_SLIDE
        # 两指都定义成"q 增大 = 向外张开"：左指沿 +y，右指沿 -y
        joint.axis = [0.0, sign, 0.0]
        joint.range = [0.0, FINGER_OPEN]
        joint.armature = 0.005
        joint.damping = [0.02, 0.0, 0.0]

        geom = finger.add_geom()
        geom.name = f"finger_{side}_pad"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [0.025, 0.004, 0.012]
        geom.pos = [0.0, sign * 0.004, 0.0]
        geom.rgba = [0.85, 0.35, 0.15, 1.0]
        geom.contype = GRIPPER_CONTYPE
        geom.conaffinity = GRIPPER_CONAFFINITY
        geom.friction = [2.0, 0.02, 0.002]      # 指尖要吃住物体，摩擦给高一点
        geom.solref = [0.005, 0.8]              # 硬一点的接触刚度，否则物体像橡皮

    # 联动：j6_l 与 j6_r 同号等值（两者轴向相反，所以看起来是镜像开合）
    eq = spec.add_equality()
    eq.name = "finger_couple"
    eq.type = mujoco.mjtEq.mjEQ_JOINT
    eq.name1 = "j6_finger_l"
    eq.name2 = "j6_finger_r"

    act = spec.add_actuator()
    act.name = "a_gripper"
    act.set_to_position(kp=FINGER_KP, kv=FINGER_KV)
    act.target = "j6_finger_l"
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.ctrlrange = [0.0, FINGER_OPEN]
    act.ctrllimited = True
    # 夹持力上限。真舵机堵转力矩有限，不设上限的话位置伺服会一直加力，
    # 轻物体直接被弹飞。3 N 大致相当于 STS3215 在 7.4V 下的合理上限。
    act.forcelimited = True
    act.forcerange = [-GRIPPER_FORCE, GRIPPER_FORCE]


def add_floor(spec):
    """
    地面。它和被抓物属于同一接触组（2/3），所以物体落得住；
    而连杆的 affinity 是 0，永远不会和地面接触 —— 臂可以"穿"进地板，
    这是故意的：第 2 级没开接触，臂的姿态里本来就有低于地面的解，
    一旦让地面接住它就变成"卡住"，问题会虚构成几何错误。
    """
    geom = spec.worldbody.add_geom()
    geom.name = "floor"
    geom.type = mujoco.mjtGeom.mjGEOM_PLANE
    geom.size = [1.0, 1.0, 0.05]
    geom.rgba = [0.56, 0.58, 0.62, 1.0]
    geom.contype = OBJECT_CONTYPE
    geom.conaffinity = OBJECT_CONAFFINITY
    geom.friction = [1.0, 0.005, 0.0001]


def add_cube(spec, pos=(0.20, 0.0, 0.015), half=0.015, mass=0.030,
             friction=(1.0, 0.01, 0.001)):
    """被抓的立方体。freejoint 让它有 6 个自由度，能被夹起来、也能掉下去。"""
    body = spec.worldbody.add_body()
    body.name = "cube"
    body.pos = list(pos)
    body.add_freejoint()

    geom = body.add_geom()
    geom.name = "cube_geom"
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [half, half, half]
    geom.rgba = [0.20, 0.65, 0.85, 1.0]
    geom.contype = OBJECT_CONTYPE
    geom.conaffinity = OBJECT_CONAFFINITY
    geom.friction = list(friction)
    geom.mass = mass
    # 接触刚度：太软物体像橡皮（陷进指尖），太硬容易抖
    geom.solref = [0.004, 0.9]


def add_lights(spec):
    """
    加光。灯是"看"的需求，不是"算"的需求，所以放代码里，不写进 so101.xml。
    没有灯的话离屏渲染出来接近全黑（只剩 MuJoCo 那个很弱的头灯）。
    """
    key = spec.worldbody.add_light()
    key.pos = [0.45, -0.35, 0.75]
    key.dir = [-0.5, 0.4, -0.75]
    key.diffuse = [0.95, 0.95, 0.95]
    key.specular = [0.25, 0.25, 0.25]
    key.castshadow = True

    fill = spec.worldbody.add_light()
    fill.pos = [-0.55, 0.45, 0.45]
    fill.dir = [0.6, -0.5, -0.45]
    fill.diffuse = [0.35, 0.38, 0.42]
    fill.specular = [0.0, 0.0, 0.0]
    fill.castshadow = False
    fill.intensity = 0.8


def build_scene(gravity_on=True, with_gripper=True, with_cube=False):
    """把机器人、夹爪、（可选的）地面和立方体拼成一个 spec。"""
    spec = mujoco.MjSpec.from_file(MODEL_PATH)
    spec.option.gravity = [0.0, 0.0, -9.81] if gravity_on else [0.0, 0.0, 0.0]

    if with_gripper:
        add_gripper(spec)
    if with_cube:
        add_floor(spec)
        add_cube(spec)

    return spec


def compile_scene(gravity_on=True, with_gripper=True, with_cube=False):
    """常用的那一种：直接给编译好的 MjModel + MjData。"""
    spec = build_scene(gravity_on=gravity_on, with_gripper=with_gripper,
                       with_cube=with_cube)
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


def addresses(model, names):
    """按名字取 qpos 下标。找不到就返回 None，方便判断有没有夹爪。"""
    out = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        out.append(None if jid < 0 else model.jnt_qposadr[jid])
    return out


def self_test():
    """最省事的自检：模型能不能编译、维度对不对、夹爪能不能开合。"""
    model, data = compile_scene()

    print(f"nq={model.nq}  nu={model.nu}  neq={model.neq}  "
          f"gravity={model.opt.gravity}")
    print("关节：", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    for i in range(model.njnt)])
    print("执行器：", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                      for i in range(model.nu)])

    qadr = addresses(model, ARM_JOINTS + ["j6_finger_l", "j6_finger_r"])
    gid = model.actuator("a_gripper").id

    # 只动夹爪：舵臂保持在零位，看两指间距能不能跟上指令
    # 注意两指的 q 是「同号等值」不是「一正一负」：轴向本身相反（+y / -y），
    # 所以 q_l = q_r = 0.01 时，左指往外 +10 mm、右指往外 -10 mm，间距 20 mm。
    # 因此 间距 = q_l + q_r，而 同步误差 = |q_l - q_r|。
    print(f"\n{'指令(m)':>10}{'左指(m)':>10}{'右指(m)':>10}"
          f"{'实际间距(mm)':>14}{'同步误差(mm)':>14}")
    for cmd in (0.0, 0.005, 0.010, 0.015, 0.020):
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.ctrl[gid] = cmd
        for _ in range(1500):
            mujoco.mj_step(model, data)
        ql, qr = data.qpos[qadr[5]], data.qpos[qadr[6]]
        print(f"{cmd:>10.3f}{ql:>10.4f}{qr:>10.4f}"
              f"{(ql + qr) * 1000:>14.2f}{abs(ql - qr) * 1000:>14.3f}")

    print("\n间距 = 左指 + 右指（两指内表面），4 cm 是最大开口。")
    print("同步误差接近 0 说明 equality 把两指锁住了；偏大就说明约束太软。")


if __name__ == "__main__":
    self_test()
