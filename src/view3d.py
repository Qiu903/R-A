# -*- coding: utf-8 -*-
"""
3D 可视化 · 把 models/so101.xml 真正渲染出来
================================================================

前面两个脚本都只吐数字和曲线：
    level1_fk.py    matplotlib 画的是"数学骨架"（几根线段 + 关节黑点）
    level2_mujoco.py 只有角度曲线，MuJoCo 的渲染器一次都没调过

这个脚本补上「看得见」这一块。两种用法：

    python src/view3d.py                 # 弹实时窗口，鼠标左键拖视角、滚轮缩放
    python src/view3d.py --record        # 无窗口，离屏渲染成视频文件

两种演示动作（--mode）：
    wave   五个关节各走不同频率的正弦，对应 level2 的跟随测试（看臂整体动起来）
    reach  让 TCP 在空中画圆，每一帧现算逆解，对应 level1 的逆解（看解出来的姿态连不连贯）

为什么要单独做一个脚本，而不是塞进 level2：
    离屏渲染和 matplotlib 是两套图形栈，混在一起会让 level2 变成"什么都干一点、什么都不干净"。
    第 3 级做抓取时，这个脚本会直接改成录抓取过程，所以它本身是个可复用的模块。

依赖说明：
    mp4 需要 imageio + imageio-ffmpeg；没装就自动退回 gif（PIL 合成，零额外依赖）。
    gif 体积大、颜色少，适合快速看；正式报告和 60 秒演示视频用 mp4。
"""

import os
import sys
import time
import argparse
import numpy as np

import mujoco
import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

MODEL_PATH = os.path.join(ROOT, "models", "so101.xml")
OUT_DIR = os.path.join(ROOT, "outputs")

JOINTS = ["j1_pan", "j2_lift", "j3_elbow", "j4_wrist_flex", "j5_wrist_roll"]

import grasp_scene  # noqa: E402  （放在常量后面，避免和 ROOT 定义抢位置）


# ============================================================
# 一、模型加载
# ============================================================
def load_model(kind="arm"):
    """
    kind="arm"   第 2 级的模型：5 关节、关重力、没有夹爪
    kind="grasp" 第 3 级的模型：多一对夹爪、开重力
    """
    # 用 MjSpec 而不是 from_xml_path：XML 里只描述机器人本体，灯是"看"的需求、
    # 不是"算"的需求，所以放在脚本里加，不污染物理模型（也不影响任何物理量）。
    if kind == "grasp":
        spec = grasp_scene.build_scene(gravity_on=True, with_gripper=True)
    else:
        spec = mujoco.MjSpec.from_file(MODEL_PATH)

    grasp_scene.add_lights(spec)

    model = spec.compile()
    # 离屏渲染默认只给 640x480 的缓冲，录制想用更大分辨率就会直接报错。
    # 在创建 Renderer 之前把缓冲抬高，就不用往 XML 里塞 <visual> 配置。
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1280)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 960)

    data = mujoco.MjData(model)
    # 每个关节在 qpos（关节角数组）里的下标，后面按名字取值用
    adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
           for n in JOINTS]
    return model, data, adr


def make_camera(model, distance=0.46):
    """
    自由相机。默认相机是"贴着原点看"，而这个模型横向伸出去 0.4 米，
    不调的话画面里基本只有一根黑柱子（第一次跑出来平均亮度 2/255 就是这原因）。

    distance 按模式给：reach 只在原地画小圆，可以贴近看细节；
    wave 会整条臂转来转去，贴太近末端会扫出画面。
    """
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.16, 0.0, 0.15]   # 看向工作空间中心，不是原点
    cam.distance = distance
    cam.azimuth = 132                    # 从左前方看，三段连杆不会叠成一条线
    cam.elevation = -18
    return cam


def cam_distance(mode):
    return 0.62 if mode == "wave" else 0.46


# ============================================================
# 二、两种演示动作
# ============================================================
class WaveMotion:
    """
    五个关节各走正弦，频率不同，所以姿态每秒都在变。

    加 bias 是为了"看得见"而不是"跑得对"：第 2 级关了重力和碰撞，
    J2 中点 0° 时手臂正好水平，正弦摆到正半周就直接插进地面里，
    末端夹爪整个消失在地板下面。把基准姿态抬起来、肘部弯一点，
    整条臂才在桌面上方活动。
    """

    def __init__(self, model, data, adr):
        self.amp = np.array([0.30, 0.30, 0.30, 0.25, 0.60])
        self.freq = np.array([0.30, 0.40, 0.50, 0.60, 0.30])
        self.bias = np.array([0.0, -0.75, 0.55, 0.0, 0.0])

    def ctrl(self, t):
        return self.bias + self.amp * np.sin(2.0 * np.pi * self.freq * t)

    def reset(self, t=0.0):
        """给一个"第一步就该在的位置"，避免开局从零位猛跳。"""
        return self.ctrl(t)


class ReachMotion:
    """
    让 TCP 沿一段圆弧走，每一帧现算逆解。

    逆解有 4 个分支（肘上/肘下 × 正对/背对），同一个目标点有 2~4 组合法解。
    这里的规矩是：**开局选一次分支，之后锁死**。
    每帧重新挑分支的写法试过，结果两个分支的评分咬得很近，来回切，
    实际轨迹变成满天乱飞的一团毛线 —— 位置每一帧都对，但关节来回抽 180°。
    """

    def __init__(self, model, data, adr):
        from level1_fk import solve_ik, forward_kinematics, JOINT_LIMITS
        self.solve_ik = solve_ik
        self.fk = forward_kinematics
        self.limits = JOINT_LIMITS
        self.q = np.zeros(4)
        self.tag = None                                # None = 还没选分支
        self.target = np.array([0.22, 0.0, 0.26])
        # 每个控制周期最多转 0.02 rad（= 20 rad/s，舵机级别的速度上限）。
        # 逆解算得再快，执行器也跟不上，加上这个限制才像真机。
        self.max_step = 0.02

    def _target(self, t):
        # 在 x-z 平面画圆，y 方向小幅摆一下，避免整段轨迹退化成一条直线。
        # 圆心定在 0.26 m 高：再低这个臂就只能把大臂垂下去够，画面里会穿过地面。
        ang = 2.0 * np.pi * 0.15 * t
        return np.array([0.22 + 0.040 * np.cos(ang),
                         0.030 * np.sin(ang),
                         0.26 + 0.040 * np.sin(ang)])

    def reset(self, t=0.0):
        """定位到 t 时刻的解，不走限速。录制/viewer 的初始化用它。"""
        self.target = self._target(t)
        q, tag = self._solve(self.target)
        if q is not None:
            self.q, self.tag = q[:4].copy(), tag
        return self._pack(self.q)

    def ctrl(self, t):
        self.target = self._target(t)

        q, tag = self._solve(self.target)
        if q is None:
            return self._pack(self.q)          # 这一步无解，保持姿态

        # 限速逼近，不是"跳变就拒绝"。
        # 早先写的是"变化超过 0.7 rad 就沿用上一帧"，结果第一帧从零位跳到解，
        # 直接被拒；下一帧还是拿零位去比，永远被拒 —— 手臂一动不动。
        # 限速的好处是：偶发的分支切换也只是平滑地转过去，不会卡死也不会抽。
        delta = q[:4] - self.q
        biggest = np.max(np.abs(delta))
        if biggest > self.max_step:
            self.q = self.q + delta * (self.max_step / biggest)
        else:
            self.q = q[:4].copy()

        self.tag = tag
        return self._pack(self.q)

    def _branch(self, target, tag):
        facing, elbow = tag.split("/")
        q = self.solve_ik(target, elbow=elbow, facing=facing)
        if q is None:
            return None
        if not (np.all(q >= self.limits[:, 0]) and np.all(q <= self.limits[:, 1])):
            return None
        return q

    def _solve(self, target):
        """
        已经锁定分支就只解那一个；只有锁定失败（换了分支）时才重新选。

        选法不按分支名字，按"肘关节实际有多高"：
        elbow="up" 在数学上并不等于画面上肘朝上 —— 这个臂底座到肩只有 95 mm、
        臂展 361 mm，够前方低处时唯一可行解往往是大臂整个垂下去，画面里直接穿地。
        与其猜命名，不如算一次正解，量一下肘的 z 坐标。
        """
        if self.tag is not None:
            q = self._branch(target, self.tag)
            if q is not None:
                return q, self.tag

        best = None
        for facing in ("front", "back"):
            for elbow in ("up", "down"):
                tag = f"{facing}/{elbow}"
                q = self._branch(target, tag)
                if q is None:
                    continue
                score = float(self.fk(q)[3][2, 3])      # frames[3] 原点 = 肘关节位置
                if best is None or score > best[0]:
                    best = (score, q, tag)

        if best is None:
            return None, None
        return best[1], best[2]

    @staticmethod
    def _pack(q4):
        """逆解只有 4 个关节（J5 不影响位置），补一个 0 给 J5 凑成 5 维控制量。"""
        return np.array([q4[0], q4[1], q4[2], q4[3], 0.0])


def make_motion(mode, model, data, adr):
    return ReachMotion(model, data, adr) if mode == "reach" else WaveMotion(model, data, adr)


def gripper_joint_ids(model):
    """夹爪两个手指的关节 id；模型没夹爪就返回空列表。"""
    ids = []
    for name in ("j6_finger_l", "j6_finger_r"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            ids.append(jid)
    return ids


def set_ctrl(model, data, arm_ctrl, t):
    """
    写控制量。臂的 5 个关节来自 motion，夹爪则自己开合。

    夹爪这里只是"让画面动起来"的展示：0.4~1.8 cm 来回，对应两指间距 8~36 mm。
    真正的抓取逻辑（什么时候合、合到多大、判定是否夹住）要到步骤 4 才写。
    """
    data.ctrl[:len(arm_ctrl)] = arm_ctrl
    gids = gripper_joint_ids(model)
    if gids:
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_gripper")] = \
            0.004 + 0.007 * (1.0 - np.cos(2.0 * np.pi * 0.3 * t))


def init_pose(model, data, adr, arm_ctrl):
    """把臂摆到第一帧姿态，夹爪半开。避免视频开头从零位猛地抽过去。"""
    data.qpos[adr] = arm_ctrl
    for jid in gripper_joint_ids(model):
        data.qpos[model.jnt_qposadr[jid]] = 0.008
    mujoco.mj_forward(model, data)


# ============================================================
# 三、画在场景里的额外东西：地面 + TCP 轨迹 + 目标点
# ============================================================
# 这些不属于物理模型（不参与碰撞），只是给眼睛看的参照物。
# 直接往场景里塞 geom 而不是改 XML，好处是 models/so101.xml 保持"只描述机器人"。
def draw_extras(scn, trail, marker=None):
    """scn 是 mjvScene（渲染时的场景描述），往里面追加几何体。"""

    # 地面：MuJoCo 的 plane 是无限大平面，size 前两个数是视觉上的纹理格尺寸
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_PLANE,
            np.array([0.35, 0.35, 0.0]),
            np.array([0.13, 0.0, 0.0]),
            np.eye(3).reshape(-1).astype(np.float64),
            np.array([0.42, 0.44, 0.48, 1.0]),
        )
        scn.ngeom += 1

    # TCP 走过的轨迹：一段段胶囊连起来，比散点连贯得多
    for i in range(len(trail) - 1):
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_connector(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.003,                      # 半价径，太粗会把轨迹糊成一团
            trail[i], trail[i + 1],
        )
        scn.geoms[scn.ngeom].rgba[:] = [0.10, 0.85, 0.55, 1.0]
        scn.ngeom += 1

    # 逆解模式的目标点：红色小球，能一眼看出 TCP 有没有跟上
    if marker is not None and scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.010, 0.010, 0.010]),
            np.asarray(marker, dtype=np.float64),
            np.eye(3).reshape(-1).astype(np.float64),
            np.array([0.90, 0.20, 0.20, 1.0]),
        )
        scn.ngeom += 1


# ============================================================
# 四、实时窗口
# ============================================================
def run_viewer(mode, seconds=None, kind="arm"):
    model, data, adr = load_model(kind)
    motion = make_motion(mode, model, data, adr)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

    # 从第一帧的姿态起步，否则 0 位到目标姿态是一次阶跃，画面开头会抽一下。
    # 注意用 reset 而不是 ctrl：ctrl 受限速约束，只会挪 0.02 rad，
    # 初始化时拿到的仍接近零位，于是视频第一秒全是"手臂还没到位"的空白。
    init_pose(model, data, adr, motion.reset(0.0))

    trail = [data.site_xpos[sid].copy()]
    sim_t = 0.0
    step = model.opt.timestep
    wall0 = time.time()

    print("窗口已打开：左键拖动转视角，滚轮缩放，右键平移。关窗口即退出。")

    # launch_passive：MuJoCo 自己开窗口和 GL 上下文，物理步进仍由我们控制
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.16, 0.0, 0.15]
        viewer.cam.distance = cam_distance(mode)
        viewer.cam.azimuth = 132
        viewer.cam.elevation = -18

        while viewer.is_running():
            for _ in range(10):                       # 每 10 步（10ms）刷一次画面
                set_ctrl(model, data, motion.ctrl(sim_t), sim_t)
                mujoco.mj_step(model, data)
                sim_t += step

            trail.append(data.site_xpos[sid].copy())
            if len(trail) > 400:                      # 轨迹只留最近一段，否则画面变糊
                trail.pop(0)

            # user_scn 是留给用户自绘的临时场景，每帧清空重画
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                marker = getattr(motion, "target", None) if mode == "reach" else None
                # 轨迹只在 reach 模式画：wave 是五个关节各摆各的，末端走出来的是
                # 一团没意义的李萨如曲线，还会扫满整个画面。
                draw_extras(viewer.user_scn, trail if mode == "reach" else [], marker)
            viewer.sync()

            # 让仿真时间跟真实时间对齐，否则五轴 1ms 步长会跑成快进
            lag = (time.time() - wall0) - sim_t
            if lag < 0:
                time.sleep(-lag)

            if seconds is not None and sim_t > seconds:
                break


# ============================================================
# 五、离屏录制
# ============================================================
def save_frames(frames, fps, out_base):
    """优先 mp4，没装 imageio 就退 gif。返回实际写出的路径。"""
    try:
        import imageio.v2 as imageio
        path = out_base + ".mp4"
        imageio.mimsave(path, frames, fps=fps, quality=8, macro_block_size=None)
        return path
    except ImportError:
        from PIL import Image
        path = out_base + ".gif"
        imgs = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=192)
                for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0, optimize=True)
        print("（未装 imageio，已退回 gif。要 mp4：pip install imageio imageio-ffmpeg）")
        return path


def run_record(mode, seconds, fps, width, height, kind="arm"):
    model, data, adr = load_model(kind)
    motion = make_motion(mode, model, data, adr)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    cam = make_camera(model, cam_distance(mode))

    # Renderer 开一块离屏缓冲，不需要窗口；Windows 上走系统 GL，能纯 CPU 跑
    renderer = mujoco.Renderer(model, height, width)
    substep = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    n = int(seconds / model.opt.timestep)

    init_pose(model, data, adr, motion.reset(0.0))

    frames = []
    trail = [data.site_xpos[sid].copy()]
    sim_t = 0.0
    for k in range(n):
        set_ctrl(model, data, motion.ctrl(sim_t), sim_t)
        mujoco.mj_step(model, data)
        sim_t += model.opt.timestep

        if k % substep == 0:
            trail.append(data.site_xpos[sid].copy())
            if len(trail) > int(fps * 6):
                trail.pop(0)
            renderer.update_scene(data, camera=cam)
            marker = getattr(motion, "target", None) if mode == "reach" else None
            # update_scene 之后，renderer.scene 就是这一帧要渲染的场景
            draw_extras(renderer.scene, trail if mode == "reach" else [], marker)
            frames.append(renderer.render())
    renderer.close()

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = "level3" if kind == "grasp" else "level2"
    out = os.path.join(OUT_DIR, f"{tag}_view3d_{mode}")
    path = save_frames(frames, fps, out)
    print(f"共 {len(frames)} 帧，已保存：{path}")
    return path


def run_still(mode, out_name=None, kind="arm"):
    """出一张静态图，报告里要放截图时用。文件名带模式，免得两种模式互相覆盖。"""
    tag = "level3" if kind == "grasp" else "level2"
    out_name = out_name or f"{tag}_view3d_{mode}.png"
    model, data, adr = load_model(kind)
    motion = make_motion(mode, model, data, adr)
    cam = make_camera(model, cam_distance(mode))
    renderer = mujoco.Renderer(model, 640, 880)

    # 先空跑一小段，让手臂离开零位，并把执行器的跟踪滞后甩掉
    init_pose(model, data, adr, motion.reset(0.0))
    for k in range(900):
        set_ctrl(model, data, motion.ctrl(k * model.opt.timestep),
                 k * model.opt.timestep)
        mujoco.mj_step(model, data)
    renderer.update_scene(data, camera=cam)
    draw_extras(renderer.scene, [], getattr(motion, "target", None))
    img = renderer.render()
    renderer.close()

    from PIL import Image
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, out_name)
    Image.fromarray(img).save(path)
    print(f"静态图已保存：{path}")
    return path


def main():
    ap = argparse.ArgumentParser(description="SO-101 仿真 3D 可视化")
    ap.add_argument("--mode", choices=["wave", "reach"], default="wave",
                    help="wave=关节正弦摆动；reach=逆解让 TCP 画圆")
    ap.add_argument("--model", choices=["arm", "grasp"], default="arm",
                    help="arm=第 2 级模型（5 关节、关重力）；grasp=第 3 级模型（带夹爪、开重力）")
    ap.add_argument("--record", action="store_true", help="离屏录制，不弹窗口")
    ap.add_argument("--still", action="store_true", help="只出一张静态截图")
    # 默认 None = 实时窗口一直开着，直到你自己关掉。
    # 只有录制模式才需要一个时长（不给就录 6 秒）。
    ap.add_argument("--seconds", type=float, default=None,
                    help="运行时长（秒）。实时模式不给就一直跑；录制模式不给则录 6 秒")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    if args.still:
        run_still(args.mode, kind=args.model)
    elif args.record:
        seconds = 6.0 if args.seconds is None else args.seconds
        run_record(args.mode, seconds, args.fps, args.width, args.height,
                   kind=args.model)
    else:
        run_viewer(args.mode, args.seconds, kind=args.model)


if __name__ == "__main__":
    main()
