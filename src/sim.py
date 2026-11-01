# -*- coding: utf-8 -*-
"""
机械臂抓取仿真 · 完整闭环
================================================================

一个能直接跑、能看、能判定成败的机械臂仿真：
    桌子上的 3 cm 立方体 → 机械臂从上方接近 → 夹住 → 抬起 → 判定有没有抓起来

和第 3 级前面几步的关系：
    grasp_scene.py  提供场景（臂 + 夹爪 + 地面 + 立方体）
    grasp_ik.py     提供"夹爪朝下"的逆解（这是能抓住的前提）
    本文件          把它们串成时间序列，变成一个动作

为什么抓取必须让夹爪朝下：
    这个臂底座只有 95 mm，腕部俯仰如果锁在 0，够桌面上 30 mm 的物体时
    夹爪只能斜 47° 杵过去，指尖一端先磕到桌面。所以规划时把腕角固定成
    1.0 rad（约 57°），让 TCP 的轴向 z 分量降到 -1.0，也就是垂直向下。

用法：
    python src/sim.py                  # 弹实时窗口看整个过程
    python src/sim.py --record         # 无窗口，录成 gif/mp4
    python src/sim.py --seconds 8      # 改时长
"""

import os
import sys
import time
import argparse
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import grasp_scene  # noqa: E402
import grasp_ik  # noqa: E402
from view3d import make_camera, draw_extras, save_frames  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(ROOT, "outputs")

ARM_JOINTS = grasp_scene.ARM_JOINTS
CUBE_POS = (0.20, 0.0, 0.015)

# 腕角固定 1.0 rad：这是让夹爪垂直朝下的那个值，也是能同时够到
# "抓取高度"和"抬起高度"的唯一解区间（再高就没解了，实测 0.09 m 以上无解）
Q4_FIXED = 1.0
GRASP_POS = (0.20, 0.0, 0.012)     # TCP 略低于物体中心，让指尖罩住立方体上半部
PRE_POS = (0.20, 0.0, 0.075)       # 抓取点正上方 5.5 cm
LIFT_POS = PRE_POS                  # 抬起来就回到预抓取高度

OPEN = 0.020                        # 夹爪张开：两指间距 40 mm
CLOSED = 0.004                      # 合拢指令：间距 8 mm，物体 30 mm，靠力限兜住

# 各阶段时长（秒）。累计成时间轴：
#   0.0 起手 → 1.8 到预抓取 → 2.8 到抓取点 → 3.7 合爪完成
#   → 4.2 停顿结束 → 5.4 抬起完成 → 6.9 观察结束
T_APPROACH, T_DESCEND, T_CLOSE, T_HOLD, T_LIFT, T_SETTLE = 1.8, 1.0, 0.9, 0.5, 1.2, 1.5
T1 = T_APPROACH
T2 = T1 + T_DESCEND
T3 = T2 + T_CLOSE
T4 = T3 + T_HOLD
T5 = T4 + T_LIFT
T_END = T5 + T_SETTLE

# 判定用"相对初始位置抬了多少"，不用绝对高度：
# 预抓取高度会降级，绝对高度阈值会跟着失效。
SUCCESS_LIFT = 0.020                # 抬高 2 cm 以上算抓起来


def smooth(u):
    """三次平滑：两端速度为 0。线性插值会让关节急停急起，物体容易被甩掉。"""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def lerp(a, b, s):
    return np.asarray(a) + (np.asarray(b) - np.asarray(a)) * s


class GraspDemo:
    """
    一次完整抓取。位置、质量、摩擦都可以从外面传进来 ——
    成功率统计就是靠"换一组参数重建一个 demo"来做的。

    注意每次重建都要重新 compile + 重新规划逆解，所以别在实时循环里反复建。
    """

    def __init__(self, cube_xy=(0.20, 0.0), place_offset=(0.0, 0.0),
                 cube_mass=0.030, cube_friction=1.0, verbose=True):
        """
        cube_xy      规划时"认为"物体在哪 —— 逆解按这个位置算
        place_offset 物体实际被放的位置相对规划的偏移

        分开这两个是有意的：真机上你永远不会知道物体精确在哪（视觉有误差），
        规划用的是估计值，执行时物体却在别处。合在一起的话测出来的成功率
        会虚高，因为"偏差"被当成"已知信息"吃掉了。
        """
        self.cube_xy = (float(cube_xy[0]), float(cube_xy[1]))
        self.place_xy = (self.cube_xy[0] + float(place_offset[0]),
                         self.cube_xy[1] + float(place_offset[1]))
        self.cube_half = 0.015
        # 底面贴地再抬 0.2 mm，避免初始就嵌进地面（会产生一个瞬间的弹力）
        self.cube_z = self.cube_half + 0.0002
        self.verbose = verbose

        spec = grasp_scene.build_scene(with_gripper=True, with_cube=False)
        grasp_scene.add_floor(spec)
        grasp_scene.add_cube(spec, pos=(self.place_xy[0], self.place_xy[1], self.cube_z),
                             half=self.cube_half, mass=cube_mass,
                             friction=(cube_friction, 0.01, 0.001))
        grasp_scene.add_lights(spec)

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.qadr = grasp_scene.addresses(self.model, ARM_JOINTS)

        cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_bid = cube_body
        # freejoint 的 qpos 是 7 个数：前 3 个是位置，后 4 个是四元数
        cube_jid = self.model.body_jntadr[cube_body]
        self.cube_qadr = self.model.jnt_qposadr[cube_jid]

        self.gid = self.model.actuator("a_gripper").id
        self.tcp_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

        self.plan()
        self.reset()

    def plan(self):
        """
        用逆解把三个关键姿态算出来。算不出来就直接报错 ——
        抓取点无解意味着物体放的位置根本够不着，那是场景问题，不是控制问题。
        """
        x, y = self.cube_xy
        # TCP 略低于物体中心，让指尖罩住立方体上半部
        grasp_pos = (x, y, self.cube_z - 0.003)

        q_grasp = grasp_ik.solve_q123(grasp_pos, Q4_FIXED)
        if q_grasp is None:
            raise RuntimeError(f"抓取点 {np.round(grasp_pos, 3)} 在腕角 {Q4_FIXED} 下无解")

        # 预抓取高度按从高到低试。腕角锁死后工作空间会收窄：
        # 物体放得近（x < 0.20）时 7.5 cm 的预抓取点直接无解，
        # 但降两厘米就又能解出来 —— 与其报错，不如退一步用更低的接近高度。
        q_pre, pre_pos = None, None
        for z in (PRE_POS[2], 0.065, 0.055, 0.045):
            q_try = grasp_ik.solve_q123((x, y, z), Q4_FIXED, seed=q_grasp)
            if q_try is not None:
                q_pre, pre_pos = q_try, (x, y, z)
                break
        if q_pre is None:
            raise RuntimeError("预抓取点在任何高度都无解")

        # 抬起高度就回到预抓取高度，原路返回
        q_lift = q_pre
        self.pre_z = pre_pos[2]
        if self.verbose and pre_pos[2] < PRE_POS[2]:
            print(f"  （预抓取高度降级到 {pre_pos[2] * 1000:.0f} mm）")

        # 注意维度：逆解只给到 q4，而臂的执行器有 5 个（J5 也在里面）。
        # J5 是腕部自转，不影响 TCP 位置，这里固定为 0，补一维凑齐 5 个。
        self.q_start = np.array([0.0, -0.70, 0.50, 0.30, 0.0])
        self.q_pre = np.array([q_pre[0], q_pre[1], q_pre[2], Q4_FIXED, 0.0])
        self.q_grasp = np.array([q_grasp[0], q_grasp[1], q_grasp[2], Q4_FIXED, 0.0])
        self.q_lift = np.array([q_lift[0], q_lift[1], q_lift[2], Q4_FIXED, 0.0])

        # 自检：关键姿态的 TCP 轴向必须朝下，否则不是"从上往下抓"
        if self.verbose:
            for name, q in (("预抓取", self.q_pre), ("抓取", self.q_grasp)):
                _, axis = grasp_ik.fk(q[3], q[:3])
                print(f"  {name}姿态 q=({np.degrees(q[:4]).round(1)})  "
                      f"夹爪轴向 z={axis[2]:+.3f}")

    def reset(self):
        d = self.data
        # mj_resetData 会把 qpos/qvel/时间全部归零，比手写清零可靠。
        # 之前漏了 time，导致录制时 apply(data.time) 拿到的是上一次跑完的 6.9s，
        # 整套动作被直接跳到最后一帧，视频里臂一动不动。
        mujoco.mj_resetData(self.model, d)
        d.qpos[self.qadr] = self.q_start
        d.qpos[self.cube_qadr:self.cube_qadr + 3] = [self.place_xy[0], self.place_xy[1],
                                                     self.cube_z]
        d.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        for name in ("j6_finger_l", "j6_finger_r"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            d.qpos[self.model.jnt_qposadr[jid]] = OPEN
        d.ctrl[:] = 0.0
        d.ctrl[self.gid] = OPEN
        mujoco.mj_forward(self.model, d)

    # ---------- 时间 → 控制量 ----------
    def arm_at(self, t):
        if t < T1:
            return lerp(self.q_start, self.q_pre, smooth(t / T1))
        if t < T2:
            return lerp(self.q_pre, self.q_grasp, smooth((t - T1) / T_DESCEND))
        if t < T4:
            return self.q_grasp
        if t < T5:
            return lerp(self.q_grasp, self.q_lift, smooth((t - T4) / T_LIFT))
        return self.q_lift

    def grip_at(self, t):
        if t < T2:
            return OPEN
        if t < T3:
            return lerp(OPEN, CLOSED, smooth((t - T2) / T_CLOSE))
        return CLOSED

    def stage_name(self, t):
        for edge, label in ((T1, "接近"), (T2, "下降"), (T3, "合爪"),
                            (T4, "停顿"), (T5, "抬起"), (T_END, "观察")):
            if t < edge:
                return label
        return "结束"

    def apply(self, t):
        arm = self.arm_at(t)
        d = self.data
        d.ctrl[:len(self.qadr)] = arm
        d.ctrl[self.gid] = self.grip_at(t)

    # ---------- 观测 ----------
    def cube_pos(self):
        return self.data.xpos[self.cube_bid].copy()

    def tcp_pos(self):
        return self.data.site_xpos[self.tcp_sid].copy()

    def finger_gap(self):
        jl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "j6_finger_l")
        jr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "j6_finger_r")
        return (self.data.qpos[self.model.jnt_qposadr[jl]] +
                self.data.qpos[self.model.jnt_qposadr[jr]])


# ============================================================
# 跑一段仿真并记录
# ============================================================
def simulate(demo, seconds=None):
    model, data = demo.model, demo.data
    seconds = T_END if seconds is None else seconds
    n = int(seconds / model.opt.timestep)

    # cube_x / cube_y 也记下来：判定"是被夹飞了还是根本没夹住"要靠水平位移
    log = {"t": [], "cube_z": [], "cube_x": [], "cube_y": [],
           "tcp_z": [], "gap": [], "stage": []}
    for _ in range(n):
        demo.apply(data.time)
        mujoco.mj_step(model, data)
        if int(data.time / model.opt.timestep) % 5 == 0:
            c = demo.cube_pos()
            log["t"].append(data.time)
            log["cube_z"].append(c[2])
            log["cube_x"].append(c[0])
            log["cube_y"].append(c[1])
            log["tcp_z"].append(demo.tcp_pos()[2])
            log["gap"].append(demo.finger_gap())
            log["stage"].append(demo.stage_name(data.time))

    for k in log:
        log[k] = np.array(log[k])
    return log


def report(demo, log):
    cube_z = log["cube_z"]
    z0 = cube_z[0]
    lifted = cube_z[-1] - z0
    ok = lifted > SUCCESS_LIFT

    print("\n--- 抓取结果 ---")
    print(f"立方体初始高度 {z0 * 1000:.0f} mm，最终 {cube_z[-1] * 1000:.1f} mm，"
          f"抬高 {lifted * 1000:.1f} mm（判定线 {SUCCESS_LIFT * 1000:.0f} mm）")
    print(f"过程最高 {cube_z.max() * 1000:.1f} mm，  "
          f"最低 {cube_z.min() * 1000:.1f} mm")
    print(f"合爪时最小间距 {log['gap'].min() * 1000:.1f} mm（立方体宽 30 mm）")
    # 不用 emoji：Windows 控制台默认 GBK，打印 ✅ 会直接抛 UnicodeEncodeError
    print(f"判定：{'抓起来了' if ok else '没抓起来'}")
    if not ok:
        print("  排查顺序：1) 指尖有没有对准物体（看 tcp 与 cube 的 xy 偏差）"
              "2) 夹持力够不够 3) 摩擦/接触刚度是否太软")
    return ok


def plot(log, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("机械臂抓取仿真 · 立方体高度与夹爪间距", fontsize=13)

    ax = axes[0]
    ax.plot(log["t"], log["cube_z"] * 1000, lw=2, color="#2e86de", label="立方体")
    ax.plot(log["t"], log["tcp_z"] * 1000, lw=1.4, ls="--", color="#c0392b", label="TCP")
    ax.axhline((log["cube_z"][0] + SUCCESS_LIFT) * 1000, color="gray", ls=":",
               lw=1, label=f"成功线（+{SUCCESS_LIFT * 1000:.0f} mm）")
    for x in (T1, T2, T3, T4, T5):
        ax.axvline(x, color="0.8", lw=0.8)
    ax.set_xlabel("时间 (s)", fontsize=9)
    ax.set_ylabel("高度 (mm)", fontsize=9)
    ax.set_title("抬高轨迹", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(log["t"], log["gap"] * 1000, lw=2, color="#10ac84")
    ax.axhline(30, color="#c0392b", ls=":", lw=1, label="立方体宽度 30 mm")
    for x in (T1, T2, T3, T4, T5):
        ax.axvline(x, color="0.8", lw=0.8)
    ax.set_xlabel("时间 (s)", fontsize=9)
    ax.set_ylabel("两指间距 (mm)", fontsize=9)
    ax.set_title("合爪过程", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"曲线已保存：{path}")


# ============================================================
# 实时窗口 / 录制
# ============================================================
def run_viewer(demo, seconds=None):
    model, data = demo.model, demo.data
    print("窗口已打开：左键拖动转视角，滚轮缩放。动作结束后自动重来，关窗口退出。")
    start = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.16, 0.0, 0.10]
        viewer.cam.distance = 0.55
        viewer.cam.azimuth = 118
        viewer.cam.elevation = -20

        wall0 = time.time()
        trail = []
        while viewer.is_running():
            for _ in range(10):
                demo.apply(data.time)
                mujoco.mj_step(model, data)
            trail.append(demo.tcp_pos())
            if len(trail) > 500:
                trail.pop(0)

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw_extras(viewer.user_scn, trail)
            viewer.sync()

            lag = (time.time() - wall0) - data.time
            if lag < 0:
                time.sleep(-lag)

            if data.time > T_END:
                demo.reset()
                trail.clear()
                wall0 = time.time()

            # seconds 按真实时间算，不是仿真时间：演示时你可能中途在拖视角
            if seconds is not None and time.time() - start > seconds:
                break


def run_record(demo, seconds, fps, width, height):
    model, data = demo.model, demo.data
    renderer = mujoco.Renderer(model, height, width)
    cam = make_camera(model, 0.55)
    cam.azimuth = 118
    cam.elevation = -20
    cam.lookat[:] = [0.16, 0.0, 0.10]

    substep = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    n = int(seconds / model.opt.timestep)

    frames, trail = [], []
    for k in range(n):
        demo.apply(data.time)
        mujoco.mj_step(model, data)
        if k % substep == 0:
            trail.append(demo.tcp_pos())
            if len(trail) > fps * 3:
                trail.pop(0)
            renderer.update_scene(data, camera=cam)
            draw_extras(renderer.scene, trail)
            frames.append(renderer.render())
    renderer.close()

    os.makedirs(OUT_DIR, exist_ok=True)
    path = save_frames(frames, fps, os.path.join(OUT_DIR, "level3_grasp"))
    print(f"共 {len(frames)} 帧，视频已保存：{path}")
    return path


def main():
    ap = argparse.ArgumentParser(description="机械臂抓取仿真")
    ap.add_argument("--record", action="store_true", help="离屏录制，不弹窗口")
    ap.add_argument("--seconds", type=float, default=None,
                    help=f"运行时长；默认跑完整套动作 {T_END:.1f}s")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    print("规划关键姿态：")
    demo = GraspDemo()

    if args.record:
        seconds = T_END if args.seconds is None else args.seconds
        log = simulate(demo, seconds)
        report(demo, log)
        plot(log, os.path.join(OUT_DIR, "level3_grasp.png"))
        demo.reset()
        run_record(demo, seconds, args.fps, args.width, args.height)
    else:
        run_viewer(demo, args.seconds)


if __name__ == "__main__":
    main()
