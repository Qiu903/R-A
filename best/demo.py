# -*- coding: utf-8 -*-
"""
SO-101 高质量仿真 · 主程序
================================================================

把场景、控制、执行器模型串成一个能看、能录、能判定成败的东西。

用法：
    python best/demo.py                 实时窗口，循环播放
    python best/demo.py --record        离屏录制（1080p）
    python best/demo.py --still         渲一张静帧

和 src/sim.py 的区别：
    sim.py 是"把物理跑通"的最小闭环，box 几何 + 理想执行器。
    best 这里是"拿得出手"的版本：程序化圆角网格、PBR 材质、三点布光、
    非理想执行器（延迟/背隙/量化）。
    两者的运动学完全相同（同一份 so101.xml），所以位置数据可以互相引用。
"""

import os
import sys
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
from PIL import Image  # noqa: E402

import control  # noqa: E402
import postfx  # noqa: E402
import grasp_ik  # noqa: E402
import scene  # noqa: E402

OUT_DIR = os.path.join(ROOT, "outputs")

ARM_JOINTS = ["j1_pan", "j2_lift", "j3_elbow", "j4_wrist_flex", "j5_wrist_roll"]
Q4_FIXED = 1.0
OPEN_WIDTH = 0.020
CLOSED_WIDTH = 0.004
SUCCESS_LIFT = 0.020


class BestDemo:
    def __init__(self, object_xy=(0.20, 0.0), place_offset=(0.0, 0.0),
                 mass=0.030, friction=1.0, verbose=True):
        self.object_xy = object_xy
        self.place_xy = (object_xy[0] + place_offset[0],
                         object_xy[1] + place_offset[1])
        self.cube_z = 0.0152

        self.model, self.data = scene.compile_scene(
            object_pos=(self.place_xy[0], self.place_xy[1], self.cube_z),
            mass=mass, friction=friction)

        self.qadr = [self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ARM_JOINTS]
        self.gid = self.model.actuator("a_gripper").id
        self.tcp_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.cube_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_jid = self.model.body_jntadr[self.cube_bid]
        self.cube_qadr = self.model.jnt_qposadr[cube_jid]

        self.plan(verbose)
        self.servo = control.ServoModel(n_joint=5, dt=self.model.opt.timestep)
        self.reset()

    # ---------------- 规划 ----------------
    def plan(self, verbose=True):
        x, y = self.object_xy
        grasp_pos = (x, y, self.cube_z - 0.003)

        q_grasp = grasp_ik.solve_q123(grasp_pos, Q4_FIXED)
        if q_grasp is None:
            raise RuntimeError(f"抓取点 {np.round(grasp_pos, 3)} 无解")

        q_pre, pre_z = None, None
        for z in (0.075, 0.065, 0.055, 0.045):
            q_try = grasp_ik.solve_q123((x, y, z), Q4_FIXED, seed=q_grasp)
            if q_try is not None:
                q_pre, pre_z = q_try, z
                break
        if q_pre is None:
            raise RuntimeError("预抓取点无解")

        self.q_start = np.array([0.0, -0.70, 0.50, 0.30, 0.0])
        self.q_pre = np.array([q_pre[0], q_pre[1], q_pre[2], Q4_FIXED, 0.0])
        self.q_grasp = np.array([q_grasp[0], q_grasp[1], q_grasp[2], Q4_FIXED, 0.0])
        self.q_lift = self.q_pre.copy()

        self.seq = control.GraspSequencer(self.q_start, self.q_pre,
                                          self.q_grasp, self.q_lift,
                                          OPEN_WIDTH, CLOSED_WIDTH)

        if verbose:
            for name, q in (("预抓取", self.q_pre), ("抓取", self.q_grasp)):
                _, axis = grasp_ik.fk(q[3], q[:3])
                print(f"  {name} q=({np.degrees(q[:4]).round(1)}) "
                      f"夹爪轴向 z={axis[2]:+.3f}")
            print(f"  预抓取高度 {pre_z * 1000:.0f} mm，"
                  f"整套动作 {self.seq.total:.1f} s")

    # ---------------- 状态 ----------------
    def reset(self):
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[self.qadr] = self.q_start
        d.qpos[self.cube_qadr:self.cube_qadr + 3] = [self.place_xy[0],
                                                     self.place_xy[1], self.cube_z]
        d.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        for name in ("j6_finger_l", "j6_finger_r"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            d.qpos[self.model.jnt_qposadr[jid]] = OPEN_WIDTH
        d.ctrl[:] = 0.0
        d.ctrl[self.gid] = OPEN_WIDTH
        self.servo.reset(self.q_start)
        mujoco.mj_forward(self.model, d)

    def step(self):
        t = self.data.time
        arm = self.seq.arm(t)
        self.data.ctrl[:5] = self.servo(arm)
        self.data.ctrl[self.gid] = self.seq.grip(t)
        mujoco.mj_step(self.model, self.data)

    # ---------------- 观测 ----------------
    def cube_pos(self):
        return self.data.xpos[self.cube_bid].copy()

    def tcp_pos(self):
        return self.data.site_xpos[self.tcp_sid].copy()

    def success(self):
        return (self.cube_pos()[2] - self.cube_z) > SUCCESS_LIFT

    def run_to_end(self):
        z0 = self.cube_pos()[2]
        n = int(self.seq.total / self.model.opt.timestep)
        trail = []
        for k in range(n):
            self.step()
            if k % 10 == 0:
                trail.append((self.data.time, self.cube_pos()[2], self.tcp_pos()[2]))
        cube_z = self.cube_pos()[2]
        return {
            "lift": cube_z - z0,
            "ok": (cube_z - z0) > SUCCESS_LIFT,
            "trail": np.array(trail),
        }


# ---------------------------------------------------------------- 相机
def make_camera(model, aspect=16 / 9):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.19, 0.0, 0.085]
    cam.distance = 0.42
    cam.azimuth = 124
    cam.elevation = -14
    return cam


# ---------------------------------------------------------------- 渲染
def _render_pair(renderer, data, cam):
    """渲一帧 RGB + 一帧深度。深度用来做景深，切模式要重新 render。"""
    renderer.update_scene(data, camera=cam)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
    rgb = renderer.render()

    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()
    return rgb, depth


def render_still(demo, width=1280, height=720, out=None, film=True):
    """跑到"抬起完成"再截图，展示的是夹着物体悬空的状态，比空爪有说服力。"""
    renderer = mujoco.Renderer(demo.model, height, width)
    t_lift_end = 2.0 + 1.1 + 0.9 + 0.5 + 1.9
    for _ in range(int(t_lift_end / demo.model.opt.timestep)):
        demo.step()
    cam = make_camera(demo.model)
    rgb, depth = _render_pair(renderer, demo.data, cam)
    renderer.close()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = out or os.path.join(OUT_DIR, "best_still.png")

    # 同时存一张"未处理"的，方便对比后处理到底做了什么
    raw = out.replace(".png", "_raw.png")
    Image.fromarray(rgb).save(raw)

    img = postfx.apply(rgb, depth, dof_focus=0.52) if film else rgb
    Image.fromarray(img).save(out)
    print("静帧已保存：", out)
    print("原图（未后处理）：", raw)
    return out


def render_video(demo, fps=30, width=1280, height=720, out_name="best_grasp",
                 film=True):
    renderer = mujoco.Renderer(demo.model, height, width)
    cam = make_camera(demo.model)
    dt = demo.model.opt.timestep
    substep = max(1, int(round(1.0 / (fps * dt))))
    n = int(demo.seq.total / dt)

    frames = []
    for k in range(n):
        demo.step()
        if k % substep == 0:
            rgb, depth = _render_pair(renderer, demo.data, cam)
            frames.append(postfx.apply(rgb, depth, dof_focus=0.52) if film else rgb)
    renderer.close()

    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, out_name)
    path = save_video(frames, fps, base)
    print(f"{len(frames)} 帧 → {path}")
    return path


def save_video(frames, fps, base):
    try:
        import imageio.v2 as imageio
        path = base + ".mp4"
        imageio.mimsave(path, frames, fps=fps, quality=9, macro_block_size=None)
        return path
    except ImportError:
        from PIL import Image
        path = base + ".gif"
        imgs = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=256)
                for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0, optimize=True)
        print("（未装 imageio，输出 gif。要 mp4：pip install imageio imageio-ffmpeg）")
        return path


def run_viewer(demo, seconds=None):
    print("窗口已打开：左键拖动转视角，滚轮缩放。动作结束后自动重来。")
    start = time.time()
    with mujoco.viewer.launch_passive(demo.model, demo.data) as viewer:
        viewer.cam.lookat[:] = [0.19, 0.0, 0.085]
        viewer.cam.distance = 0.42
        viewer.cam.azimuth = 124
        viewer.cam.elevation = -14

        wall0 = time.time()
        while viewer.is_running():
            for _ in range(10):
                demo.step()
            viewer.sync()

            lag = (time.time() - wall0) - demo.data.time
            if lag < 0:
                time.sleep(-lag)

            if demo.data.time > demo.seq.total:
                demo.reset()
                wall0 = time.time()

            if seconds is not None and time.time() - start > seconds:
                break


def main():
    ap = argparse.ArgumentParser(description="SO-101 高质量仿真")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--still", action="store_true")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    demo = BestDemo()

    if args.still:
        render_still(demo, args.width, args.height)
        return

    if args.record:
        result = demo.run_to_end()
        print(f"\n抓取结果：抬高 {result['lift'] * 1000:.1f} mm → "
              f"{'成功' if result['ok'] else '失败'}")
        demo.reset()
        render_video(demo, args.fps, args.width, args.height)
        return

    run_viewer(demo, args.seconds)


if __name__ == "__main__":
    main()
