# -*- coding: utf-8 -*-
"""
第 2 级 · MuJoCo 建模 + 关重力跟随验证
========================================

目标不是"跑起来一个仿真"，而是验证一件很具体的事：
    MJCF 模型和第 1 级那张 DH 表，是不是同一个几何。

为什么这一步值得单独做：
    后面第 3 级一开重力、一加接触，任何异常都会同时有三种嫌疑人
    （模型错 / 逆解错 / 接触参数错）。只有先把"模型本身对不对"钉死，
    后面的调试才有意义。

本脚本做三件事：
    1. 跨校验：同一组关节角下，MuJoCo 算出的 TCP 与第 1 级正解是否一致
    2. 跟随测试：给正弦目标，看关节能不能跟上（关重力）
    3. 阶跃测试：给一个固定目标，看稳态误差是不是零
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import mujoco  # noqa: E402

# 让脚本能 import 同目录的 level1_fk.py
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from level1_fk import forward_kinematics, DEG  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(HERE), "models", "so101.xml")
FIG_DIR = os.path.join(os.path.dirname(HERE), "outputs")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

JOINTS = ["j1_pan", "j2_lift", "j3_elbow", "j4_wrist_flex", "j5_wrist_roll"]
LABELS = ["J1 肩回转", "J2 肩俯仰", "J3 肘俯仰", "J4 腕俯仰", "J5 腕自转"]


def load_model():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    # jnt_qposadr[i] = 第 i 个关节在 qpos 数组里的下标（进阶语法：model 的数组属性直接按关节 id 索引）
    adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
           for n in JOINTS]
    return model, data, adr


def mj_tcp(model, data, adr):
    """从当前 qpos 算一次前向运动学，返回 TCP 的世界坐标。"""
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    return data.site_xpos[sid].copy()


def cross_check(model, data, adr):
    """核心校验：MJCF 与 DH 正解的 TCP 必须一致。"""
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(10):
        q = rng.uniform(-1.0, 1.0, 5)   # 含 J5，但 J5 是自转，不该影响 TCP 位置
        for k, i in enumerate(adr):
            data.qpos[i] = q[k]
        tcp_mj = mj_tcp(model, data, adr)

        frames = forward_kinematics(q[:4])
        tcp_dh = frames[-1][:3, 3]
        errs.append(np.linalg.norm(tcp_mj - tcp_dh))

    print("--- 跨校验：MJCF vs 第 1 级 DH 正解 ---")
    print(f"10 组随机姿态下 TCP 最大偏差 = {max(errs):.3e} m")
    print("（1e-3 以下算通过；超过说明两边的长度/轴向写岔了）\n")
    return max(errs)


def run_tracking(model, data, adr, duration=4.0):
    """正弦跟随测试：5 个关节各给不同频率的正弦目标。"""
    dt = model.opt.timestep
    n = int(duration / dt)
    t = np.arange(n) * dt

    amp = np.array([0.60, 0.50, 0.40, 0.30, 0.60])
    freq = np.array([0.30, 0.40, 0.50, 0.60, 0.30])

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    target_log = np.zeros((n, 5))
    qpos_log = np.zeros((n, 5))

    for k in range(n):
        target = amp * np.sin(2.0 * np.pi * freq * t[k])
        data.ctrl[:] = target
        mujoco.mj_step(model, data)
        target_log[k] = target
        qpos_log[k] = data.qpos[adr]

    return t, target_log, qpos_log


def run_step(model, data, adr, target_value=0.6, hold=2.0):
    """阶跃测试：目标突然跳到固定角度，看最后能不能精确停在目标上。"""
    dt = model.opt.timestep
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    data.ctrl[:] = 0.0
    for _ in range(int(0.3 / dt)):
        mujoco.mj_step(model, data)

    data.ctrl[:] = target_value
    last = None
    for _ in range(int(hold / dt)):
        mujoco.mj_step(model, data)
        last = data.qpos[adr].copy()
    return last


def main():
    model, data, adr = load_model()
    print(f"模型加载成功：nq={model.nq}  nu={model.nu}  dt={model.opt.timestep}s  "
          f"gravity={model.opt.gravity}\n")

    worst = cross_check(model, data, adr)

    t, tgt, act = run_tracking(model, data, adr)
    err = tgt - act

    # 前 0.5s 是起步过渡，统计误差从 0.5s 之后开始算
    settle = int(0.5 / model.opt.timestep)
    print("--- 正弦跟随误差（0.5s 之后）---")
    for i, name in enumerate(LABELS):
        print(f"{name:<10} RMS = {np.sqrt(np.mean(err[settle:, i]**2)):.4f} rad"
              f"   最大 = {np.max(np.abs(err[settle:, i])):.4f} rad")

    final = run_step(model, data, adr, target_value=0.6)
    step_err = np.abs(0.6 - final)
    print("\n--- 阶跃测试（目标 0.6 rad，保持 2s）---")
    for i, name in enumerate(LABELS):
        print(f"{name:<10} 稳态残差 = {step_err[i]:.2e} rad")
    print("（这个数接近 0 就对了：无重力、无外力时，理想位置伺服的终点就是目标角）\n")

    # ---------- 出图 ----------
    fig = plt.figure(figsize=(11, 10))
    fig.suptitle("第 2 级 · MuJoCo 关重力跟随验证（dt = 1 ms）", fontsize=13)

    colors = ["#2e86de", "#10ac84", "#ee9f27", "#8e44ad", "#c0392b"]
    for i in range(5):
        ax = fig.add_subplot(3, 2, i + 1)
        ax.plot(t, np.degrees(tgt[:, i]), "--", lw=1.2, color="gray", label="目标")
        ax.plot(t, np.degrees(act[:, i]), lw=1.6, color=colors[i], label="实际")
        rms = np.sqrt(np.mean(err[settle:, i]**2))
        ax.set_title(f"{LABELS[i]}   RMS 误差 {np.degrees(rms):.2f}°", fontsize=9)
        ax.set_xlabel("时间 (s)", fontsize=8)
        ax.set_ylabel("角度 (°)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="upper right")

    ax = fig.add_subplot(3, 2, 6)
    for i in range(5):
        ax.plot(t, np.degrees(err[:, i]), lw=1.2, color=colors[i], label=LABELS[i])
    ax.set_title(f"跟随误差（跨校验最大偏差 {worst:.1e} m）", fontsize=9)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("误差 (°)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "level2_tracking.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"图片已保存：{out}")


if __name__ == "__main__":
    main()
