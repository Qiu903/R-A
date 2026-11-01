# -*- coding: utf-8 -*-
"""
第 3 级 · 步骤 1：打开重力，量清楚"位置伺服在自重下差多少"
================================================================

第 3 级分四步，每步只加一个变量：
    步骤 1（本文件）：开重力，什么都不加，先量出重力造成的定位误差
    步骤 2：加夹爪（还不夹东西）        → grasp_scene.py
    步骤 3：加物体和接触参数            → grasp_scene.py
    步骤 4：闭环抓取                    → sim.py

为什么第一步必须单独做：
    第 2 级关重力的阶跃残差是 1e-7 rad，看着完美。一开重力，位置伺服就变成
    "弹簧吊着重物"——误差不再是零，而是 重力力矩 ÷ 刚度。这个误差有多大、
    随姿态怎么变，决定了后面要不要做重力补偿。不先量，第 4 步抓不住时
    你会在"接触参数"和"重力"之间反复怀疑。

本脚本不修改 models/so101.xml：重力用 MjSpec 在内存里覆写，
第 2 级那份"关重力"的模型保持原样，两级的对照才有意义。
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import mujoco  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from level1_fk import forward_kinematics  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "so101.xml")
FIG_DIR = os.path.join(ROOT, "outputs")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

JOINTS = ["j1_pan", "j2_lift", "j3_elbow", "j4_wrist_flex", "j5_wrist_roll"]
LABELS = ["J1 肩回转", "J2 肩俯仰", "J3 肘俯仰", "J4 腕俯仰", "J5 腕自转"]

# 关节软限位（弧度），与 MJCF 的 range 一致
LIMITS = np.array([[-np.pi, np.pi], [-1.6, 1.6], [-2.4, 2.4],
                   [-1.9, 1.9], [-np.pi, np.pi]])

GRAVITY_ON = [0.0, 0.0, -9.81]
GRAVITY_OFF = [0.0, 0.0, 0.0]


# ============================================================
# 一、模型
# ============================================================
def build_model(gravity):
    """
    同一份 XML 编译两次，只改重力。

    用 MjSpec 而不是去改 XML：第 2 级那份文件必须保持"关重力"，
    否则跨校验和跟随曲线全部失去对照意义。
    """
    spec = mujoco.MjSpec.from_file(MODEL_PATH)
    spec.option.gravity = list(gravity)
    return spec.compile()


def joint_addresses(model):
    """qpos 下标（关节角）和 dof 下标（力矩）要分开取，这两个数组不是一回事。"""
    qadr, dadr = [], []
    for name in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr.append(model.jnt_qposadr[jid])
        dadr.append(model.jnt_dofadr[jid])
    return qadr, dadr


def tcp_pos(model, data):
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    return data.site_xpos[sid].copy()


def hold(model, data, qadr, target, settle=2500):
    """
    把关节拉到 target 后一直保持，等它稳定，返回最终的关节角。

    为什么要跑满 2.5 秒：欠阻尼系统会来回晃几下才停，
    采样太早拿到的是振荡中的瞬时值，不是稳态误差。
    """
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = target
    for _ in range(settle):
        mujoco.mj_step(model, data)
    return data.qpos[qadr].copy()


# ============================================================
# 二、测试 1：阶跃响应（开重力 vs 关重力）
# ============================================================
def step_response(model, data, qadr, target_value=0.6, seconds=1.5):
    """给 J2 一个固定目标，记录全过程。两条曲线叠在一起，能直接看出稳态偏差。"""
    dt = model.opt.timestep
    n = int(seconds / dt)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    data.ctrl[:] = 0.0
    for _ in range(int(0.2 / dt)):        # 先静止 0.2 秒
        mujoco.mj_step(model, data)

    t_log = np.zeros(n)
    q_log = np.zeros((n, len(qadr)))
    data.ctrl[:] = target_value
    for k in range(n):
        mujoco.mj_step(model, data)
        t_log[k] = data.time
        q_log[k] = data.qpos[qadr]
    return t_log, q_log


# ============================================================
# 三、测试 2：稳态误差 + 重力力矩
# ============================================================
def steady_state_report(model, data, qadr, dadr, target=None):
    """
    target 给定时保持该姿态；不给则用零位。
    返回：实际角、误差、以及支撑这个姿态需要多大力矩（qfrc_bias）。
    """
    target = np.zeros(5) if target is None else np.asarray(target, float)
    q = hold(model, data, qadr, target)

    # 静止时 qfrc_bias 就是重力项：执行器必须输出这么大的力矩才能抗住
    mujoco.mj_forward(model, data)
    gravity_torque = data.qfrc_bias[dadr].copy()
    return q, q - target, gravity_torque


# ============================================================
# 四、测试 3：随机姿态下的 TCP 偏差
# ============================================================
def tcp_error_sweep(model, data, qadr, n=60, seed=3, settle=2500):
    """
    随机给 60 组目标姿态，比较"想要的 TCP"和"伺服实际送到的 TCP"。

    这里比的是末端毫米数，不是关节弧度：
    报告里能写"开重力后末端平均偏 8 mm"，比"关节误差 0.03 rad"有说服力得多。
    """
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n):
        target = np.array([rng.uniform(-2.5, 2.5),
                           rng.uniform(-1.2, 1.2),
                           rng.uniform(-1.8, 1.8),
                           rng.uniform(-1.4, 1.4),
                           0.0])
        q_actual = hold(model, data, qadr, target, settle=settle)
        want = forward_kinematics(target[:4])[-1][:3, 3]
        errs.append(np.linalg.norm(tcp_pos(model, data) - want))
    return np.array(errs)


# ============================================================
# 五、主流程
# ============================================================
def main():
    m_on = build_model(GRAVITY_ON)
    m_off = build_model(GRAVITY_OFF)
    d_on = mujoco.MjData(m_on)
    d_off = mujoco.MjData(m_off)

    qadr, dadr = joint_addresses(m_on)
    print(f"模型：nq={m_on.nq}  nu={m_on.nu}  dt={m_on.opt.timestep}s  "
          f"重力={np.round(m_on.opt.gravity, 2)}\n")

    # ---------- 测试 1 ----------
    t, q_on = step_response(m_on, d_on, qadr, target_value=0.6)
    _, q_off = step_response(m_off, d_off, qadr, target_value=0.6)

    print("--- 测试 1：J2 目标 0.6 rad 的阶跃响应（保持 1.5 s）---")
    print(f"{'':<12}{'关重力终值':>12}{'开重力终值':>12}{'重力造成的残差':>16}")
    for i, name in enumerate(LABELS):
        print(f"{name:<10}{q_off[-1, i]:>14.4f}{q_on[-1, i]:>14.4f}"
              f"{q_off[-1, i] - q_on[-1, i]:>16.4f}")

    # ---------- 测试 2 ----------
    print("\n--- 测试 2：零位姿态下各关节的稳态误差与所需力矩 ---")
    q0, err0, tau0 = steady_state_report(m_on, d_on, qadr, dadr)
    print(f"{'关节':<12}{'目标':>8}{'实际':>9}{'误差(°)':>11}"
          f"{'重力力矩(N·m)':>15}{'kp':>6}")
    for i, name in enumerate(LABELS):
        kp = m_on.actuator_gainprm[i, 0]
        print(f"{name:<10}{0.0:>8.2f}{q0[i]:>9.4f}{np.degrees(err0[i]):>11.2f}"
              f"{tau0[i]:>15.4f}{kp:>6.0f}")

    # ---------- 测试 3 ----------
    print("\n--- 测试 3：60 组随机姿态的末端偏差 ---")
    e_on = tcp_error_sweep(m_on, d_on, qadr)
    e_off = tcp_error_sweep(m_off, d_off, qadr)
    print(f"关重力：最大 {e_off.max():.2e} m，平均 {e_off.mean():.2e} m（只是浮点噪声）")
    print(f"开重力：最大 {e_on.max() * 1000:.2f} mm，平均 {e_on.mean() * 1000:.2f} mm")
    print("（真机的最终精度 = 重力偏差 + 舵机背隙 + 打印件刚度，这三项各自都有毫米级分量。"
          "仿真里只出现了第一项，所以别把这里的数字当成真机精度）\n")

    # ---------- 出图 ----------
    fig = plt.figure(figsize=(12, 7.5))
    fig.suptitle("第 3 级 · 步骤 1：打开重力后的位置伺服误差", fontsize=13)

    ax = fig.add_subplot(2, 2, 1)
    ax.plot(t, q_off[:, 1], "--", lw=1.3, color="gray", label="关重力")
    ax.plot(t, q_on[:, 1], lw=1.8, color="#c0392b", label="开重力")
    ax.axhline(0.6, color="#2e86de", lw=1.0, ls=":", label="目标 0.6 rad")
    ax.set_title("J2 阶跃响应：关重力能精确到位，开重力差一截", fontsize=9)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("角度 (rad)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")

    # 稳态差异只有 0.0076 rad，在全尺度图里两条线糊成一条，必须放大才能看见
    tail = int(0.35 / (t[1] - t[0]))
    axins = ax.inset_axes([0.40, 0.16, 0.56, 0.40])
    axins.plot(t[-tail:], q_off[-tail:, 1], "--", lw=1.2, color="gray")
    axins.plot(t[-tail:], q_on[-tail:, 1], lw=1.6, color="#c0392b")
    axins.axhline(0.6, color="#2e86de", lw=0.9, ls=":")
    axins.set_title("稳态放大（0.35 s）", fontsize=7)
    axins.tick_params(labelsize=6)
    axins.grid(alpha=0.2)
    ax.indicate_inset_zoom(axins, edgecolor="0.5")

    ax = fig.add_subplot(2, 2, 2)
    colors = ["#2e86de", "#10ac84", "#ee9f27", "#8e44ad", "#c0392b"]
    ax.bar(range(5), np.degrees(np.abs(err0)), color=colors, width=0.6)
    ax.set_xticks(range(5))
    ax.set_xticklabels([l.split()[0] for l in LABELS], fontsize=8)
    ax.set_title("零位下各关节被重力压出的角度偏差", fontsize=9)
    ax.set_ylabel("误差 (°)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, axis="y")
    for i, v in enumerate(np.degrees(np.abs(err0))):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax = fig.add_subplot(2, 2, 3)
    ax.bar(range(5), np.abs(tau0), color=colors, width=0.6)
    ax.set_xticks(range(5))
    ax.set_xticklabels([l.split()[0] for l in LABELS], fontsize=8)
    ax.set_title("各关节静止时需要的力矩（qfrc_bias，即重力项）", fontsize=9)
    ax.set_ylabel("力矩 (N·m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, axis="y")
    for i, v in enumerate(np.abs(tau0)):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax = fig.add_subplot(2, 2, 4)
    bins = np.linspace(0, max(e_on.max(), 1e-3) * 1000, 20)
    ax.hist(e_off * 1000, bins=bins, color="gray", alpha=0.6, label="关重力")
    ax.hist(e_on * 1000, bins=bins, color="#c0392b", alpha=0.75, label="开重力")
    ax.set_title("60 组随机姿态的末端偏差分布", fontsize=9)
    ax.set_xlabel("TCP 偏差 (mm)", fontsize=8)
    ax.set_ylabel("组数", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "level3_gravity.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"图片已保存：{out}")


if __name__ == "__main__":
    main()
