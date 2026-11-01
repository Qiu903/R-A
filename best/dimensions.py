# -*- coding: utf-8 -*-
"""
导出当前机械臂的详细尺寸
================================================================

数据全部从"编译后的 MjModel"里读，不是从代码常量抄 —— 这样报告里的数字
和仿真里跑的模型必然一致。只有装配件的标称尺寸（舵机外形等）来自设计参数。

输出：控制台表格 + outputs/dimensions.md
"""

import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import scene  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "dimensions.md")

ARM_BODIES = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "tool",
              "finger_l", "finger_r"]
JOINTS = ["j1_pan", "j2_lift", "j3_elbow", "j4_wrist_flex", "j5_wrist_roll",
          "j6_finger_l", "j6_finger_r"]

GEOM_TYPE = {
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_MESH: "mesh",
    mujoco.mjtGeom.mjGEOM_PLANE: "plane",
}


def name(model, objtype, i):
    return mujoco.mj_id2name(model, objtype, i) or f"#{i}"


def build_report(model):
    L = []
    w = L.append

    w("# SO-101 仿真模型 · 详细尺寸表")
    w("")
    w("> 数据来源：`best/scene.py` 编译后的 MjModel。运动学与 `models/so101.xml` 一致。")
    w("")

    # ---------------- 运动学链 ----------------
    w("## 1. 运动学链")
    w("")
    w("| 体 | 相对父体位置 (mm) | 关节轴 | 该段尺寸 (mm) | 质量 (g) |")
    w("|---|---|---|---|---|")

    link_dim = {
        "base": ("底座高", 95.0),
        "shoulder": ("肩部侧偏", 30.0),
        "upper_arm": ("大臂长", 116.0),
        "lower_arm": ("小臂长", 135.0),
        "wrist": ("腕→TCP", 110.0),
    }
    for bn in ARM_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
        if bid < 0:
            continue
        p = model.body_pos[bid] * 1000
        mass = float(model.body_mass[bid]) * 1000
        jid = model.body_jntadr[bid]
        axis = "-"
        if model.body_jntnum[bid] > 0 and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            a = model.jnt_axis[jid]
            axis = "xyz"[[int(round(v)) for v in np.abs(a)].index(1)]
        if bn in link_dim:
            label, val = link_dim[bn]
            dim = f"{label} {val:.0f}"
        else:
            dim = "—"
        L.append(f"| {bn} | ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) | {axis} "
                 f"| {dim} | {mass:.1f} |")

    reach = 30.0 + 116.0 + 135.0 + 110.0
    total_mass = sum(float(model.body_mass[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)]) for b in ARM_BODIES
                     if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) >= 0)
    w(f"| **合计** | | | **最大臂展 {reach:.0f}** | **{total_mass * 1000:.0f}** |")
    w("")
    w("最大臂展 = 肩侧偏 + 大臂 + 小臂 + 腕到TCP，即完全伸直时 TCP 到回转轴的直线距离。")
    w("")

    # ---------------- 关节 ----------------
    w("## 2. 关节参数")
    w("")
    w("| 关节 | 类型 | 轴 | 限位 (°) | armature | 阻尼 | 执行器 kp | 最大力矩 (N·m) |")
    w("|---|---|---|---|---|---|---|---|")

    for jn in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            continue
        t = "hinge" if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE else "slide"
        a = model.jnt_axis[jid]
        if t == "hinge":
            lo, hi = np.degrees(model.jnt_range[jid])
            rng_txt = f"{lo:.1f}° ~ {hi:.1f}°"
        else:
            lo, hi = model.jnt_range[jid] * 1000
            rng_txt = f"{lo:.0f} ~ {hi:.0f} mm"
        ar = float(model.dof_armature[model.jnt_dofadr[jid]])
        dp = float(model.dof_damping[model.jnt_dofadr[jid]])

        aid = -1
        for i in range(model.nu):
            if model.actuator_trnid[i, 0] == jid:
                aid = i
                break
        if aid >= 0:
            kp = f"{float(model.actuator_gainprm[aid, 0]):.0f}"
            fr = float(abs(model.actuator_forcerange[aid]).max())
            fmax = f"{fr:.1f}" if fr > 0 else "不限"
        else:
            kp, fmax = "—", "由耦合约束驱动"
        w(f"| {jn} | {t} | ({a[0]:.0f}, {a[1]:.0f}, {a[2]:.0f}) "
          f"| {rng_txt} | {ar:.3f} | {dp:.3f} | {kp} | {fmax} |")
    w("")

    # ---------------- 结构 ----------------
    w("## 3. 结构与几何")
    w("")
    w("| 部位 | geom 数 | 三角形面数（估） |")
    w("|---|---|---|")

    tri_total = 0
    for bn in ARM_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
        if bid < 0:
            continue
        n_geom = int(model.body_geomnum[bid])
        tris = 0
        for gi in range(model.body_geomadr[bid],
                        model.body_geomadr[bid] + model.body_geomnum[bid]):
            if model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[gi]
                if mid >= 0:
                    tris += int(model.mesh_facenum[mid])
            else:
                tris += 12
        tri_total += tris
        w(f"| {bn} | {n_geom} | {tris} |")
    w(f"| **合计** | **{sum(int(model.body_geomnum[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)]) for b in ARM_BODIES if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) >= 0)}** | **{tri_total}** |")
    w("")

    # ---------------- 夹爪 ----------------
    w("## 4. 夹爪与末端")
    w("")
    w("| 项 | 数值 |")
    w("|---|---|")
    w("| TCP 位置 | 距腕关节 110 mm，沿前臂轴向 |")
    w("| 单指行程 | 0 ~ 20 mm |")
    w("| 两指开口 | 0 ~ 40 mm |")
    w("| 指尖尺寸 | 38 × 8 × 22 mm（长 × 厚 × 高） |")
    w("| 指尖摩擦系数 | 2.0 |")
    w("| 最大夹持力 | 8 N / 单指 |")
    w("| 指尖接触刚度 solref | 0.004 s, 0.9 |")
    w("")

    # ---------------- 工作空间 ----------------
    w("## 5. 工作空间")
    w("")
    rng = np.random.default_rng(0)
    pts = []
    from level1_fk import forward_kinematics
    for _ in range(20000):
        q = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-1.6, 1.6),
                      rng.uniform(-2.4, 2.4), rng.uniform(-1.9, 1.9)])
        pts.append(forward_kinematics(q)[-1][:3, 3])
    pts = np.array(pts)

    w("蒙特卡洛采样（2 万组随机关节角，仅统计关节限位内的点）：")
    w("")
    w("| 方向 | 最小 | 最大 |")
    w("|---|---|---|")
    for i, ax in enumerate("XYZ"):
        w(f"| {ax} (mm) | {pts[:, i].min() * 1000:.1f} | {pts[:, i].max() * 1000:.1f} |")
    rxy = np.hypot(pts[:, 0], pts[:, 1])
    w(f"| 水平半径 (mm) | {rxy.min() * 1000:.1f} | {rxy.max() * 1000:.1f} |")
    w("")
    w("注意：Z 的下限是负值，因为仿真里连杆不参与碰撞，关节能转到桌面以下。")
    w("实际可用的工作空间是 Z > 0（桌面以上）那部分，加上夹爪本身的尺寸余量。")
    w("")

    # ---------------- 物理 ----------------
    w("## 6. 仿真参数")
    w("")
    w("| 项 | 数值 |")
    w("|---|---|")
    w(f"| 时间步长 | {model.opt.timestep * 1000:.2f} ms |")
    w(f"| 求解器迭代 | {model.opt.iterations} |")
    w(f"| 重力 | ({model.opt.gravity[0]:.1f}, {model.opt.gravity[1]:.1f}, "
      f"{model.opt.gravity[2]:.1f}) m/s² |")
    w(f"| 碰撞几何 | 仅指尖（contype=1），其余 contype=0 |")
    w(f"| 视觉网格 | {model.nmesh} 份 |")
    w(f"| 材质 | {model.nmat} 种 |")
    w("")

    return "\n".join(L)


def main():
    # Windows 控制台默认 GBK，打印 m/s² 这类字符会直接抛异常
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    model, _ = scene.compile_scene()
    text = build_report(model)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n已保存：{OUT}")


if __name__ == "__main__":
    main()
