# -*- coding: utf-8 -*-
"""
带腕部姿态的逆解（数值法）
================================================================

为什么第 1 级那个解析逆解不够用：

    level1_fk.solve_ik 把 J4（腕俯仰）钉死在 0，理由是"J4 不影响位置"。
    做运动学验证时这没问题，但一抓东西就露馅：
    q4 = 0 意味着夹爪永远顺着前臂方向伸出去，夹爪的朝向完全由 q2+q3 决定。
    而这个臂底座只有 95 mm，要够到桌面上 30 mm 的物体时，唯一可行的位置解
    会让夹爪斜着 47° 杵下去（实测），指尖一端先磕到桌面，夹不住。

    要让夹爪垂直朝下，必须把 q4 当成自由参数。可 q4 一偏离 0，
    "J4 之后 110 mm 不影响位置"这条就不成立了，解析式要重推。

所以这里换数值法：固定 q4 猜一个值，只对 q1/q2/q3 解位置，
用阻尼最小二乘迭代。好处是不用重推公式、天然支持关节限位，
坏处是慢 —— 离线规划用，不进实时循环。

选姿态的策略：把 q4 从 -1.6 扫到 1.6，对每个 q4 解一次位置，
挑"夹爪轴向的 z 分量最小"的那个，也就是最朝下的姿态。
"""

import numpy as np

from level1_fk import forward_kinematics, JOINT_LIMITS

LO = JOINT_LIMITS[:3, 0]      # q1/q2/q3 的下限
HI = JOINT_LIMITS[:3, 1]


def fk(q4, q123):
    """返回 (TCP 位置, TCP 的 x 轴方向)。q123 是 q1/q2/q3。"""
    frames = forward_kinematics(np.array([q123[0], q123[1], q123[2], q4]))
    T = frames[-1]
    return T[:3, 3].copy(), T[:3, 0].copy()


def _jacobian(q4, q, p0, eps=1e-6):
    """
    数值雅可比：一个元素一个元素地扰动，看 TCP 怎么动。
    解析雅可比更快，但这里每个目标只解一次，没必要为它多写一套公式。
    """
    J = np.zeros((3, 3))
    for i in range(3):
        dq = q.copy()
        dq[i] += eps
        p, _ = fk(q4, dq)
        J[:, i] = (p - p0) / eps
    return J


def solve_q123(target, q4, seed=None, iters=120, tol=2e-4):
    """
    固定腕角 q4，解 q1/q2/q3 让 TCP 到达 target。
    解不出（超限位或落在工作空间外）返回 None。
    """
    target = np.asarray(target, dtype=float)
    q = np.array([0.0, -0.8, 1.0]) if seed is None else np.array(seed, dtype=float)
    q = np.clip(q, LO, HI)

    for _ in range(iters):
        p, _ = fk(q4, q)
        e = target - p
        if np.linalg.norm(e) < tol:
            break
        J = _jacobian(q4, q, p)
        # 阻尼最小二乘：J 在某些姿态接近奇异，直接求逆会炸，
        # 加个 1e-6 的阻尼相当于"尽量动，但别动太猛"
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
        q = np.clip(q + np.clip(dq, -0.25, 0.25), LO, HI)

    p, _ = fk(q4, q)
    if np.linalg.norm(p - target) > 5e-4:
        return None
    return q


def best_pose(target, q4_grid=None, seed=None):
    """
    在 q4 网格上找一个"夹爪最朝下"的可达姿态。

    返回 dict：q4 / q123 / q（4 维）/ axis（TCP 轴向）/ z_axis（轴向 z 分量）
    找不到返回 None。
    """
    if q4_grid is None:
        q4_grid = np.arange(-1.5, 1.51, 0.05)

    best = None
    for q4 in q4_grid:
        q123 = solve_q123(target, q4, seed=seed)
        if q123 is None:
            continue
        _, axis = fk(q4, q123)
        if best is None or axis[2] < best["z_axis"]:
            best = {
                "q4": float(q4),
                "q123": q123,
                "q": np.array([q123[0], q123[1], q123[2], q4]),
                "axis": axis,
                "z_axis": float(axis[2]),
            }
    return best


def pose_report(target, pose):
    """把一次求解结果打成一行人能看的话，调参时用。"""
    if pose is None:
        return f"目标 {np.round(target, 3)}：无解"
    q = pose["q"]
    return (f"目标 {np.round(target, 3)} → "
            f"q=({np.degrees(q[0]):.1f}°, {np.degrees(q[1]):.1f}°, "
            f"{np.degrees(q[2]):.1f}°, {np.degrees(q[3]):.1f}°)  "
            f"夹爪轴向 z={pose['z_axis']:.3f}")


if __name__ == "__main__":
    # 自检：抓取点必须解出一个夹爪明显朝下的姿态，否则后面不可能夹住
    for tgt in [(0.20, 0.0, 0.015), (0.20, 0.0, 0.115), (0.24, 0.0, 0.015)]:
        pose = best_pose(tgt)
        print(pose_report(tgt, pose))
        if pose is not None:
            p, _ = fk(pose["q4"], pose["q123"])
            print(f"    正解回代误差 = {np.linalg.norm(p - np.array(tgt)):.2e} m")
