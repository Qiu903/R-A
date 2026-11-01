# -*- coding: utf-8 -*-
"""
控制层：轨迹、执行器模型、抓取状态机
================================================================

这一层解决"实际能力"：仿真里默认的位置执行器太理想了，说 30° 立刻在 30°，
没有延迟、没有间隙、没有量化误差。真机上这三样都在，而且直接决定成败。

这里刻意把它们加回去：
    延迟    舵机从收到指令到开始动，12 ms 左右
    背隙    齿轮咬合的空程，反向运动时有一段"白转"
    量化    编码器分辨率，最小可控角度 2π/4096
    噪声    总线抖动带来的读数波动

加了这些之后成功率一定会掉。但那才是真机上的数字 ——
一个 100% 的仿真结果，对后面迁移到实物没有任何参考价值。
"""

import math
from collections import deque

import numpy as np


# ---------------------------------------------------------------- 插值
def quintic(u):
    """
    五次多项式：位置、速度、加速度在两端都为零。
    对比之前用的 smoothstep（三次），五次不会在起步/停止时留下加速度突变，
    真机上那个突变就是"抖动"的来源。
    """
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


class Trajectory:
    """
    分段关节轨迹。每段给定起止关节角和时长，段内用五次多项式过渡。

    用关键帧而不是"每步重新算逆解"，是因为逆解会有分支跳变；
    关键帧之间插值能保证关节连续、速度连续。
    """

    def __init__(self, start):
        self.start = np.asarray(start, dtype=float)
        self.segments = []          # (t0, t1, q_from, q_to)
        self._t = 0.0

    def add(self, q_to, duration):
        q_from = self.start if not self.segments else self.segments[-1][3]
        self.segments.append((self._t, self._t + duration,
                              np.asarray(q_from, float), np.asarray(q_to, float)))
        self._t += duration
        return self

    @property
    def total(self):
        return self._t

    def at(self, t):
        if not self.segments:
            return self.start.copy()
        if t <= self.segments[0][0]:
            return self.segments[0][2].copy()
        for t0, t1, qa, qb in self.segments:
            if t < t1:
                s = quintic((t - t0) / max(t1 - t0, 1e-9))
                return qa + (qb - qa) * s
        return self.segments[-1][3].copy()


# ---------------------------------------------------------------- 执行器
class ServoModel:
    """
    把理想指令变成"舵机实际能做到的指令"。

    注意这里不是模拟舵机的力矩输出（那仍然是 MuJoCo 的位置伺服在做），
    而是在指令侧加进真实存在的几种失真。这样做的原因：
    MuJoCo 的 actuator 是刚体动力学里的理想位置源，改不了它的延迟和间隙，
    但指令失真对抓取的影响和真实间隙是同量级的。

    每个关节独立维护状态，因为背隙方向和延迟累积是各关节自己的事。
    """

    def __init__(self, n_joint=5, dt=0.001, delay=0.012, backlash=0.007,
                 resolution=2 * math.pi / 4096, noise=0.0004, seed=0):
        self.n = n_joint
        self.delay_steps = max(1, int(round(delay / dt)))
        self.backlash = np.full(n_joint, backlash, dtype=float)
        self.resolution = resolution
        self.noise = noise
        self.rng = np.random.default_rng(seed)

        self._buffer = deque(maxlen=self.delay_steps + 1)
        self._engaged = np.zeros(n_joint, dtype=bool)   # 当前是否贴着某一侧齿面
        self._out = np.zeros(n_joint, dtype=float)

    def reset(self, q0):
        q0 = np.asarray(q0, float).copy()
        self._buffer.clear()
        for _ in range(self.delay_steps + 1):
            self._buffer.append(q0.copy())
        self._out = q0.copy()
        self._engaged = np.zeros(self.n, dtype=bool)

    def __call__(self, target):
        target = np.asarray(target, float)
        self._buffer.append(target.copy())
        delayed = self._buffer[0].copy()      # 到点的指令

        # 背隙：只有超出死区才带动负载，否则舵机在原地"空转"
        delta = delayed - self._out
        for i in range(self.n):
            if abs(delta[i]) > self.backlash[i]:
                self._out[i] = delayed[i] - math.copysign(self.backlash[i], delta[i])

        # 编码器量化 + 读数噪声
        q = np.round(self._out / self.resolution) * self.resolution
        q = q + self.rng.standard_normal(self.n) * self.noise
        return q


# ---------------------------------------------------------------- 状态机
class GraspSequencer:
    """
    抓取动作的时间表：接近 → 下降 → 合爪 → 停顿 → 抬起 → 观察。

    夹爪这边用一个"合到多窄"的指令，靠力限兜住物体；
    真机没有力反馈的话也只能这么做 —— 指令给到底，靠堵转力矩夹住。
    """

    STAGES = [
        ("approach", 2.0),
        ("descend", 1.1),
        ("close", 0.9),
        ("hold", 0.5),
        ("lift", 1.9),
        ("settle", 1.6),
    ]

    def __init__(self, q_start, q_pre, q_grasp, q_lift, open_width, closed_width):
        self.traj = Trajectory(q_start)
        self.traj.add(q_pre, 2.0)
        self.traj.add(q_grasp, 1.1)
        self.traj.add(q_grasp, 0.9 + 0.5)     # 合爪 + 停顿，臂不动
        self.traj.add(q_lift, 1.9)
        self.traj.add(q_lift, 1.6)

        self.open = open_width
        self.closed = closed_width
        self.t_close = 2.0 + 1.1
        self.t_close_end = self.t_close + 0.9

        self.edges = []
        t = 0.0
        for name, dur in self.STAGES:
            t += dur
            self.edges.append((t, name))

    @property
    def total(self):
        return self.traj.total

    def arm(self, t):
        return self.traj.at(t)

    def grip(self, t):
        if t < self.t_close:
            return self.open
        if t < self.t_close_end:
            return self.open + (self.closed - self.open) * quintic(
                (t - self.t_close) / (self.t_close_end - self.t_close))
        return self.closed

    def stage(self, t):
        for edge, name in self.edges:
            if t < edge:
                return name
        return "done"
