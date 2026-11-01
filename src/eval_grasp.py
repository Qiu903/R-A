# -*- coding: utf-8 -*-
"""
第 3 级 · 成功率和失败模式统计
================================================================

前一步只证明了"能抓一次"。这一次要回答"多少次里能抓几次，抓不住时是怎么失败的"。

三次不同的扰动，都来自真机上的不确定性：
    位置偏移  规划时以为物体在 A，实际放在 A ± 2.5 mm  —— 对应视觉定位误差
    质量      30 g × U(0.8, 1.2)                      —— 对应未知物体重量
    摩擦      1.0 × U(0.7, 1.3)                       —— 对应表面材质差异

为什么位置偏移要单独做一个参数（place_offset）：
    如果让"规划位置"和"实际位置"一起抖，那等于假设机械臂知道物体精确在哪，
    测出来的成功率会虚高。真机上你永远不知道精确位置，规划用的是估计值。

判定与失败分类：
    抬高 > 20 mm                  → 成功
    曾经超过 20 mm 又掉回来        → 中途掉落
    水平被推走 > 20 mm 且没抬起    → 夹飞
    抬了但不到 20 mm              → 打滑
    基本没动                      → 没夹住
    逆解算不出姿态                → 规划无解

用法：
    python src/eval_grasp.py              # 5×5 网格 × 3 次
    python src/eval_grasp.py --trials 5   # 每格 5 次
"""

import os
import sys
import argparse
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import sim  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(ROOT, "outputs")

XS = np.array([0.18, 0.20, 0.22, 0.24, 0.26])
YS = np.array([-0.04, -0.02, 0.00, 0.02, 0.04])

# 两档扰动。温和档全通过说明不了什么 —— 必须有一档能把成功率压下来，
# 才知道系统在哪里崩、为什么崩，这样才叫"有失败模式可分析"。
PRESETS = {
    "mild":  dict(jitter=0.0025, mass=(0.8, 1.2), fric=(0.7, 1.3)),   # 定位精度好的情况
    "harsh": dict(jitter=0.0080, mass=(0.5, 2.0), fric=(0.4, 1.3)),   # 定位差 + 未知物体
}
JITTER = PRESETS["mild"]["jitter"]
MASS_RANGE = PRESETS["mild"]["mass"]
FRIC_RANGE = PRESETS["mild"]["fric"]

MODES = ["成功", "夹飞", "中途掉落", "打滑", "没夹住", "规划无解"]


def classify(log):
    """把一条轨迹归类成一种失败模式。返回 (模式, 抬升高度, 水平漂移)。"""
    z = log["cube_z"]
    lift = float(z[-1] - z[0])
    peak = float(z.max() - z[0])
    drift = float(np.hypot(log["cube_x"][-1] - log["cube_x"][0],
                           log["cube_y"][-1] - log["cube_y"][0]))

    if lift > sim.SUCCESS_LIFT:
        return "成功", lift, drift
    if peak > sim.SUCCESS_LIFT:
        return "中途掉落", lift, drift
    if drift > 0.020:
        return "夹飞", lift, drift
    if lift > 0.005:
        return "打滑", lift, drift
    return "没夹住", lift, drift


def trial(x, y, rng):
    """一次完整试验：随机化参数 → 建场景 → 跑完 → 分类。"""
    offset = rng.uniform(-JITTER, JITTER, 2)
    mass = 0.030 * rng.uniform(*MASS_RANGE)
    fric = 1.0 * rng.uniform(*FRIC_RANGE)

    try:
        demo = sim.GraspDemo(cube_xy=(x, y), place_offset=offset,
                             cube_mass=mass, cube_friction=fric, verbose=False)
    except RuntimeError:
        # 逆解不出姿态属于"这个位置这个腕角够不着"，是规划层的失败，
        # 和"够得着但没夹住"要分开统计
        return "规划无解", 0.0, 0.0

    log = sim.simulate(demo)
    mode, lift, drift = classify(log)
    return mode, lift, drift


def run(trials, seed=0):
    rng = np.random.default_rng(seed)
    records = []
    total = len(XS) * len(YS) * trials
    n = 0

    for x in XS:
        for y in YS:
            for _ in range(trials):
                mode, lift, drift = trial(x, y, rng)
                records.append({"x": float(x), "y": float(y),
                                "mode": mode, "lift": lift, "drift": drift})
                n += 1
                print(f"\r进度 {n}/{total}  当前 x={x:.2f} y={y:+.2f} → {mode}"
                      + " " * 6, end="", flush=True)
    print()
    return records


def sweep_run(n, seed, x=0.22, y=0.0):
    """
    只扫一个变量：定位误差。质量和摩擦固定在温和档。

    为什么值得单独做：两档对比只能说明"8 mm 比 2.5 mm 差"，
    而这条曲线能直接读出工程指标 —— "定位误差必须优于多少毫米"，
    那句话是能写进报告结论、也能指导真机标定精度的。
    """
    global JITTER
    js = np.array([0.0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.015])
    rates = []
    rng = np.random.default_rng(seed)

    for j in js:
        JITTER = float(j)
        ok = 0
        for _ in range(n):
            mode, _, _ = trial(x, y, rng)
            ok += mode == "成功"
        rates.append(ok / n)
        print(f"  定位误差 ±{j * 1000:>4.1f} mm → 成功率 {100 * ok / n:>3.0f}%  ({ok}/{n})")
    return js, np.array(rates)


def plot_tolerance(js, rates, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(js * 1000, rates * 100, "o-", lw=2, color="#c0392b")
    # 理论容差 = (夹爪开口 40 - 物体 30) / 2 = 5 mm，但实测能撑到 6~8 mm：
    # 因为合爪时指尖会把物体往中间推一点，等于自带一点对中能力。
    ax.axvline(6.0, color="gray", ls="--", lw=1, label="实测拐点 ~6 mm（理论 5 mm）")
    ax.set_xlabel("定位误差 ±(mm)", fontsize=9)
    ax.set_ylabel("成功率 (%)", fontsize=9)
    ax.set_title("抓取成功率随定位误差的变化（x=0.22, y=0）", fontsize=10)
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图已保存：{path}")


def summarize(records, trials):
    n = len(records)
    succ = sum(r["mode"] == "成功" for r in records)
    print(f"\n===== 总成功率 {succ}/{n} = {100 * succ / n:.1f}% =====")

    print(f"\n按位置（每格 {trials} 次）：")
    header = "        " + "".join(f"y={y:+.2f}  " for y in YS)
    print(header)
    for x in XS:
        row = f"x={x:.2f}  "
        for y in YS:
            cell = [r for r in records if r["x"] == x and r["y"] == y]
            ok = sum(r["mode"] == "成功" for r in cell)
            row += f"  {ok}/{len(cell)}    "
        print(row)

    print("\n失败模式分布：")
    for m in MODES:
        c = sum(r["mode"] == m for r in records)
        if c:
            print(f"  {m:<8} {c:>3} 次  {100 * c / n:>5.1f}%")

    lifts = np.array([r["lift"] for r in records])
    print(f"\n抬升高度：成功组均值 {1000 * lifts[lifts > sim.SUCCESS_LIFT].mean():.1f} mm"
          if np.any(lifts > sim.SUCCESS_LIFT) else "\n抬升高度：无成功样本")
    return succ / n


def plot(records, trials, path):
    grid = np.zeros((len(XS), len(YS)))
    for i, x in enumerate(XS):
        for j, y in enumerate(YS):
            cell = [r for r in records if r["x"] == x and r["y"] == y]
            grid[i, j] = sum(r["mode"] == "成功" for r in cell) / len(cell)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                            gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("第 3 级 · 抓取成功率（位置 × 参数扰动）", fontsize=13)

    ax = axes[0]
    im = ax.imshow(grid * 100, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto",
                   origin="lower")
    ax.set_xticks(range(len(YS)))
    ax.set_xticklabels([f"{y:+.2f}" for y in YS], fontsize=8)
    ax.set_yticks(range(len(XS)))
    ax.set_yticklabels([f"{x:.2f}" for x in XS], fontsize=8)
    ax.set_xlabel("y 方向偏移 (m)", fontsize=9)
    ax.set_ylabel("x 方向距离 (m)", fontsize=9)
    ax.set_title(f"成功率 %（每格 {trials} 次）", fontsize=10)
    for i in range(len(XS)):
        for j in range(len(YS)):
            ax.text(j, i, f"{grid[i, j] * 100:.0f}", ha="center", va="center",
                    fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[1]
    counts = [sum(r["mode"] == m for r in records) for m in MODES]
    colors = ["#10ac84", "#c0392b", "#e67e22", "#f1c40f", "#7f8c8d", "#34495e"]
    bars = ax.barh(range(len(MODES)), counts, color=colors)
    ax.set_yticks(range(len(MODES)))
    ax.set_yticklabels(MODES, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("次数", fontsize=9)
    ax.set_title(f"失败模式分布（共 {len(records)} 次）", fontsize=10)
    ax.grid(alpha=0.25, axis="x")
    for b, c in zip(bars, counts):
        if c:
            ax.text(c, b.get_y() + b.get_height() / 2, f" {c}", va="center", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图已保存：{path}")


def main():
    global JITTER, MASS_RANGE, FRIC_RANGE

    ap = argparse.ArgumentParser(description="抓取成功率统计")
    ap.add_argument("--trials", type=int, default=3, help="每个位置重复几次")
    ap.add_argument("--preset", choices=list(PRESETS), default="mild",
                    help="扰动强度：mild=定位精度好，harsh=定位差+未知物体")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help="只扫定位误差这一个变量，出容差曲线")
    args = ap.parse_args()

    p = PRESETS[args.preset]
    JITTER, MASS_RANGE, FRIC_RANGE = p["jitter"], p["mass"], p["fric"]

    if args.sweep:
        print(f"定位误差扫描：每档 {args.trials * 4} 次")
        js, rates = sweep_run(args.trials * 4, args.seed)
        plot_tolerance(js, rates, os.path.join(OUT_DIR, "level3_tolerance.png"))
        return

    print(f"扰动档位 {args.preset}：放置偏移 ±{JITTER * 1000:.1f} mm，"
          f"质量 ×{MASS_RANGE}，摩擦 ×{FRIC_RANGE}")
    print(f"网格 {len(XS)}×{len(YS)}，每格 {args.trials} 次，共 "
          f"{len(XS) * len(YS) * args.trials} 次")

    records = run(args.trials, args.seed)
    summarize(records, args.trials)
    plot(records, args.trials,
         os.path.join(OUT_DIR, f"level3_success_{args.preset}.png"))


if __name__ == "__main__":
    main()
