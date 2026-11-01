# -*- coding: utf-8 -*-
"""
第 1 级 · 纯数学正解 + 逆解 + matplotlib 3D 线框
================================================================

这一级不装任何物理引擎，只做两件事：
    正解：给一组关节角 -> 算出每个关节、末端 TCP 在空间中的坐标 -> 画出来
    逆解：给一个目标点 -> 反推这组关节角 -> 再用正解验证回没回到目标

它为什么必须是第一步？
    后面 MuJoCo 里的模型、逆解、标定，全都以这张 DH 参数表为基准。
    基准错了，后面所有调试都在给错误打补丁。所以这一级宁可慢，不要跳。

尺寸来源：SO-101 官方 URDF 实测几何（单位 m）
    底座到肩回转轴 0.095 | 肩侧偏 0.030 | 大臂 0.116 | 小臂 0.135 | 腕到 TCP 0.110
买了实物后把上面这几个数改成实测值即可，其余代码不用动。
"""

import os
import numpy as np
import matplotlib

# Agg 后端 = 不弹窗、直接把图存成文件。必须在 import pyplot 之前设置，否则无效。
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 让图上的中文正常显示（Windows 自带雅黑；换成 SimHei 也行）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 负号用 ASCII 减号，避免显示成方块


# ============================================================
# 一、机械尺寸（全部为米）
# ============================================================
BASE_HEIGHT = 0.095        # 底座地面 -> J1 肩回转轴高度
SHOULDER_OFFSET = 0.030    # J1 轴线 -> J2 肩俯仰轴的横向距离
UPPER_ARM = 0.116          # J2 -> J3 肘（大臂）
LOWER_ARM = 0.135          # J3 -> J4 腕（小臂）
TOOL_LENGTH = 0.110        # J4 腕 -> 夹爪中心 TCP

DEG = np.pi / 180.0        # 角度 -> 弧度，后面统一用角度写参数，读起来直观

# 图片输出目录（项目根目录下的 outputs/），已被 .gitignore 忽略
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "outputs")


# ============================================================
# 二、DH 参数表
# ============================================================
# 采用 Craig 的「改进 DH」写法，每一行是 4 个数 + 1 个关节角：
#     a     上一根连杆的长度
#     alpha 上一根连杆绕 x 轴转过的角度（扭角）
#     d     沿上一关节 z 轴的偏置
#     theta 本关节转角（我们的变量 q）
#
# 记忆方法：改一行只改一个几何量，出问题就只怀疑这一个。
DH_ROWS = [
    # a        alpha(deg)  d             说明
    (0.0,             0.0,  BASE_HEIGHT),   # 行1：J1 肩回转（绕垂直轴）
    (SHOULDER_OFFSET, -90.0, 0.0),          # 行2：J2 肩俯仰（扭 -90° 把轴放平）
    (UPPER_ARM,        0.0,  0.0),          # 行3：J3 肘俯仰
    (LOWER_ARM,        0.0,  0.0),          # 行4：J4 腕俯仰
]

# 说明：J5 腕部自转是「绕前臂长轴」转的。
# 标准 DH 只描述绕 z 轴的旋转，描述不了这种与前一根连杆共线的轴，
# 所以它不参与位置计算（转它不影响 TCP 在哪），只在画夹爪朝向时用到。
# 这不是偷懒，是 DH 表达能力的边界——真机里 J5 也确实只改朝向、不改位置。


def dh_matrix(a, alpha, d, theta):
    """由 4 个 DH 参数算出 4x4 齐次变换矩阵（当前帧 -> 下一帧）。"""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    T = np.array([
        [ct,      -st,      0.0, a],
        [ca * st,  ca * ct, -sa, -d * sa],
        [sa * st,  sa * ct,  ca,  d * ca],
        [0.0,      0.0,     0.0, 1.0],
    ])
    return T


def forward_kinematics(q):
    """
    正解：输入 4 个关节角（弧度），返回每个关节坐标系的原点和最终 TCP。

    @ 是 numpy 的矩阵乘法运算符（进阶语法）：A @ B 等价于矩阵乘，
    不要写成 A * B，那个是逐元素相乘，结果完全不同。
    """
    T = np.eye(4)                 # 基座坐标系，就是世界原点
    frames = [T.copy()]           # frames[i] = 第 i 个关节坐标系在世界中的位姿

    for (a, alpha_deg, d), theta in zip(DH_ROWS, q):
        T = T @ dh_matrix(a, alpha_deg * DEG, d, theta)
        frames.append(T.copy())

    # 最后一段：从腕关节再往前 110mm 到 TCP。
    # 这里只做平移、不转动，所以单独写，不塞进 DH 表以免概念混乱。
    T_tool = T @ dh_matrix(TOOL_LENGTH, 0.0, 0.0, 0.0)
    frames.append(T_tool)
    return frames


def joint_positions(frames):
    """取每个坐标系的平移部分，就是各关节在世界系里的 xyz。"""
    return [F[:3, 3] for F in frames]


# ============================================================
# 三、自检：几何对不对，先用三个能心算的结论验一遍
# ============================================================
def self_check():
    frames = forward_kinematics(np.zeros(4))
    tcp = frames[-1][:3, 3]

    reach = SHOULDER_OFFSET + UPPER_ARM + LOWER_ARM + TOOL_LENGTH
    expect = np.array([reach, 0.0, BASE_HEIGHT])

    # 真正的刚体不变量：每一段连杆自己的长度不会变。
    # （注意：肩到 TCP 的直线距离会随弯肘变化，那不是 bug，是几何本身如此。）
    seg_errors = {"大臂": 0.0, "小臂": 0.0, "腕->TCP": 0.0}
    # 单独转 J1 时，TCP 的高度和水平半径必须不变（J1 轴是垂直的）
    radius_errors = []

    rng = np.random.default_rng(0)   # 固定随机种子，保证每次结果一样
    for _ in range(20):
        q = rng.uniform(-1.0, 1.0, 4)
        f = forward_kinematics(q)
        pts = joint_positions(f)
        seg_errors["大臂"] = max(seg_errors["大臂"], abs(np.linalg.norm(pts[3] - pts[2]) - UPPER_ARM))
        seg_errors["小臂"] = max(seg_errors["小臂"], abs(np.linalg.norm(pts[4] - pts[3]) - LOWER_ARM))
        seg_errors["腕->TCP"] = max(seg_errors["腕->TCP"], abs(np.linalg.norm(pts[5] - pts[4]) - TOOL_LENGTH))

        tcp_0 = forward_kinematics(np.array([0.0, q[1], q[2], q[3]]))[-1][:3, 3]
        radius_errors.append(abs(np.linalg.norm(tcp_0[:2]) - np.linalg.norm(f[-1][:3, 3][:2])))

    print("--- 自检 ---")
    print(f"[1] 零位 TCP = {np.round(tcp, 4)}，理论值 = {np.round(expect, 4)}，"
          f"误差 = {np.linalg.norm(tcp - expect):.2e} m")
    print("[2] 随机姿态下各段连杆长度最大偏差："
          + "，".join(f"{k} {v:.2e} m" for k, v in seg_errors.items()))
    print(f"[3] 只转 J1 时 TCP 水平半径最大偏差 = {max(radius_errors):.2e} m")
    print()


# ============================================================
# 四、逆解（给目标点，反推关节角）
# ============================================================
def _ik_planar(dr, dz, sign):
    """平面两连杆求解：返回 (psi1, psi2)，其中 psi 是从 +r 轴量起、z 向上的角度。"""
    L2c = LOWER_ARM + TOOL_LENGTH   # J4 固定时，小臂与腕到 TCP 共线，合成一根
    d = np.hypot(dr, dz)

    # 够不着 / 落在死区（比 |L1 - L2c| 还近）时无解。
    # 容差加在边界外侧：完全伸直的手臂正好压在可达边界上，
    # 留 0.1mm 余量让边界附近的合法目标不会被浮点误差误判（后面的 clip 会兜住这点误差）。
    if d > UPPER_ARM + L2c + 1e-4 or d < abs(UPPER_ARM - L2c) - 1e-4:
        return None

    # 余弦定理求肘关节角；clip 防浮点误差把 cos 顶出 [-1, 1]
    c3 = (d * d - UPPER_ARM**2 - L2c**2) / (2.0 * UPPER_ARM * L2c)
    c3 = np.clip(c3, -1.0, 1.0)
    s3 = sign * np.sqrt(1.0 - c3 * c3)

    psi1 = np.arctan2(dz, dr) - np.arctan2(L2c * s3, UPPER_ARM + L2c * c3)
    psi2 = psi1 + np.arctan2(s3, c3)
    return psi1, psi2


def solve_ik(target, elbow="up", facing="front", q4=0.0):
    """
    逆解：给 TCP 目标位置 (x, y, z)，返回 [q1, q2, q3, q4]（弧度）。无解就返回 None。

    两个必须一起试的自由度：
      elbow  = "up" / "down"    肘往上翻还是往下翻（同一目标的镜像解）
      facing = "front" / "back" 正对着目标，还是把肩转 180° 背对着目标去够

    为什么"背对"也要试：
        肩侧偏让肩俯仰轴偏离底座轴线 30mm。目标落在底座轴线附近时，正对着解会把肘折过头、
        超出关节限位；换个方向（J1 差 180°、径向距离取负）反而有合法解。
        漏掉这个分支，逆解会误判一堆"够不着"——第一次验证时 500 个目标里就有 20 个是这么丢的。

    剩下的 1 个冗余自由度（J4）默认固定为 0：位置不受影响，抓取时可以拿它调夹爪朝向。
    """
    x, y, z = target
    r = np.hypot(x, y)
    dz = z - BASE_HEIGHT

    if facing == "front":
        q1 = np.arctan2(y, x)
        dr = r - SHOULDER_OFFSET
    else:
        # 背对：转 180°，径向变成负的
        q1 = np.arctan2(y, x) + np.pi
        dr = -(r + SHOULDER_OFFSET)

    # 归一化到 (-π, π]，避免输出 3π 这种看着奇怪的角度
    q1 = (q1 + np.pi) % (2.0 * np.pi) - np.pi

    sign = 1.0 if elbow == "up" else -1.0
    planar = _ik_planar(dr, dz, sign)
    if planar is None:
        return None

    psi1, psi2 = planar
    # 换算回本项目约定：正解里 psi = -theta，且 theta3 = psi1 - psi2
    q2 = -psi1
    q3 = psi1 - psi2
    return np.array([q1, q2, q3, q4])


# 关节软限位（弧度），和 models/so101.xml 里的 range 保持一致
JOINT_LIMITS = np.array([[-np.pi, np.pi], [-1.6, 1.6], [-2.4, 2.4], [-1.9, 1.9]])


def solve_ik_best(target):
    """把 2 种肘姿态 × 2 种朝向都试一遍，返回第一个不超关节限位的解。"""
    for facing in ("front", "back"):
        for elbow in ("up", "down"):
            q = solve_ik(target, elbow=elbow, facing=facing)
            if q is None:
                continue
            if np.all(q >= JOINT_LIMITS[:, 0]) and np.all(q <= JOINT_LIMITS[:, 1]):
                return q, f"{facing}/{elbow}"
    return None, None


def verify_ik():
    """
    闭环验证：正解 -> 目标点 -> 逆解 -> 正解，看 TCP 能不能原样回到目标点。
    这是检验逆解唯一靠谱的方法——算出来的角度对不对，让正解自己说。
    """
    rng = np.random.default_rng(2)
    pos_err = []
    joint_err = []
    unreachable = 0

    for _ in range(500):
        # 在关节限位内随机采一个姿态（J4 固定为 0，保证逆解一定有解）
        q0 = np.array([rng.uniform(-np.pi, np.pi),
                       rng.uniform(-1.5, 1.5),
                       rng.uniform(-2.3, 2.3),
                       0.0])
        target = forward_kinematics(q0)[-1][:3, 3]

        q_sol, _ = solve_ik_best(target)
        if q_sol is None:
            unreachable += 1
            continue

        tcp_back = forward_kinematics(q_sol)[-1][:3, 3]
        pos_err.append(np.linalg.norm(tcp_back - target))
        # 关节角只在 q2/q3 上可比：q1 可能差 2π，两解也可能镜像
        joint_err.append(np.linalg.norm(q_sol[1:3] - q0[1:3]))

    print("--- 逆解闭环验证（500 组随机目标）---")
    print(f"[4] 位置闭环最大误差 = {max(pos_err):.3e} m"
          f"（{len(pos_err)} 组成功，{unreachable} 组无解）")
    print(f"[5] 关节角最大偏差  = {max(joint_err):.3f} rad"
          f"（出现镜像解时这个数会很大，但位置对得上就说明解是对的）")
    print()


# ============================================================
# 五、可视化
# ============================================================
def fmt(v):
    """格式化坐标，并把 -0.000 这种显示成 0.000。"""
    return f"{0.0 if abs(v) < 1e-9 else v:.3f}"


def draw_gripper(ax, frames, roll_deg, color="crimson"):
    """在 TCP 处画一个简易两指夹爪，roll 用来演示 J5 的效果。"""
    T = frames[-1]
    tcp = T[:3, 3]
    x_axis = T[:3, 0]              # 前臂前伸方向
    y_axis = T[:3, 1]              # 与开合方向垂直的参考轴

    # 罗德里格斯旋转：绕任意轴 u 转 theta。
    # 进阶语法：这里用 3x3 的反对称矩阵 K 拼出旋转，比手写三角函数短得多。
    roll = roll_deg * DEG
    w = y_axis * np.cos(roll) + np.cross(x_axis, y_axis) * np.sin(roll)

    base = tcp + 0.010 * x_axis
    tip_a = tcp + 0.045 * x_axis + 0.020 * w
    tip_b = tcp + 0.045 * x_axis - 0.020 * w

    for p, q in ((base, tip_a), (base, tip_b), (tip_a, tip_b)):
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=color, lw=2)


def draw_arm(ax, q_deg, roll_deg, title, target=None):
    q = np.array(q_deg) * DEG
    frames = forward_kinematics(q)
    pts = joint_positions(frames)

    # 桌面参考面，用来建立"上下"直觉
    gx, gy = np.meshgrid(np.linspace(-0.15, 0.45, 2), np.linspace(-0.25, 0.25, 2))
    ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.08, color="gray")

    # 目标点（逆解时用），画成空心方块，能直观看出 TCP 有没有对准
    if target is not None:
        ax.scatter([target[0]], [target[1]], [target[2]], s=90,
                   marker="s", facecolors="none", edgecolors="#c0392b",
                   depthshade=False, zorder=6)

    # 连杆
    seg_color = ["#4a4a4a", "#2e86de", "#2e86de", "#10ac84", "#10ac84"]
    for i in range(len(pts) - 1):
        p, q2 = pts[i], pts[i + 1]
        ax.plot([p[0], q2[0]], [p[1], q2[1]], [p[2], q2[2]],
                color=seg_color[i], lw=4, solid_capstyle="round")

    # 关节
    joints = np.array(pts[:-1])
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               s=45, c="black", depthshade=False, zorder=5)

    # 末端夹爪 + TCP
    draw_gripper(ax, frames, roll_deg)
    tcp = pts[-1]
    ax.scatter([tcp[0]], [tcp[1]], [tcp[2]], s=70, c="crimson",
               depthshade=False, zorder=6, label="TCP")

    # 坐标写进标题，避免压在坐标轴刻度上
    ax.set_title(f"{title}\nTCP = ({fmt(tcp[0])}, {fmt(tcp[1])}, {fmt(tcp[2])}) m", fontsize=9)
    ax.set_xlim(-0.1, 0.45)
    ax.set_ylim(-0.25, 0.25)
    ax.set_zlim(-0.05, 0.45)
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_zlabel("Z (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=22, azim=-62)


def plot_ik_demo():
    """给 4 个目标点，让逆解自己算出姿态，再用正解画出来对比。"""
    targets = [
        (0.30, 0.10, 0.20, "① 目标 (0.30, 0.10, 0.20)"),
        (0.32, -0.15, 0.25, "② 目标 (0.32, -0.15, 0.25)"),
        (0.20, 0.22, 0.12, "③ 目标 (0.20, 0.22, 0.12)"),
        (0.25, 0.00, 0.33, "④ 目标 (0.25, 0.00, 0.33)"),
    ]

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("第 1 级 · 逆解演示（方块是目标点，圆点是实际 TCP）", fontsize=13)

    for i, (x, y, z, title) in enumerate(targets, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        q, tag = solve_ik_best((x, y, z))
        if q is None:
            ax.set_title(f"{title}\n够不着", fontsize=9)
            continue
        label = {"front/up": "正对·肘上", "front/down": "正对·肘下",
                 "back/up": "背对·肘上", "back/down": "背对·肘下"}[tag]
        draw_arm(ax, np.degrees(q), 0, f"{title} · {label}", target=(x, y, z))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "level1_ik.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    self_check()
    verify_ik()
    ik_fig = plot_ik_demo()

    # 四个典型姿态：手臂伸直 / 抬起 / 前伸下压 / 转身并拧腕
    poses = [
        ([0, 0, 0, 0],      0,   "① 零位（全伸直）"),
        ([0, -35, 0, 0],    0,   "② J2 抬起 35°"),
        ([25, -55, 65, -15], 0,  "③ 前伸下压（弯肘）"),
        ([70, -20, 45, 0],  60,  "④ 转 70° + J5 腕滚 60°"),
    ]

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("第 1 级 · 正解可视化（纯数学，无物理引擎）", fontsize=13)

    for i, (q_deg, roll, title) in enumerate(poses, start=1):
        # subplot(2,2,i, projection="3d") 一次建一个 3D 子图（进阶语法：关键字参数控制投影类型）
        ax = fig.add_subplot(2, 2, i, projection="3d")
        draw_arm(ax, q_deg, roll, title)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "level1_fk.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print("--- 各姿态 TCP 坐标（米）---")
    for q_deg, _, title in poses:
        frames = forward_kinematics(np.array(q_deg) * DEG)
        tcp = frames[-1][:3, 3]
        print(f"{title:<22} x={fmt(tcp[0])}  y={fmt(tcp[1])}  z={fmt(tcp[2])}")

    print(f"\n图片已保存：{out_path}")
    print(f"逆解演示图已保存：{ik_fig}")


if __name__ == "__main__":
    main()
