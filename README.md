# SO-101 五轴机械臂仿真

在 MuJoCo 里从零搭一台 5 轴舵机机械臂的仿真：DH 参数 → 正逆解 → 物理验证 → 闭环抓取 → 成功率统计。

每个阶段都配了可复现的数值验证，所以这不是"看起来能动"的演示，而是一条能逐级定位错误的调试链。

## 为什么这样组织

把同一台臂分成三级来建，每级只增加一个变量：

| 级别 | 内容 | 只增加的那一个变量 |
|---|---|---|
| 1 | 纯数学正解 / 逆解 + matplotlib 线框 | —— |
| 2 | MuJoCo 建模，关重力，测跟随 | 物理引擎 |
| 3 | 开重力 → 加夹爪 → 加物体 → 闭环抓取 | 重力 / 接触 |

这样出问题时嫌疑人只有一个。一上来就"开重力 + 接触 + 强化学习"是初学者最常见的失败方式：90% 的时间在调参，不是在学东西。

## 快速开始

需要 Python 3.10+。

```bash
pip install -r requirements.txt
```

```bash
python src/level1_fk.py      # 正解/逆解自检 + 两张 3D 线框图
python src/level2_mujoco.py  # MJCF 与 DH 表的跨校验 + 跟随曲线
python src/level3_grasp.py   # 开重力后的定位误差标定
python src/grasp_scene.py    # 夹爪模型自检（开合精度、耦合误差）

python src/sim.py            # 完整抓取仿真，弹实时窗口，动作结束后自动重来
python src/sim.py --record   # 无窗口录制 + 输出高度曲线

python src/eval_grasp.py --preset mild  --trials 3   # 成功率（定位误差 ±2.5 mm）
python src/eval_grasp.py --preset harsh --trials 3   # 成功率（定位误差 ±8 mm）
python src/eval_grasp.py --sweep --trials 3          # 定位误差 vs 成功率曲线

python src/view3d.py                       # 3D 可视化（可拖视角）
python src/view3d.py --model grasp         # 带夹爪的版本
python src/view3d.py --mode reach --record # 逆解画圆并录像
```

所有图片和视频输出到 `outputs/`（已加入 .gitignore）。

## 项目结构

```
src/
  level1_fk.py     改进 DH 参数表、正解、解析逆解（4 分支）、matplotlib 可视化
  level2_mujoco.py MJCF 与 DH 的跨校验、关重力下的正弦跟随与阶跃响应
  level3_grasp.py  开重力后的稳态误差、重力力矩、末端偏差统计
  grasp_scene.py   第 3 级场景：夹爪、地面、立方体、接触分组、灯光
  grasp_ik.py      带腕部姿态的数值逆解（固定 q4，阻尼最小二乘解 q1–q3）
  sim.py           完整抓取动作：接近 → 下降 → 合爪 → 抬起 → 成败判定
  eval_grasp.py    成功率统计：位置网格 × 参数扰动，含失败模式分类
  view3d.py        实时窗口 / 离屏录制 / 静态截图
models/
  so101.xml        5 关节 MJCF（关重力、无接触，作为第 2 级的基准模型）
best/              高质量版本（结构组装 + PBR 材质 + 非理想执行器）
  scene.py         场景：纹理、桌面、灯光、相机
  robot.py         SO-101 结构组装（94 个零件，运动学不变）
  parts.py         零件库：舵机、螺栓、法兰、镂空梁、线缆
  meshgen.py       程序化网格生成（超椭球、圆柱、圆环、棱柱）
  textures.py      程序化纹理（层纹、木纹、金属）
  control.py       轨迹、舵机模型、抓取状态机
  demo.py          主程序
data/              预留：将来放示教数据与实验记录
outputs/           运行产物（自动生成，不入库）
```

## 高质量版本（best/）

`src/` 是"把物理跑通"的最小闭环：box 几何、理想执行器。数字可信，画面像示意图。

`best/` 是同一套运动学上的"看得过去"的版本：

```bash
python best/demo.py            # 实时窗口
python best/demo.py --record   # 录 1080p 视频
python best/demo.py --still    # 渲一张静帧
```

- 几何：程序化超椭球圆角网格（不是原生 box）
- 材质：程序化纹理（3D 打印层纹、木纹、拉丝金属）+ PBR
- 光照：三点布光、4096 阴影贴图、8× 抗锯齿
- 执行器：延迟 12 ms、背隙 7 mrad、编码器量化 —— 成功率会因此下降，那才是真机的数字

细节和踩坑记录见 [`best/README.md`](best/README.md)。

## 结果

抓的是 3 cm 立方体，物体放在桌面上，臂从上方垂直下探夹住再抬起。

| 条件 | 成功率 |
|---|---|
| 定位误差 ±2.5 mm，物体 30 g | 100%（75/75） |
| 定位误差 ±8 mm，物体 15–60 g，摩擦 0.4–1.3 | 83%（62/75） |

定位误差扫描：0–6 mm 全过，8 mm 92%，10 mm 67%，15 mm 42%。

归因实验（单变量）：

| 变量 | 结果 |
|---|---|
| 物体质量 15 g / 30 g / 60 g | 12/12 · 10/12 · 9/12 |
| 摩擦系数 0.4 / 0.7 / 1.0 / 1.3 | 9/12 · 9/12 · 9/12 · 12/12 |

早期版本的瓶颈是**夹持力**：夹爪刚度只有 40、力限 3 N 时，60 g 的物体一次都抓不起来（0/12）。
把刚度提到 300、力限提到 8 N 之后到了 9/12，质量不再是主要瓶颈，剩下的失败集中在大定位误差下。

摩擦系数的影响很小 —— 说明限制因素一直是法向夹持力，不是表面材质。

## 几个关键设计

**不修改基础模型，场景用 MjSpec 在内存里追加。**
`models/so101.xml` 是第 2 级的基准（关重力、5 关节）。直接往里加夹爪会让它的 `ctrl` 维度从 5 变 6，之前所有验证作废。而夹爪必须挂在嵌套的 `tool` body 下，XML 的 `include` 做不到往已有 body 里追加子元素，所以用 `MjSpec` 在编译前追加。

**接触用位掩码分组，而不是全局开关。**
指尖 `contype=1/conaffinity=2`，被抓物 `contype=2/conaffinity=3`。这样指尖↔物体会碰、指尖↔指尖不碰（否则两指合拢时会被自己的接触力顶开）、连杆↔任何东西都不碰。

**逆解要能控制腕部姿态。**
解析逆解把 J4 固定在 0，位置算得对，但夹爪永远顺着前臂方向伸出去。这个臂底座只有 95 mm，够桌面上 30 mm 的物体时夹爪会斜 47° 杵下去，指尖一端先磕桌面。`grasp_ik.py` 改成数值解：固定 q4，解 q1–q3，再扫 q4 网格挑"夹爪最朝下"的姿态。

**成功率实验里，规划位置和实际放置位置必须是两个参数。**
合在一起等于假设机械臂知道物体的精确位置，测出来的成功率会虚高。真机上你永远不知道精确位置。

## 依赖

- Python 3.10+
- MuJoCo 3.2+（本项目在 3.17.0 上验证）
- numpy / matplotlib / pillow
- 可选：imageio + imageio-ffmpeg（输出 mp4，否则退回 gif）

不需要显卡，纯 CPU 可跑。

## 几何来源

机械臂的连杆尺寸（底座到肩轴 95 mm、肩侧偏 30 mm、大臂 116 mm、小臂 135 mm、腕到 TCP 110 mm）取自开源项目 [SO-101](https://github.com/huggingface/lerobot)（Hugging Face LeRobot，Apache-2.0）的公开 URDF，按实测几何简化成 DH 参数与 MJCF。夹爪是简化的平行两指模型，不对应具体实物结构。

## 许可证

MIT，见 [LICENSE](LICENSE)。机械臂几何尺寸来自 SO-101（Hugging Face LeRobot，Apache-2.0）。

## 局限

- 只有位置控制，没有力控；夹爪的夹持力靠一个固定的力上限近似
- 视觉部分尚未接入：物体位置是直接给定的，没有相机、检测和手眼标定
- 执行器是理想位置伺服，没有延迟、背隙和温漂
- 只测了单个立方体、桌面平放、单一抓取姿态
- 所有数字都来自仿真，没有真机数据
