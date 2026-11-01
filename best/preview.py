# -*- coding: utf-8 -*-
"""快速渲染一张静帧，用来调外观和灯光，不参与主流程。"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

import scene  # noqa: E402


def main():
    model, data = scene.compile_scene()
    print(f"nq={model.nq} nu={model.nu} ngeom={model.ngeom} "
          f"nmesh={model.nmesh} ntex={model.ntex} nmat={model.nmat}")

    # 一个好看的展示姿态：抬起、弯肘、夹爪朝下
    data.qpos[0] = 0.25
    data.qpos[1] = -0.75
    data.qpos[2] = 0.95
    data.qpos[3] = 0.55
    data.qpos[4] = 0.3
    data.qpos[5] = 0.012
    data.qpos[6] = 0.012
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.15, 0.0, 0.08]
    cam.distance = 0.52
    cam.azimuth = 122
    cam.elevation = -16

    renderer = mujoco.Renderer(model, 720, 1080)
    renderer.update_scene(data, camera=cam)
    # 关掉天空盒，背景变成纯色。默认那个浅灰渐变会抢走注意力，
    # 深色背景反而让白色机身和阴影更突出。
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
    img = renderer.render()
    renderer.close()

    out = os.path.join(ROOT, "outputs", "best_preview.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.fromarray(img).save(out)
    print("saved", out, img.shape, "mean", round(float(img.mean()), 1))


if __name__ == "__main__":
    main()
