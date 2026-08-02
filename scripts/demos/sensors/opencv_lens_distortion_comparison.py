#!/usr/bin/env python3
"""Render a cube grid with OVRTX and Newton under three OpenCV lens models."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 640, 480
SIM_DT = 1.0 / 60.0
WARMUP_STEPS = 5
DEVICE = "cuda:0"
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"

CALIBRATION = dict(fx=430.0, fy=430.0, cx=WIDTH / 2.0, cy=HEIGHT / 2.0)
PINHOLE_COEFFICIENTS = dict(
    k1=-0.38,
    k2=0.16,
    k3=-0.035,
    k4=0.0,
    k5=0.0,
    k6=0.0,
    p1=0.008,
    p2=-0.008,
    s1=0.0,
    s2=0.0,
    s3=0.0,
    s4=0.0,
)
FISHEYE_COEFFICIENTS = dict(k1=0.08, k2=-0.025, k3=0.003, k4=0.0)


def distortion_cfg(model: str):
    """Return a common calibration with the requested lens model."""
    from isaaclab.sim.spawners.sensors.sensors_cfg import (
        OpenCvFisheyeDistortionCfg,
        OpenCvPinholeDistortionCfg,
    )

    common = dict(image_size=(WIDTH, HEIGHT), **CALIBRATION)
    if model == "none":
        return OpenCvPinholeDistortionCfg(
            apply_lens_distortion=False,
            **common,
            **PINHOLE_COEFFICIENTS,
        )
    if model == "fisheye":
        return OpenCvFisheyeDistortionCfg(
            apply_lens_distortion=True,
            **common,
            **FISHEYE_COEFFICIENTS,
        )
    if model == "opencv":
        return OpenCvPinholeDistortionCfg(
            apply_lens_distortion=True,
            **common,
            **PINHOLE_COEFFICIENTS,
        )
    raise ValueError(f"Unknown model: {model}")


def make_scene_cfg():
    """Create a static checkerboard-like grid of colored cubes."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg

    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=20.0)
    cfg.ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.12, 0.12, 0.12)),
    )
    cfg.light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3200.0, color=(0.95, 0.95, 0.95)),
    )

    palette = (
        (0.95, 0.12, 0.10),
        (0.10, 0.75, 0.20),
        (0.10, 0.35, 0.95),
        (0.95, 0.75, 0.08),
        (0.75, 0.12, 0.90),
        (0.05, 0.85, 0.85),
    )
    grid_size = 7
    spacing = 0.78
    cube_size = 0.55
    center = (grid_size - 1) / 2.0
    for row in range(grid_size):
        for col in range(grid_size):
            x = (col - center) * spacing
            y = (row - center) * spacing
            color = palette[(row + 2 * col) % len(palette)]
            cube = AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Cube_{row}_{col}",
                spawn=sim_utils.CuboidCfg(
                    size=(cube_size, cube_size, 0.30),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=color,
                        metallic=0.05,
                        roughness=0.55,
                    ),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, 0.15)),
            )
            setattr(cfg, f"cube_{row}_{col}", cube)

    cfg.anchor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Anchor",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
    )
    return cfg


def renderer_cfg(renderer: str):
    if renderer == "ovrtx":
        from isaaclab_ov.renderers import OVRTXRendererCfg

        return OVRTXRendererCfg()
    if renderer == "newton":
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        return NewtonWarpRendererCfg()
    raise ValueError(f"Unknown renderer: {renderer}")


def to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    """Normalize a renderer's RGB output into an H×W×3 uint8 image."""
    array = np.asarray(array)[..., :3]
    if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.5:
        array = array * 255.0
    return np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0).clip(0, 255).astype(np.uint8)


def depth_path_for(rgb_path: Path) -> Path:
    return rgb_path.with_name(f"{rgb_path.stem}_depth.npy")


def colorize_depth(depth: np.ndarray, near: float, far: float) -> Image.Image:
    """Colorize depth consistently: near is yellow, far is dark purple."""
    depth = np.asarray(depth).squeeze()
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = 1.0 - np.clip((depth[valid] - near) / max(far - near, 1e-6), 0.0, 1.0)

    # Compact inferno-like color ramp without adding a plotting dependency.
    stops = np.array(
        [
            [0.0, 0.0, 4.0],
            [40.0, 11.0, 84.0],
            [101.0, 21.0, 110.0],
            [159.0, 42.0, 99.0],
            [212.0, 72.0, 66.0],
            [245.0, 125.0, 21.0],
            [252.0, 255.0, 164.0],
        ],
        dtype=np.float32,
    )
    position = normalized * (len(stops) - 1)
    lower = np.floor(position).astype(np.int32)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (position - lower)[..., None]
    rgb = stops[lower] * (1.0 - fraction) + stops[upper] * fraction
    rgb[~valid] = 0.0
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8), mode="RGB")


def render_one(renderer: str, model: str, modality: str, output_path: Path) -> None:
    """Render one renderer/model/modality combination in an isolated process."""
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
    from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(
        SimulationCfg(
            dt=SIM_DT,
            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1),
            device=DEVICE,
        )
    )
    scene = InteractiveScene(make_scene_cfg())
    camera_eye = (0.0, -0.8, 5.2)
    camera_target = (0.0, 0.0, 0.0)
    camera_rotation = tuple(
        quat_from_matrix(
            create_rotation_matrix_from_view(
                torch.tensor([camera_eye]),
                torch.tensor([camera_target]),
                up_axis="Z",
            )
        )[0].tolist()
    )
    camera = Camera(
        CameraCfg(
            prim_path="/World/envs/env_.*/Camera",
            update_period=0.0,
            height=HEIGHT,
            width=WIDTH,
            data_types=["rgb" if modality == "rgb" else "distance_to_camera"],
            offset=CameraCfg.OffsetCfg(pos=camera_eye, rot=camera_rotation, convention="opengl"),
            spawn=PinholeCameraCfg(
                focal_length=20.0,
                clipping_range=(0.01, 20.0),
                distortion=distortion_cfg(model),
            ),
            renderer_cfg=renderer_cfg(renderer),
        )
    )

    try:
        sim.reset()
        for _ in range(WARMUP_STEPS):
            sim.step()
            camera.update(SIM_DT, force_recompute=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if modality == "rgb":
            rgb = camera.data.output["rgb"].torch[0].detach().cpu().numpy().copy()
            Image.fromarray(to_uint8_rgb(rgb), mode="RGB").save(output_path)
        else:
            depth = camera.data.output["distance_to_camera"].torch[0].detach().cpu().float().numpy().copy()
            np.save(output_path, depth)
        print(f"saved {renderer}/{model}/{modality}: {output_path}")
    finally:
        del camera
        del scene
        sim.stop()
        sim.clear_instance()


def load_font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def compose(
    rgb_paths: dict[tuple[str, str], Path],
    depth_paths: dict[tuple[str, str], Path],
) -> Path:
    """Compose RGB and depth into a labeled 3×4 comparison."""
    models = ("none", "fisheye", "opencv")
    renderers = ("ovrtx", "newton")
    depths = {key: np.load(path) for key, path in depth_paths.items()}
    valid_values = np.concatenate(
        [depth[np.isfinite(depth) & (depth > 0.0)].ravel() for depth in depths.values()]
    )
    near, far = np.percentile(valid_values, [1.0, 99.0])

    title_h, label_h = 58, 42
    canvas = Image.new("RGB", (3 * WIDTH, title_h + 4 * (HEIGHT + label_h)), (25, 25, 28))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    label_font = load_font(21)
    title = "OpenCV Lens Distortion — OVRTX vs Newton — RGB and Depth"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((canvas.width - (title_box[2] - title_box[0])) / 2, 12), title, fill="white", font=title_font)

    model_titles = {"none": "No distortion", "fisheye": "OpenCV fisheye", "opencv": "OpenCV pinhole"}
    for renderer_index, renderer in enumerate(renderers):
        for modality_index, modality in enumerate(("RGB", "Depth")):
            row = renderer_index * 2 + modality_index
            for col, model in enumerate(models):
                key = (renderer, model)
                x = col * WIDTH
                y = title_h + row * (HEIGHT + label_h)
                if modality == "RGB":
                    image = Image.open(rgb_paths[key]).convert("RGB")
                else:
                    image = colorize_depth(depths[key], float(near), float(far))
                    image.save(rgb_paths[key].with_name(f"{rgb_paths[key].stem}_depth.png"))
                canvas.paste(image, (x, y + label_h))
                label = f"{renderer.upper()} {modality} — {model_titles[model]}"
                box = draw.textbbox((0, 0), label, font=label_font)
                draw.text(
                    (x + (WIDTH - (box[2] - box[0])) / 2, y + 8),
                    label,
                    fill=(245, 245, 245),
                    font=label_font,
                )

    output = OUTPUT_DIR / "opencv_distortion_comparison.png"
    canvas.save(output)
    return output


def run_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rgb_paths: dict[tuple[str, str], Path] = {}
    depth_paths: dict[tuple[str, str], Path] = {}
    for renderer in ("ovrtx", "newton"):
        for model in ("none", "fisheye", "opencv"):
            key = (renderer, model)
            rgb_paths[key] = OUTPUT_DIR / f"{renderer}_{model}.png"
            depth_paths[key] = depth_path_for(rgb_paths[key])
            for modality, output in (("rgb", rgb_paths[key]), ("depth", depth_paths[key])):
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        renderer,
                        model,
                        modality,
                        str(output),
                    ],
                    check=True,
                )
    comparison = compose(rgb_paths, depth_paths)
    print(f"\ncomparison: {comparison}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=4, metavar=("RENDERER", "MODEL", "MODALITY", "OUTPUT"))
    args = parser.parse_args()
    if args.worker:
        renderer, model, modality, output = args.worker
        render_one(renderer, model, modality, Path(output))
    else:
        run_all()


if __name__ == "__main__":
    main()
