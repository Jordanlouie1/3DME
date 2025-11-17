from __future__ import annotations

"""Interactive viewer to nudge left/right scans closer or farther apart.

Usage:
    python interactive_offset.py --left aligned_scans/aligned_left1.ply \
        --right aligned_scans/aligned_right1.ply --range 0.0 0.2 \
        --save offset_choice.json

Drag the slider to change the offset. Click "Save offset" to write the current
value (in meters) to JSON so you can reuse it in `align_to_model.py` via the
`--manual-translate` option.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True, help="Path to left/aligned scan")
    parser.add_argument("--right", type=Path, required=True, help="Path to right/aligned scan")
    parser.add_argument(
        "--range",
        nargs=2,
        type=float,
        default=(0.0, 0.2),
        help="Slider limits in meters (min max). Use positive numbers.",
    )
    parser.add_argument(
        "--initial-offset",
        type=float,
        default=0.05,
        help="Initial offset distance between clouds (meters).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("offset_choice.json"),
        help="Where to write the selected offset when clicking 'Save offset'.",
    )
    return parser.parse_args()


@dataclass
class GeometryRecord:
    name: str
    base_center: np.ndarray


class OffsetViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.left = o3d.io.read_point_cloud(str(args.left))
        self.right = o3d.io.read_point_cloud(str(args.right))
        if self.left.is_empty() or self.right.is_empty():
            raise SystemExit("One of the scans is empty. Ensure the inputs are valid PLY files.")

        self.app = gui.Application.instance
        self.window = self.app.create_window("Left/Right Offset Viewer", 1280, 900)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.enable_scene_caching(True)
        self.window.add_child(self.scene_widget)

        self.panel = gui.Vert(0.25 * gui.theme.font_size, gui.Margins(10, 10, 10, 10))
        self.window.add_child(self.panel)

        self.slider = gui.Slider(gui.Slider.DOUBLE)
        low, high = args.range
        self.slider.set_limits(low, high)
        self.slider.double_value = np.clip(args.initial_offset, low, high)
        self.slider.set_on_value_changed(self._on_slider)

        self.label = gui.Label(f"Offset: {self.slider.double_value:.3f} m")
        self.save_button = gui.Button("Save offset")
        self.save_button.set_on_clicked(self._on_save)

        self.panel.add_child(gui.Label("Adjust spacing between clouds"))
        self.panel.add_child(self.slider)
        self.panel.add_child(self.label)
        self.panel.add_fixed(0.5 * gui.theme.font_size)
        self.panel.add_child(self.save_button)

        self._setup_scene()
        self._apply_offset(self.slider.double_value)

    def _setup_scene(self) -> None:
        center = (self.left.get_center() + self.right.get_center()) / 2.0
        self.scene_widget.scene.scene.set_sun_light(
            [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 75000, True
        )
        self.scene_widget.scene.scene.enable_sun_light(True)

        self.left_record = self._add_geometry(self.left, "left", [1.0, 0.2, 0.2])
        self.right_record = self._add_geometry(self.right, "right", [0.2, 0.6, 1.0])
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60.0, bounds, center)
        self.scene_widget.scene.show_axes(True)

    def _add_geometry(self, pcd: o3d.geometry.PointCloud, name: str, color) -> GeometryRecord:
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 2.0
        pcd_colored = o3d.geometry.PointCloud(pcd)
        pcd_colored.paint_uniform_color(color)
        self.scene_widget.scene.add_geometry(name, pcd_colored, material)
        return GeometryRecord(name=name, base_center=np.asarray(pcd.get_center()))

    def _on_slider(self, value: float) -> None:
        self.label.text = f"Offset: {value:.3f} m"
        self._apply_offset(value)

    def _apply_offset(self, offset: float) -> None:
        left_tf = np.eye(4)
        right_tf = np.eye(4)
        left_tf[0, 3] = -offset / 2.0
        right_tf[0, 3] = offset / 2.0
        self.scene_widget.scene.set_geometry_transform(self.left_record.name, left_tf)
        self.scene_widget.scene.set_geometry_transform(self.right_record.name, right_tf)
        self.scene_widget.force_redraw()

    def _on_save(self) -> None:
        value = self.slider.double_value
        data = {
            "offset_meters": value,
            "manual_translate": {
                Path(self.args.left).stem: [-value / 2.0, 0.0, 0.0],
                Path(self.args.right).stem: [value / 2.0, 0.0, 0.0],
            },
        }
        self.args.save.parent.mkdir(parents=True, exist_ok=True)
        with self.args.save.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        gui.Application.instance.post_to_main_thread(
            self.window,
            lambda: gui.Dialog.show_message("Saved", f"Offset {value:.3f} m saved to {self.args.save}"),
        )


def main() -> None:
    args = parse_args()
    gui.Application.instance.initialize()
    OffsetViewer(args)
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
