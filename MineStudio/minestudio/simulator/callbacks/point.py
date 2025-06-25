"""
Date: 2024-11-18 20:37:50
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-24 08:23:45
FilePath: /MineStudio/minestudio/simulator/callbacks/point.py
"""

import os

import cv2
import numpy as np
from rich import print

from minestudio.simulator.callbacks import MinecraftCallback
from minestudio.simulator.utils import MinecraftGUI
from minestudio.simulator.utils.gui import MultiPointDrawCall, PointDrawCall, SegmentDrawCall


class PointCallback(MinecraftCallback):
    """
    Callback to get the pointing position from the player on a GUI window.
    """

    def __init__(self):
        super().__init__()

    def after_reset(self, sim, obs, info):
        sim.callback_messages.add("Press 'P' to start pointing.")
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        if info.get("P", False):
            print("[green]Start pointing[/green]")
        else:
            return obs, reward, terminated, truncated, info

        gui = MinecraftGUI(extra_draw_call=[PointDrawCall], show_info=False)
        gui.window.activate()

        while True:
            gui.window.dispatch_events()
            gui.window.switch_to()
            gui.window.set_mouse_visible(True)
            gui.window.set_exclusive_mouse(False)
            gui.window.flip()
            released_keys = gui._capture_all_keys()
            if "ESCAPE" in released_keys:
                break
            if gui.mouse_position is not None:
                info["point"] = gui.mouse_position
            gui._show_image(info)

        gui.close_gui()

        if info["point"] is not None:
            print(f"[red]Stop pointing at {info['point']}[/red]")
        info["P"] = False
        return obs, reward, terminated, truncated, info


class PlaySegmentCallback(MinecraftCallback):
    """
    Callback for generating segment using segment anything 2 with human
    @Notice: This callback should be put before the play callback
    """

    def __init__(self, sam_path, sam_choice="base"):
        super().__init__()
        self.sam_path = sam_path
        self._clear()
        self.sam_choice = "base"
        self._load_sam()

        # TODO: add different segment types

    def _load_sam(self):
        ckpt_mapping = {
            "large": [os.path.join(self.sam_path, "sam2_hiera_large.pt"), "sam2_hiera_l.yaml"],
            "base": [os.path.join(self.sam_path, "sam2_hiera_base_plus.pt"), "sam2_hiera_b+.yaml"],
            "small": [os.path.join(self.sam_path, "sam2_hiera_small.pt"), "sam2_hiera_s.yaml"],
            "tiny": [os.path.join(self.sam_path, "sam2_hiera_tiny.pt"), "sam2_hiera_t.yaml"],
        }
        sam_ckpt, model_cfg = ckpt_mapping[self.sam_choice]
        # first realease the old predictor
        if hasattr(self, "predictor"):
            del self.predictor
        from sam2.build_sam import build_sam2_camera_predictor

        self.predictor = build_sam2_camera_predictor(model_cfg, sam_ckpt)
        print(f"Successfully loaded SAM2 from {sam_ckpt}")
        self.able_to_track = False

    def _get_message(self, info):
        message = info.get("message", {})
        message["SegmentCallback"] = (
            f"Segment: {'On' if self.tracking else 'Off'}, Tracking Time: {self.tracking_time}"
        )
        return message

    def _clear(self):
        self.positive_points = []
        self.negative_points = []
        self.segment = None
        self.able_to_track = False
        self.tracking = False
        self.tracking_time = 0

    def after_reset(self, sim, obs, info):
        self._clear()
        sim.callback_messages.add("Press 'S' to start/stop segmenting.")
        info["message"] = self._get_message(info)
        return obs, info

    def before_step(self, sim, action):
        if sim.info.get("S", False) and not self.tracking:
            return sim.noop_action()
        return action

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        if self.tracking and (not info.get("S", False)):
            # stop tracking
            print("[red]Stop tracking[/red]")
            self._clear()
            info["segment"] = None
        elif (not self.tracking) and info.get("S", False):
            # start tracking
            print("[green]Start segmenting[/green]")
            info["segment"] = None
            info["positive_points"] = []
            info["negative_points"] = []
            info = self._segment_gui(info)
            if not self.tracking:
                info["S"] = False
        elif self.tracking and info.get("S", False):
            self.tracking_time += 1
            info["segment"] = self._segment(info)

        if info.get("segment", None) is not None and self.tracking:
            # resize the segment to the size of the obs
            segment = cv2.resize(
                info["segment"].astype(np.uint8),
                dsize=(obs["image"].shape[0], obs["image"].shape[1]),
                interpolation=cv2.INTER_NEAREST,
            )
            obs["segment"] = {}
            obs["segment"]["obj_mask"] = segment
            obs["segment"]["obj_id"] = 2
        else:
            obs["segment"] = {}
            obs["segment"]["obj_mask"] = np.zeros((obs["image"].shape[0], obs["image"].shape[1]), dtype=np.uint8)
            obs["segment"]["obj_id"] = -1

        info["message"] = self._get_message(info)
        return obs, reward, terminated, truncated, info

    def _segment_gui(self, info):
        info = info.copy()
        gui = MinecraftGUI(extra_draw_call=[SegmentDrawCall, MultiPointDrawCall], show_info=True)
        help_message = [
            ["Press 'C' to clear points."],
            ["Press mouse left button to add points."],
            ["Press mouse right button to add negative points."],
            ["Press 'Enter' to start tracking."],
            ["Press 'ESC' to exit."],
        ]

        gui.window.activate()
        refresh = False
        last_mouse_position = None

        while True:
            gui.window.dispatch_events()
            gui.window.switch_to()
            gui.window.set_mouse_visible(True)
            gui.window.set_exclusive_mouse(False)
            gui.window.flip()

            released_keys = gui._capture_all_keys()
            if "ESCAPE" in released_keys:
                self._clear()
                info["segment"] = None
                info["positive_points"] = self.positive_points
                info["negative_points"] = self.negative_points
                self.tracking = False
                print("[red]Exit segmenting[/red]")
                break

            if "C" in released_keys:
                self._clear()
                info["segment"] = None
                info["positive_points"] = self.positive_points
                info["negative_points"] = self.negative_points
                last_mouse_position = None
                refresh = True
                print("[red]Points cleared[/red]")

            if "ENTER" in released_keys and self.able_to_track:
                assert info["segment"] is not None, "segment is not generated."
                print("[green]Start tracking[/green]")
                self.tracking = True
                break

            if gui.mouse_position is not None:
                if gui.mouse_pressed == 1 or gui.mouse_pressed == 4:
                    if gui.mouse_position != last_mouse_position:
                        last_mouse_position = gui.mouse_position
                        position = (
                            last_mouse_position[0],
                            gui.constants.FRAME_HEIGHT + gui.constants.INFO_HEIGHT - last_mouse_position[1],
                        )
                        # resize position to the size of the pov
                        position = (
                            int(position[0] * info["pov"].shape[1] / gui.constants.WINDOW_WIDTH),
                            int(position[1] * info["pov"].shape[0] / gui.constants.FRAME_HEIGHT),
                        )
                        # remember position is W * H

                        if gui.mouse_pressed == 1:
                            # left button pressed
                            self.positive_points.append(position)
                            info["positive_points"] = self.positive_points
                            print(f"[green]Positive point added at {position}[/green]")
                            refresh = True
                        elif gui.mouse_pressed == 4:
                            # right button pressed
                            self.negative_points.append(position)
                            info["negative_points"] = self.negative_points
                            print(f"[red]Negative point added at {position}[/red]")
                            refresh = True
                    gui.mouse_pressed = 0

            if len(self.positive_points) > 0:
                self.able_to_track = True

            if self.able_to_track:
                self._segment(info, refresh)
                info["segment"] = self.segment
                refresh = False

            gui._update_image(
                info,
                message=help_message,
                remap_points=(
                    gui.constants.WINDOW_WIDTH,
                    info["pov"].shape[1],
                    gui.constants.FRAME_HEIGHT,
                    info["pov"].shape[0],
                ),
            )

        gui.close_gui()
        return info

    def _segment(self, info, refresh=False):
        if (self.segment is None) or refresh:
            assert len(self.positive_points) > 0
            points = self.positive_points + self.negative_points
            self.predictor.load_first_frame(info["pov"])
            _, out_obj_ids, out_segment_logits = self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=0,
                points=points,
                labels=[1] * len(self.positive_points) + [0] * len(self.negative_points),
            )
        else:
            out_obj_ids, out_segment_logits = self.predictor.track(info["pov"])
        self.segment = (out_segment_logits[0, 0] > 0.0).cpu().numpy()  # 360 * 640
        return self.segment
