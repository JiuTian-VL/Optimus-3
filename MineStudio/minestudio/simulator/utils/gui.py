"""
Date: 2024-11-15 15:15:22
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2024-11-20 01:02:18
FilePath: /MineStudio/minestudio/simulator/utils/gui.py
"""

from minestudio.simulator.utils.constants import GUIConstants

from collections import defaultdict
from typing import List, Any, Optional, Callable
import importlib
import cv2
import time
from rich import print
import numpy as np


def RecordDrawCall(info, **kwargs):
    if "R" not in info.keys() or info.get("ESCAPE", False):
        return info
    recording = info["R"]
    if not recording:
        return info
    arr = info["pov"]
    if int(time.time()) % 2 == 0:
        cv2.circle(arr, (20, 20), 10, (255, 0, 0), -1)
        cv2.putText(arr, "Rec", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.circle(arr, (20, 20), 10, (0, 255, 0), -1)
        cv2.putText(arr, "Rec", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    info["pov"] = arr
    return info


def CommandModeDrawCall(info, **kwargs):
    if "ESCAPE" not in info.keys():
        return info
    mode = info["ESCAPE"]
    if not mode:
        return info
    # Draw a grey overlay on the screen
    arr = info["pov"]
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    cv2.putText(arr, "Command Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    info["pov"] = arr
    return info


def PointDrawCall(info, **kwargs):
    if "point" not in info.keys():
        return info
    point = info["point"]
    arr = info["pov"]
    # draw a red circle at the point, the position is relative to the bottom-left corner of arr
    cv2.circle(arr, (point[0], arr.shape[0] - point[1]), 10, (0, 0, 255), -1)
    cv2.putText(arr, f"Pointing at {point}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    info["pov"] = arr
    return info


def MultiPointDrawCall(info, **kwargs):
    if "positive_points" not in info.keys() or "negative_points" not in info.keys():
        return info
    positive_points = info["positive_points"]
    negative_points = info["negative_points"]
    if len(positive_points) == 0:
        return info
    arr = info["pov"]
    remap_points = kwargs.get("remap_points", (1, 1, 1, 1))
    for point in positive_points:
        point = (int(point[0] * remap_points[0] / remap_points[1]), int(point[1] * remap_points[2] / remap_points[3]))
        cv2.circle(arr, (point[0], point[1]), 10, (0, 255, 0), -1)

    for point in negative_points:
        point = (int(point[0] * remap_points[0] / remap_points[1]), int(point[1] * remap_points[2] / remap_points[3]))
        cv2.circle(arr, (point[0], point[1]), 10, (255, 0, 0), -1)

    info["pov"] = arr
    return info


def SegmentDrawCall(info, **kwargs):
    if "segment" not in info.keys():
        return info
    mask = info["segment"]
    if mask is None:
        return info
    arr = info["pov"]
    color = (0, 255, 0)
    color = np.array(color).reshape(1, 1, 3)[:, :, ::-1]
    mask = (mask[..., None] * color).astype(np.uint8)
    # resize the mask to the size of the obs
    mask = cv2.resize(mask, dsize=(arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_CUBIC)
    arr = cv2.addWeighted(arr, 1.0, mask, 0.5, 0.0)
    info["pov"] = arr
    return info


class MinecraftGUI:
    def __init__(self, extra_draw_call: List[Callable] = None, show_info=True, **kwargs):
        super().__init__(**kwargs)
        self.constants = GUIConstants()
        self.pyglet = importlib.import_module("pyglet")
        self.imgui = importlib.import_module("imgui")
        self.key = importlib.import_module("pyglet.window.key")
        self.mouse = importlib.import_module("pyglet.window.mouse")
        self.PygletRenderer = importlib.import_module("imgui.integrations.pyglet").PygletRenderer
        self.extra_draw_call = extra_draw_call
        self.show_info = show_info
        self.mode = "normal"
        self.create_window()

    def create_window(self):
        if self.show_info:
            self.window = self.pyglet.window.Window(
                width=self.constants.WINDOW_WIDTH,
                height=self.constants.INFO_HEIGHT + self.constants.FRAME_HEIGHT,
                vsync=False,
                resizable=False,
            )
        else:
            self.window = self.pyglet.window.Window(
                width=self.constants.WINDOW_WIDTH, height=self.constants.FRAME_HEIGHT, vsync=False, resizable=False
            )
        self.imgui.create_context()
        self.imgui.get_io().display_size = self.constants.WINDOW_WIDTH, self.constants.WINDOW_HEIGHT
        self.renderer = self.PygletRenderer(self.window)
        self.pressed_keys = defaultdict(lambda: False)
        self.released_keys = defaultdict(lambda: False)
        self.modifiers = None
        self.window.on_mouse_motion = self._on_mouse_motion
        self.window.on_mouse_drag = self._on_mouse_drag
        self.window.on_key_press = self._on_key_press
        self.window.on_key_release = self._on_key_release
        self.window.on_mouse_press = self._on_mouse_press
        self.window.on_mouse_release = self._on_mouse_release
        self.window.on_activate = self._on_window_activate
        self.window.on_deactivate = self._on_window_deactivate

        self.last_pov = None
        self.last_mouse_delta = [0, 0]
        self.capture_mouse = True
        self.mouse_position = None
        self.mouse_pressed = None
        self.chat_message = None
        self.command = None

        self.window.dispatch_events()
        self.window.switch_to()
        self.window.flip()
        self.window.clear()

        self._show_message("Waiting for start.")

    def _on_key_press(self, symbol, modifiers):
        self.pressed_keys[symbol] = True
        self.modifiers = modifiers

    def _on_key_release(self, symbol, modifiers):
        self.pressed_keys[symbol] = False
        self.released_keys[symbol] = True
        self.modifiers = modifiers

    def _on_mouse_press(self, x, y, button, modifiers):
        self.pressed_keys[button] = True
        self.mouse_pressed = button
        self.mouse_position = (x, y)

    def _on_mouse_release(self, x, y, button, modifiers):
        self.pressed_keys[button] = False

    def _on_window_activate(self):
        self.window.set_mouse_visible(False)
        self.window.set_exclusive_mouse(True)

    def _on_window_deactivate(self):
        self.window.set_mouse_visible(True)
        self.window.set_exclusive_mouse(False)

    def _on_mouse_motion(self, x, y, dx, dy):
        # Inverted
        self.last_mouse_delta[0] -= dy * self.constants.MOUSE_MULTIPLIER
        self.last_mouse_delta[1] += dx * self.constants.MOUSE_MULTIPLIER

    def _on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # Inverted
        self.last_mouse_delta[0] -= dy * self.constants.MOUSE_MULTIPLIER
        self.last_mouse_delta[1] += dx * self.constants.MOUSE_MULTIPLIER

    def _show_message(self, text):
        document = self.pyglet.text.document.FormattedDocument(text)
        document.set_style(0, len(document.text), dict(font_name="Arial", font_size=32, color=(255, 255, 255, 255)))
        document.set_paragraph_style(0, 100, dict(align="center"))
        layout = self.pyglet.text.layout.TextLayout(
            document,
            width=self.window.width // 2,
            height=self.window.height // 2,
            multiline=True,
            wrap_lines=True,
        )
        layout.update(x=self.window.width // 2, y=self.window.height // 2)
        layout.anchor_x = "center"
        layout.anchor_y = "center"
        layout.content_valign = "center"
        layout.draw()

        self.window.flip()

    def _show_additional_message(self, message: List):
        if len(message) == 0:
            return
        line_height = self.constants.INFO_HEIGHT // len(message)
        y = line_height // 2
        for i, row in enumerate(message):
            line = " | ".join(row)
            self.pyglet.text.Label(
                line,
                font_size=7 * self.constants.SCALE,
                x=self.window.width // 2,
                y=y,
                anchor_x="center",
                anchor_y="center",
            ).draw()
            y += line_height

    def _update_image(self, info, message: List = [], **kwargs):
        self.window.switch_to()
        self.window.clear()
        # Based on scaled_image_display.py
        info = info.copy()
        arr = info["pov"]
        arr = cv2.resize(
            arr, dsize=(self.constants.WINDOW_WIDTH, self.constants.FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC
        )  # type: ignore
        info["pov"] = arr

        if self.extra_draw_call is not None:
            for draw_call in self.extra_draw_call:
                info = draw_call(info, **kwargs)

        arr = info["pov"]
        image = self.pyglet.image.ImageData(arr.shape[1], arr.shape[0], "RGB", arr.tobytes(), pitch=arr.shape[1] * -3)
        texture = image.get_texture()
        texture.blit(0, self.constants.INFO_HEIGHT)

        if self.show_info:
            self._show_additional_message(message)

        self.imgui.new_frame()

        self.imgui.begin("Chat", False, self.imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        changed, command = self.imgui.input_text("Message", "")
        self.command = command
        if self.imgui.button("Send"):
            self.chat_message = command
            self.command = None
        self.imgui.end()

        self.imgui.render()
        self.renderer.render(self.imgui.get_draw_data())
        self.window.flip()

    def _show_image(self, info, **kwargs):
        self.window.switch_to()
        self.window.clear()
        info = info.copy()
        arr = info["pov"]
        arr = cv2.resize(
            arr, dsize=(self.constants.WINDOW_WIDTH, self.constants.FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC
        )
        info["pov"] = arr
        if self.extra_draw_call is not None:
            for draw_call in self.extra_draw_call:
                info = draw_call(info, **kwargs)
        arr = info["pov"]
        image = self.pyglet.image.ImageData(arr.shape[1], arr.shape[0], "RGB", arr.tobytes(), pitch=arr.shape[1] * -3)
        texture = image.get_texture()
        texture.blit(0, 0)
        self.window.flip()

    def _get_human_action(self):
        """Read keyboard and mouse state for a new action"""
        # Keyboard actions
        action: dict[str, Any] = {
            name: int(self.pressed_keys[key]) for name, key in self.constants.MINERL_ACTION_TO_KEYBOARD.items()
        }

        if not self.capture_mouse:
            self.last_mouse_delta = [0, 0]
        action["camera"] = self.last_mouse_delta
        self.last_mouse_delta = [0, 0]
        return action

    def reset_gui(self):
        self.window.clear()
        self.pressed_keys = defaultdict(lambda: False)
        self._show_message("Resetting environment...")

    def _capture_all_keys(self):
        released_keys = set()
        for key in self.released_keys.keys():
            if self.released_keys[key]:
                self.released_keys[key] = False
                released_keys.add(self.key.symbol_string(key))
        return released_keys

    def close_gui(self):
        #! WARNING: This should be checked
        self.window.close()
        self.pyglet.app.exit()


if __name__ == "__main__":
    gui = MinecraftGUI()
