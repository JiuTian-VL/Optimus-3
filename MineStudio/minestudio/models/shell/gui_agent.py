from typing import (
    Sequence, List, Mapping, Dict,
    Callable, Any, Tuple, Optional, Union
)
import copy
import random
import math
import os
import re
import json
import numpy as np
import time
from minestudio.simulator.entry import MinecraftSim

CAMERA_SCALER = 360.0 / 2400.0
MU20_BIN21_CURSOR = [7.24403611,5.21143765,3.7123409,2.60671619,1.79128785,1.18988722,0.74633787,0.41920814,0.17794105,0]
BASE_WIDTH, BASE_HEIGHT = 640, 360

# compute slot position
KEY_POS_INVENTORY_WO_RECIPE = {
    'resource_slot': {
        'left-top': (329, 114),
        'right-bottom': (365, 150),
        'row': 2,
        'col': 2,
        'prefix': 'resource',
        'start_id': 0,
    },
    'result_slot': {
        'left-top': (385, 124),
        'right-bottom': (403, 142),
        'row': 1,
        'col': 1,
        'prefix': 'result',
        'start_id': 0,
    },
    'hotbar_slot': {
        'left-top': (239, 238),
        'right-bottom': (401, 256),
        'row': 1,
        'col': 9,
        'prefix': 'inventory',
        'start_id': 0,
    },
    'inventory_slot': {
        'left-top': (239, 180),
        'right-bottom': (401, 234),
        'row': 3,
        'col': 9,
        'prefix': 'inventory',
        'start_id': 9,
    },
    'recipe_slot': {
        'left-top': (336, 158),
        'right-bottom': (356, 176),
        'row': 1,
        'col': 1,
        'prefix': 'recipe',
        'start_id': 0,
    }
}
KEY_POS_TABLE_WO_RECIPE = {
    'resource_slot': {
        'left-top': (261, 113),
        'right-bottom': (315, 167),
        'row': 3,
        'col': 3,
        'prefix': 'resource',
        'start_id': 0,
    },
    'result_slot': {
        'left-top': (351, 127),
        'right-bottom': (377, 153),
        'row': 1,
        'col': 1,
        'prefix': 'result',
        'start_id': 0,
    },
    'hotbar_slot': {
        'left-top': (239, 238),
        'right-bottom': (401, 256),
        'row': 1,
        'col': 9,
        'prefix': 'inventory',
        'start_id': 0,
    },
    'inventory_slot': {
        'left-top': (239, 180),
        'right-bottom': (401, 234),
        'row': 3,
        'col': 9,
        'prefix': 'inventory',
        'start_id': 9,
    },
    'recipe_slot': {
        'left-top': (237, 131),
        'right-bottom': (257, 149),
        'row': 1,
        'col': 1,
        'prefix': 'recipe',
        'start_id': 0,
    }
}
KEY_POS_FURNACE_WO_RECIPE = {
    'resource_slot': {
        'left-top': (287, 113),
        'right-bottom': (303, 164),
        'row': 2,
        'col': 1,
        'prefix': 'resource',
        'start_id': 0,
    },
    'result_slot': {
        'left-top': (345, 127),
        'right-bottom': (368, 152),
        'row': 1,
        'col': 1,
        'prefix': 'result',
        'start_id': 0,
    },
    'hotbar_slot': {
        'left-top': (242, 236),
        'right-bottom': (401, 256),
        'row': 1,
        'col': 9,
        'prefix': 'inventory',
        'start_id': 0,
    },
    'inventory_slot': {
        'left-top': (242, 178),
        'right-bottom': (401, 234),
        'row': 3,
        'col': 9,
        'prefix': 'inventory',
        'start_id': 9,
    },
    'recipe_slot': {
        'left-top': (254, 132),
        'right-bottom': (272, 147),
        'row': 1,
        'col': 1,
        'prefix': 'recipe',
        'start_id': 0,
    }
}

def COMPUTE_SLOT_POS(KEY_POS,WEIGHT_RATIO:int=1,HEIGHT_RADIO:int=1):
    result = {}
    for k, v in KEY_POS.items():
        left_top = [0,0]
        right_bottom = [0,0]
        left_top = [v['left-top'][0]*WEIGHT_RATIO, v['left-top'][1]*HEIGHT_RADIO]
        right_bottom = [v['right-bottom'][0]*WEIGHT_RATIO, v['right-bottom'][1]*HEIGHT_RADIO]
        row = v['row']
        col = v['col']
        prefix = v['prefix']
        start_id = v['start_id']
        width = right_bottom[0] - left_top[0]
        height = right_bottom[1] - left_top[1]
        slot_width = width // col
        slot_height = height // row
        slot_id = 0
        for i in range(row):
            for j in range(col):
                result[f'{prefix}_{slot_id + start_id}'] = (
                    left_top[0] + j * slot_width + (slot_width // 2),
                    left_top[1] + i * slot_height + (slot_height // 2),
                )
                slot_id += 1
    return result


class GUIWorker(object):

    def __init__(
        self,
        env: MinecraftSim,
        sample_ratio: float = 0.5,
        if_discrete = False,
        slow_act = True,
        **kwargs,
    )-> None:
        # print("Initializing worker...")

        self.env = env
        self.width,self.height = env.render_size
        self.width_ratio,self.height_ratio = self.width/BASE_WIDTH, self.height/BASE_HEIGHT
        self.slot_pos_inventory_wo_recipe = COMPUTE_SLOT_POS(KEY_POS_INVENTORY_WO_RECIPE,self.width_ratio,self.height_ratio)
        self.slot_pos_table_wo_recipe = COMPUTE_SLOT_POS(KEY_POS_TABLE_WO_RECIPE,self.width_ratio,self.height_ratio)
        self.slot_furnace_wo_recipe = COMPUTE_SLOT_POS(KEY_POS_FURNACE_WO_RECIPE,self.width_ratio,self.height_ratio)
        self.camera_scaler = CAMERA_SCALER / self.width_ratio

        self.if_discrete = if_discrete
        self.slow_act = slow_act
        self.sample_ratio = sample_ratio

        self.outframes, self.outactions, self.outinfos = [], [], []
        self.cursor = [self.width // 2, self.height // 2]
        self.current_gui_type = None
        self.gui = {}

        self.reset(fake_reset=True)

    def reset(self, fake_reset=True,):
        if not fake_reset:
            self._null_action(1)
            self.env.reset()

        self.resource_record = {f'resource_{x}': {'type': 'none', 'quantity': 0} for x in range(9)}
        self.crafting_slotpos = 'none'
        self.current_gui_type = None
        self._reset_cursor()
        self.gui = {}
        self.outframes, self.outactions, self.outinfos = [], [], []
        self._get_state()

    def _assert(self, condition, message=None):
        if not condition:
            print(message)
            if self.info['isGuiOpen']:
                self._call_func('inventory')
            self.current_gui_type = None
            self.crafting_slotpos = 'none'
            self.resource_record = {f'resource_{x}': {'type': 'none', 'quantity': 0} for x in range(9)}

            raise AssertionError(message)

    def _reset_cursor(self):
        """reset the cursor to middle of the screen """
        self.cursor = [self.width // 2, self.height // 2]
        return self.cursor

    # continue attack (retuen crafting table)
    def _attack_continue(self, times=1):
        action = self.env.noop_action()
        action['attack'] = 1
        for i in range(times):
            self.obs, _, _, _, self.info = self._step(action)

     # move

    # judge crafting_table / inventory
    @staticmethod
    def get_manipulate_type(target_data: Dict):
        manipulate_type = target_data.get("type")
        if manipulate_type == "minecraft:smelting":
            return "smelt"
        else:
            return "craft"

    def move_to_pos(self, x: float, y: float, speed: float = 10): #20

        if not self.if_discrete:
            camera_x = x - self.cursor[0]
            camera_y = y - self.cursor[1]
            distance =max(abs(camera_x), abs(camera_y))
            num_steps= int(random.uniform(5, 10) * math.sqrt(distance) / speed)
            if num_steps < 1:
                num_steps = 1
            for _ in range(num_steps):
                d1 = (camera_x / num_steps)
                d2 = (camera_y / num_steps)
                self.move_once(d1, d2)
            return
        #如果需要离散化的话
        for _ in range(100): #如果超过20次则不再调整

            camera_x = x - self.cursor[0]
            camera_y = y - self.cursor[1]
            distance =max(abs(camera_x), abs(camera_y))
            #print(distance)
            if distance <3: #当已经非常接近，就散了
                break
            choose_distance = random.uniform(40, 20) if distance>20 else distance

            d1 = camera_x*choose_distance / distance
            d2 = camera_y*choose_distance / distance
            temp_d = np.array([d1 * self.camera_scaler, d2 * self.camera_scaler])
            temp_d = self.env.action_transformer.quantizer.discretize(temp_d)
            temp_d = self.env.action_transformer.quantizer.undiscretize(temp_d)
            d1,d2 = temp_d / self.camera_scaler
            self.move_once(float(d1), float(d2)) #在这里改变了self.cursor[0]
            #print(distance,choose_distance,"||",d1,d2,"||")
        #print(x , self.cursor[0],y , self.cursor[1])

    def move_once(self, x: float, y: float):
        action = self.env.env.action_space.no_op()
        action['camera'] = np.array([y * self.camera_scaler, x * self.camera_scaler])
        # action["buttons"] = np.array([[0]])
        self.cursor[0] += x
        self.cursor[1] += y
        self.obs, _, _, _, self.info = self._step(action)

    # set the mouse and delete the memory
    def roam_camera(self):
        num_random = random.randint(1, 4)
        side_pixels = 30
        d1 =  random.uniform(-50, 50)
        d2 =  random.uniform(-50, 50)
        for _ in range(num_random):
            if self.cursor[0] + d1 < side_pixels or self.cursor[0] + d1 > self.width - side_pixels  or self.cursor[1] + d2 < side_pixels or self.cursor[1] + d2 > self.height-side_pixels:
                break
            self.move_once(d1, d2)
        self.forget(num=num_random)

    def _step(self, action,record_obs_only=False,forgeting=False,reserve=False):
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.info['resource'] = self.resource_record
        record_info = self.info
        record_info["cursor"] = copy.deepcopy(self.cursor)
        record_info["gui"] = copy.deepcopy(self.gui)
        record_info["place"] = copy.deepcopy(self.current_gui_type)
        record_info["reserve"] = reserve

        if forgeting:
            return self.obs, reward, terminated, truncated, self.info

        # record
        self.outframes.append(self.info['pov'].astype(np.uint8))

        self.outinfos.append(record_info)
        if not record_obs_only:
            self.outactions.append(copy.copy(action))

        return self.obs, reward, terminated, truncated, self.info

    def _null_action(self, times=1,forget=False,reserve=False):
        #if len(self.outactions):
            #print(f"{self.outactions[-1]['use']}",end= "  ")
        action = self.env.noop_action()

        for _ in range(times):
            self.obs, _, _, _, self.info = self._step(action,forgeting=forget,reserve=reserve)
        #if len(self.outactions)>1:
            # print(self.outactions[-2]["use"])

    # action wrapper
    def _call_func(self, func_name: str,no_op=False):
        action = self.env.noop_action()
        action[func_name] = 1
        for i in range(1):
            self.obs, _, _, _, self.info = self._step(action)
        if no_op:
            action[func_name] = 0
            for i in range(5):
                self.obs, _, _, _, self.info = self._step(action)

    def _press_inventory_button(self,fast=False):
        self._call_func("inventory",no_op=fast)
        self._reset_cursor()

    def _look_down(self):
        action = self.env.noop_action()
        if self.slow_act:
            self._null_action()
        for i in range(2):
            action['camera'] = np.array([88, 0])
            self.obs, _, _, _, self.info = self._step(action)

    def _jump(self):
        self._call_func('jump')
        for _ in range(5):
            self._null_action()

    def _place_down(self):
        self._look_down()
        self._jump()
        self._call_func('use')

    def _use_item(self):
        self._call_func('use')

    def _select_item(self):
        self._call_func('attack')

    def _get_state(self):
        action = self.env.noop_action()
        self.obs, _, _, _, self.info = self._step(action,record_obs_only=True)
        if self.info['isGuiOpen']:
            self.place = "inventory"

    def forget(self,num=1):
        """forget the past interaction"""
        if num==0:
            num = len(self.outactions)
        assert len(self.outactions)>=num and len(self.outframes) > num
        forget_actions = self.outactions[-num:] if len(self.outactions)>num else self.outactions
        forget_frames,forget_infos = self.outframes[-1-num:-1], self.outinfos[-1-num:-1]
        self.outactions = self.outactions[:-num] if len(self.outactions)>num else []
        self.outframes[-1-num] = self.outframes[-1]
        self.outframes = self.outframes[:-num]
        self.outinfos[-1-num] = self.outinfos[-1]
        self.outinfos = self.outinfos[:-num]
        return forget_frames,forget_infos,forget_actions

    def _take_a_screen_shot(self,store_path="output/screen_shot.png"):
        """check what happened now """
        import cv2
        self._null_action()
        screen_shot = self.outframes[-1] # 现在里面必然有值
        cv2.imwrite(store_path, screen_shot)
        print(f"Screenshot saved to {store_path}.")
