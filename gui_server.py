import asyncio
import base64
import logging
import random
import threading
import time
import uuid
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from minestudio.models import CraftWorker, EquipWorker, SmeltWorker, load_steve_one_policy  # noqa
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks.commands import CommandsCallback
from PIL import Image
from pydantic import BaseModel

from minecraftoptimus.model.agent.optimus3 import Optimus3Agent,check_inventory


paused = False
connected_clients: List[WebSocket] = []
main_loop = None
obs_queue = None
latest_obs_b64 = None
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="MinecraftOptimus API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You might want to restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request/response models
class ObservationData(BaseModel):
    pov: List[List[List[int]]]  # 3D array for image data
    inventory: Optional[Dict[str, int]] = None
    other_data: Optional[Dict[str, Any]] = None


class ResetData(BaseModel):
    device: str


class TextData(BaseModel):
    text: str | None
    task: str


# Global variables to store environment and model state
env = None
helper = None
model = None
current_obs = None
current_info = None
last_action = None
session_start_time = None
session_id = None
sub_tasks = None
goals = None
sub_task_index = 0
look_down_once = False
log_count = 0
iron_ore_count = 0
golden_ore_count = 0
diamond_ore_count = 0
redstone_ore_count = 0


def ndarray_to_base64(arr: np.ndarray) -> str:
    """
    Converts a numpy ndarray (HWC, uint8) to a base64-encoded PNG string.
    """
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _frame_pump(env, stop_event: threading.Event, interval_s: float = 0.05):
    """
    Periodically push the latest POV while long-running craft/smelt is executing.
    This avoids GUI freezes when step_hook is sparse.
    """
    while not stop_event.is_set():
        info = getattr(env, "info", None)
        if info and "pov" in info:
            try:
                enqueue_obs(ndarray_to_base64(info["pov"]))
            except Exception:
                pass
        time.sleep(interval_s)


@app.get("/gpu")
async def check_gpu():
    """
    Checks if GPUs are available and returns their name, total memory, and used memory.
    """
    gpu_available = torch.cuda.is_available()
    gpus = []
    if gpu_available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory // (1024 * 1024)
            # Get used memory via torch.cuda.memory_allocated
            used_mem = torch.cuda.memory_allocated(i) // (1024 * 1024)
            gpus.append(
                {
                    "name": props.name,
                    "total_memory_MB": total_mem,
                    "used_memory_MB": used_mem,
                }
            )
    return {"gpu_available": gpu_available, "gpus": gpus}



async def _send_to_clients(base64_png: str):
    if not connected_clients:
        return
    disconnected = []
    for ws in list(connected_clients):
        try:
            await ws.send_text(base64_png)
        except Exception as e:
            logger.warning("WebSocket send failed, dropping client: %s", e)
            disconnected.append(ws)
    for ws in disconnected:
        if ws in connected_clients:
            connected_clients.remove(ws)


def enqueue_obs(base64_png: str):
    """
    Store latest obs and enqueue for broadcast. Safe to call from any thread.
    """
    global latest_obs_b64
    latest_obs_b64 = base64_png
    if paused or obs_queue is None or main_loop is None:
        return

    def _put():
        if obs_queue.full():
            try:
                obs_queue.get_nowait()
            except Exception:
                pass
        try:
            obs_queue.put_nowait(base64_png)
        except Exception:
            pass

    if main_loop.is_running():
        main_loop.call_soon_threadsafe(_put)


async def obs_broadcast_loop():
    while True:
        if obs_queue is None:
            await asyncio.sleep(0.05)
            continue
        base64_png = await obs_queue.get()
        if paused:
            continue
        await _send_to_clients(base64_png)


async def broadcast_obs(base64_png: str):
    """
    Backward-compatible wrapper for existing call sites.
    """
    enqueue_obs(base64_png)


@app.post("/pause")
async def pause_agent():
    
    global paused
    paused = True
    # Clear pending frames so UI shows the newest snapshot.
    if obs_queue is not None:
        try:
            while not obs_queue.empty():
                obs_queue.get_nowait()
        except Exception:
            pass
    if latest_obs_b64:
        await _send_to_clients(latest_obs_b64)
    return {"status": "paused", "observation": latest_obs_b64}


@app.post("/resume")
async def resume_agent():
   
    global paused
    paused = False
    return {"status": "running"}


@app.websocket("/ws/obs")
async def websocket_observations(websocket: WebSocket):
    
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)


@app.on_event("startup")
async def startup_event():
    """ """
    global session_start_time, session_id, main_loop, obs_queue

    session_start_time = datetime.now()
    session_id = str(uuid.uuid4())
    main_loop = asyncio.get_running_loop()
    obs_queue = asyncio.Queue(maxsize=200)
    asyncio.create_task(obs_broadcast_loop())

    gpu_info = await check_gpu()
    default_device = "cuda:0"

    if gpu_info.get("gpu_available") and gpu_info.get("gpus"):
        for idx, gpu in enumerate(gpu_info.get("gpus")):
            free_mem = gpu.get("total_memory_MB") - gpu.get("used_memory_MB")
            if free_mem >= 40000:  # 40GB
                default_device = f"cuda:{idx}"
                logger.info(f"choose GPU{idx}, free memory {free_mem}MB, device: {default_device}")
                break
        else:
            logger.info("No GPU with >= 40GB of available memory found, use CPU")
    else:
        logger.info("GPU not available, use CPU")

    await reset(ResetData(device=default_device))


@app.post("/reset")
async def reset(reset_data: ResetData):
    """
    Initializes or resets the MinecRL environment and loads the model.
    """
    global env, model, current_obs, session_start_time, session_id, helper, current_info

    try:
        # Close existing environment if one exists
        if env:
            env.close()

        logger.info("Initializing environment")
        env = MinecraftSim(
            obs_size=(128, 128),
            preferred_spawn_biome="forest",
            callbacks=[
                CommandsCallback(
                    [
                        "/gamerule sendCommandFeedback false",
                        "/gamerule commandBlockOutput false",
                        "/gamerule keepInventory true",
                        "/effect give @a night_vision 99999 250 true",
                        "/gamerule doDaylightCycle false",
                        "/time set 0",
                        "/gamerule doImmediateRespawn true",
                        "/spawnpoint",
                    ]
                ),
            ],
            seed=random.randint(1, 100000000),
        )

        # Reset the environment to get initial observation
        current_obs, current_info = env.reset()
        helper = {"craft": CraftWorker(env), "smelt": SmeltWorker(env), "equip": EquipWorker(env)}
        if not model:
            model = Optimus3Agent(
                "Optimus3-Policy dir",
                "optimus3-mllm dir",
                "optimus3-task-router dir",
                device=reset_data.device,
            )
        obs_b64 = ndarray_to_base64(current_info["pov"])
        enqueue_obs(obs_b64)
        return {"status": "success", "observation": obs_b64}
        # return {"status": "success", "observation": ndarray_to_base64(current_obs["image"])}

    except Exception as e:
        logger.error(f"Error during reset: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize: {str(e)}")


def _step(env, agent, obs, task, goal, helper):
    global look_down_once, log_count, iron_ore_count, golden_ore_count, diamond_ore_count, redstone_ore_count

    while paused:
        time.sleep(0.05)
    if "craft" in task:
        helper["craft"].step_hook = lambda info: asyncio.create_task(broadcast_obs(ndarray_to_base64(info["pov"])))
        result, _ = helper["craft"].crafting(goal["item"], goal["count"])
        action = env.env.noop_action()

        pickaxe = env.find_best_pickaxe()
        if pickaxe:
            helper["equip"].equip_item(pickaxe)
        obs, reward, terminated, truncated, info = env.step(action)

    elif "smelt" in task:
        helper["smelt"].step_hook = lambda info: asyncio.create_task(broadcast_obs(ndarray_to_base64(info["pov"])))
        result, _ = helper["smelt"].smelting(goal["item"], goal["count"])
        obs, reward, terminated, truncated, info = env.step(env.env.noop_action())
    else:
        env._only_once = True
        action, memory = agent.get_action(obs, task)
        action = env.agent_action_to_env_action(action)
        action["drop"] = np.array(0)
        action["inventory"] = np.array(0)
        action["use"] = np.array(0)
        for i in range(9):
            action[f"hotbar.{i + 1}"] = np.array(0)

        if "dig down" in task:
            action["jump"] = action["left"] = action["right"] = np.array(0)
            action["sneak"] = action["sprint"] = np.array(0)
            if not look_down_once:
                pickaxe = env.find_best_pickaxe()
                helper["equip"].equip_item(pickaxe)
                helper["craft"]._look_down()
                look_down_once = True
            action["attack"] = np.array(1)
        
        if action["attack"] > 0:
            action["jump"] = action["left"] = action["right"] = np.array(0)
            action["sneak"] = action["sprint"] = np.array(0)

        obs, reward, terminated, truncated, info = env.step(action)

    check, count = check_inventory(info["inventory"], goal["item"], goal["count"])
    if check:
        if goal["item"] == "logs":
            log_count = count
        elif goal["item"] == "iron_ore":
            iron_ore_count = count
        elif goal["item"] == "gold_ore":
            golden_ore_count = count
        elif goal["item"] == "diamond":
            diamond_ore_count = count
        elif goal["item"] == "redstone":
            redstone_ore_count = count
        look_down_once = False
    return obs, info, check


@app.get("/get_obs")
async def get_obs():
    global current_obs
    if not env or not current_obs:
        raise HTTPException(status_code=400, detail="Environment not initialized or no observation.")
    try:
        obs_b64 = ndarray_to_base64(current_info["pov"])

        asyncio.create_task(broadcast_obs(obs_b64))
        
    except Exception as e:
        logger.error(f"Error returning observation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to return observation: {str(e)}")


@app.get("/initial_text")
async def initial_text():
    initial_text = """Hello! I'm Optimus-3, your Minecraft agent. I can help you with task planning, action execution, and visual perception in Minecraft (including captioning, embodied question answering, and grounding). Let's embark on an exciting journey of exploration in Minecraft!
     """
    return {
        "status": "success",
        "text": initial_text,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/send_text")
async def send_text(text_data: TextData):
    """
    Processes a text command and returns a response.
    """
    global env, model, current_obs, sub_tasks, goals, sub_task_index, last_action, current_info
    if not env or model is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    try:
       
        user_text = text_data.text.strip() if text_data.text else ""
        task_type = text_data.task.strip()
        logger.info(f"Received text '{user_text}' with task '{task_type}'")
        if current_info is None or "pov" not in current_info:
            raise HTTPException(status_code=400, detail="No current observation. Call /reset first.")

        img = Image.fromarray(current_info["pov"])
        
        if "help" in user_text:
            response_text = """
            Available commands:
            [Planning] [Captioning] [Embodied QA] [Grounding] [Long-horizon Action]
            """
        else:
            with torch.no_grad():
                print(f"Task type: {task_type}")
                if task_type == "planning":
                    response_text, _sub_plans, _goals = model.plan(user_text)
                    sub_tasks = _sub_plans
                    goals = _goals
                    sub_task_index = 0
                    model.task = None
                elif task_type == "captioning" or task_type == "embodied_qa":
                    response_text = model.answer(user_text, img)
                elif task_type == "action":
                    
                    if sub_tasks and sub_task_index < len(sub_tasks):
                        if model.task is None:
                            model.reset(sub_tasks[sub_task_index])
                        obs, info, check = _step(
                            env, model, current_obs, sub_tasks[sub_task_index], goals[sub_task_index], helper
                        )
                        if check:
                            sub_task_index += 1
                            model.tasdef _step(env, agent, obs, task, goal, helper):
    global look_down_once, log_count, iron_ore_count, golden_ore_count, diamond_ore_count, redstone_ore_count

    while paused:
        time.sleep(0.05)
    if "craft" in task:
        helper["craft"].step_hook = lambda info: enqueue_obs(ndarray_to_base64(info["pov"]))
        stop_event = threading.Event()
        pump = threading.Thread(target=_frame_pump, args=(env, stop_event), daemon=True)
        pump.start()
        try:
            result, _ = helper["craft"].crafting(goal["item"], goal["count"])
        finally:
            stop_event.set()
            pump.join(timeout=0.5)
        action = env.env.noop_action()

        pickaxe = env.find_best_pickaxe()
        if pickaxe:
            helper["equip"].equip_item(pickaxe)
        obs, reward, terminated, truncated, info = env.step(action)

    elif "smelt" in task:
        helper["smelt"].step_hook = lambda info: enqueue_obs(ndarray_to_base64(info["pov"]))
        stop_event = threading.Event()
        pump = threading.Thread(target=_frame_pump, args=(env, stop_event), daemon=True)
        pump.start()
        try:
            result, _ = helper["smelt"].smelting(goal["item"], goal["count"])
        finally:
            stop_event.set()
            pump.join(timeout=0.5)
        obs, reward, terminated, truncated, info = env.step(env.env.noop_action())
    else:
        env._only_once = True
        action, memory = agent.get_action(obs, task)
        action = env.agent_action_to_env_action(action)
        action["drop"] = np.array(0)
        action["inventory"] = np.array(0)
        action["use"] = np.array(0)
        for i in range(9):
            action[f"hotbar.{i + 1}"] = np.array(0)

        if "dig down" in task:
            action["jump"] = action["left"] = action["right"] = np.array(0)
            action["sneak"] = action["sprint"] = np.array(0)
            if not look_down_once:
                pickaxe = env.find_best_pickaxe()
                helper["equip"].equip_item(pickaxe)
                helper["craft"]._look_down()
                look_down_once = True
            action["attack"] = np.array(1)
        # attack时不乱动
        if action["attack"] > 0:
            action["jump"] = action["left"] = action["right"] = np.array(0)
            action["sneak"] = action["sprint"] = np.array(0)

        obs, reward, terminated, truncated, info = env.step(action)

    check, count = check_inventory(info["inventory"], goal["item"], goal["count"])
    if check:
        if goal["item"] == "logs":
            log_count = count
        elif goal["item"] == "iron_ore":
            iron_ore_count = count
        elif goal["item"] == "gold_ore":
            golden_ore_count = count
        elif goal["item"] == "diamond":
            diamond_ore_count = count
        elif goal["item"] == "redstone":
            redstone_ore_count = count
        look_down_once = False
    return obs, info, check


@app.get("/get_obs")
async def get_obs():
    global current_obs
    if not env or not current_obs:
        raise HTTPException(status_code=400, detail="Environment not initialized or no observation.")
    try:
        obs_b64 = ndarray_to_base64(current_info["pov"])

        enqueue_obs(obs_b64)
        return {
            "status": "success",
            "observation": obs_b64,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
        }
        # return {
        #     "status": "success",
        #     "observation": ndarray_to_base64(current_obs["image"]),
        #     "session_id": session_id,
        #     "timestamp": datetime.now().isoformat(),
        # }
    except Exception as e:
        logger.error(f"Error returning observation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to return observation: {str(e)}")


@app.get("/initial_text")
async def initial_text():
    initial_text = """Hello! I'm Optimus-3, your Minecraft agent. I can help you with task planning, action execution, and visual perception in Minecraft (including captioning, embodied question answering, and grounding). Let's embark on an exciting journey of exploration in Minecraft!
     """
    return {
        "status": "success",
        "text": initial_text,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/send_text")
async def send_text(text_data: TextData):
    """
    Processes a text command and returns a response.
    """
    global env, model, current_obs, sub_tasks, goals, sub_task_index, last_action, current_info
    if not env or model is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    try:
        # Process the text command
        # This is just a placeholder - replace with your actual text processing logic
        user_text = text_data.text.strip() if text_data.text else ""
        task_type = text_data.task.strip()
        if task_type == "action" and paused:
            return {
                "status": "paused",
                "response": "paused",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
            }
        logger.info(f"Received text '{user_text}' with task '{task_type}'")
        if current_info is None or "pov" not in current_info:
            raise HTTPException(status_code=400, detail="No current observation. Call /reset first.")

        img = Image.fromarray(current_info["pov"])
        # obs = obs.to(next(model).device)
        if "help" in user_text:
            response_text = """
            Available commands:
            [Planning] [Captioning] [Embodied QA] [Grounding] [Long-horizon Action]
            """
        else:
            with torch.no_grad():
                print(f"Task type: {task_type}")
                if task_type == "planning":
                    response_text, _sub_plans, _goals = model.plan(user_text)
                    sub_tasks = _sub_plans
                    goals = _goals
                    sub_task_index = 0
                    model.task = None
                    # print(sub_tasks)
                    # print(goals)
                elif task_type == "captioning" or task_type == "embodied_qa":
                    response_text = model.answer(user_text, img)
                elif task_type == "action":
                    if sub_tasks and sub_task_index < len(sub_tasks):
                        if model.task is None:
                            model.reset(sub_tasks[sub_task_index])
                        loop = asyncio.get_running_loop()
                        obs, info, check = await loop.run_in_executor(
                            None,
                            _step,
                            env,
                            model,
                            current_obs,
                            sub_tasks[sub_task_index],
                            goals[sub_task_index],
                            helper,
                        )
                        if check:
                            sub_task_index += 1
                            model.task = None
                        current_obs = obs
                        current_info = info

                        if sub_task_index < len(sub_tasks):
                            response_text = sub_tasks[sub_task_index]
                        else:
                            response_text = "success"
                    else:
                        response_text = "success"
                elif task_type == "grounding":
                    response_text = model.grounding(user_text, img)
                else:
                    response_text = "Unknown task type. Please try again."
                print(response_text)

        response_text = response_text.strip().lower()
        if current_info and "pov" in current_info:
            print("image")
            obs_b64 = ndarray_to_base64(current_info["pov"])

            enqueue_obs(obs_b64)

        return {
            "status": "success",
            "response": response_text,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error processing text command: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")


@app.get("/receive_text")
async def receive_text():
    """
    Retrieves any text or status updates from the model/environment.
    """
    if not env:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    try:
        # Generate status text based on current environment state
        # This is just a placeholder - customize based on what information you want to provide
        status_text = ""

        if last_action:
            status_text += f"Last action: {last_action}\n"

        if current_obs and "inventory" in current_obs:
            status_text += "Inventory:\n"
            for item, count in current_obs["inventory"].items():
                if count > 0:
                    status_text += f"- {item}: {count}\n"

        # Add session info
        if session_start_time:
            elapsed_time = (datetime.now() - session_start_time).total_seconds()
            status_text += f"\nSession running for: {elapsed_time:.1f} seconds"

        return {
            "status": "success",
            "text": status_text,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error generating status text: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate status: {str(e)}")


@app.get("/status")
async def get_status():
    """
    Returns the current status of the server, environment, and model.
    """
    return {"status": "running"}

if __name__ == "__main__":
    import uvicorn
    #  input the host(ip of server), e.g., 10.xx.xx.xx
    uvicorn.run("gui_server:app", host="", port=9500)
