"""
Date: 2024-11-29 11:05:35
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-29 11:07:37
FilePath: /MineStudio/minestudio/simulator/minerl/env/gpu_utils.py
"""

# https://nvidia.github.io/cuda-python/
from cuda import cuda, cudart
import argparse
import os


def call_and_check_error(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        assert result[0] == 0, f"cuda-python error, {result[0]}"
        if len(result) == 2:
            return result[1]
        else:
            assert len(result) == 1, "Unsupported function call"
            return None

    return wrapper


def getCudaDeviceCount():
    return call_and_check_error(cudart.cudaGetDeviceCount)()


def getPCIBusIdByCudaDeviceOrdinal(cuda_device_id):
    """
    cuda_device_id 在 0 ~ getCudaDeviceCount() - 1 之间取值，受到 CUDA_VISIBLE_DEVICES 影响
    """
    device = call_and_check_error(cuda.cuDeviceGet)(cuda_device_id)
    result = call_and_check_error(cuda.cuDeviceGetPCIBusId)(100, device)
    return result.decode("ascii").split("\0")[0]


if __name__ == "__main__":
    if os.environ.get("MINESTUDIO_GPU_RENDER", 0) != "1":
        print("cpu")
        exit(0)
    try:
        call_and_check_error(cuda.cuInit)(0)
    except:
        print("cpu")
        exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=str)
    args = parser.parse_args()
    index = int(args.index)
    num_cuda_devices = getCudaDeviceCount()
    if num_cuda_devices == 0:
        device = "cpu"
    else:
        cuda_device_id = index % num_cuda_devices
        pci_bus_id = getPCIBusIdByCudaDeviceOrdinal(cuda_device_id)
        device = os.path.realpath(f"/dev/dri/by-path/pci-{pci_bus_id.lower()}-card")
    print(device)
