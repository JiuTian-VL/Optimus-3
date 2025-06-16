from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    @abstractmethod
    def reset(self, task: str):
        pass

    @abstractmethod
    def get_action(self, **kwargs):
        pass

    @abstractmethod
    def plan(self, task: str, image: np.ndarray | str | None) -> list[str]:
        pass

    @abstractmethod
    def reflection(self, task: str, image: np.ndarray | str | None) -> list[str]:
        pass

    @abstractmethod
    def answer(self, task: str, image: np.ndarray | str | None) -> list[str]:
        pass
