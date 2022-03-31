from typing import Callable

class ActivationFunction(Callable):
    def __call__(self, inp : int) -> int:
        pass
    

class ReLU(ActivationFunction):
    def __call__(self, inp : int) -> int:
        return inp if inp > 0 else 0

class Linear(ActivationFunction):
    def __call__(self, inp : int) -> int:
        return inp