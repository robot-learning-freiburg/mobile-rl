


from abc import ABC, abstractmethod
from typing import Tuple
import logging

'''
abstract class to define the interface for the optimizer used for other framework like bohb
'''
class OptimizerInterface(ABC):
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @abstractmethod
    def evaluate_design(self, config: dict, budget: float) -> Tuple[float, float]:
        '''
        evaluate the given configuration (design) with the effort of the given budget.
        return:
            - the result of the evaluation
            - additional info of the evaluation run
        '''
        pass