from abc import ABC, abstractmethod
from typing import Tuple, Callable, Any

class Task(ABC):
    """Abstract base class for all tasks"""
    
    def __init__(self, threshold: float = 0.0):
        """Initialize task with basic parameters
        
        Args:
            threshold: Minimum validation accuracy required to run test evaluation
        """
        self.task_name = self.__class__.__name__
        self.last_test_acc = 0.0
        self.threshold = threshold
    
    PASS = "PASS"
    FAIL = "FAIL"
    
    @property
    @abstractmethod
    def task_description(self) -> str:
        """Task description that will be provided to the agent"""
        pass
        
    @property
    @abstractmethod 
    def initial_system(self) -> str:
        """Initial system/solver code for the task"""
        pass
        
    @abstractmethod
    def evaluate(self, solver: Callable) -> Tuple[str, float, float]:
        """
        Evaluate a solver function on this task
        
        Args:
            solver: The solver function to evaluate
            
        Returns:
            Tuple containing:
                - Feedback string
                - Validation accuracy 
                - Test accuracy
        """
        pass
    
    def format_feedback(self, valid_acc: float, test_acc: float, info_list: list[str]) -> str:
        """Format evaluation feedback string
        
        Args:
            valid_acc: Validation accuracy
            test_acc: Test accuracy 
            info_list: List of evaluation details
            
        Returns:
            Formatted feedback string
        """
        if valid_acc >= self.threshold:
            return (f"Valid Accuracy: {valid_acc}\n"
                   f"Test Accuracy: {test_acc}\n"
                   f"Evaluation Info:\n" + "\n".join(info_list))
        else:
            return (f"Valid Accuracy: {valid_acc}\n"
                   f"Valid Accuracy less than {self.threshold}, no testing needed.\n"
                   f"Evaluation Info:\n" + "\n".join(info_list))