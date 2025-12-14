"""
Reinforcement Learning Agent for Operational LCA Minimization
PPO/SAC agent that controls H₂ injection, CCUS capture rate, etc.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO, SAC
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    logging.warning("gymnasium/stable_baselines3 not available")

logger = logging.getLogger(__name__)


@dataclass
class PlantState:
    """State of industrial plant"""
    h2_injection_rate: float  # 0.0 to 1.0
    ccus_capture_rate: float  # 0.0 to 1.0
    electrification_level: float  # 0.0 to 1.0
    recycling_ratio: float  # 0.0 to 1.0
    grid_carbon_intensity: float  # g CO2/kWh
    production_rate: float  # 0.0 to 1.0


@dataclass
class PlantAction:
    """Action in plant environment"""
    h2_injection_delta: float  # Change in H2 injection
    ccus_capture_delta: float  # Change in CCUS capture
    electrification_delta: float  # Change in electrification
    recycling_delta: float  # Change in recycling ratio


class PlantEnvironment:
    """
    Digital twin environment for industrial plant.
    
    State: Plant operational parameters
    Action: Control adjustments
    Reward: Negative of real-time LCA emissions
    """
    
    def __init__(
        self,
        initial_state: Optional[PlantState] = None,
        lca_calculator: Optional[Any] = None
    ):
        """
        Initialize plant environment.
        
        Args:
            initial_state: Initial plant state
            lca_calculator: LCA calculator function
        """
        if not RL_AVAILABLE:
            raise ImportError("gymnasium and stable_baselines3 required for RL")
        
        self.state = initial_state or PlantState(
            h2_injection_rate=0.5,
            ccus_capture_rate=0.0,
            electrification_level=0.0,
            recycling_ratio=0.0,
            grid_carbon_intensity=300.0,
            production_rate=1.0
        )
        
        self.lca_calculator = lca_calculator
        
        # Action space: continuous actions for each control variable
        self.action_space = spaces.Box(
            low=np.array([-0.1, -0.1, -0.1, -0.1]),
            high=np.array([0.1, 0.1, 0.1, 0.1]),
            dtype=np.float32
        )
        
        # Observation space: plant state
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1000.0, 1.0]),
            dtype=np.float32
        )
        
        logger.info("Plant environment initialized")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.state = PlantState(
            h2_injection_rate=0.5,
            ccus_capture_rate=0.0,
            electrification_level=0.0,
            recycling_ratio=0.0,
            grid_carbon_intensity=300.0,
            production_rate=1.0
        )
        return self._state_to_array()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state, reward, done, info.
        
        Args:
            action: Action array [h2_delta, ccus_delta, electrification_delta, recycling_delta]
        
        Returns:
            (observation, reward, done, info)
        """
        # Update state
        self.state.h2_injection_rate = np.clip(
            self.state.h2_injection_rate + action[0], 0.0, 1.0
        )
        self.state.ccus_capture_rate = np.clip(
            self.state.ccus_capture_rate + action[1], 0.0, 1.0
        )
        self.state.electrification_level = np.clip(
            self.state.electrification_level + action[2], 0.0, 1.0
        )
        self.state.recycling_ratio = np.clip(
            self.state.recycling_ratio + action[3], 0.0, 1.0
        )
        
        # Calculate LCA emissions (reward is negative emissions)
        emissions = self._calculate_emissions()
        reward = -emissions  # Negative emissions = reward
        
        # Done condition (optional: episode length, etc.)
        done = False
        
        info = {
            "emissions": emissions,
            "state": self.state
        }
        
        return self._state_to_array(), reward, done, info
    
    def _state_to_array(self) -> np.ndarray:
        """Convert state to observation array"""
        return np.array([
            self.state.h2_injection_rate,
            self.state.ccus_capture_rate,
            self.state.electrification_level,
            self.state.recycling_ratio,
            self.state.grid_carbon_intensity,
            self.state.production_rate
        ], dtype=np.float32)
    
    def _calculate_emissions(self) -> float:
        """Calculate real-time LCA emissions"""
        if self.lca_calculator:
            return self.lca_calculator(self.state)
        
        # Simplified emission calculation
        base_emissions = 1000.0
        
        # H2 injection reduces emissions
        h2_reduction = self.state.h2_injection_rate * 0.3
        
        # CCUS captures emissions
        ccus_reduction = self.state.ccus_capture_rate * 0.4
        
        # Electrification reduces emissions (depends on grid CI)
        grid_factor = self.state.grid_carbon_intensity / 300.0
        electrification_reduction = self.state.electrification_level * 0.2 * (1 - grid_factor)
        
        # Recycling reduces emissions
        recycling_reduction = self.state.recycling_ratio * 0.1
        
        total_reduction = h2_reduction + ccus_reduction + electrification_reduction + recycling_reduction
        
        return base_emissions * (1 - total_reduction)


class RLAgent:
    """
    Reinforcement Learning agent for operational LCA minimization.
    
    Uses PPO or SAC to learn optimal control policies.
    """
    
    def __init__(
        self,
        algorithm: str = "PPO",
        env: Optional[PlantEnvironment] = None
    ):
        """
        Initialize RL agent.
        
        Args:
            algorithm: RL algorithm ("PPO" or "SAC")
            env: Plant environment
        """
        if not RL_AVAILABLE:
            raise ImportError("gymnasium and stable_baselines3 required")
        
        self.algorithm = algorithm
        self.env = env or PlantEnvironment()
        self.model = None
        
        logger.info(f"RL Agent initialized with algorithm: {algorithm}")
    
    def train(
        self,
        total_timesteps: int = 100000,
        **kwargs
    ):
        """
        Train RL agent.
        
        Args:
            total_timesteps: Total training timesteps
            **kwargs: Additional training parameters
        """
        if self.algorithm == "PPO":
            self.model = PPO("MlpPolicy", self.env, verbose=1, **kwargs)
        elif self.algorithm == "SAC":
            self.model = SAC("MlpPolicy", self.env, verbose=1, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        self.model.learn(total_timesteps=total_timesteps)
        logger.info("RL agent training completed")
    
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Predict action given observation.
        
        Args:
            observation: Current state observation
        
        Returns:
            Action array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def save(self, path: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model"""
        if self.algorithm == "PPO":
            self.model = PPO.load(path)
        elif self.algorithm == "SAC":
            self.model = SAC.load(path)
        logger.info(f"Model loaded from {path}")

