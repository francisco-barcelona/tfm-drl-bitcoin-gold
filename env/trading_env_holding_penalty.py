import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """
    Entorn de trading orientat a estratègies de curt termini.

    A diferència d'altres entorns, aquest penalitza el manteniment prolongat
    d'una posició per evitar polítiques trivials de tipus buy-and-hold.

    Accions:
        0 = mantenir la posició actual (hold)
        1 = adoptar posició llarga (long)
        2 = adoptar posició curta (short)

    Posicions:
        -1 = short
         0 = neutral (flat)
         1 = long
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        price_column: str = "btc_close",
        transaction_cost: float = 0.001,
        volatility_column: str = "btc_vol_12",
        alpha: float = 0.05,
        beta: float = 0.001,
    ) -> None:

        super().__init__()

        if price_column not in data.columns:
            raise ValueError(f"Missing price column: {price_column}")

        missing_features = [c for c in feature_columns if c not in data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        if volatility_column not in data.columns:
            raise ValueError(f"Missing volatility column: {volatility_column}")

        self.data = data.reset_index(drop=True).copy()
        self.feature_columns = feature_columns
        self.price_column = price_column
        self.transaction_cost = transaction_cost
        self.volatility_column = volatility_column
        self.alpha = alpha
        self.beta = beta

        self.n_steps = len(self.data)
        self.current_step = 0
        self.position = 0
        self.holding_duration = 0

        self.action_space = spaces.Discrete(3)

        obs_dim = len(self.feature_columns) + 2  # features + posició + durada
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        """
        Observació actual:
        - features del mercat
        - posició actual
        - durada de la posició actual
        """
        row = self.data.iloc[self.current_step]
        features = row[self.feature_columns].to_numpy(dtype=np.float32)

        obs = np.concatenate([
            features,
            np.array([self.position, self.holding_duration], dtype=np.float32)
        ])

        return obs

    def reset(self, seed=None, options=None):
        """
        Reinicia l'entorn.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.holding_duration = 0

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action: int):
        """
        Executa una acció i calcula la recompensa.

        Reward:
            reward = posició * retorn
                     - alpha * volatilitat
                     - beta * holding_duration

        La penalització beta força l'agent a evitar mantenir
        la mateixa posició durant massa passos consecutius.
        """

        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action: {action}")

        prev_position = self.position

        if action == 0:
            new_position = prev_position
        elif action == 1:
            new_position = 1
        else:
            new_position = -1

        current_price = self.data.iloc[self.current_step][self.price_column]

        done = self.current_step >= self.n_steps - 2
        truncated = False

        next_price = self.data.iloc[self.current_step + 1][self.price_column]
        asset_return = (next_price / current_price) - 1.0

        volatility = self.data.iloc[self.current_step][self.volatility_column]

        if new_position == prev_position and new_position != 0:
            self.holding_duration += 1
        elif new_position != prev_position:
            self.holding_duration = 1 if new_position != 0 else 0
        else:
            self.holding_duration = 0

        reward = (
            new_position * asset_return
            - self.alpha * volatility
            - self.beta * self.holding_duration
        )

        if new_position != prev_position:
            reward -= self.transaction_cost

        self.position = new_position
        self.current_step += 1

        observation = self._get_observation()

        info = {
            "step": self.current_step,
            "position": self.position,
            "asset_return": asset_return,
            "volatility": volatility,
            "holding_duration": self.holding_duration,
        }

        return observation, float(reward), done, truncated, info

    def render(self):
        """
        Mostra per consola l'estat actual de l'entorn.
        """
        print(
            f"Step: {self.current_step}, "
            f"Position: {self.position}, "
            f"Holding duration: {self.holding_duration}"
        )