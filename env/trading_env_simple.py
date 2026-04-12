import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """
    Entorn de trading simplificat per a experiments amb Deep Reinforcement Learning.

    Accions:
        0 = mantenir la posició actual (hold)
        1 = adoptar una posició llarga (long)
        2 = adoptar una posició curta (short)

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
    ) -> None:

        super().__init__()

        # Comprovació que la columna de preu existeix
        if price_column not in data.columns:
            raise ValueError(f"Missing price column: {price_column}")

        # Comprovació que totes les features existeixen
        missing_features = [c for c in feature_columns if c not in data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        # Còpia del dataset i configuració bàsica de l'entorn
        self.data = data.reset_index(drop=True).copy()
        self.feature_columns = feature_columns
        self.price_column = price_column
        self.transaction_cost = transaction_cost

        # Estat intern de l'entorn
        self.n_steps = len(self.data)
        self.current_step = 0
        self.position = 0  # -1 short, 0 flat, 1 long

        # Espai d'accions discret amb tres opcions
        self.action_space = spaces.Discrete(3)

        # Espai d'observació: features + posició actual
        obs_dim = len(self.feature_columns) + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        """
        Construeix l'observació actual a partir de:
        - les variables d'entrada del pas temporal actual
        - la posició actual de l'agent
        """
        row = self.data.iloc[self.current_step]
        features = row[self.feature_columns].to_numpy(dtype=np.float32)
        obs = np.concatenate([features, np.array([self.position], dtype=np.float32)])
        return obs

    def reset(self, seed=None, options=None):
        """
        Reinicia l'entorn a l'estat inicial.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action: int):
        """
        Executa una acció, actualitza la posició de l'agent
        i calcula la recompensa segons el retorn de l'actiu.

        La recompensa simple es defineix com:
            recompensa = nova_posició * retorn_de_l_actiu

        Si hi ha canvi de posició, s'aplica un cost de transacció.
        """

        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action: {action}")

        prev_position = self.position

        # Mapatge acció -> nova posició
        if action == 0:
            new_position = prev_position
        elif action == 1:
            new_position = 1
        else:  # action == 2
            new_position = -1

        # Preu actual i preu següent
        current_price = self.data.iloc[self.current_step][self.price_column]
        next_price = self.data.iloc[self.current_step + 1][self.price_column]

        # Retorn de l'actiu entre t i t+1
        asset_return = (next_price / current_price) - 1.0

        # Recompensa simple basada en la nova posició
        reward = new_position * asset_return

        # Cost de transacció si la posició canvia
        if new_position != prev_position:
            reward -= self.transaction_cost

        # Comprovació de final de l'episodi
        done = self.current_step >= self.n_steps - 2
        truncated = False

        # Actualització de l'estat intern
        self.position = new_position
        self.current_step += 1

        observation = self._get_observation()
        info = {
            "step": self.current_step,
            "position": self.position,
            "asset_return": asset_return,
        }

        return observation, float(reward), done, truncated, info

    def render(self):
        """
        Mostra per consola l'estat actual de l'entorn.
        """
        print(
            f"Step: {self.current_step}, "
            f"Position: {self.position}"
        )