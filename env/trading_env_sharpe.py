import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """
    Entorn de trading simplificat per a experiments amb Deep Reinforcement Learning.

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
        reward_mode: str = "simple",
        volatility_column: str = "btc_vol_12",
        epsilon: float = 1e-6,
    ) -> None:
        
        super().__init__()

        # Comprovació que la columna de preus existeix
        if price_column not in data.columns:
            raise ValueError(f"Missing price column: {price_column}")

        # Comprovació que totes les features estan presents al dataset
        missing_features = [c for c in feature_columns if c not in data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        # Comprovació que la columna de volatilitat existeix
        if volatility_column not in data.columns:
            raise ValueError(f"Missing volatility column: {volatility_column}")

        # Còpia del dataset i configuració de l'entorn
        self.data = data.reset_index(drop=True).copy()
        self.feature_columns = feature_columns
        self.price_column = price_column
        self.transaction_cost = transaction_cost
        self.reward_mode = reward_mode
        self.volatility_column = volatility_column
        self.epsilon = epsilon

        # Inicialització de l'estat intern
        self.n_steps = len(self.data)
        self.current_step = 0
        self.position = 0  # -1 short, 0 flat, 1 long

        # Espai d'accions discret (3 accions possibles)
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
        - les features del pas temporal actual
        - la posició actual de l'agent
        """
        row = self.data.iloc[self.current_step]
        features = row[self.feature_columns].to_numpy(dtype=np.float32)
        obs = np.concatenate([features, np.array([self.position], dtype=np.float32)])
        return obs

    def reset(self, seed=None, options=None):
        """
        Reinicia l'entorn a l'estat inicial:
        - pas temporal = 0
        - posició = neutral (0)
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action: int):
        """
        Executa una acció i retorna:
        - nova observació
        - recompensa
        - indicador de finalització
        - info addicional

        La recompensa es calcula a partir del retorn entre el preu actual
        i el següent, segons la posició adoptada per l'agent.

        Es poden considerar diferents variants de recompensa:
        - simple: basada únicament en el retorn
        - risk_adjusted: penalitza moviments extrems
        - sharpe_like: ajusta el retorn per la volatilitat
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

        # Preu actual
        current_price = self.data.iloc[self.current_step][self.price_column]

        # Comprovació de final de l'episodi
        done = self.current_step >= self.n_steps - 2
        truncated = False

        # Preu del següent pas temporal
        next_price = self.data.iloc[self.current_step + 1][self.price_column]

        # Retorn de l'actiu entre t i t+1
        asset_return = (next_price / current_price) - 1.0

        # Volatilitat estimada en el pas actual
        volatility = self.data.iloc[self.current_step][self.volatility_column]

        # Càlcul de la recompensa segons el mode seleccionat
        if self.reward_mode == "simple":
            reward = new_position * asset_return

        elif self.reward_mode == "risk_adjusted":
            reward = (new_position * asset_return) - 0.1 * (asset_return ** 2)

        elif self.reward_mode == "sharpe_like":
            reward = (new_position * asset_return) / (volatility + self.epsilon)

        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        # Cost de transacció si es canvia de posició
        if new_position != prev_position:
            reward -= self.transaction_cost

        # Actualització de l'estat intern
        self.position = new_position
        self.current_step += 1

        observation = self._get_observation()

        # Informació addicional útil per a anàlisi
        info = {
            "step": self.current_step,
            "position": self.position,
            "asset_return": asset_return,
            "volatility": volatility,
        }

        return observation, float(reward), done, truncated, info

    def render(self):
        """
        Mostra per consola l'estat actual de l'entorn
        """
        print(
            f"Step: {self.current_step}, "
            f"Position: {self.position}"
        )