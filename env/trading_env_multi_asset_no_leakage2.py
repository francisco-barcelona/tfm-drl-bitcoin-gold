import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """
    Entorn multi-actiu sense fuga d'informació.

    L'agent selecciona quin actiu mantenir en cada pas temporal:

        0 = BTC
        1 = GLD
        2 = USO

    Les features que observa l'agent han d'estar desplaçades temporalment
    i representar informació passada.

    La recompensa es calcula amb columnes target separades:

        btc_target_ret
        gld_target_ret
        uso_target_ret

    Això evita utilitzar com a feature el mateix retorn que després
    s'empra per calcular la recompensa.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        transaction_cost: float = 0.001,
        btc_return_column: str = "btc_target_ret",
        gld_return_column: str = "gld_target_ret",
        uso_return_column: str = "uso_target_ret",
    ) -> None:
        super().__init__()

        self.data = data.reset_index(drop=True).copy()
        self.feature_columns = feature_columns
        self.transaction_cost = transaction_cost

        self.btc_return_column = btc_return_column
        self.gld_return_column = gld_return_column
        self.uso_return_column = uso_return_column

        missing_features = [c for c in feature_columns if c not in data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        required_return_columns = [
            btc_return_column,
            gld_return_column,
            uso_return_column,
        ]

        missing_returns = [c for c in required_return_columns if c not in data.columns]
        if missing_returns:
            raise ValueError(f"Missing target return columns: {missing_returns}")

        self.current_step = 0
        self.n_steps = len(self.data)

        # Comencem en BTC per defecte
        self.current_asset = 0

        # 0 = BTC, 1 = GLD, 2 = USO
        self.action_space = spaces.Discrete(3)

        # Observació = features passades + actiu actual
        obs_dim = len(feature_columns) + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        row = self.data.iloc[self.current_step]
        features = row[self.feature_columns].to_numpy(dtype=np.float32)

        obs = np.concatenate([
            features,
            np.array([self.current_asset], dtype=np.float32),
        ])

        return obs

    def reset(self, seed=None, options=None):
        """
        Reinicia l'entorn a l'estat inicial.
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.current_asset = 0

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action: int):
        """
        Executa una acció d'assignació d'actiu.

        L'agent observa informació passada i decideix quin actiu mantenir.
        La recompensa es calcula amb el retorn target del pas actual.

        Si canvia d'actiu, s'aplica un cost de transacció.
        """

        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action: {action}")

        prev_asset = self.current_asset
        new_asset = int(action)

        row = self.data.iloc[self.current_step]

        if new_asset == 0:
            reward = row[self.btc_return_column]
        elif new_asset == 1:
            reward = row[self.gld_return_column]
        else:
            reward = row[self.uso_return_column]

        if new_asset != prev_asset:
            reward -= self.transaction_cost

        self.current_asset = new_asset
        self.current_step += 1

        done = self.current_step >= self.n_steps - 1
        truncated = False

        observation = self._get_observation()

        info = {
            "step": self.current_step,
            "asset": self.current_asset,
            "selected_asset": self.current_asset,
            "reward": reward,
            "btc_target_ret": row[self.btc_return_column],
            "gld_target_ret": row[self.gld_return_column],
            "uso_target_ret": row[self.uso_return_column],
        }

        return observation, float(reward), done, truncated, info

    def render(self):
        print(
            f"Step: {self.current_step}, "
            f"Asset: {self.current_asset}"
        )