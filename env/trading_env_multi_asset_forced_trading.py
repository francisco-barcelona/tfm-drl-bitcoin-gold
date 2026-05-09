import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """
    Entorn multi-actiu amb penalització per inactivitat.

    Accions:
        0 = BTC
        1 = GLD
        2 = USO

    L'agent selecciona quin actiu mantenir. Si manté el mateix actiu
    durant massa passos, rep una penalització per evitar estratègies passives.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        transaction_cost: float = 0.001,
        inactivity_penalty: float = 0.001,
        btc_return_column: str = "btc_target_ret",
        gld_return_column: str = "gld_target_ret",
        uso_return_column: str = "uso_target_ret",
    ) -> None:
        super().__init__()

        self.data = data.reset_index(drop=True).copy()
        self.feature_columns = feature_columns
        self.transaction_cost = transaction_cost
        self.inactivity_penalty = inactivity_penalty

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

        self.current_asset = 0
        self.holding_duration = 0

        self.action_space = spaces.Discrete(3)

        # features + current_asset + holding_duration
        obs_dim = len(feature_columns) + 2
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
            np.array(
                [self.current_asset, self.holding_duration],
                dtype=np.float32,
            ),
        ])

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.current_asset = 0
        self.holding_duration = 0

        return self._get_observation(), {}

    def step(self, action: int):
        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action: {action}")

        previous_asset = self.current_asset
        new_asset = int(action)

        row = self.data.iloc[self.current_step]

        if new_asset == 0:
            reward = row[self.btc_return_column]
        elif new_asset == 1:
            reward = row[self.gld_return_column]
        else:
            reward = row[self.uso_return_column]

        if new_asset != previous_asset:
            reward -= self.transaction_cost
            self.holding_duration = 0
        else:
            self.holding_duration += 1
            reward -= self.inactivity_penalty * self.holding_duration

        self.current_asset = new_asset
        self.current_step += 1

        done = self.current_step >= self.n_steps - 1
        truncated = False

        info = {
            "step": self.current_step,
            "asset": self.current_asset,
            "reward": reward,
            "holding_duration": self.holding_duration,
            "btc_target_ret": row[self.btc_return_column],
            "gld_target_ret": row[self.gld_return_column],
            "uso_target_ret": row[self.uso_return_column],
        }

        return self._get_observation(), float(reward), done, truncated, info

    def render(self):
        print(
            f"Step: {self.current_step}, "
            f"Asset: {self.current_asset}, "
            f"Holding duration: {self.holding_duration}"
        )