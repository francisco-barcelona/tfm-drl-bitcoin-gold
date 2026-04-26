import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """
    Multi-asset trading environment

    Actions:
        0 = BTC
        1 = GLD
        2 = USO
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        transaction_cost: float = 0.001,
    ):
        super().__init__()

        self.data = data.reset_index(drop=True).copy()
        self.feature_columns = feature_columns
        self.transaction_cost = transaction_cost

        self.current_step = 0
        self.n_steps = len(self.data)

        self.current_asset = 0  # start in BTC

        self.action_space = spaces.Discrete(3)

        obs_dim = len(feature_columns) + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        features = row[self.feature_columns].values.astype(np.float32)

        return np.concatenate([features, [self.current_asset]])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.current_asset = 0

        return self._get_observation(), {}

    def step(self, action):

        prev_asset = self.current_asset
        new_asset = action

        row = self.data.iloc[self.current_step]

        btc_ret = row["btc_ret_4h"]
        gld_ret = row["gld_ret_4h"]
        uso_ret = row["uso_ret_4h"]

        if new_asset == 0:
            reward = btc_ret
        elif new_asset == 1:
            reward = gld_ret
        else:
            reward = uso_ret

        # cost if switching asset
        if new_asset != prev_asset:
            reward -= self.transaction_cost

        self.current_asset = new_asset
        self.current_step += 1

        done = self.current_step >= self.n_steps - 1
        truncated = False

        obs = self._get_observation()

        info = {
            "asset": self.current_asset,
            "reward": reward
        }

        return obs, float(reward), done, truncated, info