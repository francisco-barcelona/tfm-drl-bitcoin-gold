import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """
    Entorn multi-actiu amb:
    - features de relative strength i momentum
    - penalització per inactivitat
    - cost de canvi d'actiu dinàmic segons la volatilitat

    Accions:
        0 = BTC
        1 = GLD
        2 = USO
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        base_transaction_cost: float = 0.001,
        dynamic_cost_weight: float = 0.05,
        inactivity_penalty: float = 0.001,
        btc_return_column: str = "btc_target_ret",
        gld_return_column: str = "gld_target_ret",
        uso_return_column: str = "uso_target_ret",
        btc_vol_column: str = "btc_vol_12",
        gld_vol_column: str = "gld_vol_12",
        uso_vol_column: str = "uso_vol_12",
    ) -> None:
        super().__init__()

        self.data = data.reset_index(drop=True).copy()
        self.feature_columns = feature_columns

        self.base_transaction_cost = base_transaction_cost
        self.dynamic_cost_weight = dynamic_cost_weight
        self.inactivity_penalty = inactivity_penalty

        self.return_columns = {
            0: btc_return_column,
            1: gld_return_column,
            2: uso_return_column,
        }

        self.vol_columns = {
            0: btc_vol_column,
            1: gld_vol_column,
            2: uso_vol_column,
        }

        required_columns = (
            feature_columns
            + list(self.return_columns.values())
            + list(self.vol_columns.values())
        )

        missing_columns = [c for c in required_columns if c not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

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

        selected_return = row[self.return_columns[new_asset]]
        selected_volatility = row[self.vol_columns[new_asset]]

        reward = selected_return

        switching_cost = 0.0
        inactivity_cost = 0.0

        if new_asset != previous_asset:
            switching_cost = (
                self.base_transaction_cost
                + self.dynamic_cost_weight * selected_volatility
            )
            reward -= switching_cost
            self.holding_duration = 0
        else:
            self.holding_duration += 1
            inactivity_cost = self.inactivity_penalty * self.holding_duration
            reward -= inactivity_cost

        self.current_asset = new_asset
        self.current_step += 1

        done = self.current_step >= self.n_steps - 1
        truncated = False

        info = {
            "step": self.current_step,
            "asset": self.current_asset,
            "reward": reward,
            "selected_return": selected_return,
            "selected_volatility": selected_volatility,
            "switching_cost": switching_cost,
            "inactivity_cost": inactivity_cost,
            "holding_duration": self.holding_duration,
            "btc_target_ret": row[self.return_columns[0]],
            "gld_target_ret": row[self.return_columns[1]],
            "uso_target_ret": row[self.return_columns[2]],
        }

        return self._get_observation(), float(reward), done, truncated, info

    def render(self):
        print(
            f"Step: {self.current_step}, "
            f"Asset: {self.current_asset}, "
            f"Holding duration: {self.holding_duration}"
        )