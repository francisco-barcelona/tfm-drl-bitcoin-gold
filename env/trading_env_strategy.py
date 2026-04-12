import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """
    Entorn de trading basat en selecció d'estratègies, sense fuga d'informació.

    L'agent no tria directament una posició final, sinó una estratègia:
        0 = no operar (flat)
        1 = seguir tendència (momentum)
        2 = reversió a la mitjana (mean reversion)

    La posició es deriva a partir d'una senyal observable en el pas actual
    (per defecte, btc_ret_4h), i no del retorn futur.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        price_column: str = "btc_close",
        signal_column: str = "btc_ret_4h",
        volatility_column: str = "btc_vol_12",
        transaction_cost: float = 0.001,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()

        if price_column not in data.columns:
            raise ValueError(f"Missing price column: {price_column}")

        if signal_column not in data.columns:
            raise ValueError(f"Missing signal column: {signal_column}")

        if volatility_column not in data.columns:
            raise ValueError(f"Missing volatility column: {volatility_column}")

        missing_features = [c for c in feature_columns if c not in data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        self.data = data.reset_index(drop=True).copy()
        self.feature_columns = feature_columns
        self.price_column = price_column
        self.signal_column = signal_column
        self.volatility_column = volatility_column
        self.transaction_cost = transaction_cost
        self.alpha = alpha

        self.n_steps = len(self.data)
        self.current_step = 0
        self.position = 0  # -1 short, 0 flat, 1 long

        # 0 = flat, 1 = momentum, 2 = mean reversion
        self.action_space = spaces.Discrete(3)

        # observació = features + posició actual
        obs_dim = len(self.feature_columns) + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        """
        Construeix l'observació actual a partir de les features del dataset
        i de la posició actual de l'agent.
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

    def _strategy_to_position(self, action: int, signal_value: float) -> int:
        """
        Converteix una acció-estratègia en una posició concreta a partir
        d'una senyal observable del pas actual.

        Regles:
        - 0: no operar -> posició 0
        - 1: momentum -> segueix el signe de la senyal
        - 2: mean reversion -> pren la direcció contrària
        """
        if action == 0:
            return 0

        if signal_value >= 0:
            signal_direction = 1
        else:
            signal_direction = -1

        if action == 1:  # momentum
            return signal_direction

        if action == 2:  # mean reversion
            return -signal_direction

        raise ValueError(f"Invalid strategy action: {action}")

    def step(self, action: int):
        """
        Executa una acció-estratègia i calcula la recompensa.

        La recompensa es defineix com:
            reward = posició * retorn_futur - alpha * volatilitat

        On:
        - la posició es decideix a partir d'una senyal observable actual
        - el retorn es calcula entre el preu actual i el següent
        - s'aplica cost de transacció si hi ha canvi de posició
        """
        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action: {action}")

        prev_position = self.position

        current_row = self.data.iloc[self.current_step]
        current_price = current_row[self.price_column]
        signal_value = current_row[self.signal_column]
        volatility = current_row[self.volatility_column]

        new_position = self._strategy_to_position(action, signal_value)

        done = self.current_step >= self.n_steps - 2
        truncated = False

        next_price = self.data.iloc[self.current_step + 1][self.price_column]
        asset_return = (next_price / current_price) - 1.0

        reward = (new_position * asset_return) - self.alpha * volatility

        if new_position != prev_position:
            reward -= self.transaction_cost

        self.position = new_position
        self.current_step += 1

        observation = self._get_observation()
        info = {
            "step": self.current_step,
            "position": self.position,
            "asset_return": asset_return,
            "signal_value": signal_value,
            "volatility": volatility,
            "strategy_action": action,
        }

        return observation, float(reward), done, truncated, info

    def render(self):
        """
        Mostra l'estat actual de l'entorn.
        """
        print(
            f"Step: {self.current_step}, "
            f"Position: {self.position}"
        )