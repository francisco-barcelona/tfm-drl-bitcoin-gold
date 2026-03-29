\## Objective



The main objective is to evaluate whether adding exogenous information (gold price) improves the performance and robustness of a DRL agent compared to a baseline model using only Bitcoin data.



\## Approach



\- Two agents are trained under identical conditions:

&#x20; - BTC-only (baseline)

&#x20; - BTC + GLD (enhanced model)

\- The problem is modeled as a Markov Decision Process (MDP)

\- Training is performed using DRL algorithms (e.g., PPO)

\- Evaluation is done via backtesting with transaction costs and out-of-sample data



