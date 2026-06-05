\## Objective



The objective of this project is to study how different environment designs, reward formulations, and state representations affect the behaviour of Deep Reinforcement Learning (DRL) agents in financial markets.



Starting from a Bitcoin-only trading environment, the project progressively evolves towards a multi-asset setting including Bitcoin (BTC), gold through the SPDR Gold Shares ETF (GLD), and oil through the United States Oil Fund ETF (USO).



Special attention is given to:



\* The incorporation of exogenous information.

\* Dynamic asset allocation.

\* Relative reward formulations.

\* Risk-aware reward design.

\* The impact of transaction costs and inactivity penalties.

\* The prevention of lookahead bias through strict temporal separation between features and future returns.



The goal is not to develop a production-ready trading system, but to analyse how these design choices influence the policies learned by DRL agents.



\## Approach



\* Financial markets are modelled as Markov Decision Processes (MDPs).

\* Agents are trained using Proximal Policy Optimization (PPO) implemented with Stable-Baselines3.

\* Multiple environments are evaluated, ranging from single-asset Bitcoin trading to multi-asset asset allocation problems involving BTC, GLD and USO.

\* Different reward functions are tested, including:



&#x20; \* Simple returns

&#x20; \* Risk-adjusted rewards

&#x20; \* Sharpe-inspired rewards

&#x20; \* Relative rewards

&#x20; \* Inactivity penalties

&#x20; \* Dynamic transaction costs

\* State representations are progressively enriched using:



&#x20; \* Historical returns

&#x20; \* Volatility measures

&#x20; \* Momentum indicators

&#x20; \* Relative strength features

\* Evaluation is performed through out-of-sample backtesting using cumulative return, Sharpe ratio, Sortino ratio, maximum drawdown and asset-switching statistics.

\* All experiments are designed to avoid lookahead bias by ensuring that the agent only observes information available at decision time.



