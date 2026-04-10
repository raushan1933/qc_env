# 🛒 Quick-Commerce Dark Store AI Manager

An autonomous Agentic AI environment built for the **Meta PyTorch OpenEnv Hackathon x Scaler School of Technology**. This project simulates a real-world Quick-Commerce Dark Store where an LLM acts as an independent Inventory Manager, replacing traditional rule-based ML models with autonomous reasoning.

## 🚀 The Vision
In the hyper-competitive space of 10-minute delivery (quick commerce), inventory management is the biggest bottleneck. Traditional systems use static forecasting. This project introduces an **Agentic Workflow** where the AI actively reads daily demand, balances conflicting constraints, and makes independent reorder decisions to maximize profitability.

## 🧠 How It Works
The environment is designed to test an AI agent's ability to balance two critical financial metrics:
1. **Storage Costs:** Penalty for over-ordering and holding excess inventory.
2. **Stockout Penalties:** Severe penalty for under-ordering, leading to missed revenue and bad customer experience.

The agent interacts with the `QcEnvironment` over a 7-day episode, receiving observations (current stock, predicted demand) and returning an action (reorder quantity between 0-50).

## ⚙️ Environment Difficulty Levels (Tasks)
The simulation evaluates the agent across three distinct difficulty levels defined in `openenv.yaml`:
* **🟢 Easy Task (`task_easy`):** Stable and predictable demand. Tests basic arithmetic and logical reasoning.
* **🟡 Medium Task (`task_medium`):** Volatile demand with unexpected spikes. Tests the agent's adaptability and risk management.
* **🔴 Hard Task (`task_hard`):** High penalty spikes for stockouts. Tests if the agent can prioritize customer retention over minor storage costs.

## 🛠️ Technical Stack
* **Framework:** OpenEnv Standard, FastAPI
* **AI Model:** `Qwen/Qwen2.5-72B-Instruct` (via LiteLLM Proxy)
* **Architecture:** * `qc_env_environment.py`: Contains the physics, math, and rules of the dark store.
  * `inference.py`: The control loop connecting the environment to the LLM.
  * `grader.py`: Evaluates the agent's success rate across all tasks.

## 📈 Impact
By shifting from predictive ML to autonomous Agentic AI, this approach demonstrates how quick-commerce startups can dynamically manage dark stores without manual intervention, laying the groundwork for highly profitable, scalable operations.
