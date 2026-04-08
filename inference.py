import asyncio
import os
import textwrap
from typing import List, Optional
from openai import OpenAI

from models import QcAction
from server.qc_env_environment import QcEnvironment

# Upar wale variables rehne do, par main() ke andar hum direct os.environ use karenge
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASK_NAME = "quick-commerce-manager"
BENCHMARK = "qc_env"
MAX_STEPS = 7
TEMPERATURE = 0.7
MAX_TOKENS = 150

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI manager for a Quick-Commerce Dark Store.
    Every day, you must decide how much stock to order from the warehouse (between 0 and 50).
    Your goal is to maximize total profit by balancing 'revenue' against 'storage cost' and 'stockout penalties'.
    Reply with exactly one integer representing the reorder quantity — no quotes, no words, just the number.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, current_stock: int, predicted_demand: int, last_profit: float) -> str:
    return textwrap.dedent(
        f"""
        Day: {step}
        Current Stock in Store: {current_stock}
        Predicted Demand for Today: {predicted_demand}
        Last Day's Profit: {last_profit:.2f}
        How many items will you order today? (Enter 0-50):
        """
    ).strip()

def get_model_message(client: OpenAI, step: int, current_stock: int, predicted_demand: int, last_profit: float) -> int:
    user_prompt = build_user_prompt(step, current_stock, predicted_demand, last_profit)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return int(''.join(filter(str.isdigit, text)) or "20")
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return 20 

async def main() -> None:
    # 🔥 FIX 1: Exact wahi line jo Scaler ne mail mein maangi hai
    client = OpenAI(
        base_url=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"), 
        api_key=os.environ.get("API_KEY", "hf_your_token_here")
    )
    
    env = QcEnvironment()
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # 🔥 FIX 2: Yahan se 'await' hata diya kyunki ab environment normal function hai
        obs = env.reset()
        last_profit = 0.0

        for step in range(1, MAX_STEPS + 1):
            
            reorder_qty = get_model_message(client, step, obs.current_stock, obs.predicted_demand, last_profit)
            
            # 🔥 FIX 3: Yahan se bhi 'await' hata diya
            result = env.step(QcAction(reorder_quantity=reorder_qty))
            obs = result["observation"]
            reward = result["reward"]
            done = result["done"]
            last_profit = result["info"]["Profit"]

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=f"order({reorder_qty})", reward=reward, done=done, error=None)
            
            if done:
                success = sum(rewards) > 2.0 
                score = sum(rewards) / MAX_STEPS
                break

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    except Exception as e:
        print(f"Error occurred: {e}") # Isse hume exact error dikhega agar kuch phata toh
        log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())