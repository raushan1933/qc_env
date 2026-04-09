from openenv.core.env_server import Environment  
from models import QcAction, QcObservation, QcState
import random

SELLING_PRICE = 10.0
STORAGE_COST = 2.0
STOCKOUT_PENALTY = 5.0
MAX_DAYS = 7

class QcEnvironment(Environment[QcAction, QcObservation, QcState]):
    def __init__(self):
        super().__init__()
        self.current_state = QcState(current_stock=10, day=1, total_profit=0.0)

    def reset(self) -> QcObservation:
        self.current_state = QcState(current_stock=10, day=1, total_profit=0.0)
        return self._get_observation()

    def _get_observation(self) -> QcObservation:
        predicted = random.randint(10, 30)
        return QcObservation(
            current_stock=self.current_state.current_stock,
            predicted_demand=predicted,
            day=self.current_state.day
        )

    def step(self, action: QcAction):
        self.current_state.current_stock += action.reorder_quantity
        actual_demand = random.randint(10, 30)
        items_sold = min(self.current_state.current_stock, actual_demand)

        leftover_stock = self.current_state.current_stock - items_sold
        missed_sales = actual_demand - items_sold

        revenue = items_sold * SELLING_PRICE
        holding_cost = leftover_stock * STORAGE_COST
        penalty = missed_sales * STOCKOUT_PENALTY

        daily_profit = revenue - holding_cost - penalty
        self.current_state.total_profit += daily_profit

        self.current_state.current_stock = leftover_stock
        self.current_state.day += 1
        done = self.current_state.day > MAX_DAYS

        reward = max(0.0, min(1.0, (daily_profit + 150) / 450.0))

        return {
            "observation": self._get_observation(),
            "reward": round(reward, 2),
            "done": done,
            "info": {"Sold": items_sold, "Leftover": leftover_stock, "Missed": missed_sales, "Profit": daily_profit}
        }
    
    def state(self) -> QcState:
        return self.current_state