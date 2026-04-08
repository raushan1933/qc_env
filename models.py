from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field

class QcAction(Action):
    """Action for Quick Commerce - how much stock to order."""
    reorder_quantity: int = Field(ge=0, le=50, description="Items to order from warehouse today (0 to 50)")

class QcObservation(Observation):
    """Observation for Quick Commerce - current store status."""
    current_stock: int = Field(description="Current inventory level in dark store")
    predicted_demand: int = Field(description="Predicted customer demand for tomorrow")
    day: int = Field(description="Current day number in the simulation")

class QcState(BaseModel):
    """Internal state of the simulation."""
    current_stock: int
    day: int
    total_profit: float