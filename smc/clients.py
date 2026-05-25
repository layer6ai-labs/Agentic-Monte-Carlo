
import requests
import logging
from typing import Optional, Dict, Any

from agentenv.envs.textcraft import TextCraftTask


# Configure logging
logger = logging.getLogger(__name__)

class AgentGymClient:
    """
    A base client for interacting with AgentGym environments over HTTP.
    Handles connection logic, request wrapping, and standard endpoints (step, reset, close).
    """
    def __init__(self, base_url: str, timeout: int = 1500):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.env_id: Optional[int] = None

    def _request(self, method: str, path: str, json_data: Optional[Dict] = None, params: Optional[Dict] = None) -> Any:
        """
        Internal helper to handle HTTP requests with error handling and URL construction.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                json=json_data,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Request failed for {method} {url}: {e}")
            raise

    def _ensure_connected(self):
        """Helper to ensure the environment has been created before actions are taken."""
        if self.env_id is None:
            raise RuntimeError("Environment not initialized. Call create_env() first.")

    def create_env(self, payload) -> int:
        """
        POST /create
        Initializes the environment. Accepts arbitrary kwargs to pass as JSON payload.
        """
        # Sends kwargs as the JSON body (e.g., commands={}, goal="...")
        data = self._request("POST", "create", json_data=payload)
        
        # Robustly handle if server returns raw int ID or {"id": 123} dict
        self.env_id = data.get("id") if isinstance(data, dict) else data
        
        logger.info(f"Created environment with ID: {self.env_id}")
        return self.env_id

    def step(self, action: str) -> Dict[str, Any]:
        """
        POST /step
        Executes an action in the environment.
        """
        self._ensure_connected()
        payload = {"id": self.env_id, "action": action}
        return self._request("POST", "step", json_data=payload)

    def reset(self, data_idx: int = 0) -> Dict[str, Any]:
        """
        POST /reset
        Resets the environment to a specific state/level index.
        """
        self._ensure_connected()
        payload = {"id": self.env_id, "data_idx": data_idx}
        return self._request("POST", "reset", json_data=payload)

    def observe(self) -> Any:
        """
        GET /observation
        Retrieves the current observation.
        """
        self._ensure_connected()
        return self._request("GET", "observation", params={"id": self.env_id})

    def close(self):
        """
        POST /close
        Closes the environment session.
        """
        if self.env_id is not None:
            try:
                payload = {"id": self.env_id}
                self._request("POST", "close", json_data=payload)
                logger.info(f"Closed environment {self.env_id}")
            except Exception as e:
                logger.warning(f"Error closing environment: {e}")
            finally:
                self.env_id = None

class TextCraftClient(AgentGymClient):

    def __init__(self, base_url: str, timeout: int = 1500):
        super().__init__(base_url, timeout)
        self.create_env({})

    def __len__(self):
        return 200
    
    def observe(self):
        # textcraft is weird where they use detail instead of observe
        return self._request("GET", "detail", params={"id": self.env_id})


#TODO need to clean this up    
class CustomTextCraftTask(TextCraftTask):
    env_client_cls = TextCraftClient
    env_name = "TextCraft"

    def __init__(
        self, client_args, *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, *args, n_clients=n_clients, **kwargs)

