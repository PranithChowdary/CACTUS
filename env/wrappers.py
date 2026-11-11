import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, Tuple

class TextEmbeddingWrapper(gym.ObservationWrapper):
    """
    Wraps the MetaToolEnv to replace 'prompt_id' with a dense embedding.
    """
    def __init__(self, env: gym.Env, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(env)
        self.model = SentenceTransformer(model_name)
        
        # Get embedding dimension
        dummy_emb = self.model.encode("dummy")
        self.emb_dim = dummy_emb.shape[0]
        
        # Update observation space
        orig_space = env.observation_space
        self.observation_space = spaces.Dict({
            "selection": orig_space["selection"],
            "embedding": spaces.Box(low=-np.inf, high=np.inf, shape=(self.emb_dim,), dtype=np.float32)
        })
        
        # Cache embeddings to avoid redundant computation
        self.embedding_cache = {}
        print(f"TextEmbeddingWrapper initialized with {model_name} (dim={self.emb_dim})")

    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        prompt_id = obs["prompt_id"]
        
        if prompt_id not in self.embedding_cache:
            # Get prompt string from env info (needs careful handling as wrappers don't see info in reset/step easily)
            # However, we can access the underlying env
            prompt = self.env.unwrapped.df.iloc[prompt_id]["Query"]
            self.embedding_cache[prompt_id] = self.model.encode(prompt)
            
        return {
            "selection": obs["selection"],
            "embedding": self.embedding_cache[prompt_id]
        }

    def action_masks(self) -> np.ndarray:
        """Explicit passthrough so sb3_contrib's get_action_masks() finds this
        regardless of how many wrappers sit between the policy and the base env."""
        return self.env.unwrapped.action_masks()
