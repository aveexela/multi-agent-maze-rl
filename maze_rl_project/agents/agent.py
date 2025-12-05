import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import List, Optional

def extract_toroidal_patch(passable_map: np.ndarray, y: int, x: int, size: int) -> np.ndarray:
    S = size | 1
    H, W = passable_map.shape
    r = S // 2
    ys = (y + np.arange(-r, r + 1)) % H
    xs = (x + np.arange(-r, r + 1)) % W
    return passable_map[np.ix_(ys, xs)]

class FeatureBuilder:
    def __init__(self, patch_size: int, k_nearest: int, k_mates: int = 2):
        self.patch_size = patch_size
        self.k_nearest = k_nearest
        self.k_mates = k_mates

    def __call__(self, info: dict, world_map: np.ndarray, pred_idx: int) -> np.ndarray:
        wm = np.array(world_map)
        H, W = wm.shape[:2]
        if wm.ndim == 3:
            walls = (wm[:, :, 0] == -1) & (wm[:, :, 1] == -1)
        else:
            walls = (wm == -1)
        passable = (~walls).astype(np.float32)
        p = info["predators"][pred_idx]
        y, x = int(p["y"]), int(p["x"])
        preys = [q for q in info.get("preys", []) if q.get("alive", True)]
        alive_cnt = len(preys)
        patch = extract_toroidal_patch(passable, y, x, self.patch_size).reshape(1, -1)
        prey_feats = []
        if preys:
            dy = np.array([q["y"] - y for q in preys], dtype=np.float32)
            dx = np.array([q["x"] - x for q in preys], dtype=np.float32)
            dn = np.abs(dy) + np.abs(dx)
            idx = np.argsort(dn)[:self.k_nearest]
            Hn, Wn = max(1, H), max(1, W)
            Dn = float(H + W)
            for i in idx:
                prey_feats.extend([dy[i] / Hn, dx[i] / Wn, min(dn[i], Dn) / Dn])
        while len(prey_feats) < self.k_nearest * 3:
            prey_feats.extend([0.0, 0.0, 1.0])
        mates = [q for i, q in enumerate(info["predators"]) if i != pred_idx and q.get("alive", True)]
        mate_feats = []
        if mates:
            dy = np.array([q["y"] - y for q in mates], dtype=np.float32)
            dx = np.array([q["x"] - x for q in mates], dtype=np.float32)
            dn = np.abs(dy) + np.abs(dx)
            idx = np.argsort(dn)[:self.k_mates]
            Hn, Wn = max(1, H), max(1, W)
            Dn = float(H + W)
            for i in idx:
                mate_feats.extend([dy[i] / Hn, dx[i] / Wn, min(dn[i], Dn) / Dn])
        while len(mate_feats) < self.k_mates * 3:
            mate_feats.extend([0.0, 0.0, 1.0])
        nearest_dn = prey_feats[2] if len(prey_feats) >= 3 else 1.0
        head = np.array([[y / max(1, H),
                          x / max(1, W),
                          float(p.get("alive", True)),
                          alive_cnt / max(1, H * W),
                          nearest_dn]], dtype=np.float32)
        return np.concatenate([head, patch,
                               np.asarray(prey_feats, dtype=np.float32).reshape(1, -1),
                               np.asarray(mate_feats, dtype=np.float32).reshape(1, -1)], axis=1)

class PolicyNetLite(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, num_actions: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.act = nn.ReLU()
        self.logits = nn.Linear(hidden, num_actions)
    def forward(self, x):
        h = self.act(self.fc1(x))
        return self.logits(h)

class Agent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model: Optional[nn.Module] = None
        self.f: Optional[FeatureBuilder] = None
        self.input_dim: Optional[int] = None
        self.n_actions: int = 5

    def reset(self, state, info):
        if self.model is not None:
            return
        pkl_path = __file__[:-8] + "/agent.pkl"
        obj = torch.load(pkl_path, map_location="cpu")

        state_dict = None
        meta = {}
        if isinstance(obj, dict) and "state_dict" in obj:
            state_dict = obj["state_dict"]
            meta = obj.get("meta", {}) or {}
        elif isinstance(obj, OrderedDict):
            state_dict = obj
        self.n_actions = int(meta.get("n_actions", 5))
        patch_size = int(meta.get("patch_size", 7))
        k_nearest = int(meta.get("k_nearest", 4))
        k_mates = int(meta.get("k_mates", 2))
        self.f = FeatureBuilder(patch_size=patch_size, k_nearest=k_nearest, k_mates=k_mates)
        world_map = info.get("map", None)
        if world_map is None:
            H, W = state.shape[0], state.shape[1]
            world_map = np.zeros((H, W), dtype=np.int32)
        sample_feats = self.f(info, world_map, pred_idx=0)
        D = int(meta.get("input_dim", sample_feats.shape[1]))
        self.input_dim = D
        self.model = PolicyNetLite(D, hidden=64, num_actions=self.n_actions).to(self.device)
        if state_dict is not None:
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except Exception:
                self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def get_actions(self, state, info) -> List[int]:
        world_map = info.get("map", None)
        if world_map is None:
            H, W = state.shape[0], state.shape[1]
            world_map = np.zeros((H, W), dtype=np.int32)
        K = len(info.get("predators", [])) or 1
        feats = [self.f(info, world_map, pred_idx=k) for k in range(K)]
        x = torch.tensor(np.concatenate(feats, axis=0), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x)
            a = torch.argmax(logits, dim=-1)
        return [int(i) for i in a.cpu().numpy().tolist()]