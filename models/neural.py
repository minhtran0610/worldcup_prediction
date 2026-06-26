from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from features.form import FORM_FEATURE_DIM, FORM_N, get_form_sequence
from models.grid import build_grid, derive_markets

EMBED_DIM: int = 32
GRU_HIDDEN: int = 32
CONTEXT_DIM: int = 13

# Clip floor for NLL loss to prevent log(0)
_LOG_CLIP: float = 1e-10


class ScoreGridNet(nn.Module):
    """Neural network that predicts (lambda_home, lambda_away, rho) for the Dixon-Coles score grid.

    Architecture:
        - Team embedding for home and away (index 0 reserved for UNK)
        - Shared GRU form encoder applied to each team's recent-match sequence
        - Small MLP on 11-dim context features
        - Fusion MLP that concatenates all branches and outputs the 3 distribution parameters
    """

    def __init__(self, n_teams: int) -> None:
        super().__init__()

        # Index 0 is reserved for UNK; known teams start at 1
        self.team_embed = nn.Embedding(n_teams, EMBED_DIM, padding_idx=0)

        # Shared GRU encoder — applied to both home and away sequences
        self.gru = nn.GRU(
            input_size=FORM_FEATURE_DIM,
            hidden_size=GRU_HIDDEN,
            num_layers=1,
            batch_first=True,
        )

        # Context MLP: 11 scalar features -> 16-dim
        self.context_mlp = nn.Sequential(
            nn.Linear(CONTEXT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        # Fusion: 32+32+32+32+16 = 144 -> 3
        fusion_in = EMBED_DIM + EMBED_DIM + GRU_HIDDEN + GRU_HIDDEN + 16
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),
        )

    def forward(
        self,
        home_idx: torch.Tensor,  # (batch,) long
        away_idx: torch.Tensor,  # (batch,) long
        home_seq: torch.Tensor,  # (batch, FORM_N, 5) float
        away_seq: torch.Tensor,  # (batch, FORM_N, 5) float
        context: torch.Tensor,  # (batch, 13) float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (lambda_home, lambda_away, rho) each shape (batch,)."""
        home_emb = self.team_embed(home_idx)  # (batch, 32)
        away_emb = self.team_embed(away_idx)  # (batch, 32)

        # GRU: take the final hidden state as the form context vector
        _, h_home = self.gru(home_seq)  # h_home: (1, batch, 32)
        _, h_away = self.gru(away_seq)  # h_away: (1, batch, 32)
        h_home = h_home.squeeze(0)  # (batch, 32)
        h_away = h_away.squeeze(0)  # (batch, 32)

        ctx_out = self.context_mlp(context)  # (batch, 16)

        fused = torch.cat([home_emb, away_emb, h_home, h_away, ctx_out], dim=1)  # (batch, 144)
        out = self.fusion(fused)  # (batch, 3)

        lambda_home = F.softplus(out[:, 0])  # (batch,) — positive
        lambda_away = F.softplus(out[:, 1])  # (batch,) — positive
        rho = torch.tanh(out[:, 2]) * 0.2  # (batch,) — small magnitude

        return lambda_home, lambda_away, rho


def _build_context_tensor(
    rows: pd.DataFrame,
    device: torch.device,
) -> torch.Tensor:
    """Construct the (N, 13) normalised context tensor from a DataFrame slice.

    Features (in order):
        0:  neutral (float)
        1:  rest_days_home / 180  (NaN -> 0)
        2:  rest_days_away / 180  (NaN -> 0)
        3:  elo_diff / 400
        4:  is_knockout (float)
        5:  is_host_home (float)
        6:  is_host_away (float)
        7:  squad_top5_home (float, fraction of squad in Big 5 leagues; default 0.0)
        8:  squad_top5_away (float)
        9:  squad_caps_home (float, avg caps / 100; default 0.0)
        10: squad_caps_away (float)
        11: squad_goals_home (float, career intl goals / total squad caps; default 0.0)
        12: squad_goals_away (float)
    """
    neutral = rows["neutral"].to_numpy(dtype=np.float32)
    rest_h = rows["rest_days_home"].fillna(0.0).to_numpy(dtype=np.float32) / 180.0
    rest_a = rows["rest_days_away"].fillna(0.0).to_numpy(dtype=np.float32) / 180.0
    elo_diff = rows["elo_diff"].to_numpy(dtype=np.float32) / 400.0
    is_ko = rows["is_knockout"].to_numpy(dtype=np.float32)
    is_hh = rows["is_host_home"].to_numpy(dtype=np.float32)
    is_ha = rows["is_host_away"].to_numpy(dtype=np.float32)

    # Squad features — present when registry was used in derive_context; otherwise 0.0
    def _sq(col: str) -> np.ndarray:
        return (
            rows[col].fillna(0.0).to_numpy(dtype=np.float32)
            if col in rows.columns
            else np.zeros(len(rows), dtype=np.float32)
        )

    arr = np.stack(
        [
            neutral,
            rest_h,
            rest_a,
            elo_diff,
            is_ko,
            is_hh,
            is_ha,
            _sq("squad_top5_home"),
            _sq("squad_top5_away"),
            _sq("squad_caps_home"),
            _sq("squad_caps_away"),
            _sq("squad_goals_home"),
            _sq("squad_goals_away"),
        ],
        axis=1,
    )
    return torch.from_numpy(arr).to(device)


def _nll_loss_batch(
    lambda_home: torch.Tensor,
    lambda_away: torch.Tensor,
    rho: torch.Tensor,
    home_scores: torch.Tensor,
    away_scores: torch.Tensor,
) -> torch.Tensor:
    """Mean scoreline NLL over a batch, computed entirely in PyTorch for autograd.

    Reimplements build_grid in torch ops so gradients flow through lambda_home,
    lambda_away, and rho.
    """
    device = lambda_home.device

    goals = torch.arange(8, dtype=torch.float32, device=device)  # (8,)
    factorials = torch.tensor(
        [float(math.factorial(k)) for k in range(8)], dtype=torch.float32, device=device
    )  # (8,)

    lh = lambda_home.unsqueeze(1)  # (batch, 1)
    la = lambda_away.unsqueeze(1)  # (batch, 1)

    # Poisson PMF vectors: exp(-λ) * λ^k / k!
    pmf_h = torch.exp(-lh) * (lh**goals) / factorials  # (batch, 8)
    pmf_a = torch.exp(-la) * (la**goals) / factorials  # (batch, 8)

    # Outer product per sample → (batch, 8, 8)
    grid = pmf_h.unsqueeze(2) * pmf_a.unsqueeze(1)

    # Dixon-Coles τ corrections (modify in-place on cloned cells to preserve grad)
    rho_b = rho  # (batch,)
    lh_s = lambda_home  # (batch,)
    la_s = lambda_away  # (batch,)

    grid_00 = grid[:, 0, 0] * (1.0 - lh_s * la_s * rho_b)
    grid_10 = grid[:, 1, 0] * (1.0 + la_s * rho_b)
    grid_01 = grid[:, 0, 1] * (1.0 + lh_s * rho_b)
    grid_11 = grid[:, 1, 1] * (1.0 - rho_b)

    # Scatter corrected values back — use clone + index assignment
    grid = grid.clone()
    grid[:, 0, 0] = grid_00
    grid[:, 1, 0] = grid_10
    grid[:, 0, 1] = grid_01
    grid[:, 1, 1] = grid_11

    # Renormalise
    grid_sum = grid.sum(dim=(1, 2), keepdim=True).clamp(min=_LOG_CLIP)
    grid = grid / grid_sum

    # Gather probability at actual scoreline
    batch_size = lambda_home.shape[0]
    batch_idx = torch.arange(batch_size, device=device)
    h_idx = home_scores.clamp(0, 7).long()
    a_idx = away_scores.clamp(0, 7).long()

    p = grid[batch_idx, h_idx, a_idx].clamp(min=_LOG_CLIP)
    return -torch.log(p).mean()


class NeuralModel:
    """Wrapper around ScoreGridNet with fit/predict_batch/save/load interface.

    Matches the interface of DixonColesModel.predict_batch so it can be dropped
    into the existing backtest / eval pipeline.
    """

    def __init__(
        self,
        n_epochs: int = 150,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        patience: int = 15,
        device: str = "auto",
        checkpoint_path: str | None = None,
    ) -> None:
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.checkpoint_path = checkpoint_path

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._team_vocab: dict[str, int] = {}
        self._net: ScoreGridNet | None = None
        self._train_results: pd.DataFrame | None = None
        # Epoch at which the best val NLL occurred (set by fit). Used by callers
        # that want to refit on all data for the same number of epochs.
        self.best_epoch_: int | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _team_index(self, team: str) -> int:
        """Return vocab index; 0 for unknown teams (UNK slot)."""
        return self._team_vocab.get(team, 0)

    def _build_batch_tensors(
        self,
        rows: pd.DataFrame,
        results_for_form: pd.DataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (home_idx, away_idx, home_seq, away_seq, context) tensors for a batch."""
        n = len(rows)
        home_idxs = np.array([self._team_index(t) for t in rows["home_team"]], dtype=np.int64)
        away_idxs = np.array([self._team_index(t) for t in rows["away_team"]], dtype=np.int64)

        home_seqs = np.zeros((n, FORM_N, FORM_FEATURE_DIM), dtype=np.float32)
        away_seqs = np.zeros((n, FORM_N, FORM_FEATURE_DIM), dtype=np.float32)

        for i, row in enumerate(rows.itertuples(index=False)):
            match_date: pd.Timestamp = row.date
            home_seqs[i] = get_form_sequence(results_for_form, row.home_team, match_date)
            away_seqs[i] = get_form_sequence(results_for_form, row.away_team, match_date)

        ctx = _build_context_tensor(rows, self._device)

        return (
            torch.from_numpy(home_idxs).to(self._device),
            torch.from_numpy(away_idxs).to(self._device),
            torch.from_numpy(home_seqs).to(self._device),
            torch.from_numpy(away_seqs).to(self._device),
            ctx,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_results: pd.DataFrame,
        val_results: pd.DataFrame | None = None,
        early_stopping: bool = True,
    ) -> NeuralModel:
        """Train ScoreGridNet on train_results with early stopping on val NLL.

        Builds the team vocabulary from train_results.
        If val_results is None, uses the last 10% of train_results (chronologically) as val.
        Prints epoch loss every 10 epochs.

        If early_stopping is False, trains on ALL of train_results for the full
        n_epochs with no held-out validation and no early stopping — used for the
        production refit on all data. The val_results argument is ignored in that case.
        """
        # Store the full training data for form look-ups during predict_batch
        self._train_results = train_results.reset_index(drop=True)

        # Build team vocabulary — index 0 is reserved for UNK
        teams = sorted(
            set(train_results["home_team"].unique()) | set(train_results["away_team"].unique())
        )
        self._team_vocab = {t: i + 1 for i, t in enumerate(teams)}
        n_teams = len(teams) + 1  # +1 for UNK slot

        # Chronological val split if caller did not provide one.
        # When early_stopping is off (production refit), train on ALL data; val_df
        # is a logging proxy only (full corpus) — nothing is held out from training.
        if not early_stopping:
            train_df = train_results.reset_index(drop=True)
            val_df = train_df
        elif val_results is None:
            n_val = max(1, int(len(train_results) * 0.10))
            train_df = train_results.iloc[:-n_val].reset_index(drop=True)
            val_df = train_results.iloc[-n_val:].reset_index(drop=True)
        else:
            train_df = train_results.reset_index(drop=True)
            val_df = val_results.reset_index(drop=True)

        # Form sequences are always looked up from the full training corpus
        form_df = self._train_results

        self._net = ScoreGridNet(n_teams=n_teams).to(self._device)
        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val_nll: float = float("inf")
        best_state: dict[str, Any] = {}
        best_epoch: int = self.n_epochs
        epochs_no_improve: int = 0

        n_train = len(train_df)
        train_hs = torch.from_numpy(train_df["home_score"].to_numpy(dtype=np.int64).copy()).to(
            self._device
        )
        train_as = torch.from_numpy(train_df["away_score"].to_numpy(dtype=np.int64).copy()).to(
            self._device
        )

        # Precompute all train tensors once — avoids re-running get_form_sequence each batch
        print("Precomputing train form sequences...", flush=True)
        train_home_idx, train_away_idx, train_home_seq, train_away_seq, train_ctx = (
            self._build_batch_tensors(train_df, form_df)
        )

        # Pre-build val tensors (fixed each epoch)
        print("Precomputing val form sequences...", flush=True)
        val_home_idx, val_away_idx, val_home_seq, val_away_seq, val_ctx = self._build_batch_tensors(
            val_df, form_df
        )
        val_hs = torch.from_numpy(val_df["home_score"].to_numpy(dtype=np.int64).copy()).to(
            self._device
        )
        val_as = torch.from_numpy(val_df["away_score"].to_numpy(dtype=np.int64).copy()).to(
            self._device
        )

        for epoch in range(1, self.n_epochs + 1):
            self._net.train()
            perm = torch.randperm(n_train, device=self._device)
            epoch_loss_sum = 0.0
            n_batches = 0

            for start in range(0, n_train, self.batch_size):
                idx = perm[start : start + self.batch_size]

                home_idx = train_home_idx[idx]
                away_idx = train_away_idx[idx]
                home_seq = train_home_seq[idx]
                away_seq = train_away_seq[idx]
                ctx = train_ctx[idx]
                hs = train_hs[idx]
                as_ = train_as[idx]

                optimizer.zero_grad()
                lh, la, rho = self._net(home_idx, away_idx, home_seq, away_seq, ctx)
                loss = _nll_loss_batch(lh, la, rho, hs, as_)
                loss.backward()
                optimizer.step()

                epoch_loss_sum += loss.item()
                n_batches += 1

            train_nll = epoch_loss_sum / max(n_batches, 1)

            # Validation pass
            self._net.eval()
            with torch.no_grad():
                lh_v, la_v, rho_v = self._net(
                    val_home_idx, val_away_idx, val_home_seq, val_away_seq, val_ctx
                )
                val_nll = _nll_loss_batch(lh_v, la_v, rho_v, val_hs, val_as).item()

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:3d}/{self.n_epochs}  "
                    f"train_nll={train_nll:.4f}  val_nll={val_nll:.4f}"
                )

            if not early_stopping:
                continue  # no model selection / no stopping — keep training to n_epochs

            if val_nll < best_val_nll - 1e-6:
                best_val_nll = val_nll
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch} (best val_nll={best_val_nll:.4f})")
                    break

        # Restore the best checkpoint (early-stopping mode only; otherwise keep final weights)
        if best_state:
            self._net.load_state_dict(best_state)

        self.best_epoch_ = best_epoch if early_stopping else self.n_epochs
        self._net.eval()
        return self

    def predict_batch(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Predict for a DataFrame of matches.

        Returns the input DataFrame with added columns:
            lambda_home, lambda_away, rho,
            prob_home, prob_draw, prob_away,
            expected_home, expected_away,
            grid  (object dtype, np.ndarray per row)

        Unknown teams (absent from vocab) fall back to UNK embedding (index 0).
        Uses self._train_results for form sequence look-ups.
        """
        if self._net is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        if self._train_results is None:
            raise RuntimeError("No training results stored. Call fit() first.")

        self._net.eval()
        form_df = self._train_results
        n = len(matches)

        lambda_home_list: list[float] = []
        lambda_away_list: list[float] = []
        rho_list: list[float] = []
        prob_home_list: list[float] = []
        prob_draw_list: list[float] = []
        prob_away_list: list[float] = []
        exp_home_list: list[float] = []
        exp_away_list: list[float] = []
        grid_list: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                batch_rows = matches.iloc[start : start + self.batch_size].reset_index(drop=True)
                home_idx, away_idx, home_seq, away_seq, ctx = self._build_batch_tensors(
                    batch_rows, form_df
                )
                lh_t, la_t, rho_t = self._net(home_idx, away_idx, home_seq, away_seq, ctx)

                lh_np = lh_t.cpu().numpy()
                la_np = la_t.cpu().numpy()
                rho_np = rho_t.cpu().numpy()

                for i in range(len(batch_rows)):
                    lh = float(lh_np[i])
                    la = float(la_np[i])
                    rho = float(rho_np[i])

                    grid = build_grid(lh, la, rho)
                    markets = derive_markets(grid)

                    goals = np.arange(grid.shape[0], dtype=np.float64)
                    exp_h = float(np.dot(goals, grid.sum(axis=1)))
                    exp_a = float(np.dot(goals, grid.sum(axis=0)))

                    lambda_home_list.append(lh)
                    lambda_away_list.append(la)
                    rho_list.append(rho)
                    prob_home_list.append(markets["home_win"])
                    prob_draw_list.append(markets["draw"])
                    prob_away_list.append(markets["away_win"])
                    exp_home_list.append(exp_h)
                    exp_away_list.append(exp_a)
                    grid_list.append(grid)

        out = matches.copy()
        out["lambda_home"] = lambda_home_list
        out["lambda_away"] = lambda_away_list
        out["rho"] = rho_list
        out["prob_home"] = prob_home_list
        out["prob_draw"] = prob_draw_list
        out["prob_away"] = prob_away_list
        out["expected_home"] = exp_home_list
        out["expected_away"] = exp_away_list
        out["grid"] = grid_list
        return out

    def save(self, path: str | Path) -> None:
        """Save model checkpoint to disk."""
        if self._net is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        n_teams: int = self._net.team_embed.num_embeddings

        torch.save(
            {
                "model_state": self._net.state_dict(),
                "team_vocab": self._team_vocab,
                "n_teams": n_teams,
                "hparams": {
                    "n_epochs": self.n_epochs,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                    "patience": self.patience,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> NeuralModel:
        """Load a NeuralModel from a checkpoint saved by save()."""
        checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)

        hparams: dict[str, Any] = checkpoint["hparams"]
        model = cls(
            n_epochs=hparams["n_epochs"],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            batch_size=hparams["batch_size"],
            patience=hparams["patience"],
        )
        model._team_vocab = checkpoint["team_vocab"]

        n_teams: int = checkpoint["n_teams"]
        net = ScoreGridNet(n_teams=n_teams).to(model._device)
        net.load_state_dict(checkpoint["model_state"])
        net.eval()
        model._net = net

        return model
