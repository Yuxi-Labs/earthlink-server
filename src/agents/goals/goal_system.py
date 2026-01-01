"""Goal system - autonomous goal formation, management, and achievement tracking."""

from collections import defaultdict
from datetime import datetime
from typing import Any
from uuid import UUID

import torch
import torch.nn as nn
import torch.nn.functional as F

from .goal import Goal, GoalStatus, GoalType


class GoalEncoder(nn.Module):
    """Neural network for encoding goals to embeddings."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        goal_dim: int = 32,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
        )

    def forward(self, target_state: torch.Tensor) -> torch.Tensor:
        """Encode target state to goal embedding."""
        return F.normalize(self.network(target_state), dim=-1)


class GoalProposer(nn.Module):
    """Neural network for proposing new goals based on current state."""

    def __init__(
        self,
        state_dim: int,
        goal_dim: int = 32,
        hidden_dim: int = 128,
        num_proposals: int = 5,
    ):
        super().__init__()

        self.num_proposals = num_proposals

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_proposals * goal_dim),
        )

        self.goal_dim = goal_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Propose goal embeddings based on current state.
        
        Returns: (batch, num_proposals, goal_dim)
        """
        output = self.network(state)
        output = output.view(-1, self.num_proposals, self.goal_dim)
        return F.normalize(output, dim=-1)


class GoalValueNetwork(nn.Module):
    """Neural network for estimating goal value / achievability."""

    def __init__(
        self,
        state_dim: int,
        goal_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        state: torch.Tensor,
        goal_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate value/achievability of goal from current state."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if goal_embedding.dim() == 1:
            goal_embedding = goal_embedding.unsqueeze(0)

        if state.shape[0] == 1 and goal_embedding.shape[0] > 1:
            state = state.expand(goal_embedding.shape[0], -1)
        elif goal_embedding.shape[0] == 1 and state.shape[0] > 1:
            goal_embedding = goal_embedding.expand(state.shape[0], -1)

        combined = torch.cat([state, goal_embedding], dim=-1)
        return self.network(combined).squeeze(-1)


class GoalSystem:
    """
    Autonomous goal management system.
    
    Handles:
    - Goal formation (from curiosity, competence, external tasks)
    - Goal decomposition (breaking into subgoals)
    - Goal prioritization (balancing multiple goals)
    - Achievement tracking and learning
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int = 32,
        hidden_dim: int = 128,
        max_active_goals: int = 5,
        max_proposed_goals: int = 10,
        device: torch.device = torch.device("cpu"),
    ):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.max_active_goals = max_active_goals
        self.max_proposed_goals = max_proposed_goals
        self.device = device

        # Neural components
        self.goal_encoder = GoalEncoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            goal_dim=goal_dim,
        ).to(device)

        self.goal_proposer = GoalProposer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        self.goal_value = GoalValueNetwork(
            state_dim=state_dim,
            goal_dim=goal_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        # Goal storage
        self._goals: dict[UUID, Goal] = {}
        self._goal_embeddings: dict[UUID, torch.Tensor] = {}

        # Statistics
        self.stats = defaultdict(int)

    # -------------------------------------------------------------------------
    # Goal management
    # -------------------------------------------------------------------------

    def add_goal(
        self,
        goal: Goal | None = None,
        *,
        description: str | None = None,
        goal_type: GoalType = GoalType.EXPLORATION,
        priority: float = 0.5,
        target_state: torch.Tensor | None = None,
        status: GoalStatus = GoalStatus.PROPOSED,
        source: str = "self",
        metadata: dict[str, Any] | None = None,
    ) -> Goal:
        """Add a goal to the system (accepts Goal or parameters)."""
        if goal is None:
            goal = Goal(
                description=description or "",
                goal_type=goal_type,
                status=status,
                target_state=target_state,
                priority=priority,
                source=source,
                metadata=metadata or {},
            )

        self._goals[goal.id] = goal

        # Generate embedding if target state exists
        if goal.target_state is not None:
            with torch.no_grad():
                embedding = self.goal_encoder(
                    goal.target_state.to(self.device)
                )
                self._goal_embeddings[goal.id] = embedding
                goal.embedding = embedding

        self.stats["goals_added"] += 1
        return goal

    def remove_goal(self, goal_id: UUID) -> bool:
        """Remove a goal from the system."""
        if goal_id in self._goals:
            del self._goals[goal_id]
            self._goal_embeddings.pop(goal_id, None)
            return True
        return False

    def get_goal(self, goal_id: UUID) -> Goal | None:
        """Get goal by ID."""
        return self._goals.get(goal_id)

    def get_active_goals(self) -> list[Goal]:
        """Get all active goals, sorted by effective priority."""
        active = [g for g in self._goals.values() if g.is_active]
        return sorted(active, key=lambda g: g.effective_priority, reverse=True)

    def get_proposed_goals(self) -> list[Goal]:
        """Get proposed (uncommitted) goals."""
        return [g for g in self._goals.values() if g.status == GoalStatus.PROPOSED]

    def update_goal_status(self, goal_id: UUID, status: str | GoalStatus) -> bool:
        """Update goal status by ID."""
        goal = self.get_goal(goal_id)
        if goal is None:
            return False

        if isinstance(status, str):
            status = status.lower()
            if status in {"in_progress", "active"}:
                goal.status = GoalStatus.IN_PROGRESS
            elif status in {"proposed"}:
                goal.status = GoalStatus.PROPOSED
            elif status in {"suspended"}:
                goal.status = GoalStatus.SUSPENDED
            elif status in {"achieved", "done", "complete"}:
                goal.status = GoalStatus.ACHIEVED
            elif status in {"abandoned"}:
                goal.status = GoalStatus.ABANDONED
            elif status in {"failed"}:
                goal.status = GoalStatus.FAILED
        elif isinstance(status, GoalStatus):
            goal.status = status

        return True

    # -------------------------------------------------------------------------
    # Goal formation
    # -------------------------------------------------------------------------

    def propose_goals(
        self,
        current_state: torch.Tensor,
        curiosity_scores: torch.Tensor | None = None,
        competence_progress: torch.Tensor | None = None,
    ) -> list[Goal]:
        """
        Autonomously propose new goals based on current state.
        
        Uses:
        - Curiosity: propose goals toward novel/uncertain areas
        - Competence: propose goals for skill improvement
        - State-based: propose reachable interesting targets
        """
        proposed = []

        with torch.no_grad():
            state = current_state.to(self.device)

            # Neural goal proposals
            goal_embeddings = self.goal_proposer(state.unsqueeze(0)).squeeze(0)

            for i in range(goal_embeddings.shape[0]):
                embedding = goal_embeddings[i]

                # Estimate goal value
                value = self.goal_value(state.unsqueeze(0), embedding.unsqueeze(0)).item()

                goal = Goal(
                    description=f"Auto-proposed goal {i+1}",
                    goal_type=GoalType.EXPLORATION,
                    status=GoalStatus.PROPOSED,
                    embedding=embedding,
                    priority=value,
                    intrinsic_value=value,
                    source="curiosity",
                )

                proposed.append(goal)

        # Add curiosity-driven goals
        if curiosity_scores is not None and curiosity_scores.numel() > 0:
            high_curiosity_idx = curiosity_scores.topk(
                min(3, curiosity_scores.numel())
            ).indices

            for idx in high_curiosity_idx:
                goal = Goal(
                    description=f"Explore high-curiosity region {idx.item()}",
                    goal_type=GoalType.EXPLORATION,
                    status=GoalStatus.PROPOSED,
                    priority=curiosity_scores[idx].item(),
                    intrinsic_value=curiosity_scores[idx].item(),
                    source="curiosity",
                )
                proposed.append(goal)

        # Add competence-driven goals
        if competence_progress is not None and competence_progress.numel() > 0:
            high_progress_idx = competence_progress.topk(
                min(2, competence_progress.numel())
            ).indices

            for idx in high_progress_idx:
                goal = Goal(
                    description=f"Improve skill in area {idx.item()}",
                    goal_type=GoalType.COMPETENCE,
                    status=GoalStatus.PROPOSED,
                    priority=0.7,
                    intrinsic_value=competence_progress[idx].item(),
                    source="competence",
                )
                proposed.append(goal)

        # Limit proposals
        proposed = sorted(proposed, key=lambda g: g.intrinsic_value, reverse=True)
        proposed = proposed[:self.max_proposed_goals]

        # Add to system
        for goal in proposed:
            self.add_goal(goal)

        return proposed

    def create_goal(
        self,
        description: str,
        target_state: torch.Tensor | None = None,
        goal_type: GoalType = GoalType.TASK,
        priority: float = 0.5,
        source: str = "external",
        **metadata: Any,
    ) -> Goal:
        """Create a new goal manually."""
        goal = Goal(
            description=description,
            goal_type=goal_type,
            status=GoalStatus.PROPOSED,
            target_state=target_state,
            priority=priority,
            source=source,
            metadata=metadata,
        )

        self.add_goal(goal)
        return goal

    # -------------------------------------------------------------------------
    # Goal decomposition
    # -------------------------------------------------------------------------

    def decompose_goal(
        self,
        goal_id: UUID,
        current_state: torch.Tensor,
        num_subgoals: int = 3,
    ) -> list[Goal]:
        """
        Decompose a goal into subgoals.
        
        Creates intermediate waypoints between current state and goal target.
        """
        goal = self.get_goal(goal_id)
        if goal is None or goal.target_state is None:
            return []

        subgoals = []

        with torch.no_grad():
            state = current_state.to(self.device)
            target = goal.target_state.to(self.device)

            # Linear interpolation for simple decomposition
            for i in range(1, num_subgoals + 1):
                alpha = i / (num_subgoals + 1)
                intermediate_target = state + alpha * (target - state)

                subgoal = Goal(
                    description=f"Subgoal {i} of: {goal.description}",
                    goal_type=goal.goal_type,
                    status=GoalStatus.PROPOSED,
                    target_state=intermediate_target,
                    priority=goal.priority * 0.9,  # Slightly lower priority
                    parent_id=goal_id,
                    source="decomposition",
                )

                # Encode subgoal
                embedding = self.goal_encoder(intermediate_target.unsqueeze(0)).squeeze(0)
                subgoal.embedding = embedding

                subgoals.append(subgoal)
                self.add_goal(subgoal)
                goal.subgoal_ids.append(subgoal.id)

        return subgoals

    # -------------------------------------------------------------------------
    # Goal prioritization
    # -------------------------------------------------------------------------

    def prioritize(
        self,
        current_state: torch.Tensor,
    ) -> list[Goal]:
        """
        Prioritize goals based on current state.
        
        Returns ordered list of goals to pursue.
        """
        active = self.get_active_goals()
        proposed = self.get_proposed_goals()

        all_goals = active + proposed

        if not all_goals:
            return []

        with torch.no_grad():
            state = current_state.to(self.device)

            # Score each goal
            scored_goals = []

            for goal in all_goals:
                score = goal.effective_priority

                # Boost based on goal value prediction
                if goal.embedding is not None:
                    value = self.goal_value(
                        state.unsqueeze(0),
                        goal.embedding.unsqueeze(0),
                    ).item()
                    score *= (1 + value)

                # Penalize goals with many failures
                if goal.failures > 0:
                    score *= 0.9 ** goal.failures

                scored_goals.append((goal, score))

            # Sort by score
            scored_goals.sort(key=lambda x: x[1], reverse=True)

        return [g for g, _ in scored_goals]

    def select_goal(self, current_state: torch.Tensor) -> Goal | None:
        """Select the highest priority goal to pursue."""
        prioritized = self.prioritize(current_state)

        if not prioritized:
            return None

        goal = prioritized[0]

        # Activate if proposed
        if goal.status == GoalStatus.PROPOSED:
            goal.start()

        return goal

    # -------------------------------------------------------------------------
    # Goal achievement
    # -------------------------------------------------------------------------

    def check_achievement(
        self,
        goal_id: UUID,
        current_state: torch.Tensor,
        threshold: float = 0.1,
    ) -> bool:
        """
        Check if a goal has been achieved.
        
        Uses distance to target state or embedding similarity.
        """
        goal = self.get_goal(goal_id)
        if goal is None:
            return False

        with torch.no_grad():
            state = current_state.to(self.device)

            # Check target state distance
            if goal.target_state is not None:
                target = goal.target_state.to(self.device)
                distance = F.mse_loss(state, target).item()

                if distance < threshold:
                    goal.achieve()
                    self.stats["goals_achieved"] += 1
                    return True

                # Update progress
                max_distance = 1.0  # Normalized assumption
                progress = 1.0 - min(distance / max_distance, 1.0)
                goal.update_progress(progress)

            # Check embedding similarity
            if goal.embedding is not None:
                current_embedding = self.goal_encoder(state.unsqueeze(0)).squeeze(0)
                similarity = F.cosine_similarity(
                    current_embedding.unsqueeze(0),
                    goal.embedding.unsqueeze(0),
                ).item()

                if similarity > (1 - threshold):
                    goal.achieve()
                    self.stats["goals_achieved"] += 1
                    return True

        return False

    def update_goal_progress(
        self,
        goal_id: UUID,
        current_state: torch.Tensor | None = None,
        progress: float | None = None,
    ) -> float:
        """Update progress toward a goal."""
        goal = self.get_goal(goal_id)
        if goal is None:
            return 0.0

        # Direct progress update
        if progress is not None:
            goal.update_progress(progress)
            return goal.progress

        # Check achievement
        if current_state is not None and self.check_achievement(goal_id, current_state):
            return 1.0

        return goal.progress

    # -------------------------------------------------------------------------
    # Learning
    # -------------------------------------------------------------------------

    def update(
        self,
        states: torch.Tensor,
        goal_embeddings: torch.Tensor,
        achieved: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Update goal value network based on goal achievement data.
        
        Args:
            states: States from which goals were pursued
            goal_embeddings: Goal embeddings
            achieved: Binary achievement labels
            
        Returns:
            Training losses
        """
        predictions = self.goal_value(states, goal_embeddings)
        loss = F.binary_cross_entropy(predictions, achieved.float())

        return {"goal_value_loss": loss}

    # -------------------------------------------------------------------------
    # Current goal for policy conditioning
    # -------------------------------------------------------------------------

    def get_current_goal_embedding(
        self,
        current_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get embedding of current goal for policy conditioning.
        
        Returns zero tensor if no active goal.
        """
        goal = self.select_goal(current_state)

        if goal is not None and goal.embedding is not None:
            return goal.embedding

        # Return zero embedding
        return torch.zeros(self.goal_dim, device=self.device)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get goal system statistics."""
        return {
            "total_goals": len(self._goals),
            "active_goals": len(self.get_active_goals()),
            "proposed_goals": len(self.get_proposed_goals()),
            "goals_achieved": self.stats["goals_achieved"],
            "goals_added": self.stats["goals_added"],
        }

    def save_state(self) -> dict[str, Any]:
        """Save goal system state."""
        return {
            "goals": [g.to_dict() for g in self._goals.values()],
            "stats": dict(self.stats),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load goal system state."""
        self._goals.clear()
        self._goal_embeddings.clear()

        for goal_dict in state.get("goals", []):
            goal = Goal.from_dict(goal_dict)
            self._goals[goal.id] = goal

        self.stats = defaultdict(int, state.get("stats", {}))
