"""
Learning Module - Advanced learning capabilities beyond basic RL.

Provides:
- Meta-learning (learn how to learn faster)
- Transfer learning (knowledge across domains)
- Curriculum learning (progressive difficulty)
- Active learning (query what to learn)
- Learning strategy adaptation
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional
import numpy as np


class LearningStrategy(Enum):
    """Learning approach strategies."""
    SUPERVISED = "supervised"  # Learn from labeled examples
    REINFORCEMENT = "reinforcement"  # Learn from rewards
    IMITATION = "imitation"  # Learn from demonstrations
    SELF_SUPERVISED = "self_supervised"  # Learn from own experiences
    ACTIVE = "active"  # Query for informative examples
    META = "meta"  # Learn learning strategies


@dataclass
class Task:
    """Learning task definition."""
    id: str
    name: str
    domain: str  # "navigation", "knowledge", "social", etc.
    difficulty: float  # 0-1
    description: str = ""
    success_criteria: dict[str, Any] = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)


@dataclass
class LearningCurve:
    """Track learning progress over time."""
    task_id: str
    performance_history: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    learning_rate: float = 0.0
    plateau_detected: bool = False
    
    def add_performance(self, performance: float):
        """Add new performance measurement."""
        self.performance_history.append(performance)
        self.timestamps.append(datetime.now(UTC))
        
        # Calculate learning rate (recent improvement)
        if len(self.performance_history) >= 5:
            recent = self.performance_history[-5:]
            self.learning_rate = (recent[-1] - recent[0]) / 5
            
            # Detect plateau (no improvement)
            self.plateau_detected = abs(self.learning_rate) < 0.01


@dataclass
class TransferResult:
    """Result of knowledge transfer between domains."""
    source_domain: str
    target_domain: str
    transferred_knowledge: dict[str, Any]
    transfer_effectiveness: float  # 0-1, how well transfer worked
    adaptation_required: bool
    performance_gain: float  # Improvement from transfer


@dataclass
class MetaKnowledge:
    """Meta-level knowledge about learning itself."""
    strategy_effectiveness: dict[str, float] = field(default_factory=dict)
    optimal_learning_rate: float = 0.01
    best_exploration_rate: float = 0.1
    task_similarities: dict[str, list[str]] = field(default_factory=dict)
    learning_biases: list[str] = field(default_factory=list)


class LearningModule:
    """
    Advanced learning capabilities for autonomous agents.
    
    Goes beyond basic RL to include:
    - Learning how to learn (meta-learning)
    - Transferring knowledge between tasks
    - Adaptive curriculum design
    - Active query generation
    """
    
    def __init__(
        self,
        agent_id: str,
        base_learning_rate: float = 0.01,
        meta_learning_enabled: bool = True,
    ):
        self.agent_id = agent_id
        self.base_learning_rate = base_learning_rate
        self.meta_learning_enabled = meta_learning_enabled
        
        # Learning history
        self.task_history: dict[str, LearningCurve] = {}
        self.domain_knowledge: dict[str, dict] = {}  # domain -> learned patterns
        
        # Meta-learning
        self.meta_knowledge = MetaKnowledge()
        self.adaptation_count = 0
        
        # Curriculum state
        self.current_difficulty = 0.1  # Start easy
        self.curriculum_tasks: list[Task] = []
        
        # Active learning
        self.uncertainty_threshold = 0.7  # Query when uncertainty > this
        self.queries_made = 0
        
        # Statistics
        self.stats = {
            "total_learning_episodes": 0,
            "tasks_mastered": 0,
            "knowledge_transferred": 0,
            "meta_adaptations": 0,
            "active_queries": 0,
            "avg_learning_rate": 0.0,
        }
    
    def meta_learn(
        self,
        task_distribution: list[Task],
        episodes_per_task: int = 10,
    ) -> MetaKnowledge:
        """
        Learn how to learn from a distribution of tasks.
        
        Meta-learning enables quick adaptation to new tasks by learning
        common patterns across task families.
        
        Args:
            task_distribution: Multiple related tasks to learn from
            episodes_per_task: Training episodes per task
        
        Returns:
            Updated meta-knowledge
        """
        if not self.meta_learning_enabled:
            return self.meta_knowledge
        
        # Track performance across tasks
        task_performances = []
        
        for task in task_distribution:
            # Learn task and track speed of learning
            initial_perf = 0.0
            final_perf = 0.0
            
            for episode in range(episodes_per_task):
                # Simulate learning episode
                performance = self._simulate_learning_episode(task, episode)
                
                if episode == 0:
                    initial_perf = performance
                if episode == episodes_per_task - 1:
                    final_perf = performance
            
            improvement = final_perf - initial_perf
            task_performances.append({
                "task": task,
                "improvement": improvement,
                "final_performance": final_perf,
            })
        
        # Extract meta-knowledge from task distribution
        self._extract_meta_patterns(task_distribution, task_performances)
        
        # Adapt learning strategy based on patterns
        self._adapt_learning_strategy()
        
        self.stats["meta_adaptations"] += 1
        
        return self.meta_knowledge
    
    def transfer_knowledge(
        self,
        source_domain: str,
        target_domain: str,
        target_task: Optional[Task] = None,
    ) -> TransferResult:
        """
        Transfer learned knowledge from source to target domain.
        
        Enables leveraging experience in one area to accelerate learning
        in related areas.
        
        Args:
            source_domain: Domain to transfer from (e.g., "navigation")
            target_domain: Domain to transfer to (e.g., "foraging")
            target_task: Optional specific task in target domain
        
        Returns:
            TransferResult with effectiveness metrics
        """
        # Check if we have knowledge in source domain
        if source_domain not in self.domain_knowledge:
            return TransferResult(
                source_domain=source_domain,
                target_domain=target_domain,
                transferred_knowledge={},
                transfer_effectiveness=0.0,
                adaptation_required=True,
                performance_gain=0.0,
            )
        
        source_knowledge = self.domain_knowledge[source_domain]
        
        # Find transferable patterns
        transferable = self._identify_transferable_knowledge(
            source_knowledge,
            source_domain,
            target_domain,
        )
        
        # Estimate similarity between domains
        similarity = self._calculate_domain_similarity(source_domain, target_domain)
        
        # Transfer effectiveness depends on similarity
        effectiveness = similarity * 0.8  # Up to 80% effectiveness
        
        # Apply transfer
        if target_domain not in self.domain_knowledge:
            self.domain_knowledge[target_domain] = {}
        
        # Merge transferable knowledge with adaptation
        for key, value in transferable.items():
            adapted_value = self._adapt_knowledge(value, target_domain)
            self.domain_knowledge[target_domain][key] = adapted_value
        
        # Estimate performance gain
        performance_gain = effectiveness * 0.5  # Transfer can boost perf by up to 40%
        
        # Update meta-knowledge about task similarities
        if source_domain not in self.meta_knowledge.task_similarities:
            self.meta_knowledge.task_similarities[source_domain] = []
        if similarity > 0.5:
            self.meta_knowledge.task_similarities[source_domain].append(target_domain)
        
        self.stats["knowledge_transferred"] += 1
        
        return TransferResult(
            source_domain=source_domain,
            target_domain=target_domain,
            transferred_knowledge=transferable,
            transfer_effectiveness=effectiveness,
            adaptation_required=similarity < 0.7,
            performance_gain=performance_gain,
        )
    
    def curriculum_learn(
        self,
        all_tasks: list[Task],
        current_performance: float,
    ) -> Optional[Task]:
        """
        Select next task from curriculum based on current skill level.
        
        Implements curriculum learning: start easy, progressively increase
        difficulty as agent improves.
        
        Args:
            all_tasks: Available tasks across difficulty levels
            current_performance: Current performance level (0-1)
        
        Returns:
            Next task to learn, or None if curriculum complete
        """
        if not all_tasks:
            return None
        
        # Update difficulty based on performance
        if current_performance > 0.8:
            # Mastered current level, increase difficulty
            self.current_difficulty = min(1.0, self.current_difficulty + 0.1)
            self.stats["tasks_mastered"] += 1
        elif current_performance < 0.4:
            # Struggling, reduce difficulty
            self.current_difficulty = max(0.1, self.current_difficulty - 0.05)
        
        # Find tasks near current difficulty (zone of proximal development)
        suitable_tasks = [
            task for task in all_tasks
            if abs(task.difficulty - self.current_difficulty) < 0.15
        ]
        
        if not suitable_tasks:
            # No tasks at current level, pick closest
            suitable_tasks = sorted(all_tasks, key=lambda t: abs(t.difficulty - self.current_difficulty))
            suitable_tasks = [suitable_tasks[0]]
        
        # Prefer tasks we haven't mastered yet
        unmastered = [
            task for task in suitable_tasks
            if task.id not in self.task_history or 
            not self._is_task_mastered(task.id)
        ]
        
        if unmastered:
            # Pick task with most learning potential
            next_task = max(unmastered, key=lambda t: self._estimate_learning_potential(t))
        else:
            # All suitable tasks mastered, pick any
            next_task = suitable_tasks[0]
        
        return next_task
    
    def active_learn(
        self,
        available_examples: list[dict],
        current_uncertainty: dict[str, float],
    ) -> Optional[dict]:
        """
        Identify most informative example to query/learn next.
        
        Active learning: agent decides what to learn rather than
        passively accepting all data.
        
        Args:
            available_examples: Possible examples to learn from
            current_uncertainty: Model uncertainty for each example
        
        Returns:
            Most informative example to query, or None
        """
        if not available_examples:
            return None
        
        # Score examples by informativeness
        scored_examples = []
        for example in available_examples:
            example_id = example.get("id", str(hash(str(example))))
            uncertainty = current_uncertainty.get(example_id, 0.5)
            
            # Informativeness factors:
            # 1. High uncertainty (we're unsure about this)
            # 2. Novel (different from seen examples)
            # 3. Representative (similar to many unseen examples)
            
            novelty = self._calculate_novelty(example)
            representativeness = example.get("cluster_size", 1.0) / 100  # Normalize
            
            informativeness = (
                uncertainty * 0.5 +
                novelty * 0.3 +
                representativeness * 0.2
            )
            
            scored_examples.append({
                "example": example,
                "informativeness": informativeness,
                "uncertainty": uncertainty,
            })
        
        # Sort by informativeness
        scored_examples.sort(key=lambda x: x["informativeness"], reverse=True)
        
        # Query if most informative example exceeds threshold
        best = scored_examples[0]
        if best["informativeness"] > self.uncertainty_threshold:
            self.stats["active_queries"] += 1
            self.queries_made += 1
            return best["example"]
        
        return None
    
    def adapt_learning_rate(
        self,
        task_id: str,
        recent_performance: list[float],
    ) -> float:
        """
        Dynamically adjust learning rate based on progress.
        
        Args:
            task_id: Current task
            recent_performance: Recent performance measurements
        
        Returns:
            Adapted learning rate
        """
        if len(recent_performance) < 2:
            return self.base_learning_rate
        
        # Calculate recent progress
        progress = recent_performance[-1] - recent_performance[-5] if len(recent_performance) >= 5 else 0
        
        # Adapt rate based on progress
        if progress > 0.1:
            # Good progress, maintain or slightly increase rate
            new_rate = min(self.base_learning_rate * 1.1, 0.1)
        elif abs(progress) < 0.01:
            # Plateau, increase rate to escape
            new_rate = min(self.base_learning_rate * 1.5, 0.1)
        else:
            # Declining, reduce rate for stability
            new_rate = max(self.base_learning_rate * 0.8, 0.001)
        
        # Update meta-knowledge
        self.meta_knowledge.optimal_learning_rate = new_rate
        
        return new_rate
    
    def _simulate_learning_episode(self, task: Task, episode: int) -> float:
        """Simulate learning episode on task."""
        # Simple learning curve: performance improves with episodes
        max_performance = 0.9
        learning_speed = 0.1
        
        performance = max_performance * (1 - np.exp(-learning_speed * episode))
        
        # Add noise
        performance += np.random.normal(0, 0.05)
        performance = np.clip(performance, 0, 1)
        
        # Track in history
        if task.id not in self.task_history:
            self.task_history[task.id] = LearningCurve(task_id=task.id)
        self.task_history[task.id].add_performance(performance)
        
        self.stats["total_learning_episodes"] += 1
        
        return performance
    
    def _extract_meta_patterns(
        self,
        tasks: list[Task],
        performances: list[dict],
    ):
        """Extract common patterns across tasks."""
        # Identify which learning strategies work best
        if performances:
            avg_improvement = np.mean([p["improvement"] for p in performances])
            
            # Update meta-knowledge
            strategy = "reinforcement"  # Current strategy
            if strategy not in self.meta_knowledge.strategy_effectiveness:
                self.meta_knowledge.strategy_effectiveness[strategy] = avg_improvement
            else:
                # Running average
                current = self.meta_knowledge.strategy_effectiveness[strategy]
                self.meta_knowledge.strategy_effectiveness[strategy] = (
                    current * 0.9 + avg_improvement * 0.1
                )
        
        # Identify task similarities by domain
        domains = {}
        for task in tasks:
            if task.domain not in domains:
                domains[task.domain] = []
            domains[task.domain].append(task.id)
        
        self.meta_knowledge.task_similarities.update(domains)
    
    def _adapt_learning_strategy(self):
        """Adapt learning approach based on meta-knowledge."""
        # Find best-performing strategy
        if self.meta_knowledge.strategy_effectiveness:
            best_strategy = max(
                self.meta_knowledge.strategy_effectiveness.items(),
                key=lambda x: x[1]
            )[0]
            
            # Adapt parameters for best strategy
            self.adaptation_count += 1
    
    def _identify_transferable_knowledge(
        self,
        source_knowledge: dict,
        source_domain: str,
        target_domain: str,
    ) -> dict:
        """Identify which knowledge can transfer between domains."""
        transferable = {}
        
        # Simple heuristic: transfer general patterns, not domain-specific details
        for key, value in source_knowledge.items():
            if "general" in key or "pattern" in key or "strategy" in key:
                transferable[key] = value
        
        return transferable
    
    def _adapt_knowledge(self, knowledge: Any, target_domain: str) -> Any:
        """Adapt knowledge for new domain."""
        # Simple adaptation: keep structure, adjust parameters
        if isinstance(knowledge, (int, float)):
            return knowledge * 0.8  # Reduce confidence in transferred knowledge
        return knowledge
    
    def _calculate_domain_similarity(
        self,
        domain1: str,
        domain2: str,
    ) -> float:
        """Calculate similarity between two domains."""
        # Simple keyword-based similarity
        words1 = set(domain1.lower().split("_"))
        words2 = set(domain2.lower().split("_"))
        
        if not words1 or not words2:
            return 0.5
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        similarity = overlap / total if total > 0 else 0.5
        return similarity
    
    def _is_task_mastered(self, task_id: str) -> bool:
        """Check if task is mastered."""
        if task_id not in self.task_history:
            return False
        
        curve = self.task_history[task_id]
        if len(curve.performance_history) < 5:
            return False
        
        # Mastered if recent performance consistently high
        recent = curve.performance_history[-5:]
        return np.mean(recent) > 0.85 and np.std(recent) < 0.05
    
    def _estimate_learning_potential(self, task: Task) -> float:
        """Estimate how much we can learn from this task."""
        # More potential if we haven't tried it much
        if task.id not in self.task_history:
            return 1.0
        
        curve = self.task_history[task.id]
        episodes = len(curve.performance_history)
        
        # Less potential if we've done it many times
        potential = max(0.1, 1.0 - episodes / 100)
        
        # More potential if we're still improving
        if not curve.plateau_detected:
            potential *= 1.5
        
        return potential
    
    def _calculate_novelty(self, example: dict) -> float:
        """Calculate how novel/different an example is."""
        # Simple novelty: random for now
        # In practice, would compare to seen examples
        return np.random.uniform(0.3, 0.9)
    
    def get_learning_summary(self) -> dict:
        """Get summary of learning performance."""
        return {
            **self.stats,
            "current_difficulty": self.current_difficulty,
            "tasks_in_history": len(self.task_history),
            "domains_learned": list(self.domain_knowledge.keys()),
            "meta_learning_enabled": self.meta_learning_enabled,
            "adaptation_count": self.adaptation_count,
            "queries_made": self.queries_made,
        }
