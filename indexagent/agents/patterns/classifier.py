"""
Behavior Classifier for DEAN system.
Classifies and evaluates discovered patterns.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass

class BehaviorClass(Enum):
    """Classification of agent behaviors."""
    BENEFICIAL = "beneficial"
    NEUTRAL = "neutral"
    HARMFUL = "harmful"
    GAMING = "gaming"
    INNOVATIVE = "innovative"

@dataclass
class BehaviorClassification:
    """Result of behavior classification."""
    behavior_class: BehaviorClass
    confidence: float
    is_beneficial: bool
    reasoning: str
    metadata: Dict[str, Any]

class BehaviorClassifier:
    """Classifies agent behaviors and patterns."""
    
    def __init__(self):
        self.classification_rules = self._build_classification_rules()
    
    def _build_classification_rules(self) -> Dict[str, Any]:
        """Build rules for behavior classification."""
        return {
            "beneficial_indicators": [
                "efficiency_improvement",
                "resource_optimization", 
                "problem_solving",
                "collaboration"
            ],
            "gaming_indicators": [
                "metric_manipulation",
                "resource_hoarding",
                "false_reporting"
            ],
            "innovation_indicators": [
                "novel_approach",
                "creative_solution",
                "unexpected_success"
            ]
        }
    
    async def classify(self, behavior: Dict[str, Any]) -> BehaviorClassification:
        """Classify a behavior pattern."""
        
        # Simple rule-based classification
        effectiveness = behavior.get('effectiveness', 0.0)
        behavior_type = behavior.get('type', '')
        
        if effectiveness > 0.7:
            return BehaviorClassification(
                behavior_class=BehaviorClass.BENEFICIAL,
                confidence=0.8,
                is_beneficial=True,
                reasoning="High effectiveness score",
                metadata={"effectiveness": effectiveness}
            )
        elif effectiveness > 0.4:
            return BehaviorClassification(
                behavior_class=BehaviorClass.NEUTRAL,
                confidence=0.6,
                is_beneficial=False,
                reasoning="Moderate effectiveness",
                metadata={"effectiveness": effectiveness}
            )
        else:
            return BehaviorClassification(
                behavior_class=BehaviorClass.HARMFUL,
                confidence=0.7,
                is_beneficial=False,
                reasoning="Low effectiveness score",
                metadata={"effectiveness": effectiveness}
            )