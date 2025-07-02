#!/usr/bin/env python3
"""
Sliding Window Pattern Analyzer for DEAN System
Detects complex temporal patterns using sliding window analysis
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
import logging
import hashlib
import json

logger = logging.getLogger(__name__)

@dataclass
class WindowPattern:
    """Represents a pattern detected in a sliding window."""
    pattern_hash: str
    window_size: int
    sequence: List[Any]
    frequency: int
    positions: List[int]  # Where pattern starts in the data
    confidence: float
    temporal_consistency: float  # How consistent over time
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TemporalPattern:
    """Represents a pattern with temporal characteristics."""
    pattern_id: str
    pattern_type: str  # periodic, trending, seasonal, anomaly
    period: Optional[float] = None  # For periodic patterns
    trend_slope: Optional[float] = None  # For trending patterns
    seasonal_component: Optional[List[float]] = None  # For seasonal patterns
    anomaly_score: Optional[float] = None  # For anomaly patterns
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    confidence: float = 0.0

class SlidingWindowAnalyzer:
    """
    Advanced sliding window pattern analyzer for DEAN agents.
    Detects complex patterns across multiple time scales.
    """
    
    def __init__(self, 
                 min_window_size: int = 3,
                 max_window_size: int = 20,
                 stride: int = 1,
                 min_frequency: int = 2,
                 confidence_threshold: float = 0.7):
        """
        Initialize sliding window analyzer.
        
        Args:
            min_window_size: Minimum window size for pattern detection
            max_window_size: Maximum window size for pattern detection
            stride: Step size for sliding window
            min_frequency: Minimum occurrences for pattern recognition
            confidence_threshold: Minimum confidence for pattern acceptance
        """
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.stride = stride
        self.min_frequency = min_frequency
        self.confidence_threshold = confidence_threshold
        
        # Pattern storage
        self.detected_patterns: Dict[str, WindowPattern] = {}
        self.temporal_patterns: List[TemporalPattern] = []
        
        # Analysis cache
        self.sequence_cache: Dict[str, List[Any]] = {}
        self.analysis_results: Dict[str, Dict] = {}
    
    def analyze_sequence(self, sequence: List[Any], 
                        sequence_id: str,
                        timestamps: Optional[List[datetime]] = None) -> List[WindowPattern]:
        """
        Analyze a sequence using sliding windows of multiple sizes.
        
        Args:
            sequence: Input sequence to analyze
            sequence_id: Unique identifier for the sequence
            timestamps: Optional timestamps for temporal analysis
            
        Returns:
            List of detected patterns
        """
        if len(sequence) < self.min_window_size:
            return []
        
        # Cache sequence
        self.sequence_cache[sequence_id] = sequence
        
        all_patterns = []
        
        # Try different window sizes
        for window_size in range(self.min_window_size, 
                               min(self.max_window_size + 1, len(sequence) + 1)):
            
            # Extract windows
            windows = self._extract_windows(sequence, window_size)
            
            # Find patterns in windows
            patterns = self._find_patterns_in_windows(windows, window_size)
            
            # Filter by frequency and confidence
            valid_patterns = [
                p for p in patterns 
                if p.frequency >= self.min_frequency and p.confidence >= self.confidence_threshold
            ]
            
            all_patterns.extend(valid_patterns)
        
        # Temporal analysis if timestamps provided
        if timestamps and len(timestamps) == len(sequence):
            temporal_patterns = self._analyze_temporal_patterns(sequence, timestamps)
            self.temporal_patterns.extend(temporal_patterns)
        
        # Store results
        self.analysis_results[sequence_id] = {
            'patterns': all_patterns,
            'temporal_patterns': temporal_patterns if timestamps else [],
            'analysis_time': datetime.utcnow()
        }
        
        return all_patterns
    
    def _extract_windows(self, sequence: List[Any], window_size: int) -> List[List[Any]]:
        """Extract sliding windows from sequence."""
        windows = []
        
        for i in range(0, len(sequence) - window_size + 1, self.stride):
            window = sequence[i:i + window_size]
            windows.append(window)
        
        return windows
    
    def _find_patterns_in_windows(self, windows: List[List[Any]], 
                                window_size: int) -> List[WindowPattern]:
        """Find recurring patterns in windows."""
        # Count window occurrences
        window_counts = defaultdict(list)  # pattern -> list of positions
        
        for i, window in enumerate(windows):
            # Create hashable representation
            window_hash = self._hash_window(window)
            window_counts[window_hash].append(i * self.stride)
        
        # Create pattern objects
        patterns = []
        for window_hash, positions in window_counts.items():
            if len(positions) >= self.min_frequency:
                # Reconstruct original window
                first_pos = positions[0]
                original_window = windows[first_pos // self.stride]
                
                # Calculate confidence based on distribution
                confidence = self._calculate_pattern_confidence(positions, len(windows))
                
                # Calculate temporal consistency
                temporal_consistency = self._calculate_temporal_consistency(positions)
                
                pattern = WindowPattern(
                    pattern_hash=window_hash,
                    window_size=window_size,
                    sequence=original_window,
                    frequency=len(positions),
                    positions=positions,
                    confidence=confidence,
                    temporal_consistency=temporal_consistency,
                    metadata={
                        'window_coverage': len(positions) / len(windows),
                        'position_variance': np.var(positions) if len(positions) > 1 else 0
                    }
                )
                
                patterns.append(pattern)
                self.detected_patterns[window_hash] = pattern
        
        return patterns
    
    def _hash_window(self, window: List[Any]) -> str:
        """Create hash for window pattern."""
        # Convert window to string representation
        window_str = json.dumps(window, sort_keys=True, default=str)
        return hashlib.sha256(window_str.encode()).hexdigest()[:16]
    
    def _calculate_pattern_confidence(self, positions: List[int], 
                                    total_windows: int) -> float:
        """Calculate confidence score for a pattern."""
        # Base confidence on frequency
        frequency_score = min(len(positions) / total_windows, 1.0)
        
        # Penalize if positions are too clustered
        if len(positions) > 1:
            position_spread = (max(positions) - min(positions)) / (total_windows * self.stride)
            spread_score = min(position_spread, 1.0)
        else:
            spread_score = 0.0
        
        # Combine scores
        confidence = 0.7 * frequency_score + 0.3 * spread_score
        
        return confidence
    
    def _calculate_temporal_consistency(self, positions: List[int]) -> float:
        """Calculate how consistent a pattern is over time."""
        if len(positions) < 2:
            return 0.0
        
        # Calculate intervals between occurrences
        intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        
        if not intervals:
            return 0.0
        
        # Check for regular intervals (periodic pattern)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval > 0:
            # Lower variance means more consistent
            consistency = 1.0 - min(std_interval / mean_interval, 1.0)
        else:
            consistency = 0.0
        
        return consistency
    
    def _analyze_temporal_patterns(self, sequence: List[Any], 
                                 timestamps: List[datetime]) -> List[TemporalPattern]:
        """Analyze temporal characteristics of the sequence."""
        temporal_patterns = []
        
        # Convert to numeric if possible
        numeric_sequence = self._try_numeric_conversion(sequence)
        
        if numeric_sequence is not None:
            # Detect trending patterns
            trend_pattern = self._detect_trend(numeric_sequence, timestamps)
            if trend_pattern:
                temporal_patterns.append(trend_pattern)
            
            # Detect periodic patterns
            periodic_patterns = self._detect_periodicity(numeric_sequence, timestamps)
            temporal_patterns.extend(periodic_patterns)
            
            # Detect anomalies
            anomaly_patterns = self._detect_anomalies(numeric_sequence, timestamps)
            temporal_patterns.extend(anomaly_patterns)
        
        # Detect burst patterns (works with any sequence type)
        burst_patterns = self._detect_bursts(sequence, timestamps)
        temporal_patterns.extend(burst_patterns)
        
        return temporal_patterns
    
    def _try_numeric_conversion(self, sequence: List[Any]) -> Optional[np.ndarray]:
        """Try to convert sequence to numeric array."""
        try:
            # Try direct conversion
            return np.array(sequence, dtype=float)
        except:
            # Try converting specific attributes if objects
            if sequence and hasattr(sequence[0], '__dict__'):
                # Try common numeric attributes
                for attr in ['value', 'score', 'fitness', 'count']:
                    try:
                        values = [getattr(item, attr, 0) for item in sequence]
                        return np.array(values, dtype=float)
                    except:
                        continue
            return None
    
    def _detect_trend(self, sequence: np.ndarray, 
                     timestamps: List[datetime]) -> Optional[TemporalPattern]:
        """Detect trending patterns in numeric sequence."""
        if len(sequence) < 5:
            return None
        
        # Simple linear regression
        x = np.arange(len(sequence))
        slope, intercept = np.polyfit(x, sequence, 1)
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((sequence - y_pred) ** 2)
        ss_tot = np.sum((sequence - np.mean(sequence)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Significant trend if R-squared > threshold and slope is significant
        if r_squared > 0.5 and abs(slope) > 0.01:
            return TemporalPattern(
                pattern_id=f"trend_{datetime.utcnow().timestamp()}",
                pattern_type="trending",
                trend_slope=slope,
                start_time=timestamps[0],
                end_time=timestamps[-1],
                confidence=r_squared
            )
        
        return None
    
    def _detect_periodicity(self, sequence: np.ndarray, 
                          timestamps: List[datetime]) -> List[TemporalPattern]:
        """Detect periodic patterns using autocorrelation."""
        if len(sequence) < 10:
            return []
        
        periodic_patterns = []
        
        # Compute autocorrelation
        mean = np.mean(sequence)
        var = np.var(sequence)
        
        if var == 0:
            return []
        
        # Normalize sequence
        normalized = (sequence - mean) / np.sqrt(var)
        
        # Check different lags
        max_lag = min(len(sequence) // 2, 50)
        
        for lag in range(2, max_lag):
            # Compute autocorrelation at this lag
            c0 = np.dot(normalized[:-lag], normalized[:-lag]) / (len(sequence) - lag)
            c_lag = np.dot(normalized[:-lag], normalized[lag:]) / (len(sequence) - lag)
            
            if c0 > 0:
                autocorr = c_lag / c0
                
                # Strong periodicity if autocorrelation > threshold
                if autocorr > 0.7:
                    # Estimate period in time units
                    if len(timestamps) > lag:
                        time_diff = (timestamps[lag] - timestamps[0]).total_seconds()
                        period = time_diff / lag
                    else:
                        period = lag
                    
                    periodic_patterns.append(TemporalPattern(
                        pattern_id=f"periodic_{lag}_{datetime.utcnow().timestamp()}",
                        pattern_type="periodic",
                        period=period,
                        start_time=timestamps[0],
                        end_time=timestamps[-1],
                        confidence=autocorr
                    ))
        
        return periodic_patterns
    
    def _detect_anomalies(self, sequence: np.ndarray, 
                         timestamps: List[datetime]) -> List[TemporalPattern]:
        """Detect anomalous patterns using statistical methods."""
        if len(sequence) < 10:
            return []
        
        anomaly_patterns = []
        
        # Use rolling statistics
        window = min(10, len(sequence) // 3)
        
        for i in range(window, len(sequence)):
            # Calculate statistics for window
            window_data = sequence[i-window:i]
            mean = np.mean(window_data)
            std = np.std(window_data)
            
            if std > 0:
                # Z-score for current value
                z_score = abs((sequence[i] - mean) / std)
                
                # Anomaly if z-score > threshold
                if z_score > 3:
                    anomaly_patterns.append(TemporalPattern(
                        pattern_id=f"anomaly_{i}_{datetime.utcnow().timestamp()}",
                        pattern_type="anomaly",
                        anomaly_score=z_score,
                        start_time=timestamps[i],
                        end_time=timestamps[i],
                        confidence=min(z_score / 5, 1.0)  # Normalize confidence
                    ))
        
        return anomaly_patterns
    
    def _detect_bursts(self, sequence: List[Any], 
                      timestamps: List[datetime]) -> List[TemporalPattern]:
        """Detect burst patterns in activity."""
        if len(sequence) < 5 or len(timestamps) < 5:
            return []
        
        burst_patterns = []
        
        # Calculate inter-arrival times
        inter_arrival_times = []
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i-1]).total_seconds()
            inter_arrival_times.append(delta)
        
        if not inter_arrival_times:
            return []
        
        # Identify bursts (rapid succession of events)
        mean_interval = np.mean(inter_arrival_times)
        burst_threshold = mean_interval * 0.2  # 20% of average
        
        burst_start = None
        burst_events = []
        
        for i, interval in enumerate(inter_arrival_times):
            if interval < burst_threshold:
                if burst_start is None:
                    burst_start = i
                burst_events.append(i + 1)
            else:
                # End of burst
                if burst_start is not None and len(burst_events) >= 3:
                    burst_patterns.append(TemporalPattern(
                        pattern_id=f"burst_{burst_start}_{datetime.utcnow().timestamp()}",
                        pattern_type="burst",
                        start_time=timestamps[burst_start],
                        end_time=timestamps[burst_events[-1]],
                        confidence=min(len(burst_events) / 10, 1.0),
                        metadata={
                            'burst_size': len(burst_events),
                            'avg_interval': np.mean([
                                inter_arrival_times[j] for j in range(burst_start, min(i, len(inter_arrival_times)))
                            ])
                        }
                    ))
                
                burst_start = None
                burst_events = []
        
        return burst_patterns
    
    def find_cross_sequence_patterns(self, sequence_ids: List[str]) -> List[Dict[str, Any]]:
        """Find patterns that appear across multiple sequences."""
        if len(sequence_ids) < 2:
            return []
        
        # Collect all patterns from specified sequences
        pattern_occurrences = defaultdict(list)  # pattern_hash -> list of sequence_ids
        
        for seq_id in sequence_ids:
            if seq_id in self.analysis_results:
                patterns = self.analysis_results[seq_id].get('patterns', [])
                for pattern in patterns:
                    pattern_occurrences[pattern.pattern_hash].append(seq_id)
        
        # Find patterns that appear in multiple sequences
        cross_patterns = []
        
        for pattern_hash, seq_list in pattern_occurrences.items():
            if len(seq_list) >= 2 and pattern_hash in self.detected_patterns:
                pattern = self.detected_patterns[pattern_hash]
                
                cross_patterns.append({
                    'pattern': pattern,
                    'sequences': seq_list,
                    'prevalence': len(seq_list) / len(sequence_ids),
                    'cross_sequence_confidence': min(len(seq_list) / len(sequence_ids) + 0.3, 1.0)
                })
        
        # Sort by prevalence
        cross_patterns.sort(key=lambda x: x['prevalence'], reverse=True)
        
        return cross_patterns
    
    def get_pattern_evolution(self, pattern_hash: str) -> Dict[str, Any]:
        """Track how a pattern evolves over time."""
        if pattern_hash not in self.detected_patterns:
            return {}
        
        pattern = self.detected_patterns[pattern_hash]
        
        # Analyze pattern evolution
        evolution_data = {
            'pattern_hash': pattern_hash,
            'first_seen': min(pattern.positions) if pattern.positions else None,
            'last_seen': max(pattern.positions) if pattern.positions else None,
            'frequency_trend': self._calculate_frequency_trend(pattern.positions),
            'stability': pattern.temporal_consistency,
            'evolution_phases': self._identify_evolution_phases(pattern.positions)
        }
        
        return evolution_data
    
    def _calculate_frequency_trend(self, positions: List[int]) -> str:
        """Calculate if pattern frequency is increasing, decreasing, or stable."""
        if len(positions) < 3:
            return "insufficient_data"
        
        # Divide positions into thirds
        third = len(positions) // 3
        early = len(positions[:third])
        middle = len(positions[third:2*third])
        late = len(positions[2*third:])
        
        # Compare frequencies
        if late > early * 1.2:
            return "increasing"
        elif late < early * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _identify_evolution_phases(self, positions: List[int]) -> List[Dict[str, Any]]:
        """Identify distinct phases in pattern evolution."""
        if len(positions) < 5:
            return []
        
        phases = []
        
        # Simple phase detection based on position clustering
        sorted_positions = sorted(positions)
        
        current_phase_start = sorted_positions[0]
        current_phase_positions = [sorted_positions[0]]
        
        # Threshold for phase separation (large gap)
        threshold = np.mean([
            sorted_positions[i+1] - sorted_positions[i] 
            for i in range(len(sorted_positions)-1)
        ]) * 3
        
        for i in range(1, len(sorted_positions)):
            gap = sorted_positions[i] - sorted_positions[i-1]
            
            if gap > threshold:
                # End current phase
                phases.append({
                    'phase_id': len(phases),
                    'start_position': current_phase_start,
                    'end_position': sorted_positions[i-1],
                    'occurrences': len(current_phase_positions),
                    'density': len(current_phase_positions) / (sorted_positions[i-1] - current_phase_start + 1)
                })
                
                # Start new phase
                current_phase_start = sorted_positions[i]
                current_phase_positions = [sorted_positions[i]]
            else:
                current_phase_positions.append(sorted_positions[i])
        
        # Add final phase
        phases.append({
            'phase_id': len(phases),
            'start_position': current_phase_start,
            'end_position': sorted_positions[-1],
            'occurrences': len(current_phase_positions),
            'density': len(current_phase_positions) / (sorted_positions[-1] - current_phase_start + 1)
        })
        
        return phases
    
    def export_patterns(self, filepath: str):
        """Export detected patterns to file."""
        export_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'analyzer_config': {
                'min_window_size': self.min_window_size,
                'max_window_size': self.max_window_size,
                'stride': self.stride,
                'min_frequency': self.min_frequency,
                'confidence_threshold': self.confidence_threshold
            },
            'patterns': [],
            'temporal_patterns': []
        }
        
        # Export window patterns
        for pattern_hash, pattern in self.detected_patterns.items():
            export_data['patterns'].append({
                'hash': pattern_hash,
                'window_size': pattern.window_size,
                'frequency': pattern.frequency,
                'confidence': pattern.confidence,
                'temporal_consistency': pattern.temporal_consistency,
                'positions': pattern.positions,
                'metadata': pattern.metadata
            })
        
        # Export temporal patterns
        for temporal in self.temporal_patterns:
            export_data['temporal_patterns'].append({
                'id': temporal.pattern_id,
                'type': temporal.pattern_type,
                'period': temporal.period,
                'trend_slope': temporal.trend_slope,
                'anomaly_score': temporal.anomaly_score,
                'confidence': temporal.confidence,
                'start_time': temporal.start_time.isoformat() if temporal.start_time else None,
                'end_time': temporal.end_time.isoformat() if temporal.end_time else None
            })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(self.detected_patterns)} patterns to {filepath}")