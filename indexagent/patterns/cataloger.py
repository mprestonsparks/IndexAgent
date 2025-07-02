#!/usr/bin/env python3
"""
Pattern Cataloger for DEAN System
Stores and manages discovered patterns in a searchable catalog
"""

import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)


@dataclass 
class CatalogedPattern:
    """Pattern stored in the catalog"""
    pattern_id: str
    pattern_type: str
    action_type: str
    description: str
    pattern_data: Dict[str, Any]
    success_metrics: Dict[str, float]
    discovery_agent: str
    discovery_generation: int
    reuse_count: int = 0
    avg_success_delta: float = 0.0
    created_at: datetime = None
    last_used: datetime = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_used is None:
            self.last_used = self.created_at
        if self.tags is None:
            self.tags = []


class PatternCataloger:
    """Manages the pattern catalog database"""
    
    def __init__(self, db_url: str):
        """
        Initialize pattern cataloger
        
        Args:
            db_url: PostgreSQL connection URL
        """
        self.db_url = db_url
        self._init_database()
        
        # Cache for frequently accessed patterns
        self.pattern_cache: Dict[str, CatalogedPattern] = {}
        self.cache_size = 100
        
        logger.info("PatternCataloger initialized")
    
    def _init_database(self):
        """Initialize database schema"""
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()
        
        # Create discovered_patterns table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS discovered_patterns (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                pattern_id VARCHAR(255) UNIQUE NOT NULL,
                pattern_type VARCHAR(100) NOT NULL,
                action_type VARCHAR(100) NOT NULL,
                description TEXT,
                pattern_data JSONB NOT NULL,
                success_metrics JSONB,
                discovery_agent VARCHAR(255),
                discovery_generation INTEGER,
                reuse_count INTEGER DEFAULT 0,
                avg_success_delta FLOAT DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags TEXT[]
            )
        """)
        
        # Create pattern usage history table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pattern_usage (
                id SERIAL PRIMARY KEY,
                pattern_id VARCHAR(255) REFERENCES discovered_patterns(pattern_id),
                agent_id VARCHAR(255) NOT NULL,
                generation INTEGER,
                success_score FLOAT,
                token_cost INTEGER,
                context JSONB,
                used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indices
        cur.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON discovered_patterns(pattern_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_patterns_action ON discovered_patterns(action_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_patterns_reuse ON discovered_patterns(reuse_count DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_patterns_success ON discovered_patterns(avg_success_delta DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_patterns_tags ON discovered_patterns USING GIN(tags)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_usage_pattern ON pattern_usage(pattern_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_usage_agent ON pattern_usage(agent_id)")
        
        conn.commit()
        conn.close()
    
    def catalog_pattern(self, pattern: CatalogedPattern) -> str:
        """
        Add a pattern to the catalog
        
        Returns:
            Pattern ID
        """
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()
        
        try:
            cur.execute("""
                INSERT INTO discovered_patterns 
                (pattern_id, pattern_type, action_type, description, pattern_data,
                 success_metrics, discovery_agent, discovery_generation, reuse_count,
                 avg_success_delta, tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (pattern_id) DO UPDATE SET
                    reuse_count = discovered_patterns.reuse_count + 1,
                    last_used = CURRENT_TIMESTAMP,
                    avg_success_delta = (
                        discovered_patterns.avg_success_delta * discovered_patterns.reuse_count +
                        EXCLUDED.avg_success_delta
                    ) / (discovered_patterns.reuse_count + 1)
                RETURNING pattern_id
            """, (
                pattern.pattern_id,
                pattern.pattern_type,
                pattern.action_type,
                pattern.description,
                Json(pattern.pattern_data),
                Json(pattern.success_metrics),
                pattern.discovery_agent,
                pattern.discovery_generation,
                pattern.reuse_count,
                pattern.avg_success_delta,
                pattern.tags
            ))
            
            pattern_id = cur.fetchone()[0]
            conn.commit()
            
            # Update cache
            self.pattern_cache[pattern_id] = pattern
            self._maintain_cache_size()
            
            logger.info(f"Cataloged pattern: {pattern_id}")
            return pattern_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to catalog pattern: {e}")
            raise
        finally:
            conn.close()
    
    def record_pattern_usage(self, pattern_id: str, agent_id: str,
                           generation: int, success_score: float,
                           token_cost: int, context: Dict[str, Any] = None):
        """Record usage of a pattern by an agent"""
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()
        
        try:
            # Record usage
            cur.execute("""
                INSERT INTO pattern_usage 
                (pattern_id, agent_id, generation, success_score, token_cost, context)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (pattern_id, agent_id, generation, success_score, token_cost, Json(context or {})))
            
            # Update pattern statistics
            cur.execute("""
                UPDATE discovered_patterns 
                SET reuse_count = reuse_count + 1,
                    last_used = CURRENT_TIMESTAMP,
                    avg_success_delta = (
                        avg_success_delta * reuse_count + %s
                    ) / (reuse_count + 1)
                WHERE pattern_id = %s
            """, (success_score, pattern_id))
            
            conn.commit()
            
            # Invalidate cache entry
            if pattern_id in self.pattern_cache:
                del self.pattern_cache[pattern_id]
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to record pattern usage: {e}")
        finally:
            conn.close()
    
    def search_patterns(self, pattern_type: Optional[str] = None,
                       action_type: Optional[str] = None,
                       min_reuse_count: int = 0,
                       min_success_delta: float = 0.0,
                       tags: Optional[List[str]] = None,
                       limit: int = 20) -> List[CatalogedPattern]:
        """Search for patterns matching criteria"""
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Build query
        query = """
            SELECT * FROM discovered_patterns 
            WHERE reuse_count >= %s AND avg_success_delta >= %s
        """
        params = [min_reuse_count, min_success_delta]
        
        if pattern_type:
            query += " AND pattern_type = %s"
            params.append(pattern_type)
        
        if action_type:
            query += " AND action_type = %s"
            params.append(action_type)
        
        if tags:
            query += " AND tags && %s"  # Array overlap operator
            params.append(tags)
        
        query += " ORDER BY avg_success_delta DESC, reuse_count DESC LIMIT %s"
        params.append(limit)
        
        cur.execute(query, params)
        results = cur.fetchall()
        conn.close()
        
        patterns = []
        for row in results:
            pattern = CatalogedPattern(
                pattern_id=row['pattern_id'],
                pattern_type=row['pattern_type'],
                action_type=row['action_type'],
                description=row['description'],
                pattern_data=row['pattern_data'],
                success_metrics=row['success_metrics'],
                discovery_agent=row['discovery_agent'],
                discovery_generation=row['discovery_generation'],
                reuse_count=row['reuse_count'],
                avg_success_delta=row['avg_success_delta'],
                created_at=row['created_at'],
                last_used=row['last_used'],
                tags=row['tags']
            )
            patterns.append(pattern)
            
            # Add to cache
            self.pattern_cache[pattern.pattern_id] = pattern
        
        self._maintain_cache_size()
        return patterns
    
    def get_pattern(self, pattern_id: str) -> Optional[CatalogedPattern]:
        """Get a specific pattern by ID"""
        # Check cache first
        if pattern_id in self.pattern_cache:
            return self.pattern_cache[pattern_id]
        
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("SELECT * FROM discovered_patterns WHERE pattern_id = %s", (pattern_id,))
        row = cur.fetchone()
        conn.close()
        
        if not row:
            return None
        
        pattern = CatalogedPattern(
            pattern_id=row['pattern_id'],
            pattern_type=row['pattern_type'],
            action_type=row['action_type'],
            description=row['description'],
            pattern_data=row['pattern_data'],
            success_metrics=row['success_metrics'],
            discovery_agent=row['discovery_agent'],
            discovery_generation=row['discovery_generation'],
            reuse_count=row['reuse_count'],
            avg_success_delta=row['avg_success_delta'],
            created_at=row['created_at'],
            last_used=row['last_used'],
            tags=row['tags']
        )
        
        # Add to cache
        self.pattern_cache[pattern_id] = pattern
        self._maintain_cache_size()
        
        return pattern
    
    def get_top_patterns_by_generation(self, generation: int, 
                                     window: int = 5,
                                     limit: int = 10) -> List[CatalogedPattern]:
        """Get top patterns discovered around a specific generation"""
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT * FROM discovered_patterns 
            WHERE discovery_generation BETWEEN %s AND %s
            ORDER BY avg_success_delta DESC, reuse_count DESC
            LIMIT %s
        """, (generation - window, generation + window, limit))
        
        results = cur.fetchall()
        conn.close()
        
        patterns = []
        for row in results:
            patterns.append(CatalogedPattern(
                pattern_id=row['pattern_id'],
                pattern_type=row['pattern_type'],
                action_type=row['action_type'],
                description=row['description'],
                pattern_data=row['pattern_data'],
                success_metrics=row['success_metrics'],
                discovery_agent=row['discovery_agent'],
                discovery_generation=row['discovery_generation'],
                reuse_count=row['reuse_count'],
                avg_success_delta=row['avg_success_delta'],
                created_at=row['created_at'],
                last_used=row['last_used'],
                tags=row['tags']
            ))
        
        return patterns
    
    def get_pattern_lineage(self, pattern_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get usage history of a pattern"""
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT * FROM pattern_usage 
            WHERE pattern_id = %s
            ORDER BY used_at DESC
            LIMIT %s
        """, (pattern_id, limit))
        
        lineage = cur.fetchall()
        conn.close()
        
        return lineage
    
    def get_catalog_statistics(self) -> Dict[str, Any]:
        """Get statistics about the pattern catalog"""
        conn = psycopg2.connect(self.db_url)
        cur = conn.cursor()
        
        stats = {}
        
        # Total patterns
        cur.execute("SELECT COUNT(*) FROM discovered_patterns")
        stats['total_patterns'] = cur.fetchone()[0]
        
        # Patterns by type
        cur.execute("""
            SELECT pattern_type, COUNT(*) 
            FROM discovered_patterns 
            GROUP BY pattern_type
        """)
        stats['patterns_by_type'] = dict(cur.fetchall())
        
        # Most reused patterns
        cur.execute("""
            SELECT pattern_id, description, reuse_count 
            FROM discovered_patterns 
            ORDER BY reuse_count DESC 
            LIMIT 5
        """)
        stats['most_reused'] = [
            {'pattern_id': row[0], 'description': row[1], 'reuse_count': row[2]}
            for row in cur.fetchall()
        ]
        
        # Most successful patterns
        cur.execute("""
            SELECT pattern_id, description, avg_success_delta 
            FROM discovered_patterns 
            WHERE reuse_count >= 3
            ORDER BY avg_success_delta DESC 
            LIMIT 5
        """)
        stats['most_successful'] = [
            {'pattern_id': row[0], 'description': row[1], 'avg_success': row[2]}
            for row in cur.fetchall()
        ]
        
        # Usage over time
        cur.execute("""
            SELECT DATE(used_at) as date, COUNT(*) as usage_count
            FROM pattern_usage
            WHERE used_at >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY DATE(used_at)
            ORDER BY date
        """)
        stats['usage_timeline'] = [
            {'date': row[0].isoformat(), 'count': row[1]}
            for row in cur.fetchall()
        ]
        
        conn.close()
        return stats
    
    def export_catalog(self, output_file: str):
        """Export entire catalog to JSON file"""
        patterns = self.search_patterns(limit=10000)  # Get all patterns
        
        export_data = {
            'export_date': datetime.now().isoformat(),
            'pattern_count': len(patterns),
            'patterns': [asdict(p) for p in patterns]
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(patterns)} patterns to {output_file}")
    
    def _maintain_cache_size(self):
        """Maintain cache within size limits"""
        if len(self.pattern_cache) > self.cache_size:
            # Remove least recently used patterns
            sorted_patterns = sorted(
                self.pattern_cache.items(),
                key=lambda x: x[1].last_used
            )
            
            # Remove oldest 20%
            remove_count = len(self.pattern_cache) - int(self.cache_size * 0.8)
            for pattern_id, _ in sorted_patterns[:remove_count]:
                del self.pattern_cache[pattern_id]


if __name__ == "__main__":
    # Demo the pattern cataloger
    import os
    
    db_url = os.environ.get('DATABASE_URL', 
                           'postgresql://dean_user:dean_password_2024@localhost:5432/agent_evolution')
    
    cataloger = PatternCataloger(db_url)
    
    print("Pattern Cataloger Demo")
    print("=" * 60)
    
    # Catalog some patterns
    print("\n1. Cataloging patterns...")
    
    patterns = [
        CatalogedPattern(
            pattern_id="opt_memoization_001",
            pattern_type="optimization",
            action_type="implement_todos",
            description="Add memoization to recursive functions",
            pattern_data={
                "before": "def fib(n): return fib(n-1) + fib(n-2)",
                "after": "@lru_cache\ndef fib(n): return fib(n-1) + fib(n-2)"
            },
            success_metrics={"speedup": 100.0, "tokens_saved": 50},
            discovery_agent="agent_001",
            discovery_generation=5,
            avg_success_delta=0.85,
            tags=["optimization", "caching", "recursion"]
        ),
        CatalogedPattern(
            pattern_id="ref_extract_method_001",
            pattern_type="refactoring",
            action_type="refactor_complexity",
            description="Extract complex logic into helper methods",
            pattern_data={
                "complexity_reduction": 0.4,
                "methods_extracted": 3
            },
            success_metrics={"maintainability": 0.8, "readability": 0.9},
            discovery_agent="agent_002",
            discovery_generation=8,
            avg_success_delta=0.75,
            tags=["refactoring", "complexity", "maintainability"]
        )
    ]
    
    for pattern in patterns:
        pattern_id = cataloger.catalog_pattern(pattern)
        print(f"   Cataloged: {pattern_id}")
    
    # Search patterns
    print("\n2. Searching patterns...")
    results = cataloger.search_patterns(pattern_type="optimization")
    for pattern in results:
        print(f"   Found: {pattern.pattern_id} - {pattern.description}")
    
    # Record usage
    print("\n3. Recording pattern usage...")
    cataloger.record_pattern_usage(
        "opt_memoization_001",
        "agent_003",
        generation=10,
        success_score=0.9,
        token_cost=1800
    )
    
    # Get statistics
    print("\n4. Catalog statistics:")
    stats = cataloger.get_catalog_statistics()
    print(f"   Total patterns: {stats['total_patterns']}")
    print(f"   By type: {stats['patterns_by_type']}")
    
    if stats['most_reused']:
        print("\n   Most reused:")
        for p in stats['most_reused']:
            print(f"   - {p['pattern_id']}: {p['reuse_count']} uses")
    
    print("\nDemo complete!")