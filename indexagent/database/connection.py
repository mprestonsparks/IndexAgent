"""
Database connection utilities for DEAN agent evolution system
Provides async database session management with connection pooling
"""

import os
import logging
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)

# Global engine instance
_engine: Optional[AsyncEngine] = None
_async_session_maker: Optional[sessionmaker] = None


def get_database_url() -> str:
    """Get database URL from environment or use default"""
    # Support multiple environment variable names
    db_url = os.getenv('DATABASE_URL') or os.getenv('DEAN_DATABASE_URL')
    
    if not db_url:
        # Default for local development
        db_url = "postgresql+asyncpg://dean_api:dean_api_password@localhost:5432/agent_evolution"
        logger.warning(f"No DATABASE_URL found, using default: {db_url}")
    
    # Convert postgresql:// to postgresql+asyncpg:// for async support
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    return db_url


async def init_database(database_url: Optional[str] = None) -> AsyncEngine:
    """Initialize the database engine and session maker
    
    Args:
        database_url: Optional database URL override
        
    Returns:
        Async database engine
    """
    global _engine, _async_session_maker
    
    if _engine is not None:
        return _engine
    
    url = database_url or get_database_url()
    
    # Create async engine with connection pooling
    _engine = create_async_engine(
        url,
        echo=os.getenv('SQL_ECHO', 'false').lower() == 'true',
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,   # Recycle connections after 1 hour
    )
    
    # Create async session maker
    _async_session_maker = sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    logger.info("Database engine initialized successfully")
    return _engine


async def close_database() -> None:
    """Close the database engine and cleanup resources"""
    global _engine, _async_session_maker
    
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_maker = None
        logger.info("Database engine closed")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session as context manager
    
    Usage:
        async with get_db_session() as session:
            result = await session.execute(query)
    """
    if _async_session_maker is None:
        await init_database()
    
    async with _async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncSession:
    """Get async database session (FastAPI dependency style)
    
    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(query)
    """
    if _async_session_maker is None:
        await init_database()
    
    async with _async_session_maker() as session:
        yield session


# Health check function
async def check_database_health() -> dict:
    """Check if database is accessible and healthy
    
    Returns:
        Dictionary with health status
    """
    try:
        async with get_db_session() as session:
            # Simple query to check connection
            from sqlalchemy import text
            result = await session.execute(text("SELECT 1"))
            result.scalar()
            
            # Check if schema exists
            schema_check = await session.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.schemata 
                    WHERE schema_name = 'agent_evolution'
                )
            """))
            schema_exists = schema_check.scalar()
            
            return {
                "status": "healthy",
                "connected": True,
                "schema_exists": schema_exists
            }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)
        }


# Transaction utilities
async def execute_in_transaction(func, *args, **kwargs):
    """Execute a function within a database transaction
    
    Args:
        func: Async function to execute
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Result of the function
    """
    async with get_db_session() as session:
        return await func(session, *args, **kwargs)