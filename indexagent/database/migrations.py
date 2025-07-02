"""
Database migration management using Alembic.
Per specifications: IndexAgent/indexagent/database/migrations.py

Migration Requirements:
- Migration capability for future updates per specification requirements
- Schema evolution management
- Automated migration execution during deployment
"""

import os
import logging
from alembic import command
from alembic.config import Config
from pathlib import Path

logger = logging.getLogger(__name__)

def get_alembic_config() -> Config:
    """
    Get Alembic configuration.
    
    Returns:
        Alembic Config object
    """
    # Get the directory containing this file
    current_dir = Path(__file__).parent
    alembic_ini_path = current_dir / "alembic.ini"
    
    # Create alembic.ini if it doesn't exist
    if not alembic_ini_path.exists():
        create_alembic_ini(alembic_ini_path)
    
    config = Config(str(alembic_ini_path))
    
    # Set database URL from environment
    database_url = os.getenv(
        "AGENT_EVOLUTION_DATABASE_URL",
        "postgresql://postgres:password@postgres:5432/agent_evolution"
    )
    config.set_main_option("sqlalchemy.url", database_url)
    
    return config

def create_alembic_ini(ini_path: Path):
    """Create alembic.ini configuration file."""
    alembic_ini_content = """# Alembic configuration for DEAN system database migrations

[alembic]
# Path to migration scripts
script_location = %(here)s/versions

# Template used to generate migration files
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s_%%(slug)s

# Sys.path path, will be prepended to sys.path if present.
prepend_sys_path = .

# Timezone to use when rendering the date within the migration file
timezone = UTC

# Max length of characters to apply to the "slug" field
truncate_slug_length = 40

# Set to 'true' to run the environment during the 'revision' command
revision_environment = false

# Set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
sourceless = false

# Version locations
version_locations = %(here)s/versions

# Version path separator; As mentioned above, this is the character used to split
# version_locations. The default within new alembic.ini files is "os", which uses
# os.pathsep. If this key is omitted entirely, it falls back to the legacy
# behavior of splitting on spaces and/or commas.
version_path_separator = :

# The output encoding used when revision files
# are written from script.py.mako
output_encoding = utf-8

sqlalchemy.url = postgresql://postgres:password@postgres:5432/agent_evolution

[post_write_hooks]
# Post-write hooks define scripts or Python functions that are run
# on newly generated revision scripts.

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""
    
    with open(ini_path, 'w') as f:
        f.write(alembic_ini_content)
    
    logger.info(f"Created alembic.ini at {ini_path}")

def initialize_alembic():
    """Initialize Alembic migration environment."""
    try:
        config = get_alembic_config()
        
        # Create versions directory
        versions_dir = Path(config.get_main_option("script_location"))
        versions_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Alembic if not already initialized
        env_py_path = versions_dir.parent / "env.py"
        if not env_py_path.exists():
            logger.info("Initializing Alembic migration environment")
            command.init(config, str(versions_dir.parent))
        
        logger.info("Alembic migration environment ready")
        
    except Exception as e:
        logger.error(f"Failed to initialize Alembic: {e}")
        raise

def create_migration(message: str, autogenerate: bool = True):
    """
    Create a new migration.
    
    Args:
        message: Description of the migration
        autogenerate: Whether to auto-generate migration from model changes
    """
    try:
        config = get_alembic_config()
        
        logger.info(f"Creating migration: {message}")
        command.revision(
            config,
            message=message,
            autogenerate=autogenerate
        )
        
        logger.info("Migration created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create migration: {e}")
        raise

def run_migrations():
    """
    Run all pending migrations.
    Per specification requirement for migration capability.
    """
    try:
        config = get_alembic_config()
        
        logger.info("Running database migrations")
        command.upgrade(config, "head")
        
        logger.info("Database migrations completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

def downgrade_migration(revision: str = "-1"):
    """
    Downgrade to a specific migration.
    
    Args:
        revision: Target revision (default: previous migration)
    """
    try:
        config = get_alembic_config()
        
        logger.info(f"Downgrading to revision: {revision}")
        command.downgrade(config, revision)
        
        logger.info("Downgrade completed successfully")
        
    except Exception as e:
        logger.error(f"Downgrade failed: {e}")
        raise

def get_migration_history():
    """
    Get migration history using Alembic API.
    
    Returns:
        List of migration revisions with details
    """
    try:
        from alembic import command
        from alembic.script import ScriptDirectory
        from alembic.runtime.environment import EnvironmentContext
        from alembic.runtime.migration import MigrationContext
        from sqlalchemy import create_engine
        import os
        
        config = get_alembic_config()
        
        # Get script directory
        script_dir = ScriptDirectory.from_config(config)
        
        # Get database URL from environment
        database_url = os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url"))
        engine = create_engine(database_url)
        
        # Get current revision from database
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            current_heads = context.get_current_heads()
        
        # Get all revisions
        revisions = []
        for revision in script_dir.walk_revisions():
            is_current = revision.revision in current_heads
            is_head = revision.revision in script_dir.get_heads()
            
            revisions.append({
                "revision": revision.revision,
                "description": revision.doc or revision.message,
                "branch_labels": list(revision.branch_labels or []),
                "down_revision": revision.down_revision,
                "is_current": is_current,
                "is_head": is_head,
                "create_date": revision.create_date.isoformat() if revision.create_date else None
            })
        
        # Sort by dependency order (newest first)
        revisions.reverse()
        
        logger.info(f"Retrieved {len(revisions)} migration revisions")
        return revisions
        
    except Exception as e:
        logger.error(f"Failed to get migration history: {e}")
        # Return empty list as fallback
        return []

def validate_database_schema():
    """
    Validate that the database schema matches the current models.
    
    Returns:
        True if schema is up to date, False otherwise
    """
    try:
        from .schema import Base, engine
        from sqlalchemy import inspect
        
        # Get current database schema
        inspector = inspect(engine)
        
        # Check if agent_evolution schema exists
        schemas = inspector.get_schema_names()
        if 'agent_evolution' not in schemas:
            logger.warning("agent_evolution schema not found")
            return False
        
        # Check if required tables exist
        required_tables = [
            'agents', 'evolution_history', 'performance_metrics',
            'discovered_patterns', 'strategy_evolution', 'audit_log',
            'token_transactions'
        ]
        
        existing_tables = inspector.get_table_names(schema='agent_evolution')
        
        for table in required_tables:
            if table not in existing_tables:
                logger.warning(f"Required table '{table}' not found")
                return False
        
        logger.info("Database schema validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        return False

def reset_database():
    """
    Reset database to initial state (development only).
    WARNING: This will drop all data!
    """
    try:
        from .schema import Base, engine
        
        # Only allow in development environment
        if os.getenv("ENVIRONMENT") != "development":
            raise ValueError("Database reset only allowed in development environment")
        
        logger.warning("Resetting database - ALL DATA WILL BE LOST")
        
        # Drop all tables in agent_evolution schema
        Base.metadata.drop_all(bind=engine)
        
        # Recreate tables
        from .schema import create_all_tables
        create_all_tables()
        
        logger.info("Database reset completed")
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        raise

def backup_database(backup_path: str):
    """
    Create database backup using pg_dump.
    
    Args:
        backup_path: Path to save backup file
    """
    try:
        import subprocess
        
        database_url = os.getenv(
            "AGENT_EVOLUTION_DATABASE_URL",
            "postgresql://postgres:password@postgres:5432/agent_evolution"
        )
        
        logger.info(f"Creating database backup at {backup_path}")
        
        # Use pg_dump to create backup
        cmd = ["pg_dump", database_url, "-f", backup_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Database backup created successfully")
        else:
            logger.error(f"Backup failed: {result.stderr}")
            raise RuntimeError(f"pg_dump failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python migrations.py <command>")
        print("Commands: init, migrate, create <message>, validate, reset")
        sys.exit(1)
    
    command_name = sys.argv[1]
    
    try:
        if command_name == "init":
            initialize_alembic()
            print("✅ Alembic initialized")
            
        elif command_name == "migrate":
            run_migrations()
            print("✅ Migrations completed")
            
        elif command_name == "create":
            if len(sys.argv) < 3:
                print("Usage: python migrations.py create <message>")
                sys.exit(1)
            message = " ".join(sys.argv[2:])
            create_migration(message)
            print(f"✅ Migration created: {message}")
            
        elif command_name == "validate":
            if validate_database_schema():
                print("✅ Database schema is valid")
            else:
                print("❌ Database schema validation failed")
                sys.exit(1)
                
        elif command_name == "reset":
            reset_database()
            print("✅ Database reset completed")
            
        else:
            print(f"Unknown command: {command_name}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)