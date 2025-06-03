"""Invoke tasks for the project."""

from invoke import Collection
from . import quality

# Create the namespace
ns = Collection()

# Add quality tasks
ns.add_collection(quality)