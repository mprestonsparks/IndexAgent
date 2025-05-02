# Start the IndexAgent stack using docker-compose in detached mode
up:
	docker-compose up -d

# Stop the stack and remove containers, networks, and volumes created by docker-compose
down:
	docker-compose down

# Trigger Zoekt to re-index the mounted repositories
# Assumes the Zoekt service is named 'zoekt' in docker-compose.yml
# and that 'zoekt-indexer' is available in the container.
index:
	docker-compose exec zoekt zoekt-indexer /data/repos

# Show the status of running containers in the stack
status:
	docker-compose ps
