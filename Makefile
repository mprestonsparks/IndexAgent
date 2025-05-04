# Start the IndexAgent stack using docker-compose in detached mode
up: build
	$(MAKE) clean
	$(MAKE) down
	docker-compose -f config/docker-compose.yml up -d

build:
	docker build . -t indexagent:latest

# Stop the stack and remove containers, networks, and volumes created by docker-compose
down:
	docker-compose -f config/docker-compose.yml down

# Trigger Zoekt to re-index the mounted repositories
# Assumes the Zoekt service is named 'zoekt' in docker-compose.yml
# and that 'zoekt-indexer' is available in the container.
index:
	docker-compose exec zoekt zoekt-indexer /data/repos

# Show the status of running containers in the stack
status:
	docker-compose ps

# Run the find_undocumented.py script to check for missing or short docstrings
find-undocumented:
	python3 scripts/find_undocumented.py

clean:
	docker rm -f zoekt-indexserver sourcebot indexagent || true
	docker network rm config_default || true
