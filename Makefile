# Docker parameters
DOCKER_IMAGE=wealthcare-ai-gateway
DOCKER_TAG=latest

FI_DOCKER_IMAGE=fi-mcp-dev
FI_DOCKER_TAG=latest

# Docker commands
build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker build -t $(FI_DOCKER_IMAGE):$(FI_DOCKER_TAG) .
	docker-compose up -d

stop:
	docker-compose down

rebuild:
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d