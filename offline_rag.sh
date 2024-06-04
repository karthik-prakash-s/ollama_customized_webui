#!/bin/bash

# Start Docker containers in detached mode
docker-compose up -d

# Check if the command was successful
if [ $? -ne 0 ]; then
  echo "Failed to start Docker containers."
  exit 1
fi

# Function to check the status of the containers
check_containers() {
  # Get the status of all containers
  STATUS=$(docker-compose ps -q | xargs docker inspect -f '{{ .State.Status }}' | grep -v 'running')
  if [ -z "$STATUS" ]; then
    return 0
  else
    return 1
  fi
}

# Wait for all containers to be up and running
echo "Checking container statuses..."
while ! check_containers; do
  echo "Waiting for containers to be up..."
  sleep 5
done

# Display the message
echo "System is ready to use."
