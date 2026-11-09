#!/usr/bin/env bash
# Run ON THE SERVER (normally via AWS SSM from .github/workflows/deploy.yml) after the repo
# has been checked out at the commit being deployed.
#
# Usage: scripts/deploy.sh <image-ref>
#   e.g. scripts/deploy.sh 321043732660.dkr.ecr.ap-southeast-2.amazonaws.com/jee-tutor:<git-sha>
#
# Rollback: run it again with the previous image tag (after checking out that commit).
#
# Server prerequisites: docker + compose plugin, AWS CLI v2, an IAM instance role that can pull
# from ECR, a populated .env (GROQ_API_KEY, OCR_SPACE_API_KEY) and data/processed (dvc pull).
set -euo pipefail

IMAGE="${1:?usage: deploy.sh <image-ref>}"
REGISTRY="${IMAGE%%/*}"
REGION="${AWS_REGION:-ap-southeast-2}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8001/health}"

# The EC2 metadata service can be slow for the first credential lookup.
export AWS_METADATA_SERVICE_TIMEOUT="${AWS_METADATA_SERVICE_TIMEOUT:-5}"
export AWS_METADATA_SERVICE_NUM_ATTEMPTS="${AWS_METADATA_SERVICE_NUM_ATTEMPTS:-3}"

cd "$(dirname "$0")/.."

echo "Deploying $IMAGE"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

export APP_IMAGE="$IMAGE"
docker compose pull api app
docker compose up -d --no-build chroma api app

echo "Waiting for the API to become healthy at $HEALTH_URL"
for _ in $(seq 1 30); do
  if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    echo "Healthy."
    docker compose ps
    exit 0
  fi
  sleep 5
done

echo "API did not become healthy within 150s" >&2
docker compose ps >&2
docker compose logs --tail 50 api >&2
exit 1
