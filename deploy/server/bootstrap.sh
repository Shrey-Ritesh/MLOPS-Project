#!/usr/bin/env bash
# One-time setup of the deployment server (Ubuntu 24.04, EC2). Safe to re-run.
#
# Run it ON THE SERVER as the normal `ubuntu` user:
#   curl -fsSL https://raw.githubusercontent.com/Shrey-Ritesh/MLOPS-Project/main/deploy/server/bootstrap.sh | bash
# or copy it over and run `bash bootstrap.sh`. Override the branch with BRANCH=... if needed.
#
# Installs: Docker Engine + compose plugin, git, AWS CLI v2, DVC (in ~/dvcenv), and clones the repo.
# Status: the Docker, git, DVC and clone steps were run by hand on the real server. The AWS CLI
# step has not been run there yet.
#
# After it finishes you still need to (see docs/devops_runbook.md section 10.3):
#   1. create ~/MLOPS-Project/.env with the real GROQ_API_KEY / OCR_SPACE_API_KEY
#   2. log out and back in (so the docker group applies), then `dvc pull` the data
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

REPO_URL="${REPO_URL:-https://github.com/Shrey-Ritesh/MLOPS-Project.git}"
BRANCH="${BRANCH:-main}"
APP_DIR="${APP_DIR:-$HOME/MLOPS-Project}"

echo "==> Base packages"
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl git unzip python3-venv python3-pip

echo "==> Docker Engine + compose plugin"
if ! command -v docker >/dev/null 2>&1; then
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
sudo usermod -aG docker "$USER"
sudo systemctl enable --now docker

echo "==> AWS CLI v2 (used by scripts/deploy.sh to log in to ECR)"
if ! command -v aws >/dev/null 2>&1; then
  tmp="$(mktemp -d)"
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "$tmp/awscliv2.zip"
  unzip -q "$tmp/awscliv2.zip" -d "$tmp"
  sudo "$tmp/aws/install"
  rm -rf "$tmp"
fi

echo "==> AWS settings for the EC2 instance role"
# Without a longer metadata timeout, DVC/boto can fail to find the role credentials and report
# misleading "missing files" errors.
if ! grep -q AWS_METADATA_SERVICE_TIMEOUT "$HOME/.profile" 2>/dev/null; then
  cat >> "$HOME/.profile" <<'EOF'

# JEE Tutor deployment: credentials come from the EC2 instance role
export AWS_DEFAULT_REGION=ap-southeast-2
export AWS_METADATA_SERVICE_TIMEOUT=5
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=3
EOF
fi

echo "==> DVC (in ~/dvcenv)"
if [ ! -x "$HOME/dvcenv/bin/dvc" ]; then
  python3 -m venv "$HOME/dvcenv"
  "$HOME/dvcenv/bin/pip" install -q "dvc[s3]"
fi

echo "==> Repository at $APP_DIR (branch $BRANCH)"
if [ ! -d "$APP_DIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

echo
echo "Done. Versions:"
docker --version
docker compose version
aws --version
"$HOME/dvcenv/bin/dvc" --version
echo
echo "Next: create $APP_DIR/.env, log out and back in, then run the dvc pull step in the runbook."
