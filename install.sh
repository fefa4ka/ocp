#!/bin/bash

# OpenAI Compatible API Proxy Installation Script for Ubuntu
# This script installs the proxy as a systemd service with autostart

set -e  # Exit on any error

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

# Configuration variables
APP_NAME="openai-proxy"
APP_USER="$APP_NAME"
APP_GROUP="$APP_NAME"
APP_DIR="/opt/$APP_NAME"
VENV_DIR="$APP_DIR/venv"
CONFIG_DIR="/etc/$APP_NAME"
LOG_DIR="/var/log/$APP_NAME"
SERVICE_FILE="/etc/systemd/system/$APP_NAME.service"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Installing OpenAI Compatible API Proxy...${NC}"

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
apt-get update
apt-get install -y python3 python3-venv python3-pip autossh

# Create app user if it doesn't exist
if ! id -u "$APP_USER" &>/dev/null; then
  echo -e "${YELLOW}Creating service user $APP_USER...${NC}"
  useradd -r -s /bin/false "$APP_USER"
fi

# Create necessary directories
echo -e "${YELLOW}Creating application directories...${NC}"
mkdir -p "$APP_DIR"
mkdir -p "$CONFIG_DIR"
mkdir -p "$LOG_DIR"

# Copy application files
echo -e "${YELLOW}Copying application files...${NC}"
cp -r ./* "$APP_DIR/"

# Create Python virtual environment
echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"

# Create .env file if it doesn't exist
if [ ! -f "$CONFIG_DIR/.env" ]; then
  echo -e "${YELLOW}Creating default .env configuration...${NC}"
  cat > "$CONFIG_DIR/.env" << EOF
MODEL_LIST_URL=https://example.com/models
# MODEL_LIST_AUTH_TOKEN=your_token_here
LOG_LEVEL=INFO
EOF
fi

# Create symlink to config
ln -sf "$CONFIG_DIR/.env" "$APP_DIR/.env"

# Create systemd service file
echo -e "${YELLOW}Creating systemd service...${NC}"
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=OpenAI Compatible API Proxy
After=network.target

[Service]
User=$APP_USER
Group=$APP_GROUP
WorkingDirectory=$APP_DIR
ExecStart=/bin/bash -c "$VENV_DIR/bin/uvicorn main:app --host 127.0.0.1 --port 8000 & autossh -i ~/.ssh/intranet_ssh -R 8888:127.0.0.1:8000 user@example.com -N"
Restart=always
RestartSec=5
SyslogIdentifier=$APP_NAME
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# Set proper permissions
echo -e "${YELLOW}Setting permissions...${NC}"
chown -R "$APP_USER:$APP_GROUP" "$APP_DIR"
chown -R "$APP_USER:$APP_GROUP" "$CONFIG_DIR"
chown -R "$APP_USER:$APP_GROUP" "$LOG_DIR"
chmod 755 "$APP_DIR"
chmod 755 "$CONFIG_DIR"
chmod 755 "$LOG_DIR"
chmod 644 "$SERVICE_FILE"

# Enable and start the service
echo -e "${YELLOW}Enabling and starting service...${NC}"
systemctl daemon-reload
systemctl enable "$APP_NAME"
systemctl start "$APP_NAME"

# Check service status
echo -e "${YELLOW}Checking service status...${NC}"
systemctl status "$APP_NAME"

echo -e "${GREEN}Installation complete!${NC}"
echo -e "${GREEN}Service is running at http://localhost:8000${NC}"
echo -e "${GREEN}Configuration file is at $CONFIG_DIR/.env${NC}"
echo -e "${GREEN}Logs can be viewed with: journalctl -u $APP_NAME${NC}"
