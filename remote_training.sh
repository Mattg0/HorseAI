#!/bin/bash
# vastai_auto_deploy.sh - Automated vast.ai instance creation and training setup

set -e

# Configuration
FILTER='reliability > 0.98 gpu_ram >= 10 dph < 0.3 cuda_vers >= 12.0 gpu_name != RTX_5070 gpu_name != RTX_5080 gpu_name != RTX_5090 geolocation != China geolocation != India geolocation != Russia'
DOCKER_IMAGE="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"
DISK_SIZE="32"
SETUP_SCRIPT_URL="https://raw.githubusercontent.com/Mattg0/HorseAI/refs/heads/fastforward-neural-net/get_horseai.sh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
VASTAI="conda run -n vastai vastai"
echo -e "${BLUE}🚀 Starting automated vast.ai deployment...${NC}"

# Step 1: Search for offers using vastai CLI with filter
echo -e "${YELLOW}Searching for suitable offers...${NC}"
echo -e "${BLUE}Using filter: $FILTER${NC}"

# Use vastai search and extract offer ID without jq
search_output=$($VASTAI search offers "$FILTER" --raw)

if [ -z "$search_output" ]; then
    echo -e "${RED}No suitable offers found with filter${NC}"
    exit 1
fi

# Extract first offer ID using grep and sed (more reliable than jq for malformed JSON)
offer_id=$(echo "$search_output" | grep -o '"id": *[0-9][0-9]*' | head -1 | sed 's/"id": *//')

if [ -z "$offer_id" ]; then
    echo -e "${RED}Could not extract offer ID from search result${NC}"
    echo -e "${YELLOW}First 200 characters of response:${NC}"
    echo "$search_output" | head -c 200
    exit 1
fi

echo -e "${GREEN}Found offer ID: $offer_id${NC}"

# Extract additional details without jq
gpu_name=$(echo "$search_output" | grep -o '"gpu_name": *"[^"]*"' | head -1 | sed 's/"gpu_name": *"//' | sed 's/"//')
dph_total=$(echo "$search_output" | grep -o '"dph_total": *[0-9.]*' | head -1 | sed 's/"dph_total": *//')
geolocation=$(echo "$search_output" | grep -o '"geolocation": *"[^"]*"' | head -1 | sed 's/"geolocation": *"//' | sed 's/"//')

echo -e "${BLUE}Selected offer: GPU: $gpu_name, Price: \$$dph_total/hour, Location: $geolocation${NC}"

# Step 2: Create instance
echo -e "${YELLOW}🏗️ Creating instance...${NC}"
create_result=$($VASTAI create instance "$offer_id" --image "$DOCKER_IMAGE" --disk "$DISK_SIZE")
echo "$create_result"
# Extract instance ID from result
# Extract instance ID from the new_contract field
instance_id=$(echo "$create_result" | grep -o '[0-9]*')

if [ -z "$instance_id" ]; then
    echo -e "${RED}Failed to create instance${NC}"
    echo "$create_result"
    exit 1
fi

echo -e "${GREEN}Instance created: $instance_id${NC}"

# Step 3: Wait for instance to be ready
echo -e "${YELLOW}⏳ Waiting for instance to become active...${NC}"
max_attempts=60
attempt=0

while [ $attempt -lt $max_attempts ]; do
    status=$($VASTAI show instance $instance_id --raw | grep "actual_status"| sed -E 's/.*"actual_status": *"([^"]+)".*/\1/p'| head -1|tr -d '[:space:]')

    if [ "$status" = 'running' ]; then
        echo -e "${GREEN}✅ Instance is running${NC}"
        break
    elif [ "$status" = "exited" ] || [ "$status" = "error" ]; then
        echo -e "${RED}❌ Instance failed to start (status: $status)${NC}"
        exit 1
    fi

    echo -e "${BLUE}Instance status: $status (attempt $((attempt + 1))/$max_attempts)${NC}"
    sleep 10
    ((attempt++))
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}❌ Timeout waiting for instance to start${NC}"
    exit 1
fi

echo -e "${YELLOW}🔑 Setting up SSH key...${NC}"
$VASTAI attach ssh $instance_id ~/.ssh/vastai_key.pub

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to set SSH key${NC}"
    exit 1
fi

echo -e "${GREEN}✅ SSH key configured${NC}"

# Get SSH connection string
echo -e "${YELLOW}🔑 Getting SSH connection details...${NC}"
ssh_url=$($VASTAI ssh-url "$instance_id")

if [ -z "$ssh_url" ]; then
    echo -e "${RED}❌ Could not get SSH URL${NC}"
    exit 1
fi

echo -e "${GREEN}✅ SSH URL: $ssh_url${NC}"

# Step 5: Wait for SSH to be ready
echo -e "${YELLOW}⏳ Waiting for SSH to be ready...${NC}"
max_ssh_attempts=30
ssh_attempt=0

while [ $ssh_attempt -lt $max_ssh_attempts ]; do
    # Test SSH connection with your private key
    if ssh -i ~/.ssh/vastai_key -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $ssh_url "echo 'SSH Ready'" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ SSH connection established!${NC}"
        break
    fi

    echo -e "${BLUE}SSH not ready yet (attempt $((ssh_attempt + 1))/$max_ssh_attempts)${NC}"
    sleep 10
    ((ssh_attempt++))
done

if [ $ssh_attempt -eq $max_ssh_attempts ]; then
    echo -e "${RED}❌ SSH connection timeout${NC}"
    exit 1
fi

# Function to parse vastai SSH URL and execute commands
execute_remote_command() {
    local ssh_url=$1
    local command=$2

    # Parse ssh://root@host:port format
    local host=$(echo $ssh_url | sed 's|ssh://root@||' | sed 's|:.*||')
    local port=$(echo $ssh_url | sed 's|.*:||')

    echo -e "${BLUE}Executing on $host:$port: $command${NC}"

    # Execute the command with proper quoting
    ssh -i ~/.ssh/vastai_key -o BatchMode=yes -o StrictHostKeyChecking=no -T -p $port root@$host "$command"
    return $?
}

# Step 6: Execute setup script with retry logic
echo -e "${YELLOW}📥 Downloading and executing setup script...${NC}"
max_setup_attempts=10
setup_attempt=0
setup_success=false

echo -e "${YELLOW}📥 Downloading and executing setup script...${NC}"

execute_remote_command "$ssh_url" "
wget -O /tmp/setup.sh $SETUP_SCRIPT_URL &&
chmod +x /tmp/setup.sh &&
/tmp/setup.sh
"

if [ $? -ne 0 ]; then
   echo -e "${RED}❌ Setup script failed${NC}"
   exit 1
fi

echo -e "${GREEN}✅ Setup script executed successfully!${NC}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "${BLUE}📊 Summary:${NC}"
    echo -e "  Instance ID: $instance_id"
    echo -e "  SSH Command: ssh $ssh_url"
    echo -e "  Project Path: /workspace/HorseAI"
    echo -e "\n${YELLOW}💡 Next steps:${NC}"
    echo -e "  1. SSH into instance: ssh $ssh_url"
    echo -e "  2. Check training logs: tail -f /workspace/HorseAI/logs/*"
    echo -e "  3. Monitor GPU usage: nvidia-smi"
    echo -e "  4. Destroy when done: vastai destroy instance $instance_id"
else
    echo -e "${RED}❌ Setup script failed${NC}"
    echo -e "${YELLOW}⚠️ Instance is still running. SSH manually to debug:${NC}"
    echo -e "  ssh $ssh_url"
    echo -e "  Or destroy: vastai destroy instance $instance_id"
    exit 1
fi