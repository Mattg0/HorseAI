#!/bin/bash
# vastai_auto_deploy.sh - Automated vast.ai instance creation and training setup

set -e

# Configuration
FILTER='reliability > 0.98 gpu_ram >= 10 dph < 0.3 cuda_vers >= 12.0 gpu_name != RTX_5070 gpu_name != RTX_5080 gpu_name != RTX_5090'
DOCKER_IMAGE="pytorch/pytorch"
DISK_SIZE="32"
SETUP_SCRIPT_URL="https://raw.githubusercontent.com/Mattg0/HorseAI/main/setup_horseai.sh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
VASTAI="conda run -n vastai vastai"
echo -e "${BLUE}🚀 Starting automated vast.ai deployment...${NC}"

# Step 1: Search for offers using curl API and extract first offer ID
echo -e "${YELLOW}Searching for suitable offers...${NC}"

offer_data=$(curl -s -X POST \
   -d '{"verified": {"eq": true}, "external": {"eq": false}, "rentable": {"eq": true}, "rented": {"eq": false}, "reliability": {"gt": "0.98"}, "gpu_ram": {"gte": 10000.0}, "dph_total": {"lt": "0.3"}, "cpu_ram": {"gte": 24000.0}, "order": [["dph_total", "asc"]], "type": "on-demand", "allocated_storage": 5.0}' \
   "https://console.vast.ai/api/v0/bundles/?api_key=$API_KEY")

offer_id=$(echo "$offer_data" | grep -o '"id": [0-9]*' | head -1 | sed 's/"id": //')

if [ -z "$offer_id" ]; then
    echo -e "${RED}No suitable offers found${NC}"
    echo "API Response: $(echo "$offer_data" | head -200)"
    exit 1
fi

echo -e "${GREEN}Found offer ID: $offer_id${NC}"

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

    echo -e "${BLUE}Executing on $host:$port${NC}"

    # Execute the command
    ssh -i ~/.ssh/vastai_key -o BatchMode=yes -o StrictHostKeyChecking=no -T -p $port root@$host "$command"
    return $?
}

# Step 6: Execute setup script with retry logic
echo -e "${YELLOW}📥 Downloading and executing setup script...${NC}"
max_setup_attempts=10
setup_attempt=0
setup_success=false

while [ $setup_attempt -lt $max_setup_attempts ] && [ "$setup_success" = false ]; do
    echo -e "${BLUE}Setup attempt $((setup_attempt + 1))/$max_setup_attempts...${NC}"

    # Execute setup script remotely
    setup_output=$(execute_remote_command "$ssh_url" "curl -sSL $SETUP_SCRIPT_URL | bash" 2>&1)
    ssh_exit_code=$?

    # Check for authentication/permission issues
    if echo "$setup_output" | grep -q "Permission denied\|authentication fail\|Connection refused"; then
        echo -e "${YELLOW}⚠️  Connection failed, retrying in 15 seconds...${NC}"
        setup_attempt=$((setup_attempt + 1))
        sleep 15
        continue
    elif [ $ssh_exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ Setup script executed successfully!${NC}"
        setup_success=true
        break
    else
        echo -e "${RED}❌ Setup script failed with exit code $ssh_exit_code${NC}"
        echo "Output: $setup_output"
        setup_attempt=$((setup_attempt + 1))
        sleep 10
    fi
done

if [ "$setup_success" = false ]; then
    echo -e "${RED}❌ Setup script failed after $max_setup_attempts attempts${NC}"
    echo "Last output: $setup_output"
    exit 1
fi

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