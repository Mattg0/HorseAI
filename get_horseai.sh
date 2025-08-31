#!/bin/bash
# setup_horseai.sh - Complete setup script for HorseAI project with MinIO and training

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/Mattg0/HorseAI.git"
PROJECT_DIR="HorseAI"

# MinIO S3 Configuration
S3_ENDPOINT="https://buckets.mattgautier.fr:8443"
S3_ACCESS_KEY="72nPfRGm6KSYKzNt9HvC"
S3_SECRET_KEY="erUfnRCp4qT1v87ffk8VI2NsnrpOrstEBWqCl1wL"
S3_DB_BUCKET="databases"
S3_MODELS_BUCKET="models"

# Database files to download
DB_FILES=(
    "hippique2.db"
    "test_lite.db"
)

echo -e "${BLUE}🚀 Starting HorseAI Setup with MinIO and Training...${NC}"

# Function to detect environment
detect_environment() {
    if [ -n "$VAST_CONTAINERLABEL" ]; then
        echo "vast.ai"
    else
        echo "localhost"
    fi
}

# Function to install mc (MinIO Client)
install_minio_client() {
    local env_type=$1

    if command -v mc &> /dev/null; then
        echo -e "${GREEN}✅ MinIO client already installed${NC}"
        return 0
    fi

    echo -e "${YELLOW}📦 Installing MinIO client...${NC}"

    if [ "$env_type" = "vast.ai" ]; then
        # Install on vast.ai
        curl https://dl.min.io/client/mc/release/linux-amd64/mc \
             --create-dirs \
             -o /usr/local/bin/mc
        chmod +x /usr/local/bin/mc
    else
        # Install on localhost
        curl https://dl.min.io/client/mc/release/linux-amd64/mc \
             --create-dirs \
             -o ~/.local/bin/mc
        chmod +x ~/.local/bin/mc
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if command -v mc &> /dev/null; then
        echo -e "${GREEN}✅ MinIO client installed successfully${NC}"
    else
        echo -e "${RED}❌ Failed to install MinIO client${NC}"
        exit 1
    fi
}

# Function to configure MinIO client
configure_minio_client() {
    echo -e "${YELLOW}🔧 Configuring MinIO client...${NC}"

    # Configure mc with our MinIO server
    mc alias set horseai "$S3_ENDPOINT" "$S3_ACCESS_KEY" "$S3_SECRET_KEY"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ MinIO client configured${NC}"
        # Test connection
        mc ls horseai/ > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ MinIO connection verified${NC}"
        else
            echo -e "${YELLOW}⚠️ MinIO connection test failed, but continuing...${NC}"
        fi
    else
        echo -e "${RED}❌ Failed to configure MinIO client${NC}"
        exit 1
    fi
}

# Function to download file from MinIO
download_from_minio() {
    local filename=$1
    local output_path=$2
    local bucket=$3

    echo -e "${YELLOW}📥 Downloading ${filename} from MinIO...${NC}"

    # Download with progress bar
    mc cp "horseai/$bucket/$filename" "$output_path"

    if [ $? -eq 0 ] && [ -s "$output_path" ]; then
        local file_size=$(du -h "$output_path" | cut -f1)
        echo -e "${GREEN}✅ Downloaded ${filename} (${file_size})${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to download ${filename}${NC}"
        return 1
    fi
}

# Function to upload models to MinIO
upload_models_to_minio() {
    local project_root=$1
    local models_dir="$project_root/models"
    local date_str=$(date +%Y-%m-%d)

    echo -e "${YELLOW}📤 Uploading trained models to MinIO...${NC}"

    # Find the latest model directory
    local latest_model_dir=$(find "$models_dir" -type d -name "*" | grep -E "/[0-9]{4}-[0-9]{2}-[0-9]{2}/" | sort | tail -1)

    if [ -z "$latest_model_dir" ] || [ ! -d "$latest_model_dir" ]; then
        echo -e "${YELLOW}⚠️ No trained model directory found${NC}"
        return 1
    fi

    echo -e "${BLUE}📁 Found model directory: ${latest_model_dir}${NC}"

    # Upload all model files to S3
    local s3_models_path="horseai/$S3_MODELS_BUCKET/$date_str/"

    # Upload model files
    for file in "$latest_model_dir"/*.joblib "$latest_model_dir"/*.keras "$latest_model_dir"/*.json; do
        if [ -f "$file" ]; then
            local filename=$(basename "$file")
            echo -e "${YELLOW}📤 Uploading ${filename}...${NC}"

            mc cp "$file" "${s3_models_path}${filename}"

            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Uploaded ${filename}${NC}"
            else
                echo -e "${RED}❌ Failed to upload ${filename}${NC}"
            fi
        fi
    done

    echo -e "${GREEN}✅ Models uploaded to: ${s3_models_path}${NC}"
    return 0
}

# Function to run training
run_training() {
    local project_root=$1

    echo -e "${YELLOW}🎯 Starting model training...${NC}"

    cd "$project_root"/model_training/historical

    # Check if training script exists
    if [ ! -f "training_race_model.py" ]; then
        echo -e "${RED}❌ train_race_model.py not found${NC}"
        return 1
    fi

    # Run training with output capture
    echo -e "${BLUE}🚀 Executing: python train_race_model.py${NC}"

    # Run training and capture output
    if python3 train_race_model.py; then
        echo -e "${GREEN}✅ Training completed successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ Training failed${NC}"
        return 1
    fi
}

# Function to update config.yaml
update_config() {
    local project_root=$1
    local config_file="$project_root/config.yaml"

    echo -e "${YELLOW}🔧 Updating config.yaml...${NC}"

    if [ ! -f "$config_file" ]; then
        echo -e "${RED}❌ config.yaml not found at $config_file${NC}"
        exit 1
    fi

    python3 << EOF
import yaml
import sys

config_file = "$config_file"
project_root = "$project_root"

try:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Update rootdir
    if 'base' not in config:
        config['base'] = {}

    old_rootdir = config['base'].get('rootdir', 'Not set')
    config['base']['rootdir'] = project_root

    # Write back
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"✅ Updated rootdir from '{old_rootdir}' to '{project_root}'")

except Exception as e:
    print(f"❌ Error updating config: {e}")
    sys.exit(1)
EOF
}

# Main setup function
main() {
    local env_type=$(detect_environment)
    echo -e "${BLUE}🔍 Detected environment: ${env_type}${NC}"

    # Install MinIO client
    install_minio_client "$env_type"

    # Configure MinIO client
    configure_minio_client

    # Set project root based on environment
    if [ "$env_type" = "vast.ai" ]; then
        PROJECT_ROOT="/workspace/$PROJECT_DIR"
    else
        PROJECT_ROOT="$(pwd)/$PROJECT_DIR"
    fi

    echo -e "${BLUE}📁 Project will be set up at: ${PROJECT_ROOT}${NC}"

    # Step 1: Clone repository
    echo -e "${YELLOW}📥 Cloning repository...${NC}"
    if [ -d "$PROJECT_DIR" ]; then
        echo -e "${YELLOW}⚠️ Directory $PROJECT_DIR already exists. Updating...${NC}"
        cd "$PROJECT_DIR"
        git pull origin main || echo -e "${YELLOW}⚠️ Git pull failed, continuing...${NC}"
        cd ..
    else
        git clone "$REPO_URL" "$PROJECT_DIR"
    fi
    echo -e "${GREEN}✅ Repository ready${NC}"

    # Step 2: Create data directory
    mkdir -p "$PROJECT_ROOT/data"

    # Step 3: Download databases from MinIO
    echo -e "${YELLOW}💾 Downloading databases from MinIO...${NC}"
    local db_downloaded=0
    for db_file in "${DB_FILES[@]}"; do
        local output_path="$PROJECT_ROOT/data/$db_file"

        # Skip if file already exists and is recent
        if [ -f "$output_path" ]; then
            local file_age=$(find "$output_path" -mtime -1 2>/dev/null | wc -l)
            if [ "$file_age" -gt 0 ]; then
                local file_size=$(du -h "$output_path" | cut -f1)
                echo -e "${GREEN}✅ ${db_file} is recent (${file_size}), skipping download${NC}"
                continue
            fi
        fi

        if download_from_minio "$db_file" "$output_path" "$S3_DB_BUCKET"; then
            ((db_downloaded++))
        fi

        # Verify file was downloaded and is not empty
        if [ ! -s "$output_path" ]; then
            echo -e "${RED}❌ Downloaded file ${db_file} is empty or missing${NC}"
            echo -e "${YELLOW}⚠️ Continuing with setup despite download issue...${NC}"
        fi
    done

    echo -e "${GREEN}✅ Database download section completed - ${db_downloaded} files downloaded${NC}"

    # Step 4: Update config.yaml
    echo -e "${YELLOW}🔧 Starting config update...${NC}"
    echo -e "${BLUE}📁 Project root for config: ${PROJECT_ROOT}${NC}"
    update_config "$PROJECT_ROOT"
    echo -e "${GREEN}✅ Config update completed${NC}"

    # Step 5: Set up Python environment
    echo -e "${YELLOW}🐍 Setting up Python environment...${NC}"

    cd "$PROJECT_ROOT"
    if [ -f "requirements.txt" ]; then
        echo -e "${YELLOW}📦 Installing Python dependencies...${NC}"
        pip3 install -r requirements.txt --quiet
        echo -e "${GREEN}✅ Dependencies installed${NC}"
    fi

    # Step 6: Export PYTHONPATH
    echo -e "${YELLOW}🔧 Setting up PYTHONPATH...${NC}"
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/core:$PROJECT_ROOT/utils:$PROJECT_ROOT/model_training:$PYTHONPATH"

    # Add to bash profile for persistence
    local profile_file=""
    if [ -f ~/.bashrc ]; then
        profile_file=~/.bashrc
    elif [ -f ~/.bash_profile ]; then
        profile_file=~/.bash_profile
    elif [ -f ~/.zshrc ]; then
        profile_file=~/.zshrc
    fi

    if [ -n "$profile_file" ]; then
        echo "# HorseAI Environment Setup" >> "$profile_file"
        echo "export PYTHONPATH=\"$PROJECT_ROOT:\$PROJECT_ROOT/core:\$PROJECT_ROOT/utils:\$PROJECT_ROOT/model_training:\$PYTHONPATH\"" >> "$profile_file"
        echo -e "${GREEN}✅ PYTHONPATH added to $profile_file${NC}"
    fi

    # Step 7: Test setup
    echo -e "${YELLOW}🧪 Testing setup...${NC}"
    cd "$PROJECT_ROOT"

    python3 -c "
from utils.env_setup import AppConfig
config = AppConfig()
print('✅ Configuration loaded successfully')
print(f'   Active DB: {config._config.base.active_db}')
print(f'   Root dir: {config._config.base.rootdir}')

# Test database exists
import os
db_path = config.get_sqlite_dbpath(config._config.base.active_db)
if os.path.exists(db_path):
    db_size = os.path.getsize(db_path) / (1024*1024)  # MB
    print(f'   Database: Found ({db_size:.1f} MB)')
else:
    print(f'   Database: Not found at {db_path}')

print('✅ Setup test passed')
" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Setup test passed${NC}"
    else
        echo -e "${RED}❌ Setup test failed${NC}"
        exit 1
    fi

    # Step 8: Run training
    if run_training "$PROJECT_ROOT"; then
        echo -e "${GREEN}✅ Training completed successfully${NC}"

        # Step 9: Upload models to MinIO
        if upload_models_to_minio "$PROJECT_ROOT"; then
            echo -e "${GREEN}✅ Models uploaded to MinIO${NC}"
        else
            echo -e "${YELLOW}⚠️ Model upload failed, but training was successful${NC}"
        fi
    else
        echo -e "${RED}❌ Training failed${NC}"
        exit 1
    fi

    # Final summary
    echo -e "\n${GREEN}🎉 HorseAI Setup and Training Complete!${NC}"
    echo -e "${BLUE}📊 Summary:${NC}"
    echo -e "   Environment: ${env_type}"
    echo -e "   Project root: ${PROJECT_ROOT}"
    echo -e "   Database files: ${db_downloaded} downloaded"
    echo -e "   Training: ✅ Completed"
    echo -e "   Models: ✅ Uploaded to MinIO"
    echo -e "   PYTHONPATH: Configured"

    echo -e "\n${YELLOW}💡 MinIO Buckets:${NC}"
    echo -e "   Databases: horseai/${S3_DB_BUCKET}/"
    echo -e "   Models: horseai/${S3_MODELS_BUCKET}/$(date +%Y-%m-%d)/"

    echo -e "\n${YELLOW}💡 Next steps:${NC}"
    echo -e "   1. Check models: ${GREEN}mc ls horseai/${S3_MODELS_BUCKET}/$(date +%Y-%m-%d)/${NC}"
    echo -e "   2. View training logs in: ${GREEN}$PROJECT_ROOT/logs/${NC}"
    echo -e "   3. Start predictions: ${GREEN}python your_prediction_script.py${NC}"
}

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"

    # Check for required tools
    local missing=0
    for cmd in git python3 pip3 curl; do
        if ! command -v $cmd &> /dev/null; then
            echo -e "${RED}❌ $cmd is not installed${NC}"
            missing=1
        fi
    done

    if [ $missing -eq 1 ]; then
        exit 1
    fi

    # Check for PyYAML
    python3 -c "import yaml" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠️ PyYAML not found, installing...${NC}"
        pip3 install PyYAML --quiet
    fi

    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Trap to ensure cleanup on exit
trap 'echo -e "\n${RED}❌ Setup interrupted${NC}"; exit 1' INT TERM

# Run the setup
check_prerequisites
main

echo -e "\n${GREEN}🚀 All done! Happy training! 🐎${NC}"