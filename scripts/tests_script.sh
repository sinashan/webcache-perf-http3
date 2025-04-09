#!/bin/bash

# Automatically detect base directory (parent of the script directory)
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPTS_DIR="$(dirname "$SCRIPT_PATH")"
BASE_DIR="$(dirname "$SCRIPTS_DIR")"

# Set script paths
DISCOVERY_SCRIPT="$SCRIPTS_DIR/discovery.py"
REAL_TESTS_SCRIPT="$SCRIPTS_DIR/run_real_tests.py"
ANALYSIS_SCRIPT="$SCRIPTS_DIR/analysis.py"

# Set test parameters
ITERATIONS=50  # Number of iterations per resource
MAX_RESOURCES=15  # Resources to test per network condition
NETWORK_CONDITIONS=("fast" "typical" "slow")

# Colors for better output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Create log directory
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/test_suite_${TIMESTAMP}.log"

# Set up trap to handle Ctrl+C and other signals
cleanup() {
    echo -e "\n${RED}Script interrupted! Cleaning up and exiting...${NC}"
    # Kill any python processes started by this script
    pkill -P $$ python3
    exit 1
}

# Trap signals
trap cleanup SIGINT SIGTERM

# Log the detected directories for verification
echo -e "${BLUE}[INFO] Script location: $SCRIPT_PATH${NC}"
echo -e "${BLUE}[INFO] Scripts directory: $SCRIPTS_DIR${NC}" 
echo -e "${BLUE}[INFO] Base directory: $BASE_DIR${NC}"

# Helper function for logging
log() {
    echo -e "${2:-$BLUE}[$(date +"%Y-%m-%d %H:%M:%S")] $1${NC}" | tee -a "$LOG_FILE"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check required dependencies
check_dependencies() {
    log "Checking dependencies..." "$GREEN"
    
    if ! command_exists python3; then
        log "Python 3 is required but not installed. Please install Python 3." "$RED"
        exit 1
    fi
    
    if ! command_exists playwright; then
        log "Playwright not found. Installing Playwright..." "$YELLOW"
        pip install playwright
        playwright install chromium
    fi
    
    log "All dependencies satisfied." "$GREEN"
}

# Run discovery on all websites
run_discovery() {
    log "Starting discovery on HTTP/3 websites..." "$GREEN"
    log "This will discover resources from websites and identify HTTP/3 support." "$BLUE"
    log "Press Ctrl+C at any time to stop the discovery process." "$YELLOW"
    
    # Run Python script directly with timeout protection
    python3 "$DISCOVERY_SCRIPT" --batch &
    DISCOVERY_PID=$!
    
    # Wait for the discovery to complete
    wait $DISCOVERY_PID
    DISCOVERY_EXIT_CODE=$?
    
    if [ $DISCOVERY_EXIT_CODE -ne 0 ]; then
        log "Discovery failed or was interrupted. Check the logs for details." "$RED"
        log "Continuing with real tests on already discovered resources..." "$YELLOW"
        # Ask if user wants to continue
        read -p "Continue with testing using available resources? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Exiting script as requested." "$RED"
            exit 1
        fi
    else
        log "Discovery completed successfully!" "$GREEN"
    fi
}

# Run real tests with specific network condition
run_real_tests() {
    local network=$1
    log "Starting real-world tests with $network network conditions..." "$GREEN"
    log "Running $ITERATIONS iterations on up to $MAX_RESOURCES resources" "$BLUE"
    
    # Run the test with specified parameters
    python3 "$REAL_TESTS_SCRIPT" \
        --network "$network" \
        --iterations "$ITERATIONS" \
        --max-resources "$MAX_RESOURCES"
    
    if [ $? -ne 0 ]; then
        log "Tests for $network network conditions failed. Check the logs for details." "$RED"
        return 1
    else
        log "Tests for $network network conditions completed successfully!" "$GREEN"
        return 0
    fi
}

# Analyze results across network conditions
analyze_results() {
    log "Analyzing results across all network conditions..." "$GREEN"
    
    # Run the network analysis for cross-condition comparison
    python3 "$ANALYSIS_SCRIPT" --network-analysis
    python3 "$ANALYSIS_SCRIPT" --cache-analysis
    
    if [ $? -ne 0 ]; then
        log "Warning: Cross-network analysis failed. Continuing with individual analyses..." "$YELLOW"
    else
        log "Cross-network analysis completed successfully!" "$GREEN"
    fi
    
    # Now analyze each individual CSV file for more detailed per-network insights
    log "Analyzing individual network condition results..." "$GREEN"
    
    # For each network condition directory
    for network in "${NETWORK_CONDITIONS[@]}"; do
        network_dir="$BASE_DIR/results/${network// /_}"
        
        if [ -d "$network_dir" ]; then
            log "Processing $network results..." "$BLUE"
            
            # Find the most recent CSV file that's not a stats file
            csv_files=$(find "$network_dir" -name "*.csv" | grep -v "_stats\|_connection_stats\|_optimizations" | sort -r)
            
            if [ -z "$csv_files" ]; then
                log "No CSV result files found in $network_dir" "$YELLOW"
                continue
            fi
            
            latest_csv=$(echo "$csv_files" | head -n 1)
            log "Analyzing: $(basename "$latest_csv")" "$BLUE"
            
            # Run the analysis script on this file
            python3 "$ANALYSIS_SCRIPT" "$latest_csv"
            
            if [ $? -ne 0 ]; then
                log "Analysis failed for $network condition. Check the logs for details." "$RED"
            else
                log "Analysis for $network condition completed successfully!" "$GREEN"
            fi
        else
            log "Directory for $network condition not found: $network_dir" "$YELLOW"
        fi
    done
    
    log "All analysis tasks completed!" "$GREEN"
    log "Analysis results saved in the results directory" "$GREEN"
    return 0
}


main() {
    log "Starting HTTP/3 vs HTTP/2 Performance Test Suite (Automated)" "$GREEN"
    log "==========================================================" "$GREEN"
    
    # Check dependencies
    check_dependencies
    
    # Change to base directory
    cd "$BASE_DIR"
    
    # Step 1: Run discovery
    run_discovery
    
    # Step 2: Run tests for each network condition
    for network in "${NETWORK_CONDITIONS[@]}"; do
        run_real_tests "$network"
    done
    
    # Step 3: Analyze results across network conditions
    analyze_results
    
    log "Automated test suite execution completed!" "$GREEN"
    log "Check results in the $BASE_DIR/results directory" "$GREEN"
    log "Logs saved to $LOG_FILE" "$BLUE"
}

# Run the main function
main

exit 0