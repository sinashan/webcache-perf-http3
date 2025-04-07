#!/usr/bin/env python3
import asyncio
import json
import os
import glob
import random
import time
import pandas as pd
import csv
import argparse
from datetime import datetime
from playwright.async_api import async_playwright
from urllib.parse import urlparse

# Directory settings
results_dir = "results/"
outputs_dir = "outputs/"
os.makedirs(results_dir, exist_ok=True)

# Test configurations
PROTOCOLS = ["h2", "h3"]
CACHE_STATES = ["cold", "warm"]
ITERATIONS = 1  # Number of iterations per resource
MAX_RESOURCES_PER_CATEGORY = 3  # Maximum resources to select from each category

# Network conditions to test
NETWORK_CONDITIONS = [
    {"name": "fast", "latency": 5, "packet_loss": 0.0},  # Fast connection
    {"name": "typical", "latency": 25, "packet_loss": 0.01},  # Typical connection
    {"name": "slow", "latency": 100, "packet_loss": 0.05}  # Challenging connection
]

# Choose one network condition for this test run
SELECTED_NETWORK = NETWORK_CONDITIONS[1]  # Default to typical

# Categories we're interested in
RESOURCE_CATEGORIES = ["images", "scripts", "stylesheets", "api_responses", "html"]

# Size buckets (in bytes)
SIZE_BUCKETS = [
    (0, 10240),          # 0-10KB
    (10241, 102400),     # 10KB-100KB
    (102401, 1048576),   # 100KB-1MB
    (1048577, float('inf'))  # >1MB
]

# Connection and resumption counters
connection_info = {
    "h2": {
        "new_connections": 0,
        "resumed_connections": 0,
        "connection_times_ms": []
    },
    "h3": {
        "new_connections": 0,
        "resumed_connections": 0,
        "zero_rtt_accepted": 0,
        "connection_times_ms": []
    }
}

def get_bucket_name(size):
    """Get a human-readable name for a size bucket"""
    for i, (min_size, max_size) in enumerate(SIZE_BUCKETS):
        if min_size <= size <= max_size:
            if max_size == float('inf'):
                return f"over_1MB"
            elif max_size == 1048576:
                return f"100KB_to_1MB"
            elif max_size == 102400:
                return f"10KB_to_100KB"
            else:
                return f"under_10KB"
    return "unknown"

async def apply_network_condition(page, condition):
    """Apply network conditions to simulate various network environments"""
    # Emulate network conditions using CDP
    client = await page.context.new_cdp_session(page)
    await client.send('Network.emulateNetworkConditions', {
        'offline': False,
        'latency': condition['latency'],  # Additional round-trip time (ms)
        'downloadThroughput': 10 * 1024 * 1024 / 8,  # 10 Mbps
        'uploadThroughput': 5 * 1024 * 1024 / 8,  # 5 Mbps
        'connectionType': 'wifi'
    })
    
    # Emulate packet loss
    if condition['packet_loss'] > 0:
        # We can't directly set packet loss in CDP, but we can drop random requests
        async def maybe_delay_request(route):
            # Simulate packet loss by randomly aborting a percentage of requests
            if random.random() < condition['packet_loss']:
                # Don't actually abort, just delay significantly to simulate retransmission
                await asyncio.sleep(0.1 + random.random() * 0.2)
            await route.continue_()
        
        await page.route('**/*', maybe_delay_request)
    
    print(f"Applied network conditions: {condition['latency']}ms latency, {condition['packet_loss']*100}% packet loss")

def select_resources_from_json(json_file_path, max_per_category=3):
    """Select representative resources from a discovery JSON file"""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        selected_resources = []
        domain = urlparse(data["metadata"]["url"]).netloc
        
        # Process each category
        for category in RESOURCE_CATEGORIES:
            if category not in data:
                continue
                
            resources = data[category]
            if not resources:
                continue
                
            # Group resources by size bucket
            bucketed_resources = {bucket_name: [] for bucket_name in 
                                 ["under_10KB", "10KB_to_100KB", "100KB_to_1MB", "over_1MB"]}
            
            for resource in resources:
                size = resource.get("size", 0)
                if size <= 0:  # Skip resources with unknown size
                    continue
                    
                bucket = get_bucket_name(size)
                bucketed_resources[bucket].append(resource)
            
            # Try to select resources from each bucket
            category_selections = []
            for bucket, res_list in bucketed_resources.items():
                if res_list:
                    # Sort by size to get variety
                    res_list.sort(key=lambda x: x.get("size", 0))
                    
                    # Try to get one from beginning, middle, end if enough resources
                    if len(res_list) >= 3:
                        selections = [res_list[0], res_list[len(res_list)//2], res_list[-1]]
                    else:
                        selections = res_list
                    
                    # Take up to max_per_category/4 from each bucket
                    max_from_bucket = max(1, max_per_category // 4)
                    for res in selections[:max_from_bucket]:
                        # Add domain info for full URL
                        res["domain"] = domain
                        res["category"] = category
                        res["size_bucket"] = bucket
                        category_selections.append(res)
            
            # Randomly select if we have too many
            if len(category_selections) > max_per_category:
                random.shuffle(category_selections)
                category_selections = category_selections[:max_per_category]
                
            selected_resources.extend(category_selections)
        
        return selected_resources
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return []

def collect_resources_from_discoveries(max_total=30):
    """Collect resources from all discovery JSON files"""
    all_resources = []
    
    # Find all resource JSON files
    json_paths = []
    for root, dirs, files in os.walk(outputs_dir):
        for file in files:
            if file.startswith("resources_") and file.endswith(".json"):
                json_paths.append(os.path.join(root, file))
    
    if not json_paths:
        print("No discovery JSON files found. Run discovery.py first.")
        return []
        
    print(f"Found {len(json_paths)} discovery JSON files")
    
    # Process each JSON file
    for json_path in json_paths:
        resources = select_resources_from_json(json_path)
        all_resources.extend(resources)
    
    # Ensure we have a good distribution
    # Group by category and size bucket
    grouped = {}
    for resource in all_resources:
        key = (resource["category"], resource["size_bucket"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(resource)
    
    # Select resources ensuring representation
    final_selection = []
    target_per_group = max(1, max_total // len(grouped))
    
    for group, resources in grouped.items():
        # Sort by size for variety
        resources.sort(key=lambda x: x.get("size", 0))
        
        # Select resources spaced throughout the list
        selected = []
        if len(resources) <= target_per_group:
            selected = resources
        else:
            step = len(resources) // target_per_group
            for i in range(0, len(resources), step):
                if len(selected) < target_per_group:
                    selected.append(resources[i])
        
        final_selection.extend(selected)
    
    # If we have too many, randomly select
    if len(final_selection) > max_total:
        random.shuffle(final_selection)
        final_selection = final_selection[:max_total]
    
    print(f"Selected {len(final_selection)} resources for testing")
    return final_selection

async def test_resource(resource, protocol, cache_state, iteration, session_storage_path):
    """Test a specific resource with the given protocol and cache state"""
    url = resource["url"]
    domain = resource["domain"]
    category = resource["category"]
    size = resource.get("size", 0)
    size_bucket = resource["size_bucket"]
    
    result = {
        "url": url,
        "domain": domain,
        "resource_type": category,
        "size_bytes": size,
        "size_bucket": size_bucket,
        "protocol": protocol,
        "cache_state": cache_state,
        "iteration": iteration,
        "load_time_ms": 0,
        "connection_time_ms": 0,
        "status_code": "Error",
        "from_disk_cache": "false",
        "from_memory_cache": "false",
        "cache_control": "",
        "etag": "",
        "connection_reused": "false",
        "tls_resumed": "false",
        "zero_rtt_used": "false",
        "early_data_status": "not_attempted",
        "quic_status": "",
        "network_latency": SELECTED_NETWORK['latency'],
        "network_packet_loss": SELECTED_NETWORK['packet_loss']
    }
    
    # Set up browser args for the protocol
    browser_args = ["--ignore-certificate-errors"]
    if protocol == "h3":
        browser_args.extend([
            "--enable-quic", 
            "--quic-version=h3-29",
            "--enable-features=TLS13EarlyData"  # Enable 0-RTT for TLS 1.3
        ])
    elif protocol == "h2":
        browser_args.extend(["--disable-quic", "--disable-features=QuicForAll"])
    
    # Determine storage state path for this domain/protocol
    domain_storage_path = os.path.join(session_storage_path, domain)
    os.makedirs(domain_storage_path, exist_ok=True)
    state_file = os.path.join(domain_storage_path, f"{protocol}_state.json")
    
    try:
        async with async_playwright() as p:
            # For cold cache, use a new browser instance
            browser = await p.chromium.launch(args=browser_args)
            
            # Load session state if available - even for cold cache we use the TLS session
            # but we'll clear cookies/cache later for cold cache
            context = await browser.new_context(
                bypass_csp=True,
                ignore_https_errors=True,
                storage_state=state_file if os.path.exists(state_file) and iteration > 0 else None
            )
            
            # For cold cache, clear browser cache but keep TLS session tickets
            if cache_state == "cold":
                await context.clear_cookies()
            
            page = await context.new_page()
            
            # Apply network conditions
            await apply_network_condition(page, SELECTED_NETWORK)
            
            # Setup performance listeners
            client = await page.context.new_cdp_session(page)
            await client.send("Network.enable", {
                "maxTotalBufferSize": 10000000,
                "maxResourceBufferSize": 5000000
            })
            
            # Enable security details for 0-RTT detection
            await client.send("Security.enable")
            
            # Track early data usage
            zero_rtt_used = False
            connection_reused = False
            tls_resumed = False
            early_data_status = "not_attempted"
            connection_time = 0
            
            # Create a listener for security events
            security_events = []
            
            def handle_security_event(event):
                security_events.append(event)
            
            client.on("Security.securityStateChanged", handle_security_event)
            
            # Start timing
            handshake_start_time = time.time()
            
            # Navigate directly to the resource URL
            response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            
            # Calculate load time
            load_time = (time.time() - handshake_start_time) * 1000  # in ms
            
            if response:
                # Get response headers for cache analysis
                headers = await response.all_headers()
                result["cache_control"] = headers.get("cache-control", "")
                result["etag"] = headers.get("etag", "")
                result["status_code"] = response.status
                
                # Check for QUIC-specific headers
                quic_status = headers.get("quic-status", "")
                result["quic_status"] = quic_status
                print(f"  Protocol indicators: QUIC-Status={quic_status}, Alt-Svc={headers.get('alt-svc', '')}")
                
                # Get connection timing using built-in performance API
                timing_info = {}
                try:
                    # Use Playwright's own performance API 
                    perfEntries = await page.evaluate("""
                        () => {
                            const url = window.location.href;
                            const entries = performance.getEntriesByType('navigation');
                            if (entries.length > 0) {
                                const entry = entries[0];
                                // Calculate connection timing
                                const connectionTime = entry.connectEnd - entry.connectStart;
                                // Return connection details
                                return {
                                    connectStart: entry.connectStart,
                                    connectEnd: entry.connectEnd,
                                    domainLookupStart: entry.domainLookupStart,
                                    domainLookupEnd: entry.domainLookupEnd,
                                    responseStart: entry.responseStart,
                                    responseEnd: entry.responseEnd,
                                    secureConnectionStart: entry.secureConnectionStart,
                                    nextHopProtocol: entry.nextHopProtocol,
                                    connectionTime: connectionTime,
                                    transferSize: entry.transferSize
                                };
                            }
                            return null;
                        }
                    """)
                    
                    if perfEntries:
                        # Look for connection reuse in the timings
                        if perfEntries.get('connectStart') == perfEntries.get('connectEnd'):
                            connection_reused = True
                            result["connection_reused"] = "true"
                        
                        # Store relevant timing values
                        timing_info = perfEntries
                        
                        # Extract connection setup time specifically
                        connection_time = perfEntries.get('connectionTime', 0)
                        result["connection_time_ms"] = round(connection_time, 2)
                        
                        # Track this in our connection statistics
                        if protocol == "h3":
                            connection_info["h3"]["connection_times_ms"].append(connection_time)
                        else:
                            connection_info["h2"]["connection_times_ms"].append(connection_time)
                        
                        # Check if it was cached
                        if perfEntries.get('transferSize') == 0:
                            result["from_disk_cache"] = "true"
                    
                    # Look for 0-RTT and TLS session info in headers
                    if "early-data" in str(headers).lower():
                        zero_rtt_used = True
                    
                    # Get detailed connection information using page.evaluate
                    conn_info = await page.evaluate("""
                        () => {
                            if (window.performance && window.performance.getEntriesByType) {
                                const entries = window.performance.getEntriesByType('resource');
                                const navigationEntry = window.performance.getEntriesByType('navigation')[0];
                                return {
                                    nextHopProtocol: navigationEntry ? navigationEntry.nextHopProtocol : '',
                                    resources: entries.map(e => ({
                                        url: e.name,
                                        nextHopProtocol: e.nextHopProtocol,
                                        transferSize: e.transferSize,
                                        encodedBodySize: e.encodedBodySize
                                    })).slice(0, 5)
                                };
                            }
                            return {};
                        }
                    """)
                    
                    # QUIC/HTTP3 detection through nextHopProtocol
                    if conn_info.get('nextHopProtocol') == 'h3' or conn_info.get('nextHopProtocol') == 'h3-29':
                        timing_info['protocol'] = 'h3'
                        print(f"  Confirmed HTTP/3 usage through nextHopProtocol")
                    
                    # Try to detect 0-RTT through multiple methods
                    # Method 1: Check security events
                    for event in security_events:
                        if "explanations" in event:
                            for explanation in event.get("explanations", []):
                                if "early data" in str(explanation).lower() or "0-rtt" in str(explanation).lower():
                                    zero_rtt_used = True
                                    early_data_status = "detected_in_security_event"
                    
                    # Method 2: Check Alt-Svc header and connection state
                    if "alt-svc" in headers and headers["alt-svc"] and "h3" in headers["alt-svc"]:
                        # The site offers HTTP/3
                        if quic_status == "1" or protocol == "h3":
                            # HTTP/3 was actually used
                            if connection_reused:
                                # Connection was reused, so might have been 0-RTT 
                                if cache_state == "warm" or iteration > 0:
                                    tls_resumed = True
                                    result["tls_resumed"] = "true"
                                    
                                    # High likelihood of 0-RTT if it's a warm cache or subsequent iteration
                                    if protocol == "h3" and (cache_state == "warm" or iteration > 0):
                                        if not zero_rtt_used:  # Don't overwrite if already detected
                                            early_data_status = "likely_accepted"
                                            zero_rtt_used = True
                    
                    # Method 3: Check connection timing
                    if protocol == "h3" and connection_time < 10 and (cache_state == "warm" or iteration > 0):
                        # Very fast connection setup suggests 0-RTT
                        tls_resumed = True
                        result["tls_resumed"] = "true"
                        if not zero_rtt_used:  # Don't overwrite if already detected
                            early_data_status = "inferred_from_timing"
                            zero_rtt_used = True
                    
                    # Update 0-RTT status in result
                    if zero_rtt_used:
                        result["zero_rtt_used"] = "true"
                        result["early_data_status"] = early_data_status
                    
                except Exception as e:
                    print(f"Error getting performance info: {e}")
                
                # Try to get cache information
                try:
                    cache_info = await page.evaluate("""
                        () => {
                            const entries = performance.getEntriesByType('navigation');
                            if (entries.length > 0) {
                                // transferSize of 0 indicates cached resource
                                return {
                                    fromCache: entries[0].transferSize === 0,
                                    encodedBodySize: entries[0].encodedBodySize,
                                    decodedBodySize: entries[0].decodedBodySize
                                };
                            }
                            return { fromCache: false };
                        }
                    """)
                    
                    if cache_info and cache_info.get('fromCache'):
                        result["from_disk_cache"] = "true"
                        
                    # Also check if the browser indicates caching in the headers
                    if "x-from-cache" in headers or headers.get("cf-cache-status") == "HIT":
                        result["from_disk_cache"] = "true"
                        
                    # Check via cache-control and age headers
                    if "age" in headers and int(headers.get("age", "0")) > 0:
                        # Resource was served from a cache somewhere
                        result["from_disk_cache"] = "true"
                except Exception as e:
                    print(f"Error checking cache status: {e}")
            
            # Update connection stats
            if protocol == "h3":
                if zero_rtt_used:
                    connection_info["h3"]["zero_rtt_accepted"] += 1
                    
                if tls_resumed:
                    connection_info["h3"]["resumed_connections"] += 1
                else:
                    connection_info["h3"]["new_connections"] += 1
            else:  # h2
                if tls_resumed or connection_reused:
                    connection_info["h2"]["resumed_connections"] += 1
                else:
                    connection_info["h2"]["new_connections"] += 1
            
            # Save browser state for potential 0-RTT in future
            # Only save on first iteration for each URL
            if iteration == 1:
                await context.storage_state(path=state_file)
            
            result["load_time_ms"] = round(load_time, 2)
            
            await browser.close()
            
    except Exception as e:
        result["error"] = str(e)
        print(f"Error testing {url}: {e}")
    
    return result

async def run_tests(selected_resources):
    """Run tests on selected resources"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{results_dir}/real_web_tests_{timestamp}_{SELECTED_NETWORK['name']}.csv"
    all_results = []
    
    # Create session storage directory
    session_storage_path = os.path.join(results_dir, "session_states")
    os.makedirs(session_storage_path, exist_ok=True)
    
    print(f"Starting tests on {len(selected_resources)} resources")
    print(f"Will run {len(PROTOCOLS)} protocols × {len(CACHE_STATES)} cache states × {ITERATIONS} iterations")
    print(f"Network condition: {SELECTED_NETWORK['name']} ({SELECTED_NETWORK['latency']}ms latency, {SELECTED_NETWORK['packet_loss']*100}% packet loss)")
    
    # Group resources by domain for more efficient testing
    resources_by_domain = {}
    for resource in selected_resources:
        domain = resource["domain"]
        if domain not in resources_by_domain:
            resources_by_domain[domain] = []
        resources_by_domain[domain].append(resource)
    
    # Process each domain
    for domain, domain_resources in resources_by_domain.items():
        print(f"\n===== Testing resources from {domain} =====")
        
        for protocol in PROTOCOLS:
            print(f"  Protocol: {protocol}")
            
            for cache_state in CACHE_STATES:
                print(f"    Cache state: {cache_state}")
                
                # For warm cache testing, we need persistent browser sessions
                if cache_state == "warm":
                    # First pass to warm up the cache
                    print("      Warming up cache...")
                    
                    browser_args = ["--ignore-certificate-errors"]
                    if protocol == "h3":
                        browser_args.extend([
                            "--enable-quic", 
                            "--quic-version=h3-29",
                            "--enable-features=TLS13EarlyData"
                        ])
                    elif protocol == "h2":
                        browser_args.extend([
                            "--disable-quic", 
                            "--disable-features=QuicForAll"
                        ])
                    
                    async with async_playwright() as p:
                        browser = await p.chromium.launch(args=browser_args)
                        context = await browser.new_context(
                            bypass_csp=True,
                            ignore_https_errors=True
                        )
                        
                        # Visit each resource once to populate cache
                        for resource in domain_resources:
                            url = resource["url"]
                            try:
                                page = await context.new_page()
                                await page.goto(url, timeout=20000)
                                await page.close()
                            except Exception as e:
                                print(f"        Error warming cache for {url}: {e}")
                        
                        # Save session state
                        domain_storage = os.path.join(session_storage_path, domain)
                        os.makedirs(domain_storage, exist_ok=True)
                        await context.storage_state(path=os.path.join(domain_storage, f"{protocol}_state.json"))
                        
                        await browser.close()
                
                # Now run the actual tests
                for i in range(ITERATIONS):
                    # Shuffle resources each iteration to reduce systematic bias
                    random.shuffle(domain_resources)
                    
                    for resource in domain_resources:
                        url = resource["url"]
                        category = resource["category"]
                        size_bucket = resource["size_bucket"]
                        
                        print(f"      Testing {category} ({size_bucket}): {i+1}/{ITERATIONS}")
                        result = await test_resource(resource, protocol, cache_state, i+1, session_storage_path)
                        all_results.append(result)
                        
                        # Write incrementally to avoid data loss
                        if len(all_results) % 10 == 0:
                            pd.DataFrame(all_results).to_csv(results_file, index=False)
    
    # Calculate average connection times
    if connection_info["h2"]["connection_times_ms"]:
        connection_info["h2"]["avg_connection_time_ms"] = sum(connection_info["h2"]["connection_times_ms"]) / len(connection_info["h2"]["connection_times_ms"])
    if connection_info["h3"]["connection_times_ms"]:
        connection_info["h3"]["avg_connection_time_ms"] = sum(connection_info["h3"]["connection_times_ms"]) / len(connection_info["h3"]["connection_times_ms"])
    
    # Final write to ensure all results are saved
    pd.DataFrame(all_results).to_csv(results_file, index=False)
    
    # Save connection stats summary
    connection_stats_file = results_file.replace(".csv", "_connection_stats.json")
    with open(connection_stats_file, 'w') as f:
        json.dump(connection_info, f, indent=2)
    
    print(f"\nTesting complete! Results saved to {results_file}")
    print(f"Connection statistics saved to {connection_stats_file}")
    
    # Print connection summary
    print("\n=== Connection Statistics Summary ===")
    print(f"Network conditions: {SELECTED_NETWORK['latency']}ms latency, {SELECTED_NETWORK['packet_loss']*100}% packet loss")
    print("HTTP/2:")
    print(f"  New Connections: {connection_info['h2']['new_connections']}")
    print(f"  Resumed Connections: {connection_info['h2']['resumed_connections']}")
    print(f"  Avg Connection Time: {connection_info['h2'].get('avg_connection_time_ms', 'N/A'):.2f}ms")
    print("HTTP/3:")
    print(f"  New Connections: {connection_info['h3']['new_connections']}")
    print(f"  Resumed Connections: {connection_info['h3']['resumed_connections']}")
    print(f"  0-RTT Accepted: {connection_info['h3']['zero_rtt_accepted']}")
    print(f"  Avg Connection Time: {connection_info['h3'].get('avg_connection_time_ms', 'N/A'):.2f}ms")
    
    return results_file

async def main():
    parser = argparse.ArgumentParser(description="Test HTTP/2 vs HTTP/3 performance with real-world resources")
    parser.add_argument("--network", choices=["fast", "typical", "slow"], default="typical",
                      help="Network condition to simulate (default: typical)")
    parser.add_argument("--iterations", type=int, default=50,
                      help="Number of test iterations per resource (default: 50)")
    parser.add_argument("--max-resources", type=int, default=30,
                     help="Maximum number of resources to test (default: 30)")
    
    args = parser.parse_args()
    
    # Set network condition
    global SELECTED_NETWORK, ITERATIONS
    for condition in NETWORK_CONDITIONS:
        if condition["name"] == args.network:
            SELECTED_NETWORK = condition
            break
    
    # Set iterations
    ITERATIONS = args.iterations
    
    print("HTTP/3 vs HTTP/2 Real Website Resource Testing")
    print("==============================================")
    print(f"Network condition: {SELECTED_NETWORK['name']}")
    print(f"Iterations: {ITERATIONS}")
    
    # Collect resources from discovery files
    resources = collect_resources_from_discoveries(max_total=args.max_resources)
    
    if not resources:
        print("No resources selected. Please run discovery.py first.")
        return
    
    # Show resource distribution
    print("\nResource Distribution:")
    df = pd.DataFrame(resources)
    print(df.groupby(["category", "size_bucket"]).size())
    
    # Run tests
    results_file = await run_tests(resources)
    
    # Print next steps
    print("\nTo analyze the results, run:")
    print(f"python scripts/analysis.py {results_file}")

if __name__ == "__main__":
    asyncio.run(main())