#!/usr/bin/env python3
import asyncio
from playwright.async_api import async_playwright
import time
import csv
import os
from datetime import datetime
import json
import random

csv_dir = "results/"
os.makedirs(csv_dir, exist_ok=True)

# Configure test parameters
TEST_URLS = [
    # Images
    ("/assets/small-image.jpg", "10KB JPEG Image"),
    ("/assets/medium-image.png", "100KB PNG Image"),
    ("/assets/large-image.webp", "1MB WebP Image"),
    
    # JavaScript files
    ("/assets/small-script.js", "5KB JavaScript"),
    ("/assets/large-script.js", "500KB JavaScript"),
    
    # JSON responses
    ("/assets/api/small.json", "2KB JSON Response"),
    ("/assets/api/large.json", "200KB JSON Response"),
    
    # HTML pages
    ("/assets/simple-page.html", "10KB Simple HTML"),
    ("/assets/complex-page.html", "80KB Complex HTML"),
    
    # CSS files
    ("/assets/styles.css", "15KB CSS"),
    ("/assets/framework.css", "60KB CSS")
]

# Test configurations
PROTOCOLS = ["h2", "h3"]  # HTTP/2 and HTTP/3
CACHE_STATES = ["cold", "warm"]  # Cold cache (first visit) and warm cache (subsequent)
ITERATIONS = 200  # Increased for statistical significance
BASE_URLS = {
    "h2": "https://199.94.61.82:8889",  # Your HTTP/2 URL
    "h3": "https://199.94.61.82:8889",  # Your HTTP/3 URL
}

# Network conditions to test
NETWORK_CONDITIONS = [
    {"name": "fast", "latency": 5, "packet_loss": 0.0},  # Fast connection
    {"name": "typical", "latency": 25, "packet_loss": 0.01},  # Typical connection
    {"name": "slow", "latency": 100, "packet_loss": 0.05},  # Challenging connection
    {"name": "very slow", "latency": 300, "packet_loss": 0.1}  # Very Challenging connection
]

# Choose one network condition for this test run
SELECTED_NETWORK = NETWORK_CONDITIONS[1]  # Default to typical

# Results file
RESULTS_FILE = f"{csv_dir}/cache_perf_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{SELECTED_NETWORK['name']}.csv"

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

async def run_test():
    results = []
    
    async with async_playwright() as p:
        for protocol in PROTOCOLS:
            # Keep browser instances for warm cache tests
            warm_browser = None
            warm_context = None
            
            # Session storage path (to store TLS tickets for 0-RTT)
            session_storage_path = os.path.join(csv_dir, f"{protocol}_session_state")
            os.makedirs(session_storage_path, exist_ok=True)
            
            for cache_state in CACHE_STATES:
                for test_path, description in TEST_URLS:
                    url = f"{BASE_URLS[protocol]}{test_path}"
                    
                    for iteration in range(ITERATIONS):
                        print(f"Testing {protocol} - {cache_state} cache - {description} - Iteration {iteration+1}/{ITERATIONS}")
                        
                        # Set up browser args for the protocol
                        browser_args = ["--ignore-certificate-errors"]
                        
                        if protocol == "h3":
                            browser_args.extend([
                                "--enable-quic", 
                                "--quic-version=h3-29",
                                "--enable-features=TLS13EarlyData",  # Enable 0-RTT for TLS 1.3
                                f"--host-resolver-rules=MAP *:443 127.0.0.1:8889"  # Force all HTTPS to use our test server
                            ])
                        elif protocol == "h2":
                            browser_args.extend([
                                "--disable-quic", 
                                "--disable-features=QuicForAll",
                                f"--host-resolver-rules=MAP *:443 127.0.0.1:8889"  # Force all HTTPS to use our test server
                            ])
                        
                        # For cold cache, use a new browser instance each time
                        # For warm cache, reuse the same browser instance
                        if cache_state == "cold":
                            # Always create a new browser for cold cache
                            browser = await p.chromium.launch(args=browser_args)
                            
                            # For first iteration of cold, start fresh
                            if iteration == 0:
                                context = await browser.new_context(
                                    bypass_csp=True,
                                    ignore_https_errors=True
                                )
                            else:
                                # For subsequent cold iterations, still reuse session state
                                # This is to test 0-RTT even on cold cache (session is remembered but HTTP cache is cleared)
                                context = await browser.new_context(
                                    bypass_csp=True,
                                    ignore_https_errors=True,
                                    storage_state=os.path.join(session_storage_path, "storage_state.json")
                                    if os.path.exists(os.path.join(session_storage_path, "storage_state.json")) else None
                                )
                            
                            # Clear cookies to simulate cold HTTP cache
                            await context.clear_cookies()
                        else:
                            # For warm cache, create browser once and reuse
                            if warm_browser is None:
                                warm_browser = await p.chromium.launch(args=browser_args)
                                
                                # Load previous session state if exists
                                warm_context = await warm_browser.new_context(
                                    bypass_csp=True,
                                    ignore_https_errors=True,
                                    storage_state=os.path.join(session_storage_path, "storage_state.json")
                                    if os.path.exists(os.path.join(session_storage_path, "storage_state.json")) else None
                                )
                            browser = warm_browser
                            context = warm_context
                        
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
                        
                        # Navigate to URL
                        response = await page.goto(url, wait_until="networkidle")
                        
                        # Calculate load time
                        load_time = (time.time() - handshake_start_time) * 1000  # in ms
                        
                        # Get detailed information about the connection
                        status = response.status if response else "Error"
                        
                        # Get response headers for cache analysis
                        headers = await response.all_headers() if response else {}
                        cache_control = headers.get("cache-control", "")
                        etag = headers.get("etag", "")
                        
                        # Check for QUIC-specific headers
                        quic_status = headers.get("quic-status", "")
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
                                
                                # Store relevant timing values
                                timing_info = perfEntries
                                
                                # Extract connection setup time specifically
                                connection_time = perfEntries.get('connectionTime', 0)
                                
                                # Track this in our connection statistics
                                if protocol == "h3":
                                    connection_info["h3"]["connection_times_ms"].append(connection_time)
                                else:
                                    connection_info["h2"]["connection_times_ms"].append(connection_time)
                                
                                # Check if it was cached
                                if perfEntries.get('transferSize') == 0:
                                    from_disk_cache = "true"
                            
                            # Look for 0-RTT and TLS session info in headers
                            # Most servers will indicate this in a custom header
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
                                            # High likelihood of 0-RTT if it's a warm cache or subsequent iteration
                                            if protocol == "h3" and (cache_state == "warm" or iteration > 0):
                                                if not zero_rtt_used:  # Don't overwrite if already detected
                                                    early_data_status = "likely_accepted"
                                                    zero_rtt_used = True
                            
                            # Method 3: Check connection timing
                            if protocol == "h3" and connection_time < 10 and (cache_state == "warm" or iteration > 0):
                                # Very fast connection setup suggests 0-RTT
                                tls_resumed = True
                                if not zero_rtt_used:  # Don't overwrite if already detected
                                    early_data_status = "inferred_from_timing"
                                    zero_rtt_used = True
                        
                        except Exception as e:
                            print(f"Error getting performance info: {e}")
                            
                        # Add cache hit/miss information if available
                        from_disk_cache = "false"
                        from_memory_cache = "false"
                        
                        # Try to get cache information through page evaluation
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
                                from_disk_cache = "true"
                                
                            # Also check if the browser indicates caching in the headers
                            if "x-from-cache" in headers or headers.get("cf-cache-status") == "HIT":
                                from_disk_cache = "true"
                                
                            # Check via cache-control and age headers
                            if "age" in headers and int(headers.get("age", "0")) > 0:
                                # Resource was served from a cache somewhere
                                from_disk_cache = "true"
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
                        if iteration == 0:
                            await context.storage_state(path=os.path.join(session_storage_path, "storage_state.json"))
                        
                        # Record results
                        results.append({
                            "protocol": protocol,
                            "cache_state": cache_state,
                            "asset_type": description,
                            "iteration": iteration + 1,
                            "load_time_ms": round(load_time, 2),
                            "connection_time_ms": round(connection_time, 2),
                            "status_code": status,
                            "cache_control": cache_control,
                            "etag": etag,
                            "from_disk_cache": from_disk_cache,
                            "from_memory_cache": from_memory_cache,
                            "connection_reused": str(connection_reused).lower(),
                            "tls_resumed": str(tls_resumed).lower(),
                            "zero_rtt_used": str(zero_rtt_used).lower(),
                            "early_data_status": early_data_status,
                            "quic_status": quic_status,
                            "network_latency": SELECTED_NETWORK['latency'],
                            "network_packet_loss": SELECTED_NETWORK['packet_loss'],
                            "url": url
                        })
                        
                        await page.close()
                        
                        # Only close the browser for cold cache tests
                        if cache_state == "cold":
                            await browser.close()
                
                # Close the warm cache browser when we're done with this cache state
                if cache_state == "warm" and warm_browser:
                    await warm_browser.close()
                    warm_browser = None
    
    # Write results to CSV
    with open(RESULTS_FILE, 'w', newline='') as csvfile:
        fieldnames = [
            "protocol", "cache_state", "asset_type", "iteration", 
            "load_time_ms", "connection_time_ms", "status_code", "cache_control", "etag", 
            "from_disk_cache", "from_memory_cache", "connection_reused", 
            "tls_resumed", "zero_rtt_used", "early_data_status", 
            "quic_status", "network_latency", "network_packet_loss", "url"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Calculate average connection times
    if connection_info["h2"]["connection_times_ms"]:
        connection_info["h2"]["avg_connection_time_ms"] = sum(connection_info["h2"]["connection_times_ms"]) / len(connection_info["h2"]["connection_times_ms"])
    if connection_info["h3"]["connection_times_ms"]:
        connection_info["h3"]["avg_connection_time_ms"] = sum(connection_info["h3"]["connection_times_ms"]) / len(connection_info["h3"]["connection_times_ms"])
    
    # Save connection stats summary
    connection_stats_file = RESULTS_FILE.replace(".csv", "_connection_stats.json")
    with open(connection_stats_file, 'w') as f:
        json.dump(connection_info, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_FILE}")
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test HTTP/2 vs HTTP/3 performance with different cache states")
    parser.add_argument("--network", choices=["fast", "typical", "slow", "very slow"], default="typical",
                        help="Network condition to simulate (default: typical)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of test iterations per configuration (default: 10)")
    
    args = parser.parse_args()
    
    # Set network condition
    for condition in NETWORK_CONDITIONS:
        if condition["name"] == args.network:
            SELECTED_NETWORK = condition
            break
    
    # Set iterations
    ITERATIONS = args.iterations
    
    # Update results file name with network condition
    RESULTS_FILE = f"{csv_dir}/cache_perf_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{SELECTED_NETWORK['name']}.csv"
    
    print(f"Starting tests with {ITERATIONS} iterations per configuration")
    print(f"Network condition: {SELECTED_NETWORK['name']} ({SELECTED_NETWORK['latency']}ms latency, {SELECTED_NETWORK['packet_loss']*100}% packet loss)")
    
    asyncio.run(run_test())