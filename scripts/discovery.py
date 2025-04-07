#!/usr/bin/env python3
import asyncio
from playwright.async_api import async_playwright
import argparse
import json
import os
import http3_detector
import time
from urllib.parse import urlparse
import re

# Create base output directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# List of known HTTP/3-supporting websites
HTTP3_WEBSITES = [
    # Cloudflare sites
    "https://www.cloudflare.com",
    "https://blog.cloudflare.com",
    "https://1.1.1.1",
    
    # Google properties
    "https://www.google.com",
    "https://www.youtube.com",
    "https://www.google.com/maps",
    "https://www.google.com/drive",
    "https://www.google.com/docs",
    "https://www.google.com/sheets",
    
    # Facebook family
    "https://www.facebook.com",
    "https://www.instagram.com",
    "https://www.whatsapp.com",
    
    
    # Test sites
    "https://quic.rocks",
    "https://http3-test.com",

    
    # E-commerce
    "https://www.shopify.com",
    "https://www.ebay.com",
    "https://www.amazon.com",
    "https://www.walmart.com",
    "https://www.target.com",
    "https://www.alibaba.com",
    
    # Media
    "https://www.nytimes.com",
    "https://www.theverge.com",
    "https://www.bbc.com",
    "https://www.cnn.com",
    "https://www.netflix.com",
    "https://www.spotify.com",
    "https://www.twitch.tv",
    "https://www.soundcloud.com",
    "https://www.reddit.com",
    
    # Technology
    "https://www.mozilla.org",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.github.com",
    "https://www.stackoverflow.com",
    "https://www.slack.com",
    "https://www.dropbox.com",
    "https://www.zoom.us",
]

# Add this function after extract_domain_name()

def identify_cdn(headers):
    """Identify CDN provider from response headers"""
    cdn_headers = {
        "Cloudflare": ["cf-ray", "server: cloudflare"],
        "Fastly": ["fastly-debug-digest", "x-served-by", "x-fastly"],
        "Akamai": ["x-akamai-transformed", "server: akamai", "x-akamai-request-id"],
        "Cloudfront": ["x-amz-cf-id", "x-amz-cf-pop"],
        "Google": ["x-goog-", "server: gws", "via: gvs"],
        "Verizon/Edgecast": ["server: edgecastcdn", "x-ec-"],
        "Limelight": ["x-limelight-", "server: llnw"],
        "StackPath/MaxCDN": ["x-hw", "server: netdna"],
        "KeyCDN": ["x-cdn:", "server: keycdn"]
    }
    
    # Convert all header names to lowercase for case-insensitive matching
    lowercase_headers = {k.lower(): v for k, v in headers.items()}
    
    # Check for CDN-specific headers
    for cdn, header_patterns in cdn_headers.items():
        for pattern in header_patterns:
            pattern_parts = pattern.lower().split(': ')
            header_name = pattern_parts[0]
            header_value = pattern_parts[1] if len(pattern_parts) > 1 else None
            
            # Check if header exists
            if header_name in lowercase_headers:
                # If we need to match a specific value
                if header_value:
                    if header_value in lowercase_headers[header_name].lower():
                        return cdn
                else:
                    return cdn
    
    # Check for common CDN CNAME patterns in headers
    for header in lowercase_headers.values():
        if isinstance(header, str):
            if ".cloudfront.net" in header:
                return "Cloudfront"
            elif ".akamaiedge.net" in header:
                return "Akamai"
            elif ".fastly.net" in header:
                return "Fastly"
            elif ".cloudflare.net" in header:
                return "Cloudflare"
    
    return "Unknown"

# Extract clean domain name from URL
def extract_domain_name(url):
    # Parse URL to get netloc (domain)
    domain = urlparse(url).netloc.lower()
    
    # Remove www. prefix if present
    domain = re.sub(r'^www\.', '', domain)
    
    # Remove port number if present
    domain = domain.split(':')[0]
    
    # Extract root domain (e.g., 'amazon' from 'amazon.com')
    parts = domain.split('.')
    if len(parts) >= 2:
        # For domains like amazon.com, return 'amazon'
        # For domains like bbc.co.uk, return 'bbc'
        return parts[0]
    return domain

async def discover_website_resources(url, domain_dir, timeout=30):
    """Discover and categorize all resources loaded by a website"""
    # Ensure URL has a proper scheme
    if not url.startswith('http'):
        url = 'https://' + url
        
    resources = []
    navigation_success = False
    performance_metrics = {}
    
    # Create domain-specific directories
    screenshots_dir = os.path.join(domain_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    
    async with async_playwright() as p:
        # Use more robust browser options
        browser_args = ['--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage']
        browser = await p.chromium.launch(
            args=browser_args,
            headless=True,
            timeout=timeout * 1000  # Convert to ms
        )
        
        # Create context with specific options to improve reliability
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
            extra_http_headers={'Accept-Language': 'en-US,en;q=0.9'},
            bypass_csp=True
        )
        
        page = await context.new_page()
        
        # Set navigation timeout
        page.set_default_navigation_timeout(timeout * 1000)
        
        # Track network requests
        async def on_request(request):
            resources.append({
                "url": request.url,
                "resource_type": request.resource_type,
                "method": request.method,
                "size": 0  # Will be updated when response is received
            })
            
        # Track response sizes        
        async def on_response(response):
            for resource in resources:
                if resource["url"] == response.request.url:
                    try:
                        body = await response.body()
                        resource["size"] = len(body)
                        resource["status"] = response.status
                        headers = await response.all_headers()
                        resource["content_type"] = headers.get("content-type", "")
                        resource["cache_control"] = headers.get("cache-control", "")
                        
                        # Identify CDN provider
                        resource["cdn"] = identify_cdn(headers)
                        
                        # Check for HTTP/3 support
                        alt_svc = headers.get("alt-svc", "")
                        if "h3" in alt_svc:
                            resource["supports_http3"] = True
                            
                        # Add caching information
                        resource["cache_control"] = headers.get("cache-control", "")
                        resource["etag"] = headers.get("etag", "")
                        resource["last_modified"] = headers.get("last-modified", "")
                        resource["expires"] = headers.get("expires", "")
                        resource["age"] = headers.get("age", "")
                        
                        # Add Server header which often indicates software/infrastructure
                        resource["server"] = headers.get("server", "")
                    except:
                        pass
                    break
        
        page.on("request", on_request)
        page.on("response", on_response)
        
        # Catch errors but continue processing
        print(f"Navigating to {url}...")
        try:
            start_time = time.time()
            response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            navigation_time = time.time() - start_time
            print(f"Initial page loaded in {navigation_time:.2f} seconds")
            
            # Wait a bit more for additional resources, but with timeout protection
            try:
                print("Waiting for network to become idle...")
                await page.wait_for_load_state("networkidle", timeout=15000)
                navigation_success = True
            except:
                print("Network didn't become fully idle, but continuing with analysis")
                navigation_success = True  # We still got some data
                
            print("Collecting page metrics...")
            performance_metrics = await page.evaluate("""() => {
                const timing = performance.timing;
                return {
                    loadTime: timing.loadEventEnd - timing.navigationStart,
                    domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
                    firstPaint: performance.getEntriesByType('paint')[0]?.startTime || 0
                }
            }""")
            
            # Take a screenshot for reference - saved to domain-specific screenshots directory
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(screenshots_dir, f"screenshot_{timestamp}.png")
            await page.screenshot(path=screenshot_path)
            print(f"Screenshot saved to {screenshot_path}")
            
        except Exception as e:
            print(f"Navigation error: {e}")
            # Try to continue anyway to process any resources that did load
        
        # Even if navigation fails, wait a bit to collect any resources that did load
        await asyncio.sleep(3)
        
        await browser.close()
    
    # Categorize resources
    categorized = {
        "metadata": {
            "url": url,
            "navigation_success": navigation_success,
            "performance": performance_metrics,
            "total_resources": len(resources),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "images": [],
        "scripts": [],
        "stylesheets": [],
        "fonts": [],
        "api_responses": [],
        "html": [],
        "other": []
    }
    
    for resource in resources:
        if resource["resource_type"] == "image":
            categorized["images"].append(resource)
        elif resource["resource_type"] == "script":
            categorized["scripts"].append(resource)
        elif resource["resource_type"] == "stylesheet":
            categorized["stylesheets"].append(resource)
        elif resource["resource_type"] == "font":
            categorized["fonts"].append(resource)
        elif resource["resource_type"] == "fetch" or resource["resource_type"] == "xhr":
            categorized["api_responses"].append(resource)
        elif resource["resource_type"] == "document":
            categorized["html"].append(resource)
        else:
            categorized["other"].append(resource)
    
    return categorized

async def process_single_website(url, timeout=45):
    # Extract clean domain name
    domain_name = extract_domain_name(url)
    print(f"Extracted domain name: {domain_name}")
    
    # Create domain-specific directory
    domain_dir = os.path.join(output_dir, domain_name)
    os.makedirs(domain_dir, exist_ok=True)
    print(f"Created directory: {domain_dir}")
    
    # Check HTTP/3 support using the detector
    if (http3_detector.detect_http3_support(url) == 0):
        print("HTTP/3 is not supported on this URL. Skipping...")
        return False
    
    # Discover resources with timeout
    resources = await discover_website_resources(url, domain_dir, timeout=timeout)
    
    # Generate output filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"resources_{timestamp}.json"
    output_path = os.path.join(domain_dir, output_filename)
    
    # Output summary
    print(f"\n📊 Resource Summary for {resources['metadata']['url']}")
    print(f"Navigation success: {'✅ Yes' if resources['metadata']['navigation_success'] else '❌ No'}")
    print(f"Total resources detected: {resources['metadata']['total_resources']}")
    
    if resources['metadata']['navigation_success']:
        print("\nPerformance metrics:")
        perf = resources['metadata']['performance']
        print(f"  Load time: {perf.get('loadTime', 'N/A')}ms")
        print(f"  DOMContentLoaded: {perf.get('domContentLoaded', 'N/A')}ms")
        print(f"  First paint: {perf.get('firstPaint', 'N/A')}ms")
    
    print("\nResource breakdown:")
    for category, items in resources.items():
        if category != "metadata":  # Skip metadata in the count
            print(f"Found {len(items)} {category}")
            if items:
                sizes = [item["size"] for item in items if item["size"] > 0]
                if sizes:
                    avg_size = sum(sizes) / len(sizes)
                    total_size = sum(sizes)
                    print(f"  Total size: {total_size / (1024 * 1024):.2f} MB")
                    print(f"  Average size: {avg_size / 1024:.1f} KB")
                    print(f"  Examples: {items[0]['url']}")
                    if len(items) > 1:
                        print(f"           {items[1]['url']}")
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump(resources, f, indent=2)
    
    print(f"\nFull resource list saved to {output_path}")
    return True

async def batch_process_websites(websites, timeout=45):
    """Process multiple websites from the predefined list"""
    results = {
        "success": [],
        "failed": []
    }
    
    for url in websites:
        print("\n" + "="*80)
        print(f"Processing website: {url}")
        print("="*80)
        
        try:
            success = await process_single_website(url, timeout)
            if success:
                results["success"].append(url)
            else:
                results["failed"].append(url)
        except Exception as e:
            print(f"Error processing {url}: {e}")
            results["failed"].append(url)
    
    # Print summary
    print("\n\n" + "="*80)
    print(f"BATCH PROCESSING SUMMARY")
    print("="*80)
    print(f"Total websites attempted: {len(websites)}")
    print(f"Successfully processed: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results["failed"]:
        print("\nFailed websites:")
        for url in results["failed"]:
            print(f"  - {url}")
    
    return results

async def main():
    parser = argparse.ArgumentParser(description="Discover resources on websites")
    parser.add_argument("--url", help="Specific URL to analyze (overrides batch mode)")
    parser.add_argument("--batch", "-b", action="store_true", help="Run batch discovery on predefined HTTP/3 websites")
    parser.add_argument("--timeout", "-t", type=int, default=45, help="Navigation timeout in seconds")
    parser.add_argument("--limit", "-l", type=int, help="Limit batch processing to N sites")
    args = parser.parse_args()
    
    if args.batch:
        websites = HTTP3_WEBSITES
        if args.limit and args.limit > 0 and args.limit < len(websites):
            websites = websites[:args.limit]
            
        print(f"Starting batch discovery on {len(websites)} websites...")
        await batch_process_websites(websites, args.timeout)
    elif args.url:
        # Process a single website
        await process_single_website(args.url, args.timeout)
    else:
        print("No URL specified and batch mode not enabled. Use --url or --batch")
        print("\nAvailable options:")
        print("  --url URL       Analyze a specific website")
        print("  --batch, -b     Process all predefined HTTP/3 websites")
        print("  --timeout, -t N Set navigation timeout in seconds (default: 45)")
        print("  --limit, -l N   Process only the first N websites in batch mode")
        
        print("\nPredefined HTTP/3 Websites:")
        for i, url in enumerate(HTTP3_WEBSITES):
            print(f"  {i+1}. {url}")

if __name__ == "__main__":
    asyncio.run(main())