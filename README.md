# HTTP/3 and 0-RTT Performance Measurement Tool

This repository contains tools for measuring and analyzing the performance benefits of HTTP/3 with 0-RTT resumption compared to HTTP/2, specifically focusing on cache validation and resource loading performance across different network conditions.

## Overview

The measurement framework evaluates how HTTP/3's 0-RTT resumption feature impacts resource loading performance compared to traditional HTTP/2 connections. It simulates various network conditions and cache states to provide a comprehensive analysis of protocol performance.

## Features

- Compare HTTP/2 vs HTTP/3 loading performance
- Measure 0-RTT resumption benefits on HTTP/3
- Simulate various network conditions (low/high latency, packet loss)
- Test across cold and warm browser cache states
- Generate detailed performance analytics and visualizations
- Analyze real-world websites or controlled test assets

## Repository Structure

```
measure/
├── scripts/
│   ├── run_local_tests.py      # Run tests with controlled local assets
│   ├── run_real_tests.py       # Run tests against real-world websites
│   ├── discovery.py            # Discover resources on websites
│   ├── generate_test_files.py  # Generate test assets of various types/sizes
│   └── analysis.py             # Generate charts from test results
├── http3_detector/             # Library to detect HTTP/3 support
├── nginx/                      # NGINX configuration for local testing
│   └── tests/
│       ├── https.conf          # NGINX HTTP/2 & HTTP/3 configuration
│       └── assets/             # Generated test assets
└── results/                    # Test results and generated charts
```

## Test Approaches

### Local Testing

Local testing uses controlled, generated assets of different types and sizes to provide a standardized environment for comparing protocol performance. This approach eliminates variables like external server configurations and network paths.

```bash
# Generate test assets of different types and sizes
python scripts/generate_test_files.py

# Run local tests with default settings (typical network, 10 iterations)
python scripts/run_local_tests.py

# Run with specific network condition and more iterations
python scripts/run_local_tests.py --network slow --iterations 50
```

### Real-World Testing

Real-world testing evaluates protocol performance against actual websites with their diverse content types, server configurations, and CDNs. This approach provides insights into real-world protocol behavior.

```bash
# Discover resources on websites with HTTP/3 support
python scripts/discovery.py --batch

# Test a single website
python scripts/discovery.py --url https://www.cloudflare.com

# Run performance tests on discovered resources
python scripts/run_real_tests.py --network typical --iterations 20
```

## Network Conditions

You can simulate different network conditions:

- `fast`: 5ms latency, 0% packet loss
- `typical`: 25ms latency, 1% packet loss
- `slow`: 100ms latency, 5% packet loss

## Analyzing Results

After tests complete, you can generate detailed visualizations:

```bash
python scripts/analysis.py results/cache_perf_results_20250405_214839_typical.csv
```

This will generate several PDF charts in the results directory:

- `*_cold_cache.pdf` - HTTP/2 vs HTTP/3 performance with cold cache
- `*_warm_cache.pdf` - HTTP/2 vs HTTP/3 performance with warm cache
- `*_improvement.pdf` - HTTP/3 performance improvement over HTTP/2 (%)
- `*_cache_benefit.pdf` - Cache performance benefit by protocol
- `*_0rtt_comparison.pdf` - HTTP/3 with vs without 0-RTT
- `*_0rtt_benefit.pdf` - Performance benefit of 0-RTT in HTTP/3
- `*_connection_types.pdf` - Comparison of different connection resumption types
- `*_optimizations.pdf` - Combined effect of protocol, 0-RTT and browser cache

## Requirements

- Python 3.8+
- Playwright for Python
- NGINX with HTTP/3 support
- SSL certificates for HTTPS
- Matplotlib, Seaborn, Pandas for analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/measure.git
cd measure

# Install Python dependencies
pip install playwright pandas matplotlib seaborn numpy

# Install browser engines
playwright install chromium

# Generate SSL certificates (required for HTTPS)
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/localhost.key -out nginx/ssl/localhost.crt
```

## Running NGINX with HTTP/3 Support

```bash
# Build custom NGINX with HTTP/3
docker build -t nginx-http3 .

# Run NGINX with mounted test assets
docker run -d --name nginx-http3-test \
  -p 8889:443/tcp -p 8889:443/udp \
  -v $(pwd)/nginx/tests/https.conf:/etc/nginx/conf.d/https.conf \
  -v $(pwd)/nginx/ssl:/etc/nginx/ssl \
  -v $(pwd)/nginx/tests/assets:/static/assets \
  nginx-http3
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
