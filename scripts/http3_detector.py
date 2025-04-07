import requests
import dns.resolver
import sys

def check_http3_via_alt_svc(url):
    """Check for HTTP/3 support using Alt-Svc header."""
    try:
        response = requests.get(url, timeout=10)
        alt_svc = response.headers.get("Alt-Svc")
        if alt_svc and "h3" in alt_svc:
            print(f"✅ [Alt-Svc] HTTP/3 support detected on {url}!")
            print(f"    Alt-Svc header: {alt_svc}")
            return True
        else:
            print(f"❌ [Alt-Svc] HTTP/3 not detected via Alt-Svc on {url}.")
            return False
    except requests.RequestException as e:
        print(f"Error checking Alt-Svc for {url}: {e}")
        return False

def check_http3_via_dns_https_record(hostname):
    """Check for HTTP/3 support using DNS HTTPS records."""
    try:
        answers = dns.resolver.resolve(hostname, "HTTPS")
        for answer in answers:
            if "h3" in str(answer):
                print(f"✅ [DNS HTTPS] HTTP/3 support detected via DNS on {hostname}!")
                print(f"    DNS record: {answer}")
                return True
        print(f"❌ [DNS HTTPS] HTTP/3 not detected via DNS on {hostname}.")
        return False
    except dns.resolver.NoAnswer:
        print(f"❌ [DNS HTTPS] No HTTPS DNS record found for {hostname}.")
        return False
    except Exception as e:
        print(f"Error checking DNS HTTPS record for {hostname}: {e}")
        return False

def detect_http3_support(url):
    # Extract hostname from URL
    hostname = url.replace("https://", "").replace("http://", "").split("/")[0]

    print(f"\n🔍 Checking HTTP/3 support for {url}...")
    
    # Step 1: Check Alt-Svc header
    alt_svc_supported = check_http3_via_alt_svc(url)
    
    # Step 2: Check DNS HTTPS record
    dns_https_supported = check_http3_via_dns_https_record(hostname)

    # Summary
    if alt_svc_supported and dns_https_supported:
        print(f"\n✅ HTTP/3 is supported on {url}!")
        return 1
    elif alt_svc_supported != dns_https_supported:
        print(f"\n✅ HTTP/3 is likely supported on {url}!")
        return 1
    else:
        print(f"\n❌ HTTP/3 does not appear to be supported on {url}.")
        return 0


# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python http3_detector.py <URL>")
        sys.exit(1)
    
    url = sys.argv[1]
    detect_http3_support(url)