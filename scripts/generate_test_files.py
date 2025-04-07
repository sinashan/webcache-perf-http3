#!/usr/bin/env python3
# filepath: /home/ubuntu/internet measurement project/scripts/generate_assets.py
import os
import json
import random
import string
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Configuration
OUTPUT_DIR = "/home/ubuntu/measure/nginx/tests/assets"  # Update this path as needed
API_DIR = os.path.join(OUTPUT_DIR, "api")

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(API_DIR, exist_ok=True)

print(f"Generating test assets in {OUTPUT_DIR}")

# Add these to your generate_all_assets() function:

# Generate HTML files
def generate_html_file(path, size_kb, complexity='simple'):
    """Generate an HTML file of specified size and complexity"""
    if complexity == 'simple':
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Test Page</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <h1>HTTP Caching Test Page</h1>
    </header>
    <main>
        <p>This is a simple test page for HTTP caching evaluation.</p>
        <div class="content">
"""
    else:
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Complex Test Page</title>
    <link rel="stylesheet" href="styles.css">
    <link rel="stylesheet" href="framework.css">
    <script src="small-script.js" defer></script>
</head>
<body>
    <header>
        <nav>
            <ul>
                <li><a href="#">Home</a></li>
                <li><a href="#">About</a></li>
                <li><a href="#">Services</a></li>
                <li><a href="#">Contact</a></li>
            </ul>
        </nav>
        <h1>HTTP Caching Evaluation Platform</h1>
    </header>
    <main>
        <section class="hero">
            <h2>Testing Web Performance</h2>
            <p>This page contains various elements to test HTTP caching behavior.</p>
        </section>
        <section class="cards">
"""
        
        # Add cards for complex page
        for i in range(20):
            html_content += f"""
            <article class="card">
                <h3>Item {i+1}</h3>
                <p>{random_string(200)}</p>
                <a href="#">Learn more</a>
            </article>"""
            
    # Fill to target size
    while len(html_content.encode('utf-8')) < size_kb * 1024:
        html_content += f"\n        <p>{random_string(100)}</p>"
    
    if complexity == 'simple':
        html_content += """
        </div>
    </main>
    <footer>
        <p>&copy; 2025 HTTP Cache Testing</p>
    </footer>
</body>
</html>"""
    else:
        html_content += """
        </section>
    </main>
    <footer>
        <div class="footer-content">
            <div class="footer-section">
                <h3>About</h3>
                <p>This is a test page for HTTP caching evaluation.</p>
            </div>
            <div class="footer-section">
                <h3>Contact</h3>
                <p>Email: test@example.com</p>
            </div>
            <div class="footer-section">
                <h3>Legal</h3>
                <p>Terms & Conditions</p>
            </div>
        </div>
        <p class="copyright">&copy; 2025 HTTP Cache Testing Platform</p>
    </footer>
    <script src="large-script.js" defer></script>
</body>
</html>"""
    
    with open(path, 'w') as f:
        f.write(html_content)
        
    print(f"Generated HTML file at {path} ({os.path.getsize(path) / 1024:.2f} KB)")

def generate_css_file(path, size_kb):
    """Generate a CSS file of specified size"""
    css_content = """/* Generated CSS for HTTP caching tests */
body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

header {
    background-color: #f4f4f4;
    padding: 1rem;
    margin-bottom: 1rem;
}

h1 {
    color: #2c3e50;
}

.content {
    display: flex;
    flex-wrap: wrap;
}

/* More styles below to reach target size */
"""
    
    # CSS class templates
    css_templates = [
        ".class-{} {{ color: #{:06x}; background-color: #{:06x}; padding: {}px; margin: {}px; }}",
        ".item-{} {{ display: flex; flex-direction: {}; align-items: {}; justify-content: {}; }}",
        "@media (min-width: {}px) {{ .responsive-{} {{ width: {}%; height: {}px; }} }}",
        "#element-{} {{ position: absolute; top: {}px; left: {}px; z-index: {}; transform: rotate({}deg); }}",
        ".grid-{} {{ display: grid; grid-template-columns: repeat({}, 1fr); grid-gap: {}px; }}"
    ]
    
    # Add styles until we reach the target size
    counter = 0
    while len(css_content.encode('utf-8')) < size_kb * 1024:
        counter += 1
        template = random.choice(css_templates)
        
        if "{:06x}" in template:
            # Color template
            css_content += template.format(
                counter,
                random.randint(0, 0xFFFFFF),
                random.randint(0, 0xFFFFFF),
                random.randint(5, 30),
                random.randint(5, 30)
            ) + "\n\n"
        elif "flex-direction" in template:
            # Flex template
            css_content += template.format(
                counter,
                random.choice(['row', 'column', 'row-reverse', 'column-reverse']),
                random.choice(['flex-start', 'center', 'flex-end', 'stretch']),
                random.choice(['flex-start', 'center', 'flex-end', 'space-between', 'space-around'])
            ) + "\n\n"
        elif "@media" in template:
            # Media query template
            css_content += template.format(
                random.randint(600, 1200),
                counter,
                random.randint(20, 100),
                random.randint(100, 600)
            ) + "\n\n"
        elif "position: absolute" in template:
            # Position template
            css_content += template.format(
                counter,
                random.randint(0, 500),
                random.randint(0, 500),
                random.randint(1, 10),
                random.randint(0, 360)
            ) + "\n\n"
        else:
            # Grid template
            css_content += template.format(
                counter,
                random.randint(2, 12),
                random.randint(5, 20)
            ) + "\n\n"
    
    with open(path, 'w') as f:
        f.write(css_content)
    
    print(f"Generated CSS file at {path} ({os.path.getsize(path) / 1024:.2f} KB)")


# Helper functions
def random_string(length):
    """Generate a random string of fixed length"""
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for i in range(length))

def generate_random_text(size_kb):
    """Generate random text of specified size in KB"""
    chars_per_kb = 1024
    return ''.join(random.choice(string.ascii_letters + string.digits + ' \n\t') 
                  for _ in range(chars_per_kb * size_kb))

def generate_js_file(path, size_kb):
    """Generate a JavaScript file of specified size in KB"""
    # Start with basic JS structure
    js_content = """
// Generated JavaScript file for HTTP/3 vs HTTP/2 caching tests
const testFunction = () => {
    console.log('This is a test JavaScript file');
    
    // Some variables and functions
    const items = [];
    const processItems = () => {
        return items.map(item => item * 2);
    };
    
    // Random data below to reach desired file size
    const randomData = [
"""
    
    # Add random data until we reach the desired size
    while len(js_content.encode('utf-8')) < size_kb * 1024:
        js_content += f'        "{random_string(20)}",\n'
    
    js_content += """
    ];
    
    return randomData;
};

// Export for potential use
if (typeof module !== 'undefined') {
    module.exports = { testFunction };
}
"""
    
    with open(path, 'w') as f:
        f.write(js_content)
    
    # Adjust file size if needed
    current_size = os.path.getsize(path)
    target_size = size_kb * 1024
    
    if current_size < target_size:
        with open(path, 'a') as f:
            f.write('\n\n/* ' + 'x' * (target_size - current_size - 8) + ' */\n')

def generate_json_file(path, size_kb):
    """Generate a JSON file of approximate size in KB"""
    data = {
        "timestamp": "2025-04-05T12:00:00Z",
        "version": "1.0.0",
        "status": "success",
        "items": []
    }
    
    # Add items until we reach desired size
    item_template = {
        "id": 0,
        "name": "",
        "description": "",
        "attributes": {
            "color": "",
            "size": "",
            "price": 0,
            "tags": [],
            "metadata": {
                "created": "2025-03-15T10:30:00Z",
                "updated": "2025-04-01T14:22:00Z",
                "hash": ""
            }
        }
    }
    
    counter = 0
    while True:
        counter += 1
        item = item_template.copy()
        item["id"] = counter
        item["name"] = f"Item {counter}"
        item["description"] = random_string(100)
        item["attributes"]["color"] = random.choice(["red", "blue", "green", "yellow"])
        item["attributes"]["size"] = random.choice(["small", "medium", "large"])
        item["attributes"]["price"] = round(random.uniform(10, 1000), 2)
        item["attributes"]["tags"] = [random_string(8) for _ in range(random.randint(3, 10))]
        item["attributes"]["metadata"]["hash"] = random_string(32)
        
        data["items"].append(item)
        
        # Check file size
        if counter % 10 == 0:
            with open(path, 'w') as f:
                json.dump(data, f)
            if os.path.getsize(path) >= size_kb * 1024:
                break
    
    print(f"Generated JSON file at {path} ({os.path.getsize(path) / 1024:.2f} KB)")

def generate_image(path, size_kb, format_type='JPEG'):
    """Generate an image file of specified size in KB"""
    # Estimate dimensions based on desired file size and format
    if format_type == 'JPEG':
        # JPEG is more compressed
        width = int(np.sqrt(size_kb * 1024 * 0.5))
        height = width
    elif format_type == 'PNG':
        # PNG requires larger dimensions for same file size
        width = int(np.sqrt(size_kb * 1024 * 0.2))
        height = width
    else:  # WebP
        width = int(np.sqrt(size_kb * 1024 * 0.4))
        height = width
    
    # Create a colorful image
    img = Image.new('RGB', (width, height), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    
    # Add some random shapes to make the image more complex
    for i in range(20):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        
        # Ensure x1 <= x2 and y1 <= y2 as required by PIL
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        d.rectangle([x1, y1, x2, y2], fill=fill_color)
    
    # Add text with file size info
    d.text((10, 10), f"{format_type} Image - Target: {size_kb}KB", fill=(255, 255, 255))
    
    # Save with appropriate format
    if format_type == 'JPEG':
        img.save(path, format='JPEG', quality=85)
    elif format_type == 'PNG':
        img.save(path, format='PNG')
    else:  # WebP
        img.save(path, format='WebP', quality=85)
    
    # Check size and adjust if needed
    current_size = os.path.getsize(path) / 1024
    
    # Adjust quality if size is not close enough
    attempts = 0
    while abs(current_size - size_kb) / size_kb > 0.1 and attempts < 5:
        if format_type == 'JPEG' or format_type == 'WebP':
            quality = int(85 * size_kb / current_size)
            quality = max(10, min(quality, 100))
            if format_type == 'JPEG':
                img.save(path, format='JPEG', quality=quality)
            else:
                img.save(path, format='WebP', quality=quality)
        else:  # PNG - resize the image
            new_width = int(width * np.sqrt(size_kb / current_size))
            new_height = int(height * np.sqrt(size_kb / current_size))
            resized = img.resize((new_width, new_height))
            resized.save(path, format='PNG')
        
        current_size = os.path.getsize(path) / 1024
        attempts += 1
    
    print(f"Generated {format_type} image at {path} ({os.path.getsize(path) / 1024:.2f} KB)")


# Generate all test assets
def generate_all_assets():
    """Generate all assets needed for the HTTP cache tests"""
    # Generate images
    generate_image(os.path.join(OUTPUT_DIR, 'small-image.jpg'), 10, 'JPEG')
    generate_image(os.path.join(OUTPUT_DIR, 'medium-image.png'), 100, 'PNG')
    generate_image(os.path.join(OUTPUT_DIR, 'large-image.webp'), 1024, 'WebP')
    
    # Generate JavaScript files
    generate_js_file(os.path.join(OUTPUT_DIR, 'small-script.js'), 5)
    generate_js_file(os.path.join(OUTPUT_DIR, 'large-script.js'), 500)
    
    # Generate JSON files
    generate_json_file(os.path.join(API_DIR, 'small.json'), 2)
    generate_json_file(os.path.join(API_DIR, 'large.json'), 200)

    generate_html_file(os.path.join(OUTPUT_DIR, 'simple-page.html'), 10, 'simple')
    generate_html_file(os.path.join(OUTPUT_DIR, 'complex-page.html'), 80, 'complex')
    
    generate_css_file(os.path.join(OUTPUT_DIR, 'styles.css'), 15)
    generate_css_file(os.path.join(OUTPUT_DIR, 'framework.css'), 60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        OUTPUT_DIR = sys.argv[1]
        API_DIR = os.path.join(OUTPUT_DIR, "api")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(API_DIR, exist_ok=True)
    
    generate_all_assets()
    print("All test assets generated successfully!")