from flask import Flask, render_template, request, jsonify, send_file
from wall_segmentation import apply_wallpaper
import os
from dotenv import load_dotenv
import openai
import base64
import requests
import json
import traceback

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("api_key")

# Load API keys
SHOP_NAME = os.getenv("SHOPIFY_STORE_NAME")
ACCESS_TOKEN = os.getenv("SHOPIFY_ADMIN_API_KEY")

def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def generate_ai_description(wall_image_path, wallpaper_image_path, width="10", height="8"):
    """Generate AI description using OpenAI Vision API with fixed prompt structure"""
    try:
        wall_b64 = encode_image(wall_image_path)
        wallpaper_b64 = encode_image(wallpaper_image_path)

        # Build fixed prompt with proper tiling information
        user_prompt = (
            f"Generate a realistic mockup where the selected wallpaper pattern (1ft x 1ft tiles) is accurately tiled "
            f"and applied only on the wall surface as per the dimensions {width}ft x {height}ft. "
            f"The wallpaper should be repeated in a grid pattern to cover the entire wall area. "
            f"Each wallpaper tile should maintain its 1ft x 1ft proportions and repeat seamlessly. "
            f"The wallpaper should follow the perspective, scale, and alignment of the wall to look natural. "
            f"Do not apply wallpaper on foreground elements such as furniture, windows, doors, picture frames, or decor. "
            f"Preserve the room's original lighting, shadows, and depth so that the wallpaper blends in seamlessly. "
            f"The final result should help customers clearly visualize how this specific wallpaper pattern would look "
            f"when properly tiled across their wall surface."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{wall_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{wallpaper_b64}"}}
                ]
            }
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating AI description: {str(e)}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/list-wallpapers", methods=["GET"])
def list_wallpapers():
    """List wallpapers from Shopify with pagination and error handling."""
    try:
        limit = min(int(request.args.get('limit', 50)), 250)  # Max 250 per Shopify API
        
        url = f"https://{SHOP_NAME}.myshopify.com/admin/api/2023-04/products.json?limit={limit}"
        headers = {"X-Shopify-Access-Token": ACCESS_TOKEN}
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        products = data.get("products", [])
        
        # Extract product info including all images and variants
        wallpaper_data = []
        for product in products:
            try:
                # Get ALL image URLs if available
                images = []
                for image in product.get("images", []):
                    try:
                        image_info = {
                            "id": image.get("id", ""),
                            "src": image.get("src", ""),
                            "alt": image.get("alt", ""),
                            "position": image.get("position", 0),
                            "variant_ids": image.get("variant_ids", [])  # Shows which variants use this image
                        }
                        images.append(image_info)
                    except Exception as image_error:
                        print(f"Error processing image {image.get('id', 'unknown')}: {image_error}")
                        continue
                
                # Process variants with error handling for missing keys
                variants = []
                for variant in product.get("variants", []):
                    try:
                        # Find the variant's specific image if it has one
                        variant_image_url = None
                        variant_image_id = variant.get("image_id")
                        if variant_image_id:
                            # Find the image that matches this variant
                            for img in images:
                                if img["id"] == variant_image_id:
                                    variant_image_url = img["src"]
                                    break
                        
                        variant_info = {
                            "title": variant.get("title", "Default"),
                            "id": variant.get("id", ""),
                            "available": variant.get("available", True),  # Default to True if missing
                            "price": variant.get("price", "0.00"),
                            "inventory_quantity": variant.get("inventory_quantity", 0),
                            "image_url": variant_image_url,  # Specific image for this variant
                            "image_id": variant.get("image_id")  # Reference to the image
                        }
                        variants.append(variant_info)
                    except Exception as variant_error:
                        print(f"Error processing variant {variant.get('id', 'unknown')}: {variant_error}")
                        # Skip problematic variants but continue processing
                        continue
                
                product_info = {
                    "title": product.get("title", "Untitled Product"),
                    "id": product.get("id", ""),
                    "images": images,  # All images for the product
                    "main_image_url": images[0]["src"] if images else None,  # First image as main
                    "handle": product.get("handle", ""),
                    "product_type": product.get("product_type", ""),
                    "created_at": product.get("created_at", ""),
                    "variants": variants
                }
                wallpaper_data.append(product_info)
                
            except Exception as product_error:
                print(f"Error processing product {product.get('id', 'unknown')}: {product_error}")
                # Skip problematic products but continue processing
                continue
        
        return jsonify({
            "success": True,
            "count": len(wallpaper_data),
            "wallpapers": wallpaper_data
        })
        
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to fetch wallpapers: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Error processing wallpapers: {str(e)}"}), 500


@app.route("/debug-shopify", methods=["GET"])
def debug_shopify():
    """Debug endpoint to see raw Shopify data structure"""
    try:
        url = f"https://{SHOP_NAME}.myshopify.com/admin/api/2023-04/products.json?limit=1"
        headers = {"X-Shopify-Access-Token": ACCESS_TOKEN}
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Return raw data for inspection
        return jsonify({
            "success": True,
            "raw_data": data,
            "sample_product": data.get("products", [{}])[0] if data.get("products") else None
        })
        
    except Exception as e:
        return jsonify({"error": f"Debug failed: {str(e)}"}), 500

@app.route('/get-wallpaper-image', methods=['POST'])
def get_wallpaper_image():
    """Download wallpaper image from Shopify and save locally"""
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        
        if not image_url:
            return jsonify({"error": "No image URL provided"}), 400
        
        # Download the image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Save the image locally
        wallpaper_path = os.path.join(app.config['UPLOAD_FOLDER'], 'selected_wallpaper.jpg')
        with open(wallpaper_path, 'wb') as f:
            f.write(response.content)
        
        return jsonify({"success": True, "local_path": wallpaper_path})
        
    except Exception as e:
        return jsonify({"error": f"Failed to download wallpaper: {str(e)}"}), 500


@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload and wallpaper application with proper sizing"""
    try:
        print("=== UPLOAD ENDPOINT CALLED ===")
        print(f"Request method: {request.method}")
        print(f"Request files: {list(request.files.keys())}")
        print(f"Request form: {dict(request.form)}")
        
        # Get the wall image (always required)
        if 'wall' not in request.files:
            print("ERROR: No wall image in request.files")
            return jsonify({"error": "No wall image provided"}), 400
        
        wall_img = request.files['wall']
        if wall_img.filename == '':
            print("ERROR: Wall image filename is empty")
            return jsonify({"error": "No wall image selected"}), 400
        
        print(f"Wall image filename: {wall_img.filename}")
        
        # Get wall dimensions from form data with better error handling
        try:
            wall_width_ft = float(request.form.get('wall_width', 10))  # Default 10 feet
            wall_height_ft = float(request.form.get('wall_height', 8))  # Default 8 feet
            print(f"Wall dimensions: {wall_width_ft}ft x {wall_height_ft}ft")
        except (ValueError, TypeError) as e:
            print(f"ERROR: Invalid wall dimensions - {e}")
            return jsonify({"error": "Invalid wall dimensions provided"}), 400
        
        # Save wall image
        wall_path = os.path.join(app.config['UPLOAD_FOLDER'], 'wall.jpg')
        try:
            wall_img.save(wall_path)
            print(f"Wall image saved to: {wall_path}")
        except Exception as e:
            print(f"ERROR: Failed to save wall image - {e}")
            return jsonify({"error": f"Failed to save wall image: {str(e)}"}), 500
        
        # Determine wallpaper source
        wallpaper_path = None
        
        # Option 1: Upload custom wallpaper
        if 'wallpaper' in request.files and request.files['wallpaper'].filename != '':
            print("Using uploaded custom wallpaper")
            wallpaper_img = request.files['wallpaper']
            wallpaper_path = os.path.join(app.config['UPLOAD_FOLDER'], 'wallpaper.jpg')
            try:
                wallpaper_img.save(wallpaper_path)
                print(f"Custom wallpaper saved to: {wallpaper_path}")
            except Exception as e:
                print(f"ERROR: Failed to save custom wallpaper - {e}")
                return jsonify({"error": f"Failed to save wallpaper: {str(e)}"}), 500
        
        # Option 2: Use Shopify wallpaper (already downloaded via /get-wallpaper-image)
        elif os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'selected_wallpaper.jpg')):
            print("Using Shopify selected wallpaper")
            wallpaper_path = os.path.join(app.config['UPLOAD_FOLDER'], 'selected_wallpaper.jpg')
        
        else:
            print("ERROR: No wallpaper source found")
            return jsonify({"error": "No wallpaper source provided"}), 400
        
        # Verify wallpaper file exists and is readable
        if not os.path.exists(wallpaper_path):
            print(f"ERROR: Wallpaper file does not exist: {wallpaper_path}")
            return jsonify({"error": "Wallpaper file not found"}), 400
        
        print(f"Using wallpaper: {wallpaper_path}")
        
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
        print(f"Output will be saved to: {output_path}")
        
        # Apply wallpaper overlay with proper sizing (1x1 feet tiles)
        print("Calling apply_wallpaper function...")
        try:
            apply_wallpaper(wall_path, wallpaper_path, output_path, 
                           wall_width_ft=wall_width_ft, wall_height_ft=wall_height_ft,
                           wallpaper_width_ft=2.0, wallpaper_height_ft=2.0)
            print("apply_wallpaper completed successfully")
        except Exception as e:
            print(f"ERROR in apply_wallpaper: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")
            return jsonify({"error": f"Wallpaper processing failed: {str(e)}"}), 500
        
        # Verify output file was created
        if not os.path.exists(output_path):
            print(f"ERROR: Output file was not created: {output_path}")
            return jsonify({"error": "Output file was not created"}), 500
        
        print(f"Sending file: {output_path}")
        return send_file(output_path, mimetype='image/jpeg')
        
    except Exception as e:
        print(f"UNEXPECTED ERROR in upload endpoint: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route('/generate-description', methods=['POST'])
def generate_description():
    """Generate AI description based on fixed pattern with proper tiling"""
    try:
        data = request.get_json()
        width = str(data.get('width', '10'))
        height = str(data.get('height', '8'))

        wall_path = os.path.join(app.config['UPLOAD_FOLDER'], 'wall.jpg')

        # Determine wallpaper path
        wallpaper_path = None
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'wallpaper.jpg')):
            wallpaper_path = os.path.join(app.config['UPLOAD_FOLDER'], 'wallpaper.jpg')
        elif os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'selected_wallpaper.jpg')):
            wallpaper_path = os.path.join(app.config['UPLOAD_FOLDER'], 'selected_wallpaper.jpg')

        if not os.path.exists(wall_path) or not wallpaper_path or not os.path.exists(wallpaper_path):
            return jsonify({"error": "Required images not found. Please upload images first."}), 400

        description = generate_ai_description(wall_path, wallpaper_path, width, height)

        return jsonify({
            "success": True,
            "description": description
        })

    except Exception as e:
        return jsonify({"error": f"Failed to generate description: {str(e)}"}), 500


# Add a debug route to check file system
@app.route('/debug-files', methods=['GET'])
def debug_files():
    """Debug endpoint to check what files exist in the upload folder"""
    try:
        files = []
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                files.append({
                    "name": file,
                    "path": file_path,
                    "exists": os.path.exists(file_path),
                    "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
                })
        
        return jsonify({
            "upload_folder": app.config['UPLOAD_FOLDER'],
            "upload_folder_exists": os.path.exists(app.config['UPLOAD_FOLDER']),
            "files": files
        })
    except Exception as e:
        return jsonify({"error": f"Debug failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
