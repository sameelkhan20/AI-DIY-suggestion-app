from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import base64
from werkzeug.utils import secure_filename
import openai
from datetime import datetime
import uuid
from PIL import Image, ImageOps

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-this-in-production')

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# OpenAI API Key - From environment variable (required for Hugging Face Spaces)
openai.api_key = os.getenv('OPENAI_API_KEY', '')
if not openai.api_key:
    print("⚠️ Warning: OPENAI_API_KEY environment variable not set. Please set it in Hugging Face Spaces secrets.")
else:
    print("✅ OpenAI API Key configured successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path, max_size=(1024, 1024), quality=85):
    """Process and optimize image for better analysis"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Auto-orient based on EXIF data
            img = ImageOps.exif_transpose(img)
            
            # Resize if too large while maintaining aspect ratio
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Create processed directory if it doesn't exist
            processed_dir = 'static/processed'
            os.makedirs(processed_dir, exist_ok=True)
            
            # Generate processed filename
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            processed_filename = f"processed_{uuid.uuid4().hex}.jpg"
            processed_path = os.path.join(processed_dir, processed_filename)
            
            # Save processed image
            img.save(processed_path, 'JPEG', quality=quality, optimize=True)
            
            return processed_path
            
    except Exception as e:
        print(f"Image processing error: {e}")
        return image_path  # Return original if processing fails

def get_image_info(image_path):
    """Get basic image information"""
    try:
        with Image.open(image_path) as img:
            return {
                'size': img.size,
                'mode': img.mode,
                'format': img.format,
                'file_size': os.path.getsize(image_path)
            }
    except Exception as e:
        print(f"Error getting image info: {e}")
        return None



def analyze_image_with_ai(image_path):
    """Analyze image using OpenAI Vision API with enhanced techniques"""
    try:
        print(f"Analyzing image: {image_path}")
        
        # Check if API key is available
        if not openai.api_key:
            raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
        
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Encode image to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        print(f"Image encoded, size: {len(base64_image)} characters")
        
        # Enhanced prompt for better analysis
        analysis_prompt = """You are an expert at identifying objects in images for creative reuse and upcycling.

Look at this image carefully and provide a detailed analysis:

1. **Object Identification**: What is the main object/item in this image? Be very specific (e.g., "wooden dining chair", "metal toolbox", "glass bottle").

2. **Material Analysis**: What materials is it made of? (wood, metal, plastic, glass, fabric, ceramic, leather, etc.)

3. **Condition Assessment**: What is the condition? (new, used, old, damaged, broken, vintage, antique, worn, etc.)

4. **Size Estimation**: What size is it approximately? (small, medium, large, or specific dimensions if visible)

5. **Style/Design**: What style or design? (modern, traditional, vintage, rustic, industrial, etc.)

6. **Potential Value**: Is it valuable, rare, collectible, or common?

7. **Creative Potential**: What makes this item suitable for creative reuse or DIY projects?

Be very accurate and descriptive. If you see multiple items, focus on the main/primary item. Don't guess - only describe what you can clearly see."""
        
        # Call OpenAI Vision API with enhanced settings
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": analysis_prompt
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"  # High detail for better analysis
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,  # Increased tokens for detailed analysis
            temperature=0.1,  # Low temperature for more consistent results
        )
        
        ai_response = response.choices[0].message.content
        print(f"AI Response: {ai_response}")
        return ai_response
        
    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        if "quota" in str(e).lower() or "insufficient_quota" in str(e).lower():
            print("⚠️ OpenAI quota exceeded. Using intelligent fallback analysis.")
            return None  # Return None to trigger fallback analysis
        else:
            raise Exception(f"AI analysis failed: {str(e)}")
    except FileNotFoundError as e:
        print(f"File Error: {e}")
        raise Exception(f"Image file not found: {str(e)}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
        raise Exception(f"Analysis failed: {str(e)}")

def generate_recommendations_with_ai(ai_analysis):
    """Generate recommendations using OpenAI based on AI analysis - NO STATIC FALLBACK"""
    try:
        if not openai.api_key:
            print("⚠️ No API key, returning empty recommendations")
            return get_empty_recommendations()
        
        if not ai_analysis or len(ai_analysis) < 50:
            print("⚠️ Analysis too short, returning empty recommendations")
            return get_empty_recommendations()
        
        # Create prompt for recommendations
        recommendations_prompt = f"""Based on this detailed image analysis, generate SPECIFIC and ACTIONABLE recommendations for this exact item:

{ai_analysis}

IMPORTANT: Provide EXACTLY 6 recommendations for each section. Format your response EXACTLY as shown below:

### DIY Creative Ideas
1. [First specific creative reuse idea for this exact item]
2. [Second specific creative reuse idea]
3. [Third specific creative reuse idea]
4. [Fourth specific creative reuse idea]
5. [Fifth specific creative reuse idea]
6. [Sixth specific creative reuse idea]

### Monetization Opportunities
1. [First monetization method specific to this item]
2. [Second monetization method]
3. [Third monetization method]
4. [Fourth monetization method]
5. [Fifth monetization method]
6. [Sixth monetization method]

### Sustainability Benefits
1. [First environmental benefit of reusing this specific item]
2. [Second environmental benefit]
3. [Third environmental benefit]
4. [Fourth environmental benefit]
5. [Fifth environmental benefit]
6. [Sixth environmental benefit]

### Helpful Tutorials
1. [First tutorial topic relevant to this item]
2. [Second tutorial topic]
3. [Third tutorial topic]
4. [Fourth tutorial topic]
5. [Fifth tutorial topic]
6. [Sixth tutorial topic]

### Marketplace Suggestions
1. [First marketplace with reason why it's good for this item]
2. [Second marketplace with reason]
3. [Third marketplace with reason]
4. [Fourth marketplace with reason]
5. [Fifth marketplace with reason]
6. [Sixth marketplace with reason]

CRITICAL: 
- Be VERY specific to this exact item based on the analysis
- Make recommendations practical, actionable, and realistic
- Use numbered lists (1., 2., 3., etc.) exactly as shown
- Each recommendation should be 1-2 sentences maximum
- Focus on the actual item described in the analysis"""
        
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": recommendations_prompt
                }
            ],
            max_tokens=2000,
            temperature=0.8,  # Slightly higher for more creative recommendations
        )
        
        recommendations_text = response.choices[0].message.content
        
        # Debug: Print AI response for troubleshooting
        print(f"\n{'='*60}")
        print(f"🤖 AI Recommendations Response:")
        print(f"{'='*60}")
        print(recommendations_text)  # Print full response
        print(f"{'='*60}\n")
        
        # Parse the AI response
        recommendations = parse_recommendations_from_text(recommendations_text)
        
        # Debug: Check what we got
        print(f"\n📋 Final Recommendations Summary:")
        for key, items in recommendations.items():
            print(f"  {key}: {len(items)} items")
            if items:
                print(f"    First item: {items[0][:60]}...")
        
        # If we got recommendations, return them
        if any(recommendations.values()):
            print("✅ Successfully parsed AI-generated recommendations!")
            return recommendations
        else:
            print("⚠️ Warning: No recommendations parsed, but returning empty (not using static fallback)")
            return recommendations
        
    except Exception as e:
        print(f"❌ AI recommendations generation failed: {e}")
        print("⚠️ Returning empty recommendations (no static fallback)")
        return get_empty_recommendations()

def parse_recommendations_from_text(text):
    """Parse recommendations from AI-generated text"""
    recommendations = {
        'diy_ideas': [],
        'monetization': [],
        'sustainability': [],
        'tutorials': [],
        'marketplace_suggestions': []
    }
    
    if not text:
        print("⚠️ Empty text received for parsing")
        return get_empty_recommendations()
    
    print(f"\n🔍 Parsing AI recommendations text (length: {len(text)} chars)")
    print(f"First 200 chars: {text[:200]}...")
    
    lines = text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        line_lower = line.lower()
            
        # Detect section headers (more flexible matching)
        # Remove markdown headers (#) for matching
        clean_line = line_lower.replace('#', '').strip()
        
        if ('diy' in clean_line and 'creative' in clean_line) or ('diy' in clean_line and 'idea' in clean_line):
            current_section = 'diy_ideas'
            print(f"📌 Found DIY section: {line}")
            continue
        elif 'monetization' in clean_line or 'monetisation' in clean_line or ('monet' in clean_line and 'opportunit' in clean_line):
            current_section = 'monetization'
            print(f"📌 Found Monetization section: {line}")
            continue
        elif 'sustainability' in clean_line or ('sustain' in clean_line and 'benefit' in clean_line):
            current_section = 'sustainability'
            print(f"📌 Found Sustainability section: {line}")
            continue
        elif ('tutorial' in clean_line or 'helpful' in clean_line) and ('tutorial' in clean_line or 'guide' in clean_line):
            current_section = 'tutorials'
            print(f"📌 Found Tutorials section: {line}")
            continue
        elif 'marketplace' in clean_line or ('market' in clean_line and 'suggest' in clean_line):
            current_section = 'marketplace_suggestions'
            print(f"📌 Found Marketplace section: {line}")
            continue
        
        # Parse list items (more flexible patterns)
        if current_section:
            # Check for numbered lists (1., 2., etc.)
            if len(line) > 0 and line[0].isdigit() and ('.' in line[:3] or ')' in line[:3]):
                item = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                if item and len(recommendations[current_section]) < 6:
                    recommendations[current_section].append(item)
                    print(f"✅ Added to {current_section}: {item[:50]}...")
            # Check for bullet points (-, *, •)
            elif line.startswith(('-', '*', '•', '→', '▶')):
                item = line.lstrip('-*•→▶ ').strip()
                if item and len(recommendations[current_section]) < 6:
                    recommendations[current_section].append(item)
                    print(f"✅ Added to {current_section}: {item[:50]}...")
            # Check for plain text lines after section header (if no numbering)
            elif len(recommendations[current_section]) < 6 and len(line) > 10:
                # Only add if it looks like a recommendation (not a header)
                if not any(word in line_lower for word in ['###', '##', '#', 'section', 'category']):
                    recommendations[current_section].append(line)
                    print(f"✅ Added to {current_section}: {line[:50]}...")
    
    # Log what we parsed
    print(f"📊 Parsed recommendations:")
    for key, items in recommendations.items():
        print(f"  {key}: {len(items)} items")
    
    # Ensure we have at least some recommendations
    if not any(recommendations.values()):
        print("⚠️ No recommendations parsed from AI response, returning empty")
        return get_empty_recommendations()
    
    # Log what we got
    for key in recommendations:
        if recommendations[key] and len(recommendations[key]) > 0:
            print(f"✅ {key}: {len(recommendations[key])} AI-generated recommendations")
        else:
            print(f"⚠️ {key}: Empty (no AI recommendations found)")
    
    return recommendations

def extract_recommendations_from_analysis(analysis_text):
    """Extract recommendations from analysis text if AI recommendations failed"""
    recommendations = {
        'diy_ideas': [],
        'monetization': [],
        'sustainability': [],
        'tutorials': [],
        'marketplace_suggestions': []
    }
    
    if not analysis_text:
        return get_empty_recommendations()
    
    analysis_lower = analysis_text.lower()
    
    # Extract DIY ideas from analysis
    if 'diy' in analysis_lower or 'creative' in analysis_lower or 'transform' in analysis_lower:
        # Look for DIY ideas patterns
        lines = analysis_text.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if any(word in line_lower for word in ['transform', 'convert', 'create', 'make', 'repurpose']) and len(line_lower) > 20:
                if line_lower.startswith(('-', '*', '•', '1.', '2.', '3.')):
                    clean_line = line.lstrip('-*•1234567890. ').strip()
                    if clean_line and len(recommendations['diy_ideas']) < 6:
                        recommendations['diy_ideas'].append(clean_line)
                elif not line_lower.startswith('**') and len(line_lower) > 15:
                    if len(recommendations['diy_ideas']) < 6:
                        recommendations['diy_ideas'].append(line.strip())
    
    # Extract monetization from analysis
    if 'monetization' in analysis_lower or 'sell' in analysis_lower or 'marketplace' in analysis_lower:
        lines = analysis_text.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if any(word in line_lower for word in ['sell', 'marketplace', 'ebay', 'facebook', 'etsy', 'craigslist']) and len(line_lower) > 15:
                if not line_lower.startswith('**') and 'monetization' not in line_lower:
                    if len(recommendations['monetization']) < 6:
                        recommendations['monetization'].append(line.strip())
    
    # Generate sustainability benefits based on analysis
    if 'reuse' in analysis_lower or 'sustain' in analysis_lower or 'environment' in analysis_lower:
        recommendations['sustainability'] = [
            "Reduces waste by reusing existing item",
            "Decreases demand for new manufacturing",
            "Supports circular economy principles",
            "Reduces carbon footprint",
            "Promotes sustainable consumption",
            "Helps conserve natural resources"
        ]
    else:
        recommendations['sustainability'] = [
            "Reduces waste in landfills",
            "Decreases manufacturing demand",
            "Supports circular economy",
            "Reduces carbon footprint",
            "Promotes sustainable living",
            "Conserves natural resources"
        ]
    
    # Generate tutorials based on material in analysis
    if 'wood' in analysis_lower or 'wooden' in analysis_lower:
        recommendations['tutorials'] = [
            "Wood sanding and refinishing techniques",
            "Wood staining and finishing methods",
            "Basic wood repair techniques",
            "Wood painting and decoration ideas",
            "Wood assembly basics",
            "Wood safety and tool handling"
        ]
    elif 'metal' in analysis_lower or 'steel' in analysis_lower:
        recommendations['tutorials'] = [
            "Metal cleaning and rust removal",
            "Metal painting and finishing techniques",
            "Basic metal repair methods",
            "Metal cutting and shaping basics",
            "Metal welding techniques",
            "Metal safety and tool handling"
        ]
    elif 'glass' in analysis_lower:
        recommendations['tutorials'] = [
            "Glass cleaning and maintenance",
            "Glass cutting and shaping techniques",
            "Glass painting and decoration methods",
            "Glass safety and handling basics",
            "Glass repair and restoration",
            "Glass crafting and DIY projects"
        ]
    else:
        recommendations['tutorials'] = [
            "Basic cleaning techniques",
            "Safe handling methods",
            "Restoration techniques",
            "Creative painting ideas",
            "Assembly basics",
            "Safety guidelines"
        ]
    
    # Generate marketplace suggestions
    recommendations['marketplace_suggestions'] = [
        "Facebook Marketplace - Best for local sales",
        "eBay - Wide audience reach",
        "Craigslist - Quick local transactions",
        "Etsy - Creative and handmade items",
        "OfferUp - Mobile-friendly marketplace",
        "Local thrift stores and consignment shops"
    ]
    
    # If we still don't have DIY ideas, generate from analysis keywords
    if not recommendations['diy_ideas']:
        if any(word in analysis_lower for word in ['chair', 'stool', 'bench']):
            recommendations['diy_ideas'] = [
                "Transform into a garden planter by adding soil and plants",
                "Create unique wall art by painting and hanging",
                "Convert into storage seating with hinged seat",
                "Make a pet bed with cushions",
                "Create a coat rack by adding hooks",
                "Transform into a side table"
            ]
        elif any(word in analysis_lower for word in ['bottle', 'glass', 'jar']):
            recommendations['diy_ideas'] = [
                "Transform bottles into decorative vases",
                "Create candle holders from glass containers",
                "Make terrariums from jars",
                "Create storage containers",
                "Transform into hanging planters",
                "Make decorative lamps"
            ]
        else:
            recommendations['diy_ideas'] = [
                "Transform into a decorative piece",
                "Create a storage solution",
                "Make it into a garden planter",
                "Convert into wall art",
                "Repurpose for pet use",
                "Create a unique display item"
            ]
    
    # If we still don't have monetization, generate from analysis
    if not recommendations['monetization']:
        if any(word in analysis_lower for word in ['vintage', 'antique', 'rare']):
            recommendations['monetization'] = [
                "Sell to antique dealers or collectors",
                "List on specialized vintage marketplaces",
                "Try auction houses for valuable items",
                "Post on Etsy for vintage items",
                "Consider consignment shops",
                "Sell to museums or collectors"
            ]
        else:
            recommendations['monetization'] = [
                "Sell on Facebook Marketplace",
                "List on eBay",
                "Post on Craigslist",
                "Try local thrift stores",
                "Consider consignment shops",
                "Rent out for events"
            ]
    
    print(f"📊 Extracted recommendations from analysis:")
    for key, items in recommendations.items():
        print(f"  {key}: {len(items)} items")
    
    return recommendations

def get_empty_recommendations():
    """Return empty recommendations structure - no static fallback"""
    return {
        'diy_ideas': [],
        'monetization': [],
        'sustainability': [],
        'tutorials': [],
        'marketplace_suggestions': []
    }

def generate_recommendations(ai_analysis):
    """Generate intelligent recommendations based on AI analysis (Fallback static method)"""
    recommendations = {
        'diy_ideas': [],
        'monetization': [],
        'sustainability': [],
        'tutorials': [],
        'marketplace_suggestions': []
    }
    
    if not ai_analysis:
        # Default recommendations if no analysis
        recommendations = {
            'diy_ideas': [
                "Transform into a decorative piece",
                "Create a storage solution",
                "Make it into a garden planter",
                "Convert into wall art",
                "Repurpose for pet use",
                "Create a unique display item"
            ],
            'monetization': [
                "Sell on Facebook Marketplace",
                "List on eBay",
                "Post on Craigslist",
                "Try local thrift stores",
                "Consider consignment shops",
                "Rent out for events"
            ],
            'sustainability': [
                "Reduces waste in landfills",
                "Decreases manufacturing demand",
                "Supports circular economy",
                "Reduces carbon footprint",
                "Promotes sustainable living",
                "Conserves natural resources"
            ],
            'tutorials': [
                "Basic cleaning techniques",
                "Safe handling methods",
                "Restoration techniques",
                "Creative painting ideas",
                "Assembly basics",
                "Safety guidelines"
            ],
            'marketplace_suggestions': [
                "Facebook Marketplace",
                "eBay",
                "Craigslist",
                "Etsy",
                "OfferUp",
                "Local stores"
            ]
        }
        return recommendations
    
    # Analyze the AI response to generate specific recommendations
    analysis_lower = ai_analysis.lower()
    
    # Generate specific DIY ideas based on what AI found
    if any(word in analysis_lower for word in ['chair', 'stool', 'bench', 'seat']):
        recommendations['diy_ideas'] = [
            "Transform into a garden planter by adding soil and plants",
            "Create a unique wall art piece by painting and hanging",
            "Convert into a storage bench by adding a hinged seat",
            "Make a pet bed by adding cushions and blankets",
            "Create a coat rack by adding hooks to the back",
            "Transform into a side table by adding a flat surface"
        ]
    elif any(word in analysis_lower for word in ['table', 'desk', 'counter']):
        recommendations['diy_ideas'] = [
            "Convert into a workbench for DIY projects",
            "Transform into a garden potting station",
            "Create a storage unit by adding drawers",
            "Make a display table for collectibles",
            "Convert into a craft table for hobbies",
            "Transform into a bar cart with wheels"
        ]
    elif any(word in analysis_lower for word in ['sofa', 'couch']):
        recommendations['diy_ideas'] = [
            "Reupholster with new fabric for a fresh look",
            "Convert into a daybed by removing back cushions",
            "Transform into outdoor seating with weather-resistant fabric",
            "Create a pet bed by adding pet-friendly cushions",
            "Convert into storage seating with hidden compartments",
            "Transform into a reading nook with pillows"
        ]
    elif any(word in analysis_lower for word in ['electronic', 'phone', 'laptop', 'computer']):
        recommendations['diy_ideas'] = [
            "Convert old phone into a security camera",
            "Transform laptop screen into external monitor",
            "Create Bluetooth speaker from old speakers",
            "Build smart home controller from tablet",
            "Make charging station from old electronics",
            "Create LED lamp from circuit boards"
        ]
    elif any(word in analysis_lower for word in ['glass', 'bottle', 'jar']):
        recommendations['diy_ideas'] = [
            "Transform bottles into decorative vases",
            "Create candle holders from glass containers",
            "Make terrariums from jars",
            "Create storage containers for small items",
            "Transform into hanging planters",
            "Make decorative lamps from bottles"
        ]
    elif any(word in analysis_lower for word in ['metal', 'steel', 'iron']):
        recommendations['diy_ideas'] = [
            "Create garden decorations from metal items",
            "Transform into wall art with paint",
            "Make storage containers from metal boxes",
            "Create wind chimes from metal pieces",
            "Transform into planters with proper drainage",
            "Make decorative hooks and hangers"
        ]
    elif any(word in analysis_lower for word in ['wood', 'wooden']):
        recommendations['diy_ideas'] = [
            "Sand and refinish for a new look",
            "Create wooden wall art or signs",
            "Transform into garden planters",
            "Make storage boxes or shelves",
            "Create decorative wooden crafts",
            "Transform into pet furniture"
        ]
    else:
        # Generic recommendations for other items
        recommendations['diy_ideas'] = [
            "Transform into a decorative piece",
            "Create a storage solution",
            "Make it into a garden planter",
            "Convert into wall art",
            "Repurpose for pet use",
            "Create a unique display item"
        ]
    
    # Generate monetization suggestions based on condition and type
    if any(word in analysis_lower for word in ['vintage', 'antique', 'rare', 'collectible']):
        recommendations['monetization'] = [
            "Sell to antique dealers or collectors",
            "List on specialized vintage marketplaces",
            "Try auction houses for valuable items",
            "Post on Etsy for vintage items",
            "Consider consignment shops",
            "Sell to museums or collectors"
        ]
    elif any(word in analysis_lower for word in ['damaged', 'broken', 'worn']):
        recommendations['monetization'] = [
            "Sell for parts or materials",
            "List as 'for repair' on marketplaces",
            "Sell to DIY enthusiasts",
            "Offer for free to artists/crafters",
            "Donate to art schools",
            "Sell scrap materials"
        ]
    else:
        recommendations['monetization'] = [
            "Sell on Facebook Marketplace",
            "List on eBay",
            "Post on Craigslist",
            "Try local thrift stores",
            "Consider consignment shops",
            "Rent out for events"
        ]
    
    # Generate sustainability benefits
    recommendations['sustainability'] = [
        "Reduces waste in landfills significantly",
        "Decreases demand for new item manufacturing",
        "Supports circular economy principles",
        "Reduces carbon footprint of production",
        "Promotes sustainable consumption habits",
        "Helps conserve natural resources"
    ]
    
    # Generate tutorials based on material
    if any(word in analysis_lower for word in ['wood', 'wooden']):
        recommendations['tutorials'] = [
            "Wood sanding and refinishing techniques",
            "Wood staining and finishing methods",
            "Basic wood repair techniques",
            "Wood painting and decoration ideas",
            "Wood assembly and construction basics",
            "Wood safety and tool handling"
        ]
    elif any(word in analysis_lower for word in ['metal', 'steel', 'iron']):
        recommendations['tutorials'] = [
            "Metal cleaning and rust removal",
            "Metal painting and finishing techniques",
            "Basic metal repair methods",
            "Metal cutting and shaping basics",
            "Metal welding and joining techniques",
            "Metal safety and tool handling"
        ]
    elif any(word in analysis_lower for word in ['glass']):
        recommendations['tutorials'] = [
            "Glass cleaning and maintenance",
            "Glass cutting and shaping techniques",
            "Glass painting and decoration methods",
            "Glass safety and handling basics",
            "Glass repair and restoration",
            "Glass crafting and DIY projects"
        ]
    else:
        recommendations['tutorials'] = [
            "Basic cleaning techniques",
            "Safe handling methods",
            "Restoration techniques",
            "Creative painting ideas",
            "Assembly basics",
            "Safety guidelines"
        ]
    
    # Generate marketplace suggestions
    recommendations['marketplace_suggestions'] = [
        "Facebook Marketplace - Best for local sales",
        "eBay - Wide audience reach",
        "Craigslist - Quick local transactions",
        "Etsy - Creative and handmade items",
        "OfferUp - Mobile-friendly marketplace",
        "Local thrift stores and consignment shops"
    ]
    
    return recommendations


def analyze_image(image_path):
    """Analyze image using OpenAI Vision API with multiple attempts"""
    try:
        print(f"Starting analysis for: {image_path}")
        
        # Process the image first
        processed_path = process_image(image_path)
        print(f"Image processed: {processed_path}")
        
        # Get image information
        image_info = get_image_info(processed_path)
        
        # Try AI analysis with multiple attempts
        ai_response = None
        max_attempts = 3
        
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1}/{max_attempts}")
            ai_response = analyze_image_with_ai(processed_path)
            
            if ai_response and len(ai_response) > 50:  # Check if we got a meaningful response
                print(f"Analysis successful on attempt {attempt + 1}")
                break
            else:
                print(f"Attempt {attempt + 1} failed or returned short response")
                if attempt < max_attempts - 1:
                    print("Retrying...")
        
        if not ai_response or len(ai_response) < 50:
            print("All AI attempts failed, using intelligent fallback analysis")
            # Intelligent fallback analysis based on filename and image info
            filename = os.path.basename(image_path).lower()
            image_info = get_image_info(processed_path)
            
            # Generate intelligent analysis based on filename patterns
            if any(word in filename for word in ['chair', 'stool', 'bench']):
                ai_response = """**Object Identification**: This appears to be a chair or seating furniture.

**Material Analysis**: Likely made of wood, metal, or a combination of materials.

**Condition Assessment**: The item appears to be in usable condition with potential for creative reuse.

**Size Estimation**: Standard furniture size suitable for various DIY projects.

**Style/Design**: Classic furniture design that can be transformed into modern pieces.

**Creative Potential**: Excellent potential for creative reuse and DIY transformations.

**DIY Ideas**: 
- Transform into a garden planter
- Create unique wall art
- Convert into storage seating
- Make a pet bed
- Create decorative display piece

**Monetization**: Can be sold on Facebook Marketplace, eBay, or local thrift stores."""
            
            elif any(word in filename for word in ['bottle', 'glass', 'jar']):
                ai_response = """**Object Identification**: This appears to be a glass bottle or container.

**Material Analysis**: Made of glass, suitable for various creative projects.

**Condition Assessment**: Glass container in good condition for reuse.

**Size Estimation**: Standard bottle size perfect for DIY projects.

**Creative Potential**: High potential for creative reuse and decoration.

**DIY Ideas**:
- Transform into decorative vases
- Create candle holders
- Make terrariums
- Create storage containers
- Transform into hanging planters

**Monetization**: Can be sold as craft supplies or decorative items."""
            
            else:
                ai_response = """**Object Identification**: This appears to be a household item with creative potential.

**Material Analysis**: The item seems to be made of common household materials suitable for reuse.

**Condition Assessment**: The item appears to be in reasonable condition for creative projects.

**Creative Potential**: Good potential for DIY transformations and creative reuse.

**DIY Ideas**:
- Transform into decorative piece
- Create storage solution
- Make garden planter
- Convert into wall art
- Repurpose for pet use

**Monetization**: Can be sold on various online marketplaces or local stores."""
            
            print("✅ Fallback analysis generated successfully")
        
        # Generate recommendations using AI
        recommendations = generate_recommendations_with_ai(ai_response)
        
        # If AI recommendations are empty, try to extract from analysis text
        if not any(recommendations.values()):
            print("⚠️ AI recommendations empty, extracting from analysis text...")
            recommendations = extract_recommendations_from_analysis(ai_response)
        
        # Determine category
        analysis_lower = ai_response.lower()
        if any(word in analysis_lower for word in ['chair', 'table', 'sofa', 'furniture', 'stool', 'bench']):
            category = "furniture"
        elif any(word in analysis_lower for word in ['phone', 'laptop', 'electronic', 'computer', 'device']):
            category = "electronics"
        elif any(word in analysis_lower for word in ['shirt', 'dress', 'clothing', 'fabric', 'textile']):
            category = "clothing"
        elif any(word in analysis_lower for word in ['glass', 'bottle', 'jar', 'container']):
            category = "kitchen"
        elif any(word in analysis_lower for word in ['metal', 'steel', 'iron', 'aluminum']):
            category = "tools"
        elif any(word in analysis_lower for word in ['wood', 'wooden']):
            category = "furniture"
        else:
            category = "general"
        
        return {
            'analysis': ai_response,
            'category': category,
            'confidence': 90 if ai_response and len(ai_response) > 100 else 60,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'original_image': image_path,
            'processed_image': processed_path,
            'image_info': image_info
        }
        
    except Exception as e:
        print(f"Analysis Error: {e}")
        return {
            'analysis': f'Analysis failed: {str(e)}',
            'category': 'error',
            'confidence': 0,
            'recommendations': get_empty_recommendations(),
            'timestamp': datetime.now().isoformat(),
            'original_image': image_path,
            'processed_image': image_path,
            'image_info': None
        }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            flash('No file selected. Please choose an image file.', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected. Please choose an image file.', 'error')
            return redirect(url_for('index'))
        
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload PNG, JPG, JPEG, GIF, or WEBP files only.', 'error')
            return redirect(url_for('index'))
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            flash(f'File too large. Maximum size allowed is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB.', 'error')
            return redirect(url_for('index'))
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze the image
        try:
            analysis = analyze_image(filepath)
            flash('Image analyzed successfully!', 'success')
            return render_template('results.html', analysis=analysis)
        except Exception as e:
            flash(f'Analysis failed: {str(e)}', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Upload failed: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/capture_image', methods=['POST'])
def capture_image():
    """Handle camera capture from frontend"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data received'}), 400
        
        # Remove data URL prefix
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Save image
        filename = f"capture_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        # Analyze the image
        analysis = analyze_image(filepath)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for image analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze the image
            analysis = analyze_image(filepath)
            
            return jsonify(analysis)
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tutorials')
def tutorials():
    """Display tutorials and DIY projects"""
    return render_template('tutorials.html')

@app.route('/batch')
def batch():
    """Batch processing page"""
    return render_template('batch.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Hugging Face Spaces uses port 7860 by default, but we'll use environment variable
    port = int(os.getenv('PORT', 7860))
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)
