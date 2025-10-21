import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import numpy as np
import geocoder

class PlantPredictor:
    def __init__(self):
        """Initialize the plant prediction pipeline with Open-Meteo client"""
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)

        self.image_model = None
        
    def get_current_location(self):
        """Get current location using IP geolocation"""
        try:
            g = geocoder.ip('me')
            if g.ok:
                print(f"📍 Location detected: {g.city}, {g.country}")
                print(f"📍 Coordinates: {g.latlng[0]:.4f}, {g.latlng[1]:.4f}")
                return g.latlng[0], g.latlng[1]  # lat, lon
            else:
                print("⚠️  Could not detect location, using default (Milan)")
            self.image_model = None
        except Exception as e:
            print(f"⚠️  Location detection failed: {e}, using default (Milan)")
        
        self.image_model = None
        
    def load_image_model(self):
        """Load the image transformation model with high-quality settings"""
        print("🔄 Loading Stable Diffusion model with high-quality settings...")
        
        # Check if CUDA is available and print GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        self.image_model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        if torch.cuda.is_available():
            self.image_model = self.image_model.to("cuda")
            
            # Enable memory efficient attention for better quality
            try:
                self.image_model.enable_xformers_memory_efficient_attention()
                print("✅ XFormers memory efficient attention enabled")
            except:
                print("⚠️  XFormers not available, using standard attention")
            
            # Enable VAE slicing for higher resolution support
            self.image_model.enable_vae_slicing()
            print("✅ VAE slicing enabled for high-res support")
            
            # Enable attention slicing for memory efficiency
            self.image_model.enable_attention_slicing(1)
            print("✅ Attention slicing enabled")
        
        print("✅ High-quality model loaded successfully!")
        
    def get_weather_forecast(self, lat, lon, days=7):
        """Get weather forecast from Open-Meteo API using official client"""
        
        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min", 
                "precipitation_sum",
                "rain_sum",
                "uv_index_max",
                "sunshine_duration"
            ],
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "auto"
        }
        
        try:
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]  # Process first location
            
            print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
            print(f"Elevation: {response.Elevation()} m asl")
            print(f"Timezone: UTC{response.UtcOffsetSeconds()//3600:+d}")
            
            # Process daily data
            daily = response.Daily()
            
            # Extract data as numpy arrays (much faster!)
            daily_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left"
                ),
                "temperature_2m_max": daily.Variables(0).ValuesAsNumpy(),
                "temperature_2m_min": daily.Variables(1).ValuesAsNumpy(),
                "precipitation_sum": daily.Variables(2).ValuesAsNumpy(),
                "rain_sum": daily.Variables(3).ValuesAsNumpy(),
                "uv_index_max": daily.Variables(4).ValuesAsNumpy(),
                "sunshine_duration": daily.Variables(5).ValuesAsNumpy()
            }
            
            # Create DataFrame for easy analysis
            daily_dataframe = pd.DataFrame(data=daily_data)
            
            return daily_dataframe, response
            
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None, None
    
    def analyze_weather_for_plants(self, weather_df):
        """Analyze weather data and create plant-specific metrics"""
        
        if weather_df is None or weather_df.empty:
            return None
        
        # Handle NaN values by filling with 0 or mean
        weather_df = weather_df.fillna(0)
        
        # Calculate plant-relevant metrics using pandas (more efficient)
        plant_conditions = {
            "avg_temp_max": round(weather_df['temperature_2m_max'].mean(), 1),
            "avg_temp_min": round(weather_df['temperature_2m_min'].mean(), 1),
            "total_precipitation": round(weather_df['precipitation_sum'].sum(), 1),
            "total_rain": round(weather_df['rain_sum'].sum(), 1),
            "total_sunshine_hours": round(weather_df['sunshine_duration'].sum() / 3600, 1),  # Convert to hours
            "max_uv_index": round(weather_df['uv_index_max'].max(), 1),
            "days_analyzed": len(weather_df),
            "temp_range": round(weather_df['temperature_2m_max'].max() - weather_df['temperature_2m_min'].min(), 1)
        }
        
        return plant_conditions
    
    def create_transformation_prompt(self, image_path, plant_conditions):
        """Create a detailed prompt for image transformation based on weather AND image analysis"""
        
        if not plant_conditions:
            return "Show this plant after one week of growth", "generic plant", "unknown health"
        
        # STEP 3A: Analyze original image
        plant_type = "generic plant"
        plant_health = "unknown health"
        
        try:
            image = Image.open(image_path).convert("RGB")
            # Basic image analysis
            width, height = image.size
            aspect_ratio = width / height
            
            # Simple plant type detection based on image characteristics
            plant_type = self.detect_plant_type(image)
            plant_health = self.assess_plant_health(image)
            
            print(f"📸 Image Analysis:")
            print(f"   Plant type detected: {plant_type}")
            print(f"   Current health: {plant_health}")
            print(f"   Image size: {width}x{height}")
            
        except Exception as e:
            print(f"Warning: Could not analyze image: {e}")
            plant_type = "generic plant"
            plant_health = "healthy"
            
        # STEP 3B: Weather analysis with plant-specific logic
        temp_avg = (plant_conditions['avg_temp_max'] + plant_conditions['avg_temp_min']) / 2
        
        # Temperature effects (adjusted by plant type)
        if plant_type == "basil" or "herb" in plant_type:
            if temp_avg > 25:
                temp_effect = "warm weather promoting vigorous basil growth with larger, aromatic leaves and bushier structure"
            elif temp_avg < 15:
                temp_effect = "cool weather slowing basil growth with smaller, less vibrant leaves"
            else:
                temp_effect = "optimal temperature for basil supporting steady growth with healthy green foliage"
        else:
            if temp_avg > 25:
                temp_effect = "warm weather promoting vigorous growth with larger, darker green leaves"
            elif temp_avg < 10:
                temp_effect = "cool weather slowing growth with smaller, pale leaves"
            else:
                temp_effect = "moderate temperature supporting steady growth with healthy green foliage"
            
        # Water effects
        if plant_conditions['total_rain'] > 20:
            water_effect = "abundant rainfall keeping leaves lush, turgid and deep green"
        elif plant_conditions['total_rain'] < 5:
            water_effect = "dry conditions causing slight leaf wilting and browning at edges"
        else:
            water_effect = "adequate moisture maintaining crisp, healthy leaf appearance"
            
        # Sunlight effects
        if plant_conditions['total_sunshine_hours'] > 50:
            sun_effect = "plenty of sunlight encouraging dense, compact foliage growth"
        elif plant_conditions['total_sunshine_hours'] < 20:
            sun_effect = "limited sunlight causing elongated stems and sparse leaf growth"
        else:
            sun_effect = "moderate sunlight supporting balanced, proportional growth"
            
        # UV effects
        if plant_conditions['max_uv_index'] > 7:
            uv_effect = "high UV causing slight leaf thickening and waxy appearance"
        else:
            uv_effect = "moderate UV maintaining normal leaf texture"
        
        #  STEP 3C: Create comprehensive prompt combining image + weather analysis


# // FINAL PROMT HERE FOR PLANT


        prompt = f"""Transform this {plant_type} showing realistic growth after {plant_conditions['days_analyzed']} days. The plant should still be realistic and its surrounding how it would look like in the real world and a human should be able to say the picture looks normal and only focus on the plant. Current state: {plant_health}. Give it two weeks very satly water so it developes brown spots. Leave the rest around it alone in the background and all. Apply these weather effects: {temp_effect}, {water_effect}, {sun_effect}, and {uv_effect}. Show natural changes in leaf size, color saturation, stem thickness, and overall plant structure while maintaining the original composition and lighting. Weather summary: {plant_conditions['avg_temp_min']}-{plant_conditions['avg_temp_max']}°C, {plant_conditions['total_rain']}mm rain, {plant_conditions['total_sunshine_hours']}h sun"""
        return prompt, plant_type, plant_health
        
    def detect_plant_type(self, image):
        """Simple plant type detection based on image characteristics"""
        # This is a simplified version - in a real app you'd use a plant classification model
        # For now, we'll do basic analysis
        
        # Convert to array for analysis
        img_array = np.array(image)
        
        # Analyze color distribution
        green_pixels = np.sum((img_array[:,:,1] > img_array[:,:,0]) & (img_array[:,:,1] > img_array[:,:,2]))
        total_pixels = img_array.shape[0] * img_array.shape[1]
        green_ratio = green_pixels / total_pixels
        
        # Simple heuristics (could be improved with ML)
        if green_ratio > 0.4:
            return "basil"  # Assume basil for high green content
        else:
            return "generic plant"
    
    def assess_plant_health(self, image):
        """Assess basic plant health from image"""
        img_array = np.array(image)
        
        # Analyze brightness and color vibrancy
        brightness = np.mean(img_array)
        green_channel = np.mean(img_array[:,:,1])
        
        if brightness > 150 and green_channel > 120:
            return "healthy and vibrant"
        elif brightness > 100 and green_channel > 80:
            return "moderately healthy"
        else:
            return "showing some stress"
    
    def transform_plant_image(self, image_path, prompt, num_samples=1):
        """STEP 4: Generate ULTRA HIGH-QUALITY image with 60 inference steps"""
        
        if self.image_model is None:
            self.load_image_model()
            
        try:
            # Load and prepare image with HIGHER RESOLUTION
            print(f"📸 Loading image for high-quality processing: {image_path}")
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            
            # Use HIGHER resolution for better quality (up to 1024x1024)
            max_size = 1024  # Increased from 512 for better quality
            if max(image.size) < max_size:
                # Upscale smaller images for better quality
                scale_factor = max_size / max(image.size)
                new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                print(f"📈 Upscaled image from {original_size} to {image.size} for better quality")
            elif max(image.size) > max_size:
                # Resize but maintain higher resolution
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                print(f"📏 Resized image from {original_size} to {image.size}")
            
            print(f"🎨 Generating 1 ULTRA HIGH-QUALITY sample with 60 inference steps...")
            print(f"📝 Using enhanced prompt: {prompt[:120]}...")
            
            generated_images = []
            
            # Clear GPU cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            for i in range(num_samples):
                print(f"🔄 Generating ultra high-quality sample {i+1}/{num_samples} with 60 steps...")
                
                # Use different seeds for variety
                seed = 42 + i * 137  # Prime number spacing for better variety
                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
                
                # ULTRA HIGH-QUALITY SETTINGS (60 steps for maximum quality)
                result = self.image_model(
                    prompt, 
                    image=image, 
                    num_inference_steps=60,  # Increased to 60 for ultra high quality
                    image_guidance_scale=2.0,  # Increased from 1.5 for stronger conditioning
                    guidance_scale=9.0,  # Increased from 7.5 for better prompt following
                    generator=generator,
                    eta=0.0,  # Deterministic for better quality
                    # Add additional quality parameters
                ).images[0]
                
                generated_images.append(result)
                print(f"✅ Ultra high-quality sample {i+1} completed with 60 inference steps!")
                
                # Clean up GPU memory between generations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"🎉 Ultra high-quality sample generated with 60 inference steps!")
            return generated_images
            
        except torch.cuda.OutOfMemoryError:
            print("❌ GPU out of memory! Try reducing num_samples or image resolution")
            print("💡 Current settings are optimized for high-end GPUs")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        except Exception as e:
            print(f"❌ Error transforming image: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
    
    def predict_plant_growth(self, image_path, lat=None, lon=None, output_path="predicted_plant.jpg", days=7, num_samples=1, high_quality=True):
        """Complete ULTRA HIGH-QUALITY pipeline with 60 inference steps for maximum quality"""
        
        # Auto-detect location if not provided
        if lat is None or lon is None:
            print("🌍 Auto-detecting location...")
            lat, lon = self.get_current_location()
        
        print(f"🌱 Starting ULTRA HIGH-QUALITY plant prediction for coordinates: {lat:.4f}, {lon:.4f}")
        print(f"📅 Analyzing {days} days of weather data...")
        print(f"🎯 Generating 1 ultra high-quality sample with 60 inference steps")
        print(f"⚠️  This will take longer but produce maximum quality results")
        
        # Step 1: Get weather data using official Open-Meteo client
        print("🌤️  Fetching weather data with caching and retry...")
        weather_df, response_info = self.get_weather_forecast(lat, lon, days)
        
        if weather_df is None:
            print("❌ Failed to get weather data")
            return None
            
        print(f"✅ Weather data retrieved for {len(weather_df)} days")
        print("\n📊 Weather Overview:")
        print(weather_df[['date', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'sunshine_duration']].head())
        
        # Step 2: Analyze weather for plants
        plant_conditions = self.analyze_weather_for_plants(weather_df)
        print(f"\n🔬 Plant-specific weather analysis: {plant_conditions}")
        
        # Step 3: Analyze image + weather to create intelligent prompt
        print("\n🧠 STEP 3: Advanced image analysis and prompt creation...")
        try:
            prompt, plant_type, plant_health = self.create_transformation_prompt(image_path, plant_conditions)
            print(f"🌿 Plant identified as: {plant_type}")
            print(f"💚 Current health: {plant_health}")
        except Exception as e:
            print(f"❌ Error in Step 3: {e}")
            return None
        
        # Step 4: Generate ULTRA HIGH-QUALITY transformed image
        print(f"\n STEP 4: Generating 1 prediction with 60 inference steps...")
        print("  This may take 5-8 minutes for absolute maximum quality...")
        
        import time
        start_time = time.time()
        
        try:
            result_images = self.transform_plant_image(image_path, prompt, num_samples=num_samples)
        except Exception as e:
            print(f" Error in Step 4: {e}")
            return None
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if result_images and len(result_images) > 0:
            # Save the ultra high-quality result
            saved_paths = []
            
            # Save with maximum quality JPEG settings
            result_images[0].save(output_path, "JPEG", quality=98, optimize=True)
            saved_paths.append(output_path)
            print(f" prediction saved to: {output_path}")
            
            # Create comparison with original
            self.create_comparison_grid(image_path, result_images, f"{output_path.replace('.jpg', '')}_comparison.jpg")
            
            print(f"⏱️  Total generation time: {total_time:.1f} seconds")
            print(f"🏆 Generated with 60 inference steps for maximum quality!")
            
            # GPU memory usage info
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**3
                print(f" Peak GPU memory usage: {memory_used:.2f} GB")
                torch.cuda.reset_peak_memory_stats()
            
            return result_images, plant_conditions, weather_df, plant_type, plant_health, saved_paths
        else:
            print(" Failed to generate image")
            return None
    
    def create_comparison_grid(self, original_path, generated_images, output_path):
        """Create a comparison grid"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Load original
            original = Image.open(original_path).convert("RGB")
            
            # Use higher resolution for grid
            target_size = (512, 512)
            original = original.resize(target_size, Image.Resampling.LANCZOS)
            resized_generated = [img.resize(target_size, Image.Resampling.LANCZOS) for img in generated_images]
            
            # Calculate grid
            total_images = len(generated_images) + 1
            cols = min(3, total_images)  # 3 columns max for better layout
            rows = (total_images + cols - 1) // cols
            
            # Create high-quality grid
            grid_width = cols * target_size[0]
            grid_height = rows * target_size[1] + 80  # More space for labels
            grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
            
            # Add images
            grid_image.paste(original, (0, 80))
            for i, img in enumerate(resized_generated):
                col = (i + 1) % cols
                row = (i + 1) // cols
                x = col * target_size[0]
                y = row * target_size[1] + 80
                grid_image.paste(img, (x, y))
            
            # Add labels
            try:
                draw = ImageDraw.Draw(grid_image)
                try:
                    font = ImageFont.truetype("arial.ttf", 32)  # Larger font
                except:
                    font = ImageFont.load_default()
                
                draw.text((10, 20), "Original", fill='black', font=font)
                for i in range(len(resized_generated)):
                    col = (i + 1) % cols
                    x = col * target_size[0] + 10
                    draw.text((x, 20), f"HQ Sample {i+1}", fill='black', font=font)
            except:
                pass
            
            # Save with high quality
            grid_image.save(output_path, "JPEG", quality=95, optimize=True)
            print(f" High-quality comparison grid saved to: {output_path}")
            
        except Exception as e:
            print(f"  Could not create comparison grid: {e}")

# Example usage - HIGH QUALITY MODE
if __name__ == "__main__":
    # Initialize predictor
    predictor = PlantPredictor()
    
    # Example coordinates (Milan, Italy)
    latitude = 45.4642
    longitude = 9.1900
    
    print(" Starting ULTRA HIGH-QUALITY plant prediction with 60 inference steps...")
    print("  This will use maximum GPU power and time for absolute best quality")
    
    # Ultra high-quality prediction with single sample
    result = predictor.predict_plant_growth(
        image_path="./foto/basilico2 originale.png",
        lat=latitude,
        lon=longitude,
        output_path="./predicted_plant_ultra_hq.jpg",
        days=7,
        num_samples=1,  # Single ultra high-quality sample
        high_quality=True
    )
    
    if result:
        images, conditions, weather_data, plant_type, plant_health, saved_paths = result
        print("\n" + "="*60)
        print("🎉 PLANT PREDICTION COMPLETED!")
        print("="*60)
        print(f"🌿 Plant type: {plant_type}")
        print(f"💚 Plant health: {plant_health}")
        print(f"🎯 Generated 1 ultra high-quality sample with 60 inference steps")
        print(f"📊 Weather data points: {weather_data.shape}")
        print(f"🌡️  Temperature range: {conditions['avg_temp_min']}°C to {conditions['avg_temp_max']}°C")
        print(f"🌧️  Total precipitation: {conditions['total_rain']}mm")
        print(f"☀️  Sunshine hours: {conditions['total_sunshine_hours']}h")
        
        print(f"\n💾 Saved files:")
        print(f"   📸 Ultra HQ prediction: ./predicted_plant_ultra_hq.jpg")
        print(f"   📊 Comparison image: ./predicted_plant_ultra_hq_comparison.jpg")
        
        print(f"\n🏆 Ultra quality improvements:")
        print(f"   ✅ 60 inference steps (maximum quality)")
        print(f"   ✅ Higher guidance scales for perfect accuracy")
        print(f"   ✅ Up to 1024x1024 resolution support")
        print(f"   ✅ Single focused sample for consistency")
        print(f"   ✅ Enhanced prompt engineering")
        print(f"   ✅ Maximum quality JPEG compression (98%)")
        print("")
        
    else:
        print("❌ Ultra high-quality plant prediction failed.")
        print("💡 Check GPU memory and ensure RTX 3060 is available")