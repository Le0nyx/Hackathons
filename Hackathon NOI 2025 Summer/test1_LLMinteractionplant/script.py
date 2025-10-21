#!/usr/bin/env python3
"""
Complete Plant Prediction Pipeline with Open-Meteo Official Client
Open source weather + AI image transformation
"""

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import numpy as np

class PlantPredictor:
    def __init__(self):
        """Initialize the plant prediction pipeline with Open-Meteo client"""
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        
        self.image_model = None
        self.device = self._get_device()
        
    def _get_device(self):
        """Determine the best available device, preferring RTX 3060"""
        if torch.cuda.is_available():
            # Check all available GPUs
            num_gpus = torch.cuda.device_count()
            print(f"🔍 Found {num_gpus} GPU(s) available:")
            
            # List all GPUs and find RTX 3060
            rtx_3060_device = None
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
                # Look for RTX 3060 specifically
                if "3060" in gpu_name or "RTX 3060" in gpu_name:
                    rtx_3060_device = i
                    print(f"  ✅ Found RTX 3060 at device {i}!")
            
            # Set the device
            if rtx_3060_device is not None:
                device_id = rtx_3060_device
                torch.cuda.set_device(device_id)
                print(f"🎯 Using RTX 3060 (GPU {device_id})")
            else:
                # Fall back to the most powerful GPU (usually the one with most memory)
                device_id = 0
                max_memory = 0
                for i in range(num_gpus):
                    memory = torch.cuda.get_device_properties(i).total_memory
                    if memory > max_memory:
                        max_memory = memory
                        device_id = i
                torch.cuda.set_device(device_id)
                print(f"🔄 RTX 3060 not found, using GPU {device_id} with most memory")
            
            device = f"cuda:{device_id}"
            
            # Display selected GPU info
            selected_gpu = torch.cuda.get_device_name(device_id)
            selected_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            print(f"🚀 Selected GPU: {selected_gpu}")
            print(f"💾 GPU Memory: {selected_memory:.1f} GB")
            
            # Clear any existing GPU cache
            torch.cuda.empty_cache()
            
            # Set memory allocation strategy for better performance
            torch.cuda.set_per_process_memory_fraction(0.85, device_id)  # Use 85% of GPU memory
            print(f"🔧 Set memory fraction to 85% for optimal performance")
            
        else:
            device = "cpu"
            print("⚠️  No GPU available, using CPU (will be slower)")
        
        return device
        
    def load_image_model(self):
        """Load the image transformation model with RTX 3060 optimization"""
        print("🔄 Loading Stable Diffusion model...")
        print(f"📍 Device: {self.device}")
        
        try:
            # Load model with appropriate precision based on device
            if "cuda" in self.device:
                print("🚀 Loading model with RTX 3060 GPU acceleration...")
                
                # For RTX 3060 (8GB VRAM), use optimized settings
                self.image_model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    "timbrooks/instruct-pix2pix",
                    torch_dtype=torch.float16,  # Use half precision for RTX 3060
                    use_safetensors=True,
                    safety_checker=None,
                    requires_safety_checker=False,
                    variant="fp16"  # Specifically request FP16 variant
                )
                
                # Move model to the specific GPU
                self.image_model = self.image_model.to(self.device)
                
                # RTX 3060 specific optimizations
                try:
                    self.image_model.enable_xformers_memory_efficient_attention()
                    print("✅ XFormers memory efficient attention enabled for RTX 3060")
                except Exception as e:
                    print(f"⚠️  XFormers not available: {e}")
                    print("💡 Consider installing xformers for better RTX 3060 performance")
                
                # Enable model CPU offload for RTX 3060's 8GB VRAM
                self.image_model.enable_model_cpu_offload()
                print("✅ Model CPU offload enabled (important for RTX 3060's 8GB VRAM)")
                
                # Enable VAE slicing for lower memory usage
                self.image_model.enable_vae_slicing()
                print("✅ VAE slicing enabled for memory efficiency")
                
                # Enable attention slicing for RTX 3060
                self.image_model.enable_attention_slicing(1)
                print("✅ Attention slicing enabled for RTX 3060")
                
            else:
                print("🐌 Loading model for CPU inference...")
                self.image_model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    "timbrooks/instruct-pix2pix",
                    torch_dtype=torch.float32,  # Use full precision for CPU
                    use_safetensors=True,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                self.image_model = self.image_model.to(self.device)
            
            print("✅ Model loaded successfully on RTX 3060!")
            
            # Display memory usage
            if "cuda" in self.device:
                device_id = int(self.device.split(':')[-1]) if ':' in self.device else 0
                allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                cached = torch.cuda.memory_reserved(device_id) / 1024**3
                print(f"📊 GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 This might be due to insufficient GPU memory or missing dependencies")
            print("💡 RTX 3060 has 8GB VRAM - try reducing image size if needed")
            raise e
        
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
    
    def create_transformation_prompt(self, plant_conditions):
        """Create a detailed prompt for image transformation based on weather"""
        
        if not plant_conditions:
            return "Show this plant after one week of growth"
            
        # Analyze conditions and create descriptive prompt
        temp_avg = (plant_conditions['avg_temp_max'] + plant_conditions['avg_temp_min']) / 2
        
        # Temperature effects
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
        
        prompt = f"""Transform this plant showing realistic growth after {plant_conditions['days_analyzed']} days with {temp_effect}, {water_effect}, {sun_effect}, and {uv_effect}. Show natural changes in leaf size, color saturation, stem thickness, and overall plant structure. Weather summary: {plant_conditions['avg_temp_min']}-{plant_conditions['avg_temp_max']}°C, {plant_conditions['total_rain']}mm rain, {plant_conditions['total_sunshine_hours']}h sun"""
        
        return prompt
    
    def transform_plant_image(self, image_path, prompt, num_inference_steps=20):
        """Transform plant image based on weather conditions with GPU acceleration"""
        
        if self.image_model is None:
            self.load_image_model()
            
        try:
            # Load and prepare image
            print(f"📸 Loading image: {image_path}")
            image = Image.open(image_path).convert("RGB")
            
            # Resize if too large (for memory efficiency)
            original_size = image.size
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                print(f"📏 Resized image from {original_size} to {image.size}")
            
            # Clear GPU cache before generation
            if "cuda" in self.device:
                torch.cuda.empty_cache()
                device_id = int(self.device.split(':')[-1]) if ':' in self.device else 0
                available_memory = torch.cuda.get_device_properties(device_id).total_memory - torch.cuda.memory_reserved(device_id)
                print(f"🧹 GPU memory cleared. Available: {available_memory / 1024**3:.2f} GB")
            
            # Transform image with optimized settings for RTX 3060
            print(f"🎨 Transforming image with prompt: {prompt[:100]}...")
            
            # Set generator for reproducible results
            device_for_generator = self.device if "cuda" in self.device else "cpu"
            generator = torch.Generator(device=device_for_generator).manual_seed(42)
            
            if "cuda" in self.device:
                # Use autocast for mixed precision on RTX 3060
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    result = self.image_model(
                        prompt, 
                        image=image, 
                        num_inference_steps=num_inference_steps,
                        image_guidance_scale=1.5,
                        guidance_scale=7.5,
                        generator=generator
                    ).images[0]
            else:
                # CPU inference without autocast
                result = self.image_model(
                    prompt, 
                    image=image, 
                    num_inference_steps=num_inference_steps,
                    image_guidance_scale=1.5,
                    guidance_scale=7.5,
                    generator=generator
                ).images[0]
            
            # Clean up GPU memory after generation
            if "cuda" in self.device:
                torch.cuda.empty_cache()
                print("🧹 RTX 3060 memory cleaned up after generation")
            
            print("✅ Image transformation completed!")
            return result
            
        except torch.cuda.OutOfMemoryError:
            print("❌ RTX 3060 out of memory!")
            print("💡 Try reducing image size or using fewer inference steps")
            print("💡 RTX 3060 has 8GB VRAM - large images may exceed this limit")
            if "cuda" in self.device:
                torch.cuda.empty_cache()
            return None
        except Exception as e:
            print(f"❌ Error transforming image: {e}")
            if "cuda" in self.device:
                torch.cuda.empty_cache()
            return None
    
    def predict_plant_growth(self, image_path, lat, lon, output_path="predicted_plant.jpg", days=7):
        """Complete pipeline: weather + image transformation with RTX 3060 acceleration"""
        
        print(f"🌱 Starting plant prediction for coordinates: {lat}, {lon}")
        print(f"📅 Analyzing {days} days of weather data...")
        print(f"🖥️  Using device: {self.device}")
        
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
        
        # Step 3: Create transformation prompt
        prompt = self.create_transformation_prompt(plant_conditions)
        print(f"\n📝 Generated transformation prompt: {prompt}")
        
        # Step 4: Transform image with RTX 3060 acceleration
        print(f"\n🎨 Transforming plant image using RTX 3060...")
        
        import time
        start_time = time.time()
        
        result_image = self.transform_plant_image(image_path, prompt)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        if result_image:
            result_image.save(output_path)
            print(f"✅ Plant growth prediction saved to: {output_path}")
            print(f"⏱️  Generation time with RTX 3060: {generation_time:.2f} seconds")
            
            # Show RTX 3060 memory usage if available
            if "cuda" in self.device:
                device_id = int(self.device.split(':')[-1]) if ':' in self.device else 0
                memory_used = torch.cuda.max_memory_allocated(device_id) / 1024**3
                total_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
                print(f"📊 RTX 3060 Peak Memory Usage: {memory_used:.2f} GB / {total_memory:.1f} GB ({memory_used/total_memory*100:.1f}%)")
                torch.cuda.reset_peak_memory_stats(device_id)
            
            return result_image, plant_conditions, weather_df
        else:
            print("❌ Failed to transform image on RTX 3060")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = PlantPredictor()
    
    # Example coordinates (Milan, Italy)
    latitude = 45.4642
    longitude = 9.1900
    
    # Predict plant growth
    # Replace 'your_plant_image.jpg' with actual image path
    result = predictor.predict_plant_growth(
        image_path=r"./foto/basilico.jpg",
        lat=latitude,
        lon=longitude,
        output_path=r"./predicted_plant_growth.jpg",
        days=7
    )
    
    if result:
        image, conditions, weather_data = result
        print("\n" + "="*50)
        print("PLANT PREDICTION COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Weather conditions analyzed: {conditions}")
        print(f"Weather data shape: {weather_data.shape}")
        print(f"Temperature range: {conditions['avg_temp_min']}°C to {conditions['avg_temp_max']}°C")
        print(f"Total precipitation: {conditions['total_rain']}mm")
        print(f"Sunshine hours: {conditions['total_sunshine_hours']}h")
    else:
        print("Plant prediction failed.")