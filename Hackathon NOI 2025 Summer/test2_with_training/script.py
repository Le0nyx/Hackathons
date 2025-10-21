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
        """Load the image transformation model"""
        print("Loading Stable Diffusion model...")
        self.image_model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            self.image_model = self.image_model.to("cuda")
        print("Model loaded successfully!")
        
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
        prompt = f"""Transform this {plant_type} showing realistic growth after {plant_conditions['days_analyzed']} days. Current state: {plant_health}. Apply these weather effects: {temp_effect}, {water_effect}, {sun_effect}, and {uv_effect}. Show natural changes in leaf size, color saturation, stem thickness, and overall plant structure while maintaining the original composition and lighting. Weather summary: {plant_conditions['avg_temp_min']}-{plant_conditions['avg_temp_max']}°C, {plant_conditions['total_rain']}mm rain, {plant_conditions['total_sunshine_hours']}h sun"""
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
    
    def transform_plant_image(self, image_path, prompt):
        """STEP 4: Generate new image based on analyzed prompt"""
        
        if self.image_model is None:
            self.load_image_model()
            
        try:
            # Load and prepare image
            image = Image.open(image_path).convert("RGB")
            
            # Resize if too large (for memory efficiency)
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            print(f" STEP 4: Generating transformed image...")
            print(f"   Using prompt: {prompt}")
            
            # Transform image
            result = self.image_model(
                prompt, 
                image=image, 
                num_inference_steps=20,
                image_guidance_scale=1.5,
                guidance_scale=7.5
            ).images[0]
            
            return result
            
        except Exception as e:
            print(f"Error transforming image: {e}")
            return None
    
    def predict_plant_growth(self, image_path, lat=None, lon=None, output_path="predicted_plant.jpg", days=7):
        """Complete pipeline: weather + image transformation"""
        
        # Auto-detect location if not provided
        if lat is None or lon is None:
            print(" Auto-detecting location...")
            lat, lon = self.get_current_location()
        
        print(f" Starting plant prediction for coordinates: {lat:.4f}, {lon:.4f}")
        print(f" Analyzing {days} days of weather data...")
        
        # Step 1: Get weather data using official Open-Meteo client
        print("Fetching weather data with caching and retry...")
        weather_df, response_info = self.get_weather_forecast(lat, lon, days)
        
        if weather_df is None:
            print("Failed to get weather data")
            return None
            
        print(f"Weather data retrieved for {len(weather_df)} days")
        print("\nWeather Overview:")
        print(weather_df[['date', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'sunshine_duration']].head())
        
        # Step 2: Analyze weather for plants
        plant_conditions = self.analyze_weather_for_plants(weather_df)
        print(f"\nPlant-specific weather analysis: {plant_conditions}")
        
        # Step 3: Analyze image + weather to create intelligent prompt
        print("\n STEP 3: Analyzing image and creating transformation prompt...")
        try:
            prompt, plant_type, plant_health = self.create_transformation_prompt(image_path, plant_conditions)
            print(f" Plant identified as: {plant_type}")
            print(f" Current health: {plant_health}")
            print(f" Generated transformation prompt: {prompt}")
        except Exception as e:
            print(f" Error in Step 3: {e}")
            return None
        
        # Step 4: Generate transformed image
        print("\nSTEP 4: Generating prediction image...")
        try:
            result_image = self.transform_plant_image(image_path, prompt)
        except Exception as e:
            print(f" Error in Step 4: {e}")
            return None
        
        if result_image:
            result_image.save(output_path)
            print(f"Plant growth prediction saved to: {output_path}")
            return result_image, plant_conditions, weather_df, plant_type, plant_health
        else:
            print("Failed to transform image")
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
        image_path="./foto/basilico.jpg",
        lat=latitude,
        lon=longitude,
        output_path="./predicted_plant_growth.jpg",
        days=7
    )
    
    if result:
        image, conditions, weather_data, plant_type, plant_health = result
        print("\n" + "="*50)
        print(" PLANT PREDICTION COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f" Plant type: {plant_type}")
        print(f" Plant health: {plant_health}")
        print(f" Weather conditions: {conditions}")
        print(f" Data points: {weather_data.shape}")
        print(f"  Temperature: {conditions['avg_temp_min']}°C to {conditions['avg_temp_max']}°C")
        print(f" Total rain: {conditions['total_rain']}mm")
        print(f" Sunshine: {conditions['total_sunshine_hours']}h")
    else:
        print("Plant prediction failed.")