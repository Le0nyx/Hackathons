import io
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import numpy as np
from torchvision import transforms
from model import PlantClassifier  # personalizzalo secondo il tuo file
import geocoder
import sys
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import os
print(sys.stdout.encoding)  # Check what encoding your console is using

# Force UTF-8 encoding for the entire script
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class PlantPredictor:
    def __init__(self):
        """Initialize the plant prediction pipeline with Open-Meteo client"""
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        
        self.image_model = None
        self.trained_model = None
        self.class_labels = ["basil", "tomato"]  # oppure caricali dinamicamente
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_trained_model(self, model_path="./models/basil_tomato_classifier.pth"):
        if not os.path.exists(model_path):
            print("⚠️ Trained model not found!")
            return

        try:
            model = PlantClassifier(num_classes=2)
            
            # Load checkpoint with proper device mapping
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # If the checkpoint is just the state dict
                state_dict = checkpoint
            
            # Fix key mismatches between training and inference models
            # The saved model has keys like "features.*" but current model expects "backbone.features.*"
            corrected_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('features.'):
                    # Add "backbone." prefix to features
                    new_key = 'backbone.' + key
                    corrected_state_dict[new_key] = value
                elif key.startswith('classifier.'):
                    # Add "backbone." prefix to classifier
                    new_key = 'backbone.' + key
                    corrected_state_dict[new_key] = value
                else:
                    # Keep other keys as they are
                    corrected_state_dict[key] = value
            
            # Load the corrected state dict
            model.load_state_dict(corrected_state_dict, strict=False)
            
            model.to(self.device)
            model.eval()
            self.trained_model = model
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"⚠️ Error loading trained model: {e}")
            self.trained_model = None

        
    def get_current_location(self):
        try:
            g = geocoder.ip('me')
            if g.ok:
                print(f"📍 Location detected: {g.city}, {g.country}")
                print(f"📍 Coordinates: {g.latlng[0]:.4f}, {g.latlng[1]:.4f}")
                return g.latlng[0], g.latlng[1]
            else:
                print("⚠️ Could not detect location, using default (Milan)")
        except Exception as e:
            print(f"⚠️ Location detection failed: {e}, using default (Milan)")

        # default Milan coords if failed
        return 45.4642, 9.1900

        
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
    
    CLASS_NAMES = {0: "basil", 1: "tomato"}  # Adatta se usi nomi diversi
    
    def create_transformation_prompt(self, image_path, plant_conditions):
        if not plant_conditions:
            return "Show this plant after one week of growth", "generic plant", "unknown health"
        
        plant_type = "generic plant"
        plant_health = "unknown health"
        
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found at {image_path}")
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                width, height = image.size
                
                try:
                    plant_type = self.detect_plant_type(image)
                except Exception as e:
                    print(f"⚠️ Plant type detection failed: {e}")
                    plant_type = "generic plant"
                    
                try:
                    plant_health = self.assess_plant_health(image)
                except Exception as e:
                    print(f"⚠️ Health assessment failed: {e}")
                    plant_health = "unknown health"
                
                print(f"📸 Image Analysis:")
                print(f"   Plant type detected: {plant_type}")
                print(f"   Current health: {plant_health}")
                print(f"   Image size: {width}x{height}")
        except Exception as e:
            print(f"⚠️ Warning: Could not analyze image: {str(e)}")
            plant_type = "generic plant"
            plant_health = "healthy"
        
        # Weather + growth prompt logic (come da tua versione)
        temp_avg = (plant_conditions['avg_temp_max'] + plant_conditions['avg_temp_min']) / 2
        
        if plant_type == "basil" or plant_type == "tomato" or ("herb" in plant_type):
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
        
        if plant_conditions['total_rain'] > 20:
            water_effect = "abundant rainfall keeping leaves lush, turgid and deep green"
        elif plant_conditions['total_rain'] < 5:
            water_effect = "dry conditions causing slight leaf wilting and browning at edges"
        else:
            water_effect = "adequate moisture maintaining crisp, healthy leaf appearance"
        
        if plant_conditions['total_sunshine_hours'] > 50:
            sun_effect = "plenty of sunlight encouraging dense, compact foliage growth"
        elif plant_conditions['total_sunshine_hours'] < 20:
            sun_effect = "limited sunlight causing elongated stems and sparse leaf growth"
        else:
            sun_effect = "moderate sunlight supporting balanced, proportional growth"
        
        if plant_conditions['max_uv_index'] > 7:
            uv_effect = "high UV causing slight leaf thickening and waxy appearance"
        else:
            uv_effect = "moderate UV maintaining normal leaf texture"
        
        prompt = (
            f"Transform this {plant_type} showing realistic growth after {plant_conditions['days_analyzed']} days. "
            f"Current state: {plant_health}. Apply these weather effects: {temp_effect}, {water_effect}, {sun_effect}, and {uv_effect}. "
            f"Show natural changes in leaf size, color saturation, stem thickness, and overall plant structure while maintaining the original composition and lighting. "
            f"Weather summary: {plant_conditions['avg_temp_min']}-{plant_conditions['avg_temp_max']}°C, "
            f"{plant_conditions['total_rain']}mm rain, {plant_conditions['total_sunshine_hours']}h sun"
        )
        
        return prompt, plant_type, plant_health
        
    def detect_plant_type(self, image):
        """Use trained model to classify the plant type"""
        if self.trained_model is None:
            self.load_trained_model()

        if self.trained_model is None:
            print("⚠️ Trained model not available, using fallback rule.")
            return "generic plant"
        
        try:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # usa la stessa dimensione del training
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],  # mean/std di ImageNet o dataset tuo
                                    [0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.trained_model(input_tensor)
                _, predicted = torch.max(output, 1)
                predicted_class = self.class_labels[predicted.item()]
                
                # Get confidence score
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence = probabilities[0][predicted].item()
                
                print(f"🌱 Plant classification: {predicted_class} (confidence: {confidence:.2f})")
                return predicted_class
                
        except Exception as e:
            print(f"⚠️ Error in plant type detection: {e}")
            return "generic plant"

    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory and move models appropriately"""
        if torch.cuda.is_available():
            # Move Stable Diffusion model to CPU if LLaVA is being used
            if hasattr(self, 'image_model') and self.image_model is not None:
                print("💾 Moving Stable Diffusion to CPU to free GPU memory...")
                self.image_model = self.image_model.to("cpu")
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Print memory stats
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"📊 GPU Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")

    def assess_plant_health(self, image):
        """Assess basic plant health from image"""
        try:
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
        except Exception as e:
            print(f"⚠️ Error in health assessment: {e}")
            return "unknown health"
        
    def describe_image_with_llava(self, image_pil, prompt=None):
        """Use LLaVA-Next to generate a description of the plant image with proper device handling."""
        try:
            from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
            import torch

            if not hasattr(self, "llava_model"):
                print("🔄 Loading LLaVA-Next model...")
                model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
                
                # Use the correct processor for LLaVA-Next
                self.llava_processor = LlavaNextProcessor.from_pretrained(model_id)
                
                # Determine optimal device configuration
                if torch.cuda.is_available():
                    # Check available GPU memory
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                    print(f"📊 Available GPU memory: {gpu_memory:.1f} GB")
                    
                    if gpu_memory >= 12:  # High memory GPU
                        device_map = "auto"
                        torch_dtype = torch.float16
                        print("🚀 Using GPU with auto device mapping")
                    else:  # Lower memory GPU - use CPU offloading
                        device_map = {"": "cpu"}
                        torch_dtype = torch.float32
                        print("💾 Using CPU due to limited GPU memory")
                else:
                    device_map = {"": "cpu"}
                    torch_dtype = torch.float32
                    print("🖥️ Using CPU (no GPU available)")
                
                # Load model with explicit device mapping
                self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    device_map=device_map,
                    offload_folder="./offload_cache",  # Explicit offload directory
                    offload_state_dict=True if device_map != "auto" else False
                )
                
                # Ensure model is in eval mode
                self.llava_model.eval()
                print("✅ LLaVA-Next loaded successfully")

            # Clear CUDA cache before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Prepare the conversation format that LLaVA-Next expects
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt or "Describe this plant's current condition, growth stage, health indicators, leaf characteristics, and any visible signs of stress or vitality. Focus on botanical details."}
                    ]
                }
            ]

            # Apply chat template and process inputs
            prompt_text = self.llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            # Process inputs properly
            inputs = self.llava_processor(
                images=image_pil,
                text=prompt_text,
                return_tensors="pt"
            )
            
            # Handle device placement more carefully
            target_device = "cpu"  # Default to CPU for stability
            if hasattr(self.llava_model, 'device'):
                target_device = self.llava_model.device
            elif hasattr(self.llava_model, 'hf_device_map'):
                # Get the device of the first layer
                for module_name, device in self.llava_model.hf_device_map.items():
                    if device != 'disk':
                        target_device = device
                        break
            
            print(f"🎯 Moving inputs to device: {target_device}")
            inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Generate with proper parameters and error handling
            with torch.no_grad():
                try:
                    output = self.llava_model.generate(
                        **inputs,
                        max_new_tokens=150,  # Reduced for stability
                        do_sample=False,     # Use greedy decoding for consistency
                        temperature=None,    # Not used with do_sample=False
                        top_p=None,         # Not used with do_sample=False
                        pad_token_id=self.llava_processor.tokenizer.eos_token_id,
                        use_cache=True,
                        repetition_penalty=1.1
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("⚠️ GPU OOM, retrying with CPU...")
                        # Move everything to CPU and retry
                        inputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                        if hasattr(self.llava_model, 'cpu'):
                            self.llava_model = self.llava_model.cpu()
                        output = self.llava_model.generate(
                            **inputs,
                            max_new_tokens=150,
                            do_sample=False,
                            pad_token_id=self.llava_processor.tokenizer.eos_token_id
                        )
                    else:
                        raise e
            
            # Decode only the new tokens (exclude input tokens)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = output[0][input_length:]
            description = self.llava_processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return description.strip()

        except ImportError as e:
            print(f"⚠️ LLaVA-Next dependencies not available: {e}")
            return "Visual description not available - missing dependencies."
        except Exception as e:
            print(f"⚠️ Error during LLaVA-Next description: {e}")
            print(f"🔍 Error details: {type(e).__name__}: {str(e)}")
            return f"Visual description failed: {str(e)}"

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
                num_inference_steps=70,
                image_guidance_scale=1.5,
                guidance_scale=7.5
            ).images[0]
            
            return result
            
        except Exception as e:
            print(f"Error transforming image: {e}")
            return None
    
    @staticmethod
    def safe_print(text):
        try:
            print(text)
        except UnicodeEncodeError:
            # Fallback for systems with limited encoding support
            print(text.encode('ascii', errors='replace').decode('ascii'))
    
    def predict_plant_growth(self, image_path, lat=None, lon=None, output_path="./predicted_plant.jpg", days=7):
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
            self.safe_print(f" Plant identified as: {plant_type}")
            self.safe_print(f" Current health: {plant_health}")
            self.safe_print(f" Generated transformation prompt: {prompt}")
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
            # Save the predicted image
            result_image.save(output_path)
            print(f"Plant growth prediction saved to: {output_path}")

            # Compose the basic description
            description = (
                f"{plant_type.capitalize()} predicted after {plant_conditions['days_analyzed']} days:\n"
                f"- Temperature: {plant_conditions['avg_temp_min']}–{plant_conditions['avg_temp_max']} °C\n"
                f"- Rain: {plant_conditions['total_rain']} mm\n"
                f"- Sunshine: {plant_conditions['total_sunshine_hours']} h\n"
                f"- UV max: {plant_conditions['max_uv_index']}\n"
                f"- Daily temperature range: {plant_conditions['temp_range']} °C\n"
                f"Estimated health: {plant_health}."
            )

            # STEP 4.5: Enhanced visual description with LLaVA-Next
            try:
                print("\n🧠 STEP 4.5: Generating detailed visual analysis...")
                
                # Clean up GPU memory before loading LLaVA
                self.cleanup_gpu_memory()
                
                llava_description = self.describe_image_with_llava(
                    result_image, 
                    f"Analyze this {plant_type} plant prediction image. Describe the visible growth changes, leaf development, overall health indicators, and how the plant appears to have responded to the weather conditions: {plant_conditions['avg_temp_min']}-{plant_conditions['avg_temp_max']}°C, {plant_conditions['total_rain']}mm rain, {plant_conditions['total_sunshine_hours']}h sun over {plant_conditions['days_analyzed']} days."
                )
                
                print("🧠 AI Visual Analysis:")
                print(llava_description)

                # Save comprehensive description
                complete_description = f"{description}\n\nAI Visual Analysis:\n{llava_description}"
                
                description_txt_path = os.path.splitext(output_path)[0] + "_analysis.txt"
                with open(description_txt_path, "w", encoding="utf-8") as f:
                    f.write(complete_description)
                print(f"📄 Complete analysis saved to: {description_txt_path}")
                
            except Exception as e:
                print(f"⚠️ Visual analysis failed: {e}")
                # Still save basic description
                basic_txt_path = os.path.splitext(output_path)[0] + "_basic_info.txt"
                with open(basic_txt_path, "w", encoding="utf-8") as f:
                    f.write(description)
                print(f"📄 Basic info saved to: {basic_txt_path}")

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
        image_path="./basilico.jpg",
        lat=latitude,
        lon=longitude,
        output_path="./basilico_new2.jpg",
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

