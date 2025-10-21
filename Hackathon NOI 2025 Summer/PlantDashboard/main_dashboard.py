import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
import os
from datetime import datetime, date
from plant_model import PlantGrowthModel
from data_handler import DataHandler
from tkcalendar import DateEntry, Calendar
from plant_meteo import HappyMeteo

class PlantGrowthDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("WeGrow")
        self.root.geometry("1000x800")  # More square dimensions
        self.root.configure(bg='#f0f0f0')
        
        image = Image.open("public/logoTransparent.png")
        
        desired_size = (128, 128)
        image = image.resize(desired_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        icon = ImageTk.PhotoImage(image)
        # Set as window icon
        self.root.iconphoto(False, icon)
        
        # Initialize components
        self.plant_model = PlantGrowthModel()
        self.data_handler = DataHandler()
        self.happyMeteo = HappyMeteo()
        
        # Variables - fixed plant type
        self.current_plant = "tomato"  # Fixed plant type
        self.counter = 0
        self.filenames = ["basilico.jpg", "pomodoro.png"]
        self.ambient_mode = tk.StringVar(value="controlled")
        self.baseline_image_path = None
        
        # Environmental parameters with defaults
        self.default_params = {
            'temperature': 22.0,
            'humidity': 65.0,
            'soil_acidity': 6.5,
            'pressure': 1013.25,
            'brightness': 30,
            'nutrients': 75.0,
            'water': 80.0,
            'co2': 850
        }
        
        self.env_params = {
            'temperature': tk.DoubleVar(value=self.default_params['temperature']),
            'humidity': tk.DoubleVar(value=self.default_params['humidity']),
            'soil_acidity': tk.DoubleVar(value=self.default_params['soil_acidity']),
            'pressure': tk.DoubleVar(value=self.default_params['pressure']),
            'brightness': tk.DoubleVar(value=self.default_params['brightness']),
            'nutrients': tk.DoubleVar(value=self.default_params['nutrients']),
            'water': tk.DoubleVar(value=self.default_params['water']),
            'co2': tk.DoubleVar(value=self.default_params['co2'])
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container with square layout
        main_frame = ttk.Frame(self.root, padding="8")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for square layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)  # Center panel wider
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🌱 Plant Growth Dashboard", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Left panel - Controls
        self.setup_control_panel(main_frame)
        
        # Center panel - Plant Visualization
        self.setup_visualization_panel(main_frame)
        
        # Right panel - Results only (no system messages)
        self.setup_results_panel(main_frame)
        
    def setup_control_panel(self, parent):
        control_frame = ttk.LabelFrame(parent, text="Environmental Controls", padding="6")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 6))
        
        # Ambient mode
        ttk.Label(control_frame, text="Environment Mode:", font=('Arial', 9, 'bold')).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 6))
        
        mode_frame = ttk.Frame(control_frame)
        mode_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 8))
        
        ttk.Radiobutton(mode_frame, text="Controlled", variable=self.ambient_mode, 
                       value="controlled", command=self.on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Open", variable=self.ambient_mode, 
                       value="open", command=self.on_mode_change).pack(anchor=tk.W)
        
        # Baseline image
        ttk.Button(control_frame, text="📷 Load Plant Image", 
                  command=self.load_baseline_image).grid(row=2, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        
        ttk.Label(control_frame, text="Parameters:", 
                font=('Arial', 9, 'bold')).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))

        param_labels = {
            'temperature': '🌡️ Temp (°C)',
            'humidity': '💧 Humidity (%)',
            'soil_acidity': '🧪 (pH)',
            'pressure': '🌬️ Pressure (Pa)',
            'brightness': '☀️ Light (DLI)',
            'nutrients': '🌿 Nutrients (%)',
            'water': '💦 Water (%)',
            'co2': '🫧 CO2 (ppm)'
        }

        # Define bounds for each parameter (min, max)
        param_bounds = {
            'temperature': (10, 40),
            'humidity': (20, 90),
            'soil_acidity': (4.0, 9.0),
            'pressure': (950, 1100),
            'brightness': (5, 50),
            'nutrients': (0, 100),
            'water': (10, 100),
            'co2': (400, 1200)
        }

        row = 4
        for param, label in param_labels.items():
            # Compact parameter layout
            param_frame = ttk.Frame(control_frame)
            param_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=1)
            
            ttk.Label(param_frame, text=label, width=11, font=('Arial', 8)).pack(side=tk.LEFT)
            
            # Get bounds for this parameter
            scale_min, scale_max = param_bounds.get(param, (0, 100))
            
            scale = ttk.Scale(param_frame, from_=scale_min, to=scale_max,
                            variable=self.env_params[param], orient=tk.HORIZONTAL,
                            command=lambda x, p=param: self.on_param_change(p))
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
            
            setattr(self, f"{param}_scale", scale)
            
            # Value label
            value_label = ttk.Label(param_frame, text=f"{self.env_params[param].get():.1f}", width=5, font=('Arial', 8))
            value_label.pack(side=tk.RIGHT)
            
            # Store reference for updates
            setattr(self, f"{param}_value_label", value_label)
            
            row += 1
        
        # Set to Default button
        ttk.Button(control_frame, text="🔄 Set to Default", 
                  command=self.set_to_default).grid(row=row, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))
        
        row += 1
        
        # ADD EMPTY SPACER
        spacer = ttk.Label(control_frame, text="")
        spacer.grid(row=row, column=0, columnspan=2, pady=10)
                
        # Add this after the parameters section
        ttk.Label(control_frame, text="Final date of growth (choose a date):", 
                font=('Arial', 9, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 4))
        
        row += 1 
        # Compact date entry with calendar popup
        # To get the selected dat simply call self.calendar.get_date()
        # self.date_entry
        self.calendar = Calendar(control_frame, selectmode='day', 
                                year=2025, month=8, day=1,
                                font=('Arial', 8))
        self.calendar.grid(row=row, column=0, columnspan=2, pady=2)

        row += 1
        
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
    
    def disable_parameter(self, param):
        scale = getattr(self, f"{param}_scale", None)
        if scale:
            scale.configure(state='disabled')

    def enable_parameter(self, param):
        scale = getattr(self, f"{param}_scale", None)
        if scale:
            scale.configure(state='normal')
            
    def setup_visualization_panel(self, parent):
        viz_frame = ttk.LabelFrame(parent, text="Plant Visualization", padding="6")
        viz_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=3)
        
        # Notebook for different views
        notebook = ttk.Notebook(viz_frame)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Plant Before Growth tab
        self.setup_before_growth_tab(notebook)
        
        # Plant Evolution tab
        self.setup_plant_evolution_tab(notebook)
        
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
    def setup_before_growth_tab(self, notebook):
        """Tab showing the plant before growth starts"""
        before_frame = ttk.Frame(notebook)
        notebook.add(before_frame, text="🌱 Initial Plant")
        
        # Create a frame for the initial plant display
        display_frame = ttk.Frame(before_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Title for initial state
        title_label = ttk.Label(display_frame, text="Plant Initial State", 
                               font=('Arial', 11, 'bold'))
        title_label.pack(pady=(0, 8))
        
        # Frame for the plant image
        self.initial_plant_frame = ttk.Frame(display_frame)
        self.initial_plant_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial plant image label
        self.initial_plant_label = ttk.Label(self.initial_plant_frame, 
                                           text="Initial plant state will appear here",
                                           font=('Arial', 9))
        self.initial_plant_label.pack(expand=True)
        
        # Plant info display
        info_frame = ttk.LabelFrame(display_frame, text="Plant Information", padding="4")
        info_frame.pack(fill=tk.X, pady=(8, 0))
        
        self.plant_info_text = tk.Text(info_frame, height=8, width=35, wrap=tk.WORD, 
                                      font=('Arial', 8), bg="#000000", fg="white")
        self.plant_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Submit button
        submit_frame = ttk.Frame(display_frame)
        submit_frame.pack(fill=tk.X, pady=(8, 0))
        
        ttk.Button(submit_frame, text="📤 Submit Plant Information & Photo", 
                  command=self.submit_plant_data).pack(fill=tk.X)
        
    def setup_plant_evolution_tab(self, notebook):
        """Evolution tab with single image display"""
        evolution_frame = ttk.Frame(notebook)
        notebook.add(evolution_frame, text="🌿 Growth Evolution")
        
        # Create main container for image display
        self.image_display_frame = ttk.Frame(evolution_frame)
        self.image_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create label for image or fallback text
        self.evolution_image_label = ttk.Label(
            self.image_display_frame, 
            text="Plant state after the prediction will appear here",
            font=('Arial', 12),
            foreground='gray',
            anchor='center'
        )
        self.evolution_image_label.pack(expand=True)

    def update_evolution_image(self, filename=None):
        """Update the evolution tab with an image from file or show fallback text"""
        if filename and os.path.exists(filename):
            try:
                print(filename)
                # Open and resize image if needed
                pil_image = Image.open(filename)
                # Optional: resize to fit the display area
                pil_image = pil_image.resize((400, 300), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage for tkinter
                photo_image = ImageTk.PhotoImage(pil_image)
                
                # Display the image
                self.evolution_image_label.config(image=photo_image, text="")
                self.evolution_image_label.image = photo_image  # Keep reference to prevent garbage collection
                
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                # Show fallback text on error
                self.evolution_image_label.config(
                    image="", 
                    text="Plant state after the prediction will appear here"
                )
        else:
            # Show fallback text when no filename or file doesn't exist
            self.evolution_image_label.config(
                image="", 
                text="Plant state after the prediction will appear here"
            )
            
    def setup_results_panel(self, parent):
        results_frame = ttk.LabelFrame(parent, text="Growth Prediction", padding="6")
        results_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(6, 0))
        
        # Prediction results only (no system messages)
        ttk.Label(results_frame, text="Forecast Results:", 
                 font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
        
        self.results_text = tk.Text(results_frame, height=10, width=26, wrap=tk.WORD, font=('Arial', 8), state='disabled')
        self.results_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        img = Image.open("public/TransparentFlower.png")
        
        # Resize if needed
        img = img.resize((180, 290), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        static_image_label = ttk.Label(results_frame, image=photo)
        static_image_label.image = photo  # Keep reference
        static_image_label.grid(row=2, column=0, pady=(6, 0))
        
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
    def on_mode_change(self):
        #KEY FOCUS IS TO IMPLEMENT THIS PART
        """Handle mode changes"""
        current_mode = self.ambient_mode.get()
        
        if current_mode == "controlled":
            print("Switched to Controlled mode")
            # Enable all parameter controls
            self.enable_parameter("humidity")
            self.enable_parameter("brightness")
            self.enable_parameter("temperature")
            
            # No need to call the meteo api
            
        elif current_mode == "open":
            print("Switched to Open mode")
            # Disable most parameter controls (temp, humidity, light)
            self.disable_parameter("humidity")
            self.disable_parameter("brightness")
            self.disable_parameter("temperature")
            
    def on_param_change(self, param):
        value = self.env_params[param].get()
        
        # Update value label
        value_label = getattr(self, f"{param}_value_label")
        if param == 'soil_acidity':
            value_label.config(text=f"{value:.1f}")
        elif param in ['brightness', 'pressure', 'co2']:
            value_label.config(text=f"{value:.0f}")
        else:
            value_label.config(text=f"{value:.1f}")
        
    def set_to_default(self):
        """Reset all parameters to default values"""
        for param_name, default_value in self.default_params.items():
            self.env_params[param_name].set(default_value)
            self.update_parameter_label(param_name, default_value)
            
        self.ambient_mode.set("controlled")
        
        self.calendar.selection_set(date.today())
    
    def update_parameter_label(self, param, value):
        """Update the value label for a specific parameter"""
        try:
            # Get the value label for this parameter
            value_label = getattr(self, f"{param}_value_label")
            
            # Format the value based on parameter type
            if param == 'soil_acidity':
                value_label.config(text=f"{value:.1f}")
            elif param in ['brightness', 'pressure', 'co2']:
                value_label.config(text=f"{value:.0f}")
            else:
                value_label.config(text=f"{value:.1f}")
        except AttributeError:
            # Handle case where label doesn't exist
            print(f"Warning: No label found for parameter {param}")
            
    def load_baseline_image(self):
        self.results_text.delete(1.0, tk.END)
        self.set_to_default()
        
        file_path = filedialog.askopenfilename(
            title="Select baseline plant image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.baseline_image_path = file_path
            self.update_initial_plant_display()
                
    def submit_plant_data(self):
        """Submit plant information and photo"""
        try:
            start_date = datetime.now().date()
        
            # Fix: Convert calendar date string to date object
            calendar_date = self.calendar.get_date()
            if isinstance(calendar_date, str):
                # Parse the string date (assuming format like "2025-08-02" or "02/08/2025")
                try:
                    if '/' in calendar_date:
                        # Handle DD/MM/YYYY format
                        end_date = datetime.strptime(calendar_date, '%d/%m/%Y').date()
                    else:
                        # Handle YYYY-MM-DD format
                        end_date = datetime.strptime(calendar_date, '%Y-%m-%d').date()
                except ValueError:
                    # Fallback: try different formats
                    for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y']:
                        try:
                            end_date = datetime.strptime(calendar_date, fmt).date()
                            break
                        except ValueError:
                            continue
                    else:
                        # If all formats fail, use today
                        end_date = datetime.now().date()
            else:
                # It's already a date object
                end_date = calendar_date
            
            time_lapse = end_date - start_date
            days_difference = time_lapse.days
            
            params = {param: var.get() for param, var in self.env_params.items()}
            params['plant_type'] = self.current_plant
            params['ambient_mode'] = self.ambient_mode.get()
                
            current_mode = self.ambient_mode.get()
            happy_data = None  # Initialize to None instead of 0
            
            if current_mode == "open":
                happy_data = self.happyMeteo.openMeteoCall(days_difference)
                
                # Filter out excluded parameters for open mode
                excluded_params = {"humidity", "temperature", "brightness"}
                params = {param: var.get() for param, var in self.env_params.items() 
                        if param not in excluded_params}
                # Re-add the metadata
                params['plant_type'] = self.current_plant
                params['ambient_mode'] = self.ambient_mode.get()
            
            # Create submission data
            submission_data = {
                'timestamp': datetime.now().isoformat(),
                'parameters': params,
                'baseline_image_path': self.baseline_image_path,
                'plant_info': self.plant_info_text.get(1.0, tk.END),
                'start_date': start_date.isoformat(),  # Fixed: was 'start date' (space)
                'end_date': end_date.isoformat(),
                'time_lapse_days': days_difference  # Added time lapse info
            }
            
            if current_mode == "open" and happy_data is not None:
                submission_data['meteoForecast'] = happy_data
            
            # Clear plant_info_text
            self.plant_info_text.delete(1.0, tk.END)
            
            # Save submission data
            data_dir = "../data"
            os.makedirs(data_dir, exist_ok=True)
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f"{current_date}-{current_date}.txt"
            filepath = os.path.join(data_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(submission_data, f, indent=4)
            
            # Here call the bot pipeline to store results on files in plant_data
            # results are in the form of (text, image)
            results = None
            
            if results is not None:  # Fixed: changed != None to is not None
                text = getattr(results, 'text', None)
                image_filename = getattr(results, 'image', None)
            else:
                text = "<<<----Here at your left you can see the results of the growth of the plant!"
                image_filename = self.filenames[self.counter] # Fixed: removed leading slash
                self.counter += 1
            
            # Create plant_data directory
            images_dir = "./plant_data"
            os.makedirs(images_dir, exist_ok=True)  # Fixed: was data_dir instead of images_dir
            
            image_path = f"public/{image_filename.split('/')[-1]}"

            # Update UI with results
            self.updating_evolution_and_forecasts(text, image_path)
            
            # Here update the informations in the last box from plant_data/texts
            # TODO: Implement reading from plant_data/texts
            
            # Here update the informations in growth evolution from plant_data/images  
            # TODO: Implement reading from plant_data/images
            
            # Show success message with better formatting
            messagebox.showinfo("Submission Successful", 
                            "Submission successful!\n\n"
                            "Go to Growth Evolution tab to see the results.")
            
            print(f"Submission data saved to: {filepath}")
            
        except Exception as e:
            messagebox.showerror("Submission Error", f"Error submitting data: {str(e)}")
            print(f"Error details: {e}")  # For debugging
        
    def updating_evolution_and_forecasts(self, text, image_path):
        self.results_text.config(state='normal')        # Enable editing
        self.results_text.delete(1.0, tk.END)           # Clear existing content
        if text != None:
            self.results_text.insert(1.0, text)         # Insert new text
        self.results_text.config(state='disabled')      # Disable editing again
        
        self.update_evolution_image(image_path)
        
    def update_initial_plant_display(self):
        """Update the initial plant state display"""
        try:
            initial_image = None
            # Generate initial plant image (stage 0)
            try:
                if self.baseline_image_path != None and os.path.exists(self.baseline_image_path):
                    initial_image = Image.open(self.baseline_image_path)
            except Exception as e:
                print(f"Error loading image from {self.baseline_image_path}: {e}")
            
            # Resize image to fit better in square layout
            
            if initial_image != None:
                initial_image = initial_image.resize((280, 210), Image.Resampling.LANCZOS)
            
                # Convert to PhotoImage and display
                photo = ImageTk.PhotoImage(initial_image)
                self.initial_plant_label.configure(image=photo, text="")
                self.initial_plant_label.image = photo  # Keep reference
            
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not generate initial plant image: {str(e)}")

    def update_results_display(self, prediction):
        self.results_text.delete(1.0, tk.END)
        
        #Here update the results display after submitting the photo and the message with the parameters and receveing the output
        results = ""
        
        self.results_text.insert(1.0, results)

def main():
    root = tk.Tk()
    app = PlantGrowthDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()
