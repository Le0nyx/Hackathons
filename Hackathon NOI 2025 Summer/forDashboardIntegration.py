from datetime import datetime
from script import PlantPredictor

def dashboard_plant_prediction(image_path, start_date, end_date, additional_notes=""):
    """
    Simple function for dashboard integration
    """
    try:
        # Calculate days
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end_dt - start_dt).days
        
        if days <= 0:
            return {"success": False, "error": "Invalid date range"}
        
        # Create predictor and run
        predictor = PlantPredictor()
        result = predictor.dashboard_plant_prediction(image_path, days, additional_notes)
        
        if result:
            return {"success": True, "result": result}
        else:
            return {"success": False, "error": "No result"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


# Test
if __name__ == "__main__":
    result = dashboard_plant_prediction(
        "./basilico.jpg", 
        "2024-08-01", 
        "2024-08-08",
        "Test plant"
    )
    
    if result["success"]:
        print(" SUCCESS!")
    else:
        print(f" ERROR: {result['error']}")