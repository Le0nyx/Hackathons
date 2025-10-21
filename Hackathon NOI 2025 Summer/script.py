from datetime import datetime
import sys
import os

class PlantPredictor:
    def dashboard_plant_prediction(
        image_path: str,
        start_date: str,
        end_date: str,
        additional_notes: str = ""
    ) -> dict:
        try:
            # Calcola giorni
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            days = (end_dt - start_dt).days
            if days <= 0:
                return {"success": False, "error": "End date must be after start date", "days": days}

            # Log
            print(f"Dashboard prediction request: {start_date} to {end_date} ({days} days) image={image_path}")
            if additional_notes:
                print(f"Notes: {additional_notes}")

            # Inizializza il predictor e chiama il metodo
            predictor = PlantPredictor()
            result = predictor.predict_plant_growth(image_path, days, additional_notes)

            # Unwrap risultato tuple
            if isinstance(result, tuple) and len(result) == 5:
                _img, conditions, weather_df, plant_type, plant_health = result
                return {
                    "success": True,
                    "plant_analysis": {"plant_type": plant_type, "plant_health": plant_health},
                    "weather_conditions": conditions,
                    "weather_data_shape": weather_df.shape,
                    "parameters_used": {"start_date": start_date, "end_date": end_date, "days": days, "notes": additional_notes, "image": image_path},
                    "prediction_summary": {
                        "temperature_range": f"{conditions['avg_temp_min']}–{conditions['avg_temp_max']}°C",
                        "total_rain": f"{conditions['total_rain']}mm",
                        "sunshine_hours": f"{conditions['total_sunshine_hours']}h"
                    }
                }
            else:
                return {"success": False, "error": "Invalid result from PlantPredictor", "result": result}

        except ValueError as e:
            return {"success": False, "error": f"Date format error: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {e}"}

# Esempio di test
if __name__ == '__main__':
    res = dashboard_plant_prediction(
        image_path='./basilico.jpg',
        start_date='2024-08-01',
        end_date='2024-08-08',
        additional_notes='Indoor day 3'
    )
    print(res)
