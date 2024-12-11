import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import zipcodes
import os
from typing import Dict, Any

class PropertyValuePredictor:
    def __init__(self):
        self.combined_data = None
        self.cities = None
        
    def load_data(self):
        """Load and prepare the initial datasets"""
        try:
            # Use raw strings for file paths
            student_data_path = r"C:\Users\jackf\Downloads\tuffrealestate\student_data.csv"
            property_data_path = r"C:\Users\jackf\Downloads\tuffrealestate\property_data.csv"
            
            # Load and merge data
            student_df = pd.read_csv(student_data_path)
            property_df = pd.read_csv(property_data_path)
            self.combined_data = pd.merge(student_df, property_df, 
                                        on=['University', 'City', 'ZIP', 'Central Campus Zip'])
            
            # Get unique cities
            self.cities = sorted(self.combined_data['City'].unique())
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def get_user_inputs(self) -> Dict[str, Any]:
        """Get all necessary inputs from the user"""
        # Display available cities
        print("\nAvailable college towns:")
        for i, city in enumerate(self.cities, 1):
            print(f"{i}. {city}")
        
        # Get city selection
        while True:
            try:
                city_choice = int(input("\nEnter the number of your chosen city: "))
                if 1 <= city_choice <= len(self.cities):
                    selected_city = self.cities[city_choice - 1]
                    break
                print("Invalid selection. Please choose a number from the list.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get university data for selected city
        city_data = self.combined_data[self.combined_data['City'] == selected_city].iloc[0]
        zip_code = city_data['ZIP']
        
        # Get property details
        print(f"\nSelected city: {selected_city} (ZIP: {zip_code})")
        
        while True:
            try:
                distance_to_campus = float(input("Enter distance to campus in miles: "))
                if 0 <= distance_to_campus <= 20:
                    break
                print("Please enter a reasonable distance (0-20 miles).")
            except ValueError:
                print("Please enter a valid number.")
        
        while True:
            try:
                bedrooms = int(input("Enter number of bedrooms (1-5): "))
                if 1 <= bedrooms <= 5:
                    break
                print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")
        
        while True:
            try:
                bathrooms = float(input("Enter number of bathrooms: "))
                if 0.5 <= bathrooms <= 5:
                    break
                print("Please enter a number between 0.5 and 5.")
            except ValueError:
                print("Please enter a valid number.")
        
        while True:
            try:
                square_footage = int(input("Enter square footage: "))
                if 100 <= square_footage <= 10000:
                    break
                print("Please enter a reasonable square footage (100-10000).")
            except ValueError:
                print("Please enter a valid number.")
        
        return {
            'zip_code': zip_code,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'square_footage': square_footage,
            'distance_to_campus': distance_to_campus,
            'city_data': city_data
        }

    def predict_values(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        """Calculate predicted values based on user inputs"""
        # Get base price for the number of bedrooms
        bedroom_col = f'MHP{inputs["bedrooms"]}B'
        base_price = inputs['city_data'][bedroom_col]
        
        # Get average square footage for this bedroom count
        avg_sqft = {
            1: 750,   # Average for 1BR
            2: 1000,  # Average for 2BR
            3: 1500,  # Average for 3BR
            4: 2000,  # Average for 4BR
            5: 2500   # Average for 5BR
        }[inputs['bedrooms']]
        
        # Calculate adjustment factors
        bathroom_factor = 1 + (0.05 * (inputs['bathrooms'] - 1))
        sqft_factor = (inputs['square_footage'] / avg_sqft) ** 0.8
        
        # Distance factor (3% reduction per mile, max 15% reduction)
        distance_factor = max(0.85, 1 - (inputs['distance_to_campus'] * 0.03))
        
        # Calculate final price with all factors
        final_price = base_price * bathroom_factor * sqft_factor * distance_factor
        
        # Calculate monthly rent using P/R ratio
        # Get average rent for this bedroom count in the area
        base_monthly_rent = (base_price / inputs['city_data']['P/R Ratio']) / 12
        
        # Adjust rent based on property specifics
        if inputs['distance_to_campus'] <= 0.6:
            rent_distance_factor = 1 - (inputs['distance_to_campus'] * 0.04)  # Mild reduction
        else:
            rent_distance_factor = 0.96 - (inputs['distance_to_campus'] * 0.15)  # Steeper reduction beyond 0.6 miles

        rent_size_factor = (inputs['square_footage'] / avg_sqft) ** 0.6
        rent_bathroom_factor = 1 + (0.03 * (inputs['bathrooms'] - 1))
        rent_bedroom_factor = 1 + (0.3 * (inputs['bedrooms'] - 1))

        location_premium = 1 + (0.1 if inputs['distance_to_campus'] <= 0.5 else 0)
        
        # Calculate final monthly rent (Multiply base_monthly_rent by potential demand factor(will incorperate later))
        monthly_rent = base_monthly_rent * location_premium * rent_distance_factor * rent_size_factor * rent_bathroom_factor * rent_bedroom_factor
        
        # Calculate 5-year forecast (assuming 3% annual appreciation)
        appreciation_rate = 0.03 - (inputs['distance_to_campus'] * 0.001)
        future_price = final_price * ((1 + appreciation_rate) ** 5)
        
        return {
            'estimated_price': round(final_price, 2),
            'estimated_monthly_rent': round(monthly_rent, 2),
            'forecasted_price_5yr': round(future_price, 2)
        }

    def display_results(self, results: Dict[str, float]):
        """Display the prediction results"""
        print("\n=== Property Value Predictions ===")
        print(f"Estimated Current Value: ${results['estimated_price']:,.2f}")
        print(f"Estimated Monthly Rent: ${results['estimated_monthly_rent']:,.2f}")
        print(f"Forecasted Value (5 years): ${results['forecasted_price_5yr']:,.2f}")

def main():
    try:
        # Initialize predictor
        predictor = PropertyValuePredictor()
        
        # Load initial data
        print("Loading data...")
        predictor.load_data()
        
        # Get user inputs
        inputs = predictor.get_user_inputs()
        
        # Make predictions
        results = predictor.predict_values(inputs)
        
        # Display results
        predictor.display_results(results)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()