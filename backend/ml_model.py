"""
Machine Learning Model for Ghost Net Accumulation Zone Prediction
Uses scikit-learn with Random Forest and Gradient Boosting
Includes model training, persistence, and validation
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json


class GhostNetMLModel:
    """
    Machine Learning model for predicting ghost net accumulation zones.
    
    Features:
    - Historical sighting density
    - Fishing ground proximity
    - Ocean current convergence
    - Wind patterns
    - Gyre zone proximity
    - Ocean conditions (temperature, wave height, etc.)
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.model_version = "1.0.0"
        self.feature_names = [
            'sighting_density',
            'fishing_proximity',
            'current_convergence',
            'gyre_proximity',
            'wind_speed',
            'wind_direction_sin',
            'wind_direction_cos',
            'current_speed',
            'sea_surface_temp',
            'wave_height',
            'distance_to_shore',
            'bathymetry_depth'  # Would need bathymetry data
        ]
        
        self.training_history = []
        self.is_trained = False
    
    def create_features(
        self,
        sighting_density: float,
        fishing_proximity: float,
        current_convergence: float,
        gyre_proximity: float,
        wind_speed: float,
        wind_direction: float,
        current_speed: float,
        sea_surface_temp: float,
        wave_height: float,
        distance_to_shore: float = 0.0,
        bathymetry_depth: float = 0.0
    ) -> np.ndarray:
        """Create feature vector from input data"""
        # Convert wind direction to sin/cos for cyclical encoding
        wind_dir_rad = np.radians(wind_direction)
        
        features = np.array([
            sighting_density,
            fishing_proximity,
            current_convergence,
            gyre_proximity,
            wind_speed,
            np.sin(wind_dir_rad),
            np.cos(wind_dir_rad),
            current_speed,
            sea_surface_temp,
            wave_height,
            distance_to_shore,
            bathymetry_depth
        ])
        
        return features.reshape(1, -1)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 10
    ) -> Dict[str, float]:
        """
        Train the ML model on provided data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,) - confidence scores
            test_size: Proportion of data for testing
            random_state: Random seed
            n_estimators: Number of trees in ensemble
            max_depth: Maximum tree depth
        
        Returns:
            Dictionary with training metrics
        """
        # Remove or impute NaN values (GradientBoostingRegressor doesn't accept NaN)
        # Drop rows with NaN in y to avoid issues
        nan_mask = np.isnan(y)
        if np.any(nan_mask):
            X = X[~nan_mask]
            y = y[~nan_mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Impute NaN in features (e.g., from missing ocean API data)
        X_train = self.imputer.fit_transform(X_train)
        X_test = self.imputer.transform(X_test)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Use Gradient Boosting for better performance
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=random_state,
            loss='squared_error'
        )
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        metrics = {
            "train_mse": float(train_mse),
            "test_mse": float(test_mse),
            "train_mae": float(train_mae),
            "test_mae": float(test_mae),
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "model_version": self.model_version,
            "training_date": datetime.utcnow().isoformat()
        }
        
        self.training_history.append(metrics)
        self.is_trained = True
        
        return metrics
    
    def predict(self, features: np.ndarray) -> float:
        """
        Predict confidence score for a location.
        
        Args:
            features: Feature vector (1, n_features)
        
        Returns:
            Predicted confidence score (0-100)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Impute NaN and scale features
        features = self.imputer.transform(features)
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        # Clamp to reasonable range
        prediction = max(0, min(100, prediction))
        
        return float(prediction)
    
    def predict_batch(self, features_list: List[np.ndarray]) -> np.ndarray:
        """Predict for multiple locations at once"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Stack features
        X = np.vstack(features_list)
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Clamp to reasonable range
        predictions = np.clip(predictions, 0, 100)
        
        return predictions
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save model and scaler to disk.
        
        Returns:
            Path to saved model file
        """
        if filename is None:
            filename = f"ghostnet_model_v{self.model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        filepath = os.path.join(self.model_dir, filename)
        
        model_data = {
            'model': self.model,
            'imputer': self.imputer,
            'scaler': self.scaler,
            'model_version': self.model_version,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'saved_at': datetime.utcnow().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        
        return filepath
    
    def load_model(self, filepath: str) -> bool:
        """
        Load model and scaler from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.imputer = model_data.get('imputer')
            if self.imputer is None:
                # Old model without imputer - create and fit so transform won't fail
                n_features = len(self.feature_names)
                self.imputer = SimpleImputer(strategy='constant', fill_value=0)
                self.imputer.fit(np.zeros((1, n_features)))
            self.scaler = model_data['scaler']
            self.model_version = model_data.get('model_version', '1.0.0')
            self.feature_names = model_data.get('feature_names', self.feature_names)
            self.training_history = model_data.get('training_history', [])
            self.is_trained = model_data.get('is_trained', False)
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from trained model"""
        if not self.is_trained or self.model is None:
            return {}
        
        importances = self.model.feature_importances_
        
        return dict(zip(self.feature_names, importances.tolist()))
    
    def generate_training_data(
        self,
        historical_sightings: List,
        fishing_grounds: List[Dict],
        gyre_zones: List[Dict],
        ocean_conditions_func
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data from historical sightings.
        Creates positive examples (where nets were found) and negative examples.
        """
        X = []
        y = []
        
        # Positive examples: locations where nets were actually found
        for sighting in historical_sightings:
            lat = sighting.location.lat if hasattr(sighting.location, 'lat') else sighting['location']['lat']
            lon = sighting.location.lon if hasattr(sighting.location, 'lon') else sighting['location']['lon']
            
            # Calculate features
            sighting_density = self._calculate_sighting_density(lat, lon, historical_sightings, 25.0)
            fishing_proximity = self._calculate_fishing_proximity(lat, lon, fishing_grounds)
            current_convergence = self._estimate_current_convergence(lat, lon)
            gyre_proximity = self._calculate_gyre_proximity(lat, lon, gyre_zones)
            
            # Get ocean conditions - use safe fallbacks for NaN/None from API
            ocean_data = ocean_conditions_func(lat, lon) or {}
            wind_speed = ocean_data.get('wind_speed', 5.0) or 5.0
            wind_direction = ocean_data.get('wind_direction', 180.0) or 180.0
            current_speed = ocean_data.get('current_speed', 0.5) or 0.5
            sst = ocean_data.get('sea_surface_temp')
            sea_surface_temp = 15.0 if (sst is None or (isinstance(sst, float) and np.isnan(sst))) else float(sst)
            wh = ocean_data.get('wave_height', 1.5)
            wave_height = 1.5 if (wh is None or (isinstance(wh, (int, float)) and np.isnan(wh))) else float(wh)
            
            features = self.create_features(
                sighting_density=sighting_density,
                fishing_proximity=fishing_proximity,
                current_convergence=current_convergence,
                gyre_proximity=gyre_proximity,
                wind_speed=wind_speed,
                wind_direction=wind_direction,
                current_speed=current_speed,
                sea_surface_temp=sea_surface_temp,
                wave_height=wave_height
            )
            
            X.append(features[0])
            # High confidence for actual sightings
            y.append(75.0 + np.random.uniform(0, 20))  # 75-95 confidence
        
        # Negative examples: random locations where no nets were found
        n_negative = len(historical_sightings) * 2
        
        for _ in range(n_negative):
            # Random location in California coast region
            lat = np.random.uniform(32.0, 40.0)
            lon = np.random.uniform(-125.0, -117.0)
            
            # Check if near any sighting (if so, skip)
            near_sighting = False
            for sighting in historical_sightings:
                sighting_lat = sighting.location.lat if hasattr(sighting.location, 'lat') else sighting['location']['lat']
                sighting_lon = sighting.location.lon if hasattr(sighting.location, 'lon') else sighting['location']['lon']
                dist = self._haversine_distance(lat, lon, sighting_lat, sighting_lon)
                if dist < 10.0:  # Within 10km
                    near_sighting = True
                    break
            
            if near_sighting:
                continue
            
            # Calculate features
            sighting_density = self._calculate_sighting_density(lat, lon, historical_sightings, 25.0)
            fishing_proximity = self._calculate_fishing_proximity(lat, lon, fishing_grounds)
            current_convergence = self._estimate_current_convergence(lat, lon)
            gyre_proximity = self._calculate_gyre_proximity(lat, lon, gyre_zones)
            
            ocean_data = ocean_conditions_func(lat, lon) or {}
            wind_speed = ocean_data.get('wind_speed', 5.0) or 5.0
            wind_direction = ocean_data.get('wind_direction', 180.0) or 180.0
            current_speed = ocean_data.get('current_speed', 0.5) or 0.5
            sst = ocean_data.get('sea_surface_temp')
            sea_surface_temp = 15.0 if (sst is None or (isinstance(sst, float) and np.isnan(sst))) else float(sst)
            wh = ocean_data.get('wave_height', 1.5)
            wave_height = 1.5 if (wh is None or (isinstance(wh, (int, float)) and np.isnan(wh))) else float(wh)
            
            features = self.create_features(
                sighting_density=sighting_density,
                fishing_proximity=fishing_proximity,
                current_convergence=current_convergence,
                gyre_proximity=gyre_proximity,
                wind_speed=wind_speed,
                wind_direction=wind_direction,
                current_speed=current_speed,
                sea_surface_temp=sea_surface_temp,
                wave_height=wave_height
            )
            
            X.append(features[0])
            # Low confidence for negative examples
            y.append(np.random.uniform(5, 40))
        
        return np.array(X), np.array(y)
    
    def _calculate_sighting_density(self, lat: float, lon: float, sightings: List, radius_km: float) -> float:
        """Helper to calculate sighting density"""
        nearby = 0
        for sighting in sightings:
            sighting_lat = sighting.location.lat if hasattr(sighting.location, 'lat') else sighting['location']['lat']
            sighting_lon = sighting.location.lon if hasattr(sighting.location, 'lon') else sighting['location']['lon']
            dist = self._haversine_distance(lat, lon, sighting_lat, sighting_lon)
            if dist <= radius_km:
                nearby += 1
        area = np.pi * radius_km ** 2
        return (nearby / area) * 1000 if area > 0 else 0
    
    def _calculate_fishing_proximity(self, lat: float, lon: float, fishing_grounds: List[Dict]) -> float:
        """Helper to calculate fishing ground proximity"""
        max_score = 0
        for ground in fishing_grounds:
            center = ground.get('center', (0, 0))
            dist = self._haversine_distance(lat, lon, center[0], center[1])
            radius = ground.get('radius_km', 25)
            if dist <= radius:
                score = ground.get('intensity', 0.5) * (1 - dist / radius)
                max_score = max(max_score, score)
        return max_score
    
    def _calculate_gyre_proximity(self, lat: float, lon: float, gyre_zones: List[Dict]) -> float:
        """Helper to calculate gyre zone proximity"""
        max_score = 0
        for zone in gyre_zones:
            center = zone.get('center', (0, 0))
            dist = self._haversine_distance(lat, lon, center[0], center[1])
            radius = zone.get('radius_km', 200)
            if dist <= radius:
                score = zone.get('intensity', 0.4) * (1 - dist / radius)
                max_score = max(max_score, score)
        return max_score
    
    def _estimate_current_convergence(self, lat: float, lon: float) -> float:
        """Helper to estimate current convergence (simplified)"""
        # Simplified pattern based on latitude/longitude
        lat_factor = abs(np.sin(np.radians(lat * 3)))
        lon_factor = abs(np.cos(np.radians(lon * 2)))
        return (lat_factor + lon_factor) / 2
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Helper to calculate distance"""
        R = 6371
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c


