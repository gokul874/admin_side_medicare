import math
import numpy as np
import logging

logger = logging.getLogger(__name__)

class GeospatialAnalyzer:
    """Handles geospatial calculations and analysis"""
    
    def __init__(self):
        self.earth_radius_km = 6371.0  # Earth's radius in kilometers
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees) using the Haversine formula
        Returns distance in kilometers
        """
        try:
            # Convert decimal degrees to radians
            lat1_rad = math.radians(lat1)
            lon1_rad = math.radians(lon1)
            lat2_rad = math.radians(lat2)
            lon2_rad = math.radians(lon2)
            
            # Haversine formula
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = (math.sin(dlat/2)**2 + 
                 math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
            c = 2 * math.asin(math.sqrt(a))
            
            # Distance in kilometers
            distance = self.earth_radius_km * c
            return distance
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Error calculating distance: {str(e)}")
            return float('inf')  # Return infinite distance for invalid coordinates
    
    def haversine_distance_vectorized(self, lat1, lon1, lat2_array, lon2_array):
        """
        Vectorized version of haversine distance calculation
        Calculate distance from one point to multiple points
        """
        try:
            # Convert to numpy arrays for vectorized operations
            lat2_array = np.array(lat2_array)
            lon2_array = np.array(lon2_array)
            
            # Convert decimal degrees to radians
            lat1_rad = math.radians(lat1)
            lon1_rad = math.radians(lon1)
            lat2_rad = np.radians(lat2_array)
            lon2_rad = np.radians(lon2_array)
            
            # Haversine formula
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = (np.sin(dlat/2)**2 + 
                 math.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(a))
            
            # Distance in kilometers
            distances = self.earth_radius_km * c
            return distances
            
        except Exception as e:
            logger.warning(f"Error in vectorized distance calculation: {str(e)}")
            return np.full(len(lat2_array), float('inf'))
    
    def find_providers_within_radius(self, member_lat, member_lon, providers_df, radius_km=15.0):
        """
        Find all providers within a specified radius of a member
        Returns filtered providers dataframe with distance column
        """
        try:
            # Calculate distances to all providers
            distances = self.haversine_distance_vectorized(
                member_lat, member_lon,
                providers_df['Latitude'].values,
                providers_df['Longitude'].values
            )
            
            # Filter providers within radius
            within_radius_mask = distances <= radius_km
            nearby_providers = providers_df[within_radius_mask].copy()
            nearby_providers['distance_km'] = distances[within_radius_mask]
            
            return nearby_providers.sort_values('distance_km')
            
        except Exception as e:
            logger.error(f"Error finding providers within radius: {str(e)}")
            return providers_df.iloc[0:0].copy()  # Return empty dataframe with same structure
    
    def calculate_coverage_areas(self, providers_df, radius_km=15.0):
        """
        Calculate theoretical coverage areas for providers
        Returns dictionary with provider coverage statistics
        """
        coverage_stats = {}
        
        try:
            total_coverage_area = 0
            for _, provider in providers_df.iterrows():
                # Each provider covers a circular area with radius_km
                coverage_area = math.pi * (radius_km ** 2)
                total_coverage_area += coverage_area
                
                coverage_stats[provider['ProviderID']] = {
                    'coverage_area_sq_km': coverage_area,
                    'latitude': provider['Latitude'],
                    'longitude': provider['Longitude'],
                    'type': provider['Type'],
                    'rating': provider['CMS Rating']
                }
            
            coverage_stats['total_theoretical_coverage'] = total_coverage_area
            coverage_stats['average_coverage_per_provider'] = coverage_area  # Same for all providers
            
            return coverage_stats
            
        except Exception as e:
            logger.error(f"Error calculating coverage areas: {str(e)}")
            return {}
    
    def analyze_geographic_distribution(self, df, entity_type='members'):
        """
        Analyze the geographic distribution of members or providers
        """
        try:
            lat_col = 'Latitude'
            lon_col = 'Longitude'
            
            analysis = {
                'total_count': len(df),
                'coordinates': {
                    'center_lat': float(df[lat_col].mean()),
                    'center_lon': float(df[lon_col].mean()),
                    'min_lat': float(df[lat_col].min()),
                    'max_lat': float(df[lat_col].max()),
                    'min_lon': float(df[lon_col].min()),
                    'max_lon': float(df[lon_col].max()),
                    'lat_range': float(df[lat_col].max() - df[lat_col].min()),
                    'lon_range': float(df[lon_col].max() - df[lon_col].min())
                }
            }
            
            # Calculate approximate geographic span
            max_distance = self.haversine_distance(
                analysis['coordinates']['min_lat'], analysis['coordinates']['min_lon'],
                analysis['coordinates']['max_lat'], analysis['coordinates']['max_lon']
            )
            analysis['max_geographic_span_km'] = max_distance
            
            # Add entity-specific analysis
            if entity_type == 'members':
                if 'SourceType' in df.columns:
                    analysis['source_type_distribution'] = df['SourceType'].value_counts().to_dict()
            elif entity_type == 'providers':
                if 'Type' in df.columns:
                    analysis['provider_type_distribution'] = df['Type'].value_counts().to_dict()
                if 'CMS Rating' in df.columns:
                    analysis['rating_distribution'] = df['CMS Rating'].value_counts().sort_index().to_dict()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing geographic distribution: {str(e)}")
            return {'error': str(e)}
    
    def check_coordinate_validity(self, lat, lon):
        """Check if coordinates are valid"""
        try:
            lat = float(lat)
            lon = float(lon)
            
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return True, None
            else:
                return False, f"Coordinates out of range: lat={lat}, lon={lon}"
                
        except (ValueError, TypeError):
            return False, f"Invalid coordinate format: lat={lat}, lon={lon}"
