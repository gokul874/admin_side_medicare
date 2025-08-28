import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data cleaning and preprocessing for member and provider datasets"""
    
    def __init__(self):
        self.required_member_columns = ['MemberID', 'SourceType', 'Latitude', 'Longitude', 'cost']
        self.required_provider_columns = ['ProviderID', 'Source', 'Location', 'Type', 'Latitude', 'Longitude', 'CMS Rating', 'Cost']
    
    def clean_cost_column(self, cost_series):
        """Clean cost column by removing $ signs and converting to numeric"""
        try:
            # Handle string costs with $ signs
            if cost_series.dtype == 'object':
                # Remove $ signs and any other non-numeric characters except decimal points
                cleaned = cost_series.astype(str).str.replace(r'[$,]', '', regex=True)
                # Convert to numeric, invalid values become NaN
                return pd.to_numeric(cleaned, errors='coerce')
            else:
                return pd.to_numeric(cost_series, errors='coerce')
        except Exception as e:
            logger.warning(f"Error cleaning cost column: {str(e)}")
            return cost_series
    
    def validate_coordinates(self, lat_series, lon_series):
        """Validate latitude and longitude values"""
        errors = []
        
        # Convert to numeric if they're not already
        lat_numeric = pd.to_numeric(lat_series, errors='coerce')
        lon_numeric = pd.to_numeric(lon_series, errors='coerce')
        
        # Check for NaN values
        lat_nan_count = lat_numeric.isna().sum()
        lon_nan_count = lon_numeric.isna().sum()
        
        if lat_nan_count > 0:
            errors.append(f"{lat_nan_count} invalid latitude values found")
        if lon_nan_count > 0:
            errors.append(f"{lon_nan_count} invalid longitude values found")
        
        # Check coordinate ranges
        invalid_lat = ((lat_numeric < -90) | (lat_numeric > 90)).sum()
        invalid_lon = ((lon_numeric < -180) | (lon_numeric > 180)).sum()
        
        if invalid_lat > 0:
            errors.append(f"{invalid_lat} latitude values outside valid range (-90 to 90)")
        if invalid_lon > 0:
            errors.append(f"{invalid_lon} longitude values outside valid range (-180 to 180)")
        
        return lat_numeric, lon_numeric, errors
    
    def process_members_data(self, filepath):
        """Process and validate members dataset"""
        errors = []
        
        try:
            # Read CSV file
            df = pd.read_csv(filepath)
            logger.info(f"Loaded members dataset with {len(df)} records")
            
            # Check required columns
            missing_columns = [col for col in self.required_member_columns if col not in df.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {', '.join(missing_columns)}")
                return None, errors
            
            # Clean cost column
            df['cost'] = self.clean_cost_column(df['cost'])
            
            # Validate coordinates
            df['Latitude'], df['Longitude'], coord_errors = self.validate_coordinates(df['Latitude'], df['Longitude'])
            errors.extend(coord_errors)
            
            # Check for missing values in critical columns
            for col in self.required_member_columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    if col in ['Latitude', 'Longitude', 'cost']:
                        # These are critical - remove rows with missing values
                        df = df.dropna(subset=[col])
                        logger.warning(f"Removed {missing_count} rows with missing {col}")
                    else:
                        errors.append(f"{missing_count} missing values in {col}")
            
            # Validate SourceType values
            valid_source_types = ['Hospital', 'Nursing Home', 'Scan Center', 'Supply Directory']
            invalid_source_types = df[~df['SourceType'].isin(valid_source_types)]['SourceType'].unique()
            if len(invalid_source_types) > 0:
                logger.warning(f"Found invalid source types: {invalid_source_types}")
                # Keep them but log as warning
            
            # Ensure MemberID is unique
            duplicate_ids = df['MemberID'].duplicated().sum()
            if duplicate_ids > 0:
                errors.append(f"{duplicate_ids} duplicate Member IDs found")
                df = df.drop_duplicates(subset=['MemberID'], keep='first')
            
            logger.info(f"Processed members dataset: {len(df)} valid records")
            return df, errors
            
        except Exception as e:
            error_msg = f"Error processing members dataset: {str(e)}"
            logger.error(error_msg)
            return None, [error_msg]
    
    def process_providers_data(self, filepath):
        """Process and validate providers dataset"""
        errors = []
        
        try:
            # Read CSV file
            df = pd.read_csv(filepath)
            logger.info(f"Loaded providers dataset with {len(df)} records")
            
            # Check required columns
            missing_columns = [col for col in self.required_provider_columns if col not in df.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {', '.join(missing_columns)}")
                return None, errors
            
            # Clean cost column
            df['Cost'] = self.clean_cost_column(df['Cost'])
            
            # Validate coordinates
            df['Latitude'], df['Longitude'], coord_errors = self.validate_coordinates(df['Latitude'], df['Longitude'])
            errors.extend(coord_errors)
            
            # Validate CMS Rating (should be 1-5)
            df['CMS Rating'] = pd.to_numeric(df['CMS Rating'], errors='coerce')
            invalid_ratings = ((df['CMS Rating'] < 1) | (df['CMS Rating'] > 5)).sum()
            if invalid_ratings > 0:
                logger.warning(f"{invalid_ratings} invalid CMS ratings found (should be 1-5)")
                # Clip ratings to valid range
                df['CMS Rating'] = df['CMS Rating'].clip(1, 5)
            
            # Check for missing values in critical columns
            critical_columns = ['ProviderID', 'Latitude', 'Longitude', 'Cost', 'CMS Rating']
            for col in critical_columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    df = df.dropna(subset=[col])
                    logger.warning(f"Removed {missing_count} rows with missing {col}")
            
            # Ensure ProviderID is unique
            duplicate_ids = df['ProviderID'].duplicated().sum()
            if duplicate_ids > 0:
                errors.append(f"{duplicate_ids} duplicate Provider IDs found")
                df = df.drop_duplicates(subset=['ProviderID'], keep='first')
            
            # Clean and validate provider types
            df['Type'] = df['Type'].fillna('Unknown')
            
            logger.info(f"Processed providers dataset: {len(df)} valid records")
            return df, errors
            
        except Exception as e:
            error_msg = f"Error processing providers dataset: {str(e)}"
            logger.error(error_msg)
            return None, [error_msg]
    
    def get_dataset_summary(self, df, dataset_type):
        """Generate summary statistics for a dataset"""
        summary = {
            'total_records': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        if dataset_type == 'members':
            summary['source_types'] = df['SourceType'].value_counts().to_dict()
            summary['cost_stats'] = {
                'min': float(df['cost'].min()),
                'max': float(df['cost'].max()),
                'mean': float(df['cost'].mean()),
                'median': float(df['cost'].median())
            }
        elif dataset_type == 'providers':
            summary['provider_types'] = df['Type'].value_counts().to_dict()
            summary['cost_stats'] = {
                'min': float(df['Cost'].min()),
                'max': float(df['Cost'].max()),
                'mean': float(df['Cost'].mean()),
                'median': float(df['Cost'].median())
            }
            summary['rating_distribution'] = df['CMS Rating'].value_counts().sort_index().to_dict()
        
        return summary
