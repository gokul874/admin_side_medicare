import pandas as pd
import numpy as np
import logging
from collections import defaultdict
from sklearn.neighbors import BallTree

logger = logging.getLogger(__name__)

class NetworkOptimizer:
    """Handles provider network optimization and assignment logic (BallTree-backed)."""
    
    def __init__(self):
        # kept the same key terms / values as requested
        self.optimization_weights = {
            'rating': 0.5,    # Higher weight for rating
            'cost': 0.3,      # Medium weight for cost (lower is better)
            'distance': 0.2   # Lower weight for distance (shorter is better)
        }
        # Earth radius in kilometers for haversine conversions
        self._earth_radius_km = 6371.0

    def calculate_provider_score(self, rating, cost, distance):
        """
        Calculate optimization score for a provider
        Higher score is better
        """
        try:
            # Normalize rating (1-5 scale, higher is better)
            rating_score = rating / 5.0

            # Normalize cost (inverse, lower cost is better)
            # Use a reasonable cost range for normalization
            max_cost = 1000  # Assume max reasonable cost
            cost_score = max(0, (max_cost - cost) / max_cost)

            # Normalize distance (inverse, shorter distance is better)
            max_distance = 15.0  # Maximum allowed distance
            distance_score = max(0, (max_distance - distance) / max_distance)

            # Calculate weighted score
            total_score = (
                rating_score * self.optimization_weights['rating'] +
                cost_score * self.optimization_weights['cost'] +
                distance_score * self.optimization_weights['distance']
            )

            return total_score

        except Exception as e:
            logger.warning(f"Error calculating provider score: {str(e)}")
            return 0.0

    def find_best_provider(self, member_candidates):
        """
        Find the best provider for a member based on optimization criteria
        Priority: High Rating > Low Cost > Shortest Distance
        """
        if not member_candidates:
            return None

        try:
            # Convert to DataFrame for easier manipulation
            candidates_df = pd.DataFrame(member_candidates)

            # Apply optimization logic in priority order
            # 1. Filter by highest rating first
            max_rating = candidates_df['rating'].max()
            highest_rated = candidates_df[candidates_df['rating'] == max_rating]

            if len(highest_rated) == 1:
                return highest_rated.iloc[0].to_dict()

            # 2. Among highest rated, find lowest cost
            min_cost = highest_rated['cost'].min()
            lowest_cost = highest_rated[highest_rated['cost'] == min_cost]

            if len(lowest_cost) == 1:
                return lowest_cost.iloc[0].to_dict()

            # 3. Among ties, find shortest distance
            min_distance = lowest_cost['distance'].min()
            best_provider = lowest_cost[lowest_cost['distance'] == min_distance].iloc[0]

            return best_provider.to_dict()

        except Exception as e:
            logger.error(f"Error finding best provider: {str(e)}")
            return None

    def find_candidate_connections(self, members_df, providers_df, geospatial=None, max_distance=15.0):
        """
        Efficiently find candidate connections between members and providers using BallTree.
        - members_df: DataFrame with 'Latitude', 'Longitude', 'SourceType', 'MemberID'
        - providers_df: DataFrame with 'Latitude', 'Longitude', 'ProviderID', 'Cost', 'CMS Rating', 'Type', 'Source'
        - geospatial: optional existing geospatial helper. If provided and it implements
                      find_providers_within_radius, that will be used for each member (kept for compatibility).
                      Otherwise, BallTree will be used.
        - max_distance: radius in kilometers (default 15.0)
        Returns: list of candidate connection dicts (same keys as original code)
        """
        candidate_connections = []

        try:
            logger.info(f"Processing {len(members_df)} members against {len(providers_df)} providers")

            # Ensure lat/lon columns exist
            if 'Latitude' not in members_df.columns or 'Longitude' not in members_df.columns:
                raise ValueError("members_df must contain 'Latitude' and 'Longitude' columns")
            if 'Latitude' not in providers_df.columns or 'Longitude' not in providers_df.columns:
                raise ValueError("providers_df must contain 'Latitude' and 'Longitude' columns")

            # Preprocess provider coordinates for BallTree (radians)
            # Fillna and convert to numeric to avoid issues
            providers_df = providers_df.copy()
            providers_df['Latitude'] = pd.to_numeric(providers_df['Latitude'], errors='coerce')
            providers_df['Longitude'] = pd.to_numeric(providers_df['Longitude'], errors='coerce')

            # Drop providers without coordinates (can't query them)
            valid_providers_mask = providers_df['Latitude'].notna() & providers_df['Longitude'].notna()
            if not valid_providers_mask.all():
                missing = (~valid_providers_mask).sum()
                logger.warning(f"Dropping {missing} providers with missing coordinates")
            providers_valid = providers_df[valid_providers_mask].reset_index(drop=True)

            # If there are no valid providers, return empty
            if providers_valid.empty:
                logger.info("No valid providers with coordinates found")
                return []

            # Build BallTree on providers (haversine metric requires radians)
            provider_coords_rad = np.radians(providers_valid[['Latitude', 'Longitude']].values)
            tree = BallTree(provider_coords_rad, metric='haversine')

            # Convert max_distance (km) to radians for query
            radius_radians = max_distance / self._earth_radius_km

            # Batch processing of members for memory safety & logging
            batch_size = 100
            members_df = members_df.copy()
            members_df['Latitude'] = pd.to_numeric(members_df['Latitude'], errors='coerce')
            members_df['Longitude'] = pd.to_numeric(members_df['Longitude'], errors='coerce')

            for i in range(0, len(members_df), batch_size):
                batch_end = min(i + batch_size, len(members_df))
                members_batch = members_df.iloc[i:batch_end]

                # Prepare array of member coords in radians (dropping invalid rows)
                member_coords_list = []
                member_indices = []  # mapping to original batch index
                for idx, member in members_batch.iterrows():
                    if pd.isna(member['Latitude']) or pd.isna(member['Longitude']):
                        logger.debug(f"Skipping member {member.get('MemberID')} due to missing coords")
                        continue
                    member_coords_list.append([member['Latitude'], member['Longitude']])
                    member_indices.append(idx)

                if not member_coords_list:
                    # nothing to query in this batch
                    continue

                member_coords_rad = np.radians(np.array(member_coords_list))

                # If a geospatial helper is provided and has the expected method, use it for compatibility.
                use_geospatial = (
                    geospatial is not None and
                    hasattr(geospatial, 'find_providers_within_radius') and
                    callable(getattr(geospatial, 'find_providers_within_radius'))
                )

                if use_geospatial:
                    # Fallback per-member call to geospatial helper (keeps original behavior if desired)
                    for idx in member_indices:
                        member = members_df.loc[idx]
                        member_lat = member['Latitude']
                        member_lon = member['Longitude']
                        member_source_type = member.get('SourceType', '')
                        member_id = member.get('MemberID')

                        nearby_providers = geospatial.find_providers_within_radius(
                            member_lat, member_lon, providers_df, max_distance
                        )

                        for _, provider in nearby_providers.iterrows():
                            provider_type = provider.get('Type', '')
                            provider_source = provider.get('Source', '')

                            # Check if provider type matches member source type
                            type_match = False
                            if member_source_type == 'Supply Directory' and 'Supplier Directory' in str(provider_source):
                                type_match = True
                            elif member_source_type in ['Hospital', 'Nursing Home', 'Scan Center']:
                                type_match = len(str(provider_type)) > 0

                            if type_match:
                                candidate_connections.append({
                                    'member_id': member_id,
                                    'provider_id': provider['ProviderID'],
                                    'distance': provider.get('distance_km', np.nan),
                                    'cost': provider.get('Cost', np.nan),
                                    'rating': provider.get('CMS Rating', np.nan),
                                    'member_source_type': member_source_type,
                                    'provider_type': provider_type
                                })

                else:
                    # Use BallTree to query all member points in this batch at once
                    # query_radius returns array of arrays of provider indices (relative to providers_valid)
                    indices_array = tree.query_radius(member_coords_rad, r=radius_radians, return_distance=False)

                    # iterate each member result
                    for list_idx, provider_idxs in enumerate(indices_array):
                        member_idx = member_indices[list_idx]
                        member = members_df.loc[member_idx]
                        member_lat = member['Latitude']
                        member_lon = member['Longitude']
                        member_source_type = member.get('SourceType', '')
                        member_id = member.get('MemberID')

                        if len(provider_idxs) == 0:
                            # no providers within radius for this member
                            continue

                        # For these matched provider indices calculate actual distances and filter by type matching
                        # Extract provider rows once
                        matched_providers = providers_valid.iloc[provider_idxs].copy().reset_index(drop=True)
                        # Compute haversine distances between member and matched providers
                        # member_coords_rad[list_idx] corresponds to this member
                        mem_rad = np.radians([member_lat, member_lon])
                        prov_rad = np.radians(matched_providers[['Latitude', 'Longitude']].values)
                        # haversine distance formula via BallTree metric: distance (radians) = great-circle radians
                        # We'll compute using vectorized haversine:
                        lat1 = mem_rad[0]
                        lon1 = mem_rad[1]
                        lat2 = prov_rad[:, 0]
                        lon2 = prov_rad[:, 1]

                        dlat = lat2 - lat1
                        dlon = lon2 - lon1
                        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
                        great_circle_radians = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
                        distances_km = great_circle_radians * self._earth_radius_km

                        # attach distance_km to matched_providers frame
                        matched_providers = matched_providers.assign(distance_km=distances_km)

                        # Now iterate through matched providers and do type matching
                        for _, provider in matched_providers.iterrows():
                            provider_type = provider.get('Type', '')
                            provider_source = provider.get('Source', '')

                            # Check if provider type matches member source type (same logic as before)
                            type_match = False
                            if member_source_type == 'Supply Directory' and 'Supplier Directory' in str(provider_source):
                                type_match = True
                            elif member_source_type in ['Hospital', 'Nursing Home', 'Scan Center']:
                                # These member types can use various provider types
                                type_match = len(str(provider_type)) > 0

                            if type_match:
                                candidate_connections.append({
                                    'member_id': member_id,
                                    'provider_id': provider['ProviderID'],
                                    'distance': float(provider['distance_km']),
                                    'cost': provider.get('Cost', np.nan),
                                    'rating': provider.get('CMS Rating', np.nan),
                                    'member_source_type': member_source_type,
                                    'provider_type': provider_type
                                })

                if (i // batch_size + 1) % 5 == 0:  # Log progress every 5 batches
                    logger.info(f"Processed {batch_end}/{len(members_df)} members...")

            return candidate_connections

        except Exception as e:
            logger.error(f"Error finding candidate connections: {str(e)}")
            return []

    def optimize_assignments(self, candidate_connections, members_df, providers_df):
        """
        Optimize member-provider assignments based on the candidate connections
        """
        assignments = []
        member_candidates = defaultdict(list)

        try:
            # Group candidates by member
            for connection in candidate_connections:
                member_candidates[connection['member_id']].append(connection)

            logger.info(f"Processing assignments for {len(member_candidates)} members with potential matches")

            # Process each member
            for _, member in members_df.iterrows():
                member_id = member['MemberID']

                assignment = {
                    'member_id': member_id,
                    'member_source_type': member['SourceType'],
                    'provider_id': None,
                    'distance': None,
                    'cost': None,
                    'rating': None,
                    'provider_type': None
                }

                # Check if member has any candidate providers
                if member_id in member_candidates:
                    best_provider = self.find_best_provider(member_candidates[member_id])

                    if best_provider:
                        assignment.update({
                            'provider_id': best_provider['provider_id'],
                            'distance': best_provider['distance'],
                            'cost': best_provider['cost'],
                            'rating': best_provider['rating'],
                            'provider_type': best_provider.get('provider_type')
                        })

                assignments.append(assignment)

            # Log assignment statistics
            served_count = len([a for a in assignments if a['provider_id'] is not None])
            total_count = len(assignments)

            logger.info(f"Assignment completed: {served_count}/{total_count} members served ({(served_count/total_count)*100:.2f}%)")

            return assignments

        except Exception as e:
            logger.error(f"Error in optimization assignments: {str(e)}")
            return []

    def analyze_by_source_type(self, assignments, members_df):
        """
        Analyze optimization results by member source type
        """
        try:
            analysis = {}

            # Group assignments by source type
            source_type_groups = defaultdict(list)
            for assignment in assignments:
                source_type = assignment['member_source_type']
                source_type_groups[source_type].append(assignment)

            # Calculate statistics for each source type
            for source_type, group_assignments in source_type_groups.items():
                total_members = len(group_assignments)
                served_members = len([a for a in group_assignments if a['provider_id'] is not None])

                analysis[source_type] = {
                    'total_members': total_members,
                    'served_members': served_members,
                    'unserved_members': total_members - served_members,
                    'access_percentage': (served_members / total_members) * 100 if total_members > 0 else 0,
                    'average_distance': np.mean([a['distance'] for a in group_assignments if a['distance'] is not None]) if any(a['distance'] is not None for a in group_assignments) else None,
                    'average_cost': np.mean([a['cost'] for a in group_assignments if a['cost'] is not None]) if any(a['cost'] is not None for a in group_assignments) else None,
                    'average_rating': np.mean([a['rating'] for a in group_assignments if a['rating'] is not None]) if any(a['rating'] is not None for a in group_assignments) else None
                }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing by source type: {str(e)}")
            return {}

    def calculate_optimization_metrics(self, assignments, members_df, providers_df):
        """
        Calculate comprehensive optimization metrics
        """
        try:
            metrics = {}

            # Basic counts
            total_members = len(members_df)
            served_members = len([a for a in assignments if a['provider_id'] is not None])
            unserved_members = total_members - served_members

            # Access metrics
            access_percentage = (served_members / total_members) * 100 if total_members > 0 else 0

            # Cost analysis
            # Note: original_total_cost expects members_df to have a 'cost' column.
            original_total_cost = members_df['cost'].sum() if 'cost' in members_df.columns else 0
            optimized_total_cost = sum(a['cost'] for a in assignments if a['cost'] is not None and not pd.isna(a['cost']))
            cost_savings = original_total_cost - optimized_total_cost
            cost_savings_percentage = (cost_savings / original_total_cost) * 100 if original_total_cost > 0 else 0

            # Provider utilization
            used_provider_ids = set(a['provider_id'] for a in assignments if a['provider_id'] is not None)
            total_providers = len(providers_df)
            used_providers = len(used_provider_ids)
            provider_utilization = (used_providers / total_providers) * 100 if total_providers > 0 else 0

            # Distance and rating statistics
            served_assignments = [a for a in assignments if a['provider_id'] is not None]
            if served_assignments:
                average_distance = np.mean([a['distance'] for a in served_assignments])
                average_rating = np.mean([a['rating'] for a in served_assignments])
                max_distance = max([a['distance'] for a in served_assignments])
                min_distance = min([a['distance'] for a in served_assignments])
            else:
                average_distance = average_rating = max_distance = min_distance = 0

            # Network feasibility assessment
            if access_percentage == 100:
                network_status = "No need to change organization"
                network_recommendation = "Current network provides complete coverage"
            elif access_percentage >= 95:
                network_status = "Good in provider and member access"
                network_recommendation = "Network is performing well with minor optimization opportunities"
            else:
                network_status = "Organization must increase providers"
                network_recommendation = f"Network needs expansion - {unserved_members} members lack access"

            metrics = {
                'access': {
                    'total_members': total_members,
                    'served_members': served_members,
                    'unserved_members': unserved_members,
                    'access_percentage': access_percentage
                },
                'cost': {
                    'original_total_cost': original_total_cost,
                    'optimized_total_cost': optimized_total_cost,
                    'cost_savings': cost_savings,
                    'cost_savings_percentage': cost_savings_percentage
                },
                'provider_utilization': {
                    'total_providers': total_providers,
                    'used_providers': used_providers,
                    'unused_providers': total_providers - used_providers,
                    'utilization_percentage': provider_utilization
                },
                'quality_metrics': {
                    'average_distance_km': average_distance,
                    'average_provider_rating': average_rating,
                    'max_distance_km': max_distance,
                    'min_distance_km': min_distance
                },
                'network_assessment': {
                    'status': network_status,
                    'recommendation': network_recommendation
                }
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating optimization metrics: {str(e)}")
            return {}

    def generate_optimization_report(self, assignments, members_df, providers_df):
        """
        Generate a comprehensive text report of optimization results
        """
        try:
            metrics = self.calculate_optimization_metrics(assignments, members_df, providers_df)
            source_type_analysis = self.analyze_by_source_type(assignments, members_df)

            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("PROVIDER NETWORK OPTIMIZATION REPORT")
            report_lines.append("=" * 80)
            report_lines.append("")

            # Access Summary
            access = metrics.get('access', {})
            report_lines.append("ACCESS SUMMARY:")
            report_lines.append(f"  Total Members: {access.get('total_members', 0):,}")
            report_lines.append(f"  Served Members: {access.get('served_members', 0):,}")
            report_lines.append(f"  Unserved Members: {access.get('unserved_members', 0):,}")
            report_lines.append(f"  Access Percentage: {access.get('access_percentage', 0):.2f}%")
            report_lines.append("")

            # Cost Analysis
            cost = metrics.get('cost', {})
            report_lines.append("COST ANALYSIS:")
            report_lines.append(f"  Original Total Cost: ${cost.get('original_total_cost', 0):,.2f}")
            report_lines.append(f"  Optimized Total Cost: ${cost.get('optimized_total_cost', 0):,.2f}")
            report_lines.append(f"  Cost Savings: ${cost.get('cost_savings', 0):,.2f}")
            report_lines.append(f"  Savings Percentage: {cost.get('cost_savings_percentage', 0):.2f}%")
            report_lines.append("")

            # Provider Utilization
            provider_util = metrics.get('provider_utilization', {})
            report_lines.append("PROVIDER UTILIZATION:")
            report_lines.append(f"  Total Providers: {provider_util.get('total_providers', 0):,}")
            report_lines.append(f"  Used Providers: {provider_util.get('used_providers', 0):,}")
            report_lines.append(f"  Unused Providers: {provider_util.get('unused_providers', 0):,}")
            report_lines.append(f"  Utilization Rate: {provider_util.get('utilization_percentage', 0):.2f}%")
            report_lines.append("")

            # Quality Metrics
            quality = metrics.get('quality_metrics', {})
            report_lines.append("QUALITY METRICS:")
            report_lines.append(f"  Average Distance: {quality.get('average_distance_km', 0):.2f} km")
            report_lines.append(f"  Average Provider Rating: {quality.get('average_provider_rating', 0):.2f}/5")
            report_lines.append(f"  Distance Range: {quality.get('min_distance_km', 0):.2f} - {quality.get('max_distance_km', 0):.2f} km")
            report_lines.append("")

            # Source Type Analysis
            if source_type_analysis:
                report_lines.append("ANALYSIS BY SOURCE TYPE:")
                for source_type, analysis in source_type_analysis.items():
                    report_lines.append(f"  {source_type}:")
                    report_lines.append(f"    Members: {analysis['served_members']}/{analysis['total_members']} served ({analysis['access_percentage']:.1f}%)")
                    if analysis['average_distance'] is not None and not np.isnan(analysis['average_distance']):
                        report_lines.append(f"    Avg Distance: {analysis['average_distance']:.2f} km")
                    if analysis['average_rating'] is not None and not np.isnan(analysis['average_rating']):
                        report_lines.append(f"    Avg Rating: {analysis['average_rating']:.2f}/5")
                report_lines.append("")

            # Network Assessment
            assessment = metrics.get('network_assessment', {})
            report_lines.append("NETWORK ASSESSMENT:")
            report_lines.append(f"  Status: {assessment.get('status', 'Unknown')}")
            report_lines.append(f"  Recommendation: {assessment.get('recommendation', 'No recommendation available')}")
            report_lines.append("")

            report_lines.append("=" * 80)

            return "\n".join(report_lines)

        except Exception as e:
            logger.error(f"Error generating optimization report: {str(e)}")
            return f"Error generating report: {str(e)}"
