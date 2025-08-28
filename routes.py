import os
import pandas as pd
import json
from flask import render_template, request, jsonify, redirect, url_for, flash, session, send_file
from werkzeug.utils import secure_filename
from app import app, db
from models import Dataset, OptimizationResult, MemberProviderAssignment
from utils.data_processor import DataProcessor
from utils.optimizer import NetworkOptimizer
from utils.geospatial import GeospatialAnalyzer
import logging

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    datasets = Dataset.query.order_by(Dataset.upload_date.desc()).all()
    return render_template('upload.html', datasets=datasets)

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        dataset_type = request.form.get('dataset_type')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400
        
        if dataset_type not in ['members', 'providers']:
            return jsonify({'error': 'Invalid dataset type'}), 400
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        timestamp = str(int(pd.Timestamp.now().timestamp()))
        filename = f"{dataset_type}_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process and validate the dataset
        processor = DataProcessor()
        
        if dataset_type == 'members':
            df, errors = processor.process_members_data(filepath)
        else:
            df, errors = processor.process_providers_data(filepath)
        
        if errors:
            # Remove the uploaded file if there were errors
            os.remove(filepath)
            return jsonify({'error': f'Data validation errors: {"; ".join(errors)}'}), 400
        
        # Save dataset info to database
        dataset = Dataset(
            name=file.filename,
            file_type=dataset_type,
            filename=filename,
            record_count=len(df),
            is_processed=True
        )
        db.session.add(dataset)
        db.session.commit()
        
        logger.info(f"Successfully uploaded {dataset_type} dataset with {len(df)} records")
        
        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {dataset_type} dataset with {len(df)} records',
            'dataset_id': dataset.id,
            'record_count': len(df)
        })
        
    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/optimize_network', methods=['POST'])
def optimize_network():
    try:
        # Get the latest datasets
        members_dataset = Dataset.query.filter_by(file_type='members', is_processed=True).order_by(Dataset.upload_date.desc()).first()
        providers_dataset = Dataset.query.filter_by(file_type='providers', is_processed=True).order_by(Dataset.upload_date.desc()).first()
        
        if not members_dataset or not providers_dataset:
            return jsonify({'error': 'Both member and provider datasets are required'}), 400
        
        # Load data
        processor = DataProcessor()
        members_path = os.path.join(app.config['UPLOAD_FOLDER'], members_dataset.filename)
        providers_path = os.path.join(app.config['UPLOAD_FOLDER'], providers_dataset.filename)
        
        members_df, _ = processor.process_members_data(members_path)
        providers_df, _ = processor.process_providers_data(providers_path)
        
        # Initialize optimizer
        optimizer = NetworkOptimizer()
        geospatial = GeospatialAnalyzer()
        
        # Use optimized algorithm for large datasets
        logger.info("Finding candidate provider connections using optimized algorithm...")
        
        # Sample data for demonstration with large datasets (use first 1000 members and 500 providers)
        
        members_sample = members_df
        providers_sample = providers_df
        
        candidate_connections = optimizer.find_candidate_connections(
            members_sample, providers_sample, geospatial, max_distance=15.0
        )
        
        logger.info(f"Found {len(candidate_connections)} candidate connections")
        
        # Optimize assignments
        assignments = optimizer.optimize_assignments(candidate_connections, members_sample, providers_sample)
        
        # Calculate metrics
        total_members = len(members_sample)
        served_members = len([a for a in assignments if a['provider_id'] is not None])
        unserved_members = total_members - served_members
        access_percentage = (served_members / total_members) * 100
        
        # Calculate costs
        original_cost = members_sample['cost'].sum()
        optimized_cost = sum(a['cost'] for a in assignments if a['cost'] is not None)
        profit_loss_percentage = ((original_cost - optimized_cost) / original_cost) * 100 if original_cost > 0 else 0
        
        # Determine network status
        if access_percentage == 100:
            network_status = "No need to change organization"
        elif access_percentage >= 95:
            network_status = "Good in provider and member access"
        else:
            network_status = "Organization must increase providers"
        
        # Count used/unused providers
        used_provider_ids = set(a['provider_id'] for a in assignments if a['provider_id'] is not None)
        total_providers = len(providers_sample)
        used_providers = len(used_provider_ids)
        unused_providers = total_providers - used_providers
        
        # Create optimization result
        optimization_result = OptimizationResult(
            dataset_members_id=members_dataset.id,
            dataset_providers_id=providers_dataset.id,
            access_percentage=access_percentage,
            original_cost=original_cost,
            optimized_cost=optimized_cost,
            profit_loss_percentage=profit_loss_percentage,
            total_members=total_members,
            served_members=served_members,
            unserved_members=unserved_members,
            total_providers=total_providers,
            used_providers=used_providers,
            unused_providers=unused_providers,
            network_status=network_status
        )
        
        # Store detailed optimization data
        optimization_data = {
            'assignments': assignments,
            'candidate_connections': len(candidate_connections),
            'source_type_analysis': optimizer.analyze_by_source_type(assignments, members_sample)
        }
        optimization_result.set_optimization_data(optimization_data)
        
        db.session.add(optimization_result)
        db.session.commit()
        
        # Store individual assignments
        for assignment in assignments:
            member_assignment = MemberProviderAssignment(
                optimization_result_id=optimization_result.id,
                member_id=assignment['member_id'],
                provider_id=assignment['provider_id'],
                distance_km=assignment['distance'],
                cost=assignment['cost'],
                provider_rating=assignment['rating'],
                is_served=assignment['provider_id'] is not None,
                member_source_type=assignment['member_source_type'],
                provider_source_type=assignment.get('provider_type')
            )
            db.session.add(member_assignment)
        
        db.session.commit()
        
        # Store result ID in session for dashboard
        session['optimization_result_id'] = optimization_result.id
        
        logger.info(f"Optimization completed. Access: {access_percentage:.2f}%, Served: {served_members}/{total_members}")
        
        return jsonify({
            'success': True,
            'optimization_id': optimization_result.id,
            'access_percentage': access_percentage,
            'served_members': served_members,
            'total_members': total_members,
            'network_status': network_status,
            'redirect': url_for('dashboard')
        })
        
    except Exception as e:
        logger.error(f"Error in network optimization: {str(e)}")
        return jsonify({'error': f'Optimization failed: {str(e)}'}), 500

@app.route('/dashboard')
def dashboard():
    optimization_id = session.get('optimization_result_id')
    if not optimization_id:
        flash('No optimization results found. Please run optimization first.', 'warning')
        return redirect(url_for('upload_page'))
    
    result = OptimizationResult.query.get_or_404(optimization_id)
    optimization_data = result.get_optimization_data()
    
    return render_template('dashboard.html', result=result, optimization_data=optimization_data)

@app.route('/visualization')
def visualization():
    optimization_id = session.get('optimization_result_id')
    if not optimization_id:
        flash('No optimization results found. Please run optimization first.', 'warning')
        return redirect(url_for('upload_page'))
    
    result = OptimizationResult.query.get_or_404(optimization_id)
    return render_template('visualization.html', result=result)

@app.route('/api/map_data')
def get_map_data():
    optimization_id = session.get('optimization_result_id')
    if not optimization_id:
        return jsonify({'error': 'No optimization results found'}), 404
    
    try:
        result = OptimizationResult.query.get_or_404(optimization_id)
        assignments = MemberProviderAssignment.query.filter_by(optimization_result_id=optimization_id).all()
        
        # Load datasets with same sampling logic for consistency
        members_dataset = Dataset.query.get(result.dataset_members_id)
        providers_dataset = Dataset.query.get(result.dataset_providers_id)
        
        processor = DataProcessor()
        members_path = os.path.join(app.config['UPLOAD_FOLDER'], members_dataset.filename)
        providers_path = os.path.join(app.config['UPLOAD_FOLDER'], providers_dataset.filename)
        
        members_df, _ = processor.process_members_data(members_path)
        providers_df, _ = processor.process_providers_data(providers_path)
        
        # Use same sampling logic for consistency
        # if len(members_df) > 1000 or len(providers_df) > 500:
        #     members_df = members_df.sample(n=min(1000, len(members_df)), random_state=42)
        #     providers_df = providers_df.sample(n=min(500, len(providers_df)), random_state=42)
        
        # Prepare map data
        served_members = []
        unserved_members = []
        used_providers = set()
        
        for assignment in assignments:
            member_row = members_df[members_df['MemberID'] == int(assignment.member_id)].iloc[0]
            member_data = {
                'id': assignment.member_id,
                'lat': float(member_row['Latitude']),
                'lng': float(member_row['Longitude']),
                'source_type': assignment.member_source_type,
                'cost': float(member_row['cost']),
                'is_served': assignment.is_served
            }
            
            if assignment.is_served:
                member_data['provider_id'] = assignment.provider_id
                member_data['distance'] = assignment.distance_km
                member_data['provider_rating'] = assignment.provider_rating
                served_members.append(member_data)
                used_providers.add(assignment.provider_id)
            else:
                unserved_members.append(member_data)
        
        # Get provider data
        providers_data = []
        for _, provider in providers_df.iterrows():
            provider_data = {
                'id': str(provider['ProviderID']),
                'lat': float(provider['Latitude']),
                'lng': float(provider['Longitude']),
                'name': provider['Location'].split(',')[0] if ',' in provider['Location'] else provider['Location'],
                'type': provider['Type'],
                'rating': provider['CMS Rating'],
                'cost': provider['Cost'],
                'is_used': str(provider['ProviderID']) in used_providers
            }
            providers_data.append(provider_data)
        
        return jsonify({
            'served_members': served_members,
            'unserved_members': unserved_members,
            'providers': providers_data,
            'stats': {
                'total_members': result.total_members,
                'served_members': result.served_members,
                'unserved_members': result.unserved_members,
                'access_percentage': result.access_percentage
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting map data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_unused_providers')
def download_unused_providers():
    optimization_id = session.get('optimization_result_id')
    if not optimization_id:
        return jsonify({'error': 'No optimization results found'}), 404
    
    try:
        result = OptimizationResult.query.get_or_404(optimization_id)
        
        if result.unused_providers == 0:
            flash('All providers are being used. Organization is working in good condition.', 'success')
            return redirect(url_for('dashboard'))
        
        # Get used provider IDs
        assignments = MemberProviderAssignment.query.filter_by(
            optimization_result_id=optimization_id,
            is_served=True
        ).all()
        used_provider_ids = set(a.provider_id for a in assignments if a.provider_id)
        
        # Load providers data
        providers_dataset = Dataset.query.get(result.dataset_providers_id)
        processor = DataProcessor()
        providers_path = os.path.join(app.config['UPLOAD_FOLDER'], providers_dataset.filename)
        providers_df, _ = processor.process_providers_data(providers_path)
        
        # Filter unused providers
        unused_providers_df = providers_df[~providers_df['ProviderID'].astype(str).isin(used_provider_ids)]
        
        # Select relevant columns for download
        download_columns = ['ProviderID', 'Location', 'Type', 'CMS Rating', 'Cost', 'Contact Number']
        unused_providers_export = unused_providers_df[download_columns].copy()
        
        # Rename columns for better readability
        unused_providers_export.columns = ['Provider ID', 'Provider Name & Address', 'Type', 'Rating', 'Cost', 'Contact Number']
        
        # Save to CSV
        export_filename = f'unused_providers_{optimization_id}.csv'
        export_path = os.path.join(app.config['UPLOAD_FOLDER'], export_filename)
        unused_providers_export.to_csv(export_path, index=False)
        
        return send_file(export_path, as_attachment=True, download_name=export_filename)
        
    except Exception as e:
        logger.error(f"Error downloading unused providers: {str(e)}")
        flash(f'Error generating download: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/api/chart_data')
def get_chart_data():
    optimization_id = session.get('optimization_result_id')
    if not optimization_id:
        return jsonify({'error': 'No optimization results found'}), 404
    
    try:
        result = OptimizationResult.query.get_or_404(optimization_id)
        optimization_data = result.get_optimization_data()
        
        # Source type analysis
        source_type_analysis = optimization_data.get('source_type_analysis', {})
        
        return jsonify({
            'access_chart': {
                'served': result.served_members,
                'unserved': result.unserved_members,
                'access_percentage': result.access_percentage
            },
            'cost_chart': {
                'original': result.original_cost,
                'optimized': result.optimized_cost,
                'savings': result.original_cost - result.optimized_cost,
                'profit_loss_percentage': result.profit_loss_percentage
            },
            'provider_usage': {
                'used': result.used_providers,
                'unused': result.unused_providers,
                'total': result.total_providers
            },
            'source_type_analysis': source_type_analysis
        })
        
    except Exception as e:
        logger.error(f"Error getting chart data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return render_template('500.html'), 500
