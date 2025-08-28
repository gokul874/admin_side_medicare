from app import db
from datetime import datetime
import json

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    file_type = db.Column(db.String(20), nullable=False)  # 'members' or 'providers'
    filename = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    record_count = db.Column(db.Integer)
    is_processed = db.Column(db.Boolean, default=False)
    
class OptimizationResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_members_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    dataset_providers_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    access_percentage = db.Column(db.Float)
    original_cost = db.Column(db.Float)
    optimized_cost = db.Column(db.Float)
    profit_loss_percentage = db.Column(db.Float)
    total_members = db.Column(db.Integer)
    served_members = db.Column(db.Integer)
    unserved_members = db.Column(db.Integer)
    total_providers = db.Column(db.Integer)
    used_providers = db.Column(db.Integer)
    unused_providers = db.Column(db.Integer)
    network_status = db.Column(db.String(100))
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    optimization_data = db.Column(db.Text)  # JSON string for detailed results
    
    def get_optimization_data(self):
        if self.optimization_data:
            return json.loads(self.optimization_data)
        return {}
    
    def set_optimization_data(self, data):
        self.optimization_data = json.dumps(data)

class MemberProviderAssignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    optimization_result_id = db.Column(db.Integer, db.ForeignKey('optimization_result.id'), nullable=False)
    member_id = db.Column(db.String(50), nullable=False)
    provider_id = db.Column(db.String(50))  # Null if no provider assigned
    distance_km = db.Column(db.Float)
    cost = db.Column(db.Float)
    provider_rating = db.Column(db.Integer)
    is_served = db.Column(db.Boolean, default=False)
    member_source_type = db.Column(db.String(50))
    provider_source_type = db.Column(db.String(50))