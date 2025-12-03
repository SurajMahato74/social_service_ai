from rest_framework import serializers
from .models import (
    District, ServiceCategory, SentimentData, ServiceLog, 
    DailyScore, MLModelMetadata, ScrapingLog
)

class DistrictSerializer(serializers.ModelSerializer):
    class Meta:
        model = District
        fields = '__all__'

class ServiceCategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = ServiceCategory
        fields = '__all__'

class SentimentDataSerializer(serializers.ModelSerializer):
    district_name = serializers.CharField(source='district.name', read_only=True)
    service_category_name = serializers.CharField(source='service_category.get_name_display', read_only=True)
    
    class Meta:
        model = SentimentData
        fields = [
            'id', 'text_content', 'source', 'district', 'district_name',
            'service_category', 'service_category_name', 'sentiment_score',
            'confidence_score', 'is_positive', 'is_complaint', 'is_service_request',
            'post_date', 'scraped_at'
        ]

class ServiceLogSerializer(serializers.ModelSerializer):
    district_name = serializers.CharField(source='district.name', read_only=True)
    service_category_name = serializers.CharField(source='service_category.get_name_display', read_only=True)
    efficiency_score = serializers.ReadOnlyField()
    
    class Meta:
        model = ServiceLog
        fields = [
            'id', 'district', 'district_name', 'service_category', 'service_category_name',
            'service_name', 'description', 'beneficiaries_planned', 'beneficiaries_actual',
            'budget_allocated', 'budget_utilized', 'start_date', 'end_date', 'status',
            'satisfaction_score', 'completion_percentage', 'efficiency_score'
        ]

class DailyScoreSerializer(serializers.ModelSerializer):
    district_name = serializers.CharField(source='district.name', read_only=True)
    district_province = serializers.CharField(source='district.province', read_only=True)
    
    class Meta:
        model = DailyScore
        fields = [
            'id', 'district', 'district_name', 'district_province', 'date',
            'sentiment_score', 'service_delivery_score', 'demand_fulfillment_score',
            'resource_efficiency_score', 'ssei_score', 'raw_ssei_score',
            'bias_adjustment', 'predicted_demand', 'actual_demand',
            'demand_gap_percentage', 'total_posts_analyzed', 'total_services_tracked'
        ]

class DashboardSummarySerializer(serializers.Serializer):
    """Serializer for dashboard summary data"""
    national_avg_ssei = serializers.FloatField()
    total_districts = serializers.IntegerField()
    total_posts_today = serializers.IntegerField()
    total_services_active = serializers.IntegerField()
    top_performing_districts = DailyScoreSerializer(many=True)
    bottom_performing_districts = DailyScoreSerializer(many=True)
    sentiment_trend = serializers.ListField()
    service_category_breakdown = serializers.DictField()

class MLModelMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLModelMetadata
        fields = '__all__'

class ScrapingLogSerializer(serializers.ModelSerializer):
    duration_minutes = serializers.SerializerMethodField()
    
    class Meta:
        model = ScrapingLog
        fields = [
            'id', 'source', 'url', 'posts_scraped', 'posts_processed',
            'errors_count', 'start_time', 'end_time', 'status',
            'duration_minutes'
        ]
    
    def get_duration_minutes(self, obj):
        if obj.end_time and obj.start_time:
            duration = obj.end_time - obj.start_time
            return round(duration.total_seconds() / 60, 2)
        return None