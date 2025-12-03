from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import (
    District, ServiceCategory, SentimentData, ServiceLog,
    DailyScore, MLModelMetadata, ScrapingLog
)


@admin.register(District)
class DistrictAdmin(admin.ModelAdmin):
    list_display = ['name', 'name_nepali', 'province', 'population', 'area_sq_km', 'coordinates']
    list_filter = ['province']
    search_fields = ['name', 'name_nepali', 'province']
    ordering = ['name']
    
    def coordinates(self, obj):
        if obj.latitude and obj.longitude:
            return f"{obj.latitude:.4f}, {obj.longitude:.4f}"
        return "Not set"
    coordinates.short_description = "Coordinates"


@admin.register(ServiceCategory)
class ServiceCategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'name_nepali', 'priority_weight', 'description']
    list_editable = ['priority_weight']
    ordering = ['name']


@admin.register(SentimentData)
class SentimentDataAdmin(admin.ModelAdmin):
    list_display = [
        'text_preview', 'source', 'district', 'service_category',
        'sentiment_badge', 'confidence_score', 'post_date', 'scraped_at'
    ]
    list_filter = [
        'source', 'district', 'service_category', 'is_positive',
        'is_complaint', 'post_date', 'scraped_at'
    ]
    search_fields = ['text_content', 'district__name']
    readonly_fields = ['scraped_at', 'processed_at']
    date_hierarchy = 'post_date'
    ordering = ['-post_date']
    
    def text_preview(self, obj):
        return obj.text_content[:100] + "..." if len(obj.text_content) > 100 else obj.text_content
    text_preview.short_description = "Text Preview"
    
    def sentiment_badge(self, obj):
        if obj.sentiment_score > 0.1:
            color = "green"
            label = "Positive"
        elif obj.sentiment_score < -0.1:
            color = "red"
            label = "Negative"
        else:
            color = "gray"
            label = "Neutral"
        
        return format_html(
            '<span style="color: {}; font-weight: bold;">{} ({:.2f})</span>',
            color, label, obj.sentiment_score
        )
    sentiment_badge.short_description = "Sentiment"


@admin.register(ServiceLog)
class ServiceLogAdmin(admin.ModelAdmin):
    list_display = [
        'service_name', 'district', 'service_category', 'status',
        'beneficiaries_progress', 'budget_progress', 'efficiency_badge',
        'start_date', 'end_date'
    ]
    list_filter = [
        'status', 'district', 'service_category', 'start_date'
    ]
    search_fields = ['service_name', 'district__name', 'description']
    readonly_fields = ['created_at', 'updated_at', 'efficiency_score']
    date_hierarchy = 'start_date'
    ordering = ['-start_date']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('service_name', 'service_name_nepali', 'description')
        }),
        ('Location & Category', {
            'fields': ('district', 'service_category')
        }),
        ('Beneficiaries', {
            'fields': ('beneficiaries_planned', 'beneficiaries_actual')
        }),
        ('Budget', {
            'fields': ('budget_allocated', 'budget_utilized')
        }),
        ('Timeline & Status', {
            'fields': ('start_date', 'end_date', 'status')
        }),
        ('Quality Metrics', {
            'fields': ('satisfaction_score', 'completion_percentage')
        }),
        ('System Fields', {
            'fields': ('created_at', 'updated_at', 'efficiency_score'),
            'classes': ('collapse',)
        })
    )
    
    def beneficiaries_progress(self, obj):
        if obj.beneficiaries_planned > 0:
            percentage = (obj.beneficiaries_actual / obj.beneficiaries_planned) * 100
            color = "green" if percentage >= 80 else "orange" if percentage >= 50 else "red"
            return format_html(
                '<span style="color: {};">{}/{} ({:.1f}%)</span>',
                color, obj.beneficiaries_actual, obj.beneficiaries_planned, percentage
            )
        return "N/A"
    beneficiaries_progress.short_description = "Beneficiaries Progress"
    
    def budget_progress(self, obj):
        if obj.budget_allocated > 0:
            percentage = (float(obj.budget_utilized) / float(obj.budget_allocated)) * 100
            color = "green" if percentage <= 100 else "red"
            return format_html(
                '<span style="color: {};">{:.1f}%</span>',
                color, percentage
            )
        return "N/A"
    budget_progress.short_description = "Budget Utilization"
    
    def efficiency_badge(self, obj):
        score = obj.efficiency_score
        if score >= 80:
            color = "green"
            label = "Excellent"
        elif score >= 60:
            color = "orange"
            label = "Good"
        else:
            color = "red"
            label = "Needs Improvement"
        
        return format_html(
            '<span style="color: {}; font-weight: bold;">{} ({:.1f})</span>',
            color, label, score
        )
    efficiency_badge.short_description = "Efficiency"


@admin.register(DailyScore)
class DailyScoreAdmin(admin.ModelAdmin):
    list_display = [
        'district', 'date', 'ssei_badge', 'sentiment_score',
        'service_delivery_score', 'posts_analyzed', 'services_tracked'
    ]
    list_filter = ['date', 'district__province']
    search_fields = ['district__name']
    readonly_fields = ['calculation_timestamp']
    date_hierarchy = 'date'
    ordering = ['-date', 'district__name']
    
    def ssei_badge(self, obj):
        score = obj.ssei_score
        if score >= 80:
            color = "green"
            label = "Excellent"
        elif score >= 60:
            color = "orange"
            label = "Good"
        elif score >= 40:
            color = "red"
            label = "Poor"
        else:
            color = "darkred"
            label = "Critical"
        
        return format_html(
            '<span style="color: {}; font-weight: bold;">{} ({:.1f})</span>',
            color, label, score
        )
    ssei_badge.short_description = "SSEI Score"
    
    def posts_analyzed(self, obj):
        return obj.total_posts_analyzed
    posts_analyzed.short_description = "Posts"
    
    def services_tracked(self, obj):
        return obj.total_services_tracked
    services_tracked.short_description = "Services"


@admin.register(MLModelMetadata)
class MLModelMetadataAdmin(admin.ModelAdmin):
    list_display = [
        'model_type', 'version', 'is_active', 'accuracy_badge',
        'training_data_size', 'training_date'
    ]
    list_filter = ['model_type', 'is_active', 'training_date']
    readonly_fields = ['created_at']
    ordering = ['-training_date']
    
    def accuracy_badge(self, obj):
        if obj.accuracy:
            color = "green" if obj.accuracy >= 0.8 else "orange" if obj.accuracy >= 0.6 else "red"
            return format_html(
                '<span style="color: {};">{:.1%}</span>',
                color, obj.accuracy
            )
        return "N/A"
    accuracy_badge.short_description = "Accuracy"


@admin.register(ScrapingLog)
class ScrapingLogAdmin(admin.ModelAdmin):
    list_display = [
        'source', 'posts_scraped', 'posts_processed', 'status_badge',
        'duration', 'start_time', 'errors_count'
    ]
    list_filter = ['source', 'status', 'start_time']
    readonly_fields = ['start_time', 'end_time']
    ordering = ['-start_time']
    
    def status_badge(self, obj):
        colors = {
            'completed': 'green',
            'running': 'blue',
            'failed': 'red',
            'partial': 'orange'
        }
        color = colors.get(obj.status, 'gray')
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color, obj.status.title()
        )
    status_badge.short_description = "Status"
    
    def duration(self, obj):
        if obj.end_time and obj.start_time:
            duration = obj.end_time - obj.start_time
            minutes = duration.total_seconds() / 60
            return f"{minutes:.1f} min"
        return "N/A"
    duration.short_description = "Duration"


# Customize admin site
admin.site.site_header = "Social Service AI - Nepal"
admin.site.site_title = "Social Service AI Admin"
admin.site.index_title = "Dashboard Administration"

# Add custom CSS
class AdminConfig:
    def __init__(self):
        pass
    
    class Media:
        css = {
            'all': ('admin/css/custom_admin.css',)
        }

# Custom admin actions
def recalculate_ssei(modeladmin, request, queryset):
    """Recalculate SSEI for selected daily scores"""
    from .tasks import calculate_district_ssei
    
    for daily_score in queryset:
        calculate_district_ssei.delay(
            daily_score.district.id,
            daily_score.date.isoformat()
        )
    
    modeladmin.message_user(
        request,
        f"SSEI recalculation triggered for {queryset.count()} records."
    )

recalculate_ssei.short_description = "Recalculate SSEI for selected records"

# Add action to DailyScoreAdmin
DailyScoreAdmin.actions = [recalculate_ssei]
