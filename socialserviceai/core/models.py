from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone


class District(models.Model):
    """77 Districts of Nepal"""
    name = models.CharField(max_length=100, unique=True)
    name_nepali = models.CharField(max_length=100)
    province = models.CharField(max_length=50)
    population = models.IntegerField(default=0)
    area_sq_km = models.FloatField(default=0.0)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} ({self.province})"


class ServiceCategory(models.Model):
    """Categories of Social Services"""
    CATEGORY_CHOICES = [
        ('health', 'Health Services'),
        ('education', 'Education Services'),
        ('relief', 'Relief & Emergency'),
        ('infrastructure', 'Infrastructure'),
        ('agriculture', 'Agriculture Support'),
        ('employment', 'Employment Programs'),
        ('social_security', 'Social Security'),
        ('other', 'Other Services')
    ]
    
    name = models.CharField(max_length=50, choices=CATEGORY_CHOICES, unique=True)
    name_nepali = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    priority_weight = models.FloatField(default=1.0, validators=[MinValueValidator(0.1), MaxValueValidator(5.0)])
    
    def __str__(self):
        return self.get_name_display()


class SentimentData(models.Model):
    """Stores sentiment analysis results from social media"""
    SOURCE_CHOICES = [
        ('facebook', 'Facebook'),
        ('twitter', 'Twitter'),
        ('news', 'News Articles'),
        ('manual', 'Manual Entry')
    ]
    
    text_content = models.TextField()
    text_nepali = models.TextField(blank=True)
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES)
    source_url = models.URLField(blank=True)
    district = models.ForeignKey(District, on_delete=models.CASCADE, null=True, blank=True)
    service_category = models.ForeignKey(ServiceCategory, on_delete=models.CASCADE, null=True, blank=True)
    
    # Sentiment Scores
    sentiment_score = models.FloatField(validators=[MinValueValidator(-1.0), MaxValueValidator(1.0)])
    confidence_score = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    
    # Classification
    is_positive = models.BooleanField(default=False)
    is_complaint = models.BooleanField(default=False)
    is_service_request = models.BooleanField(default=False)
    
    # Metadata
    post_date = models.DateTimeField()
    scraped_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-post_date']
        indexes = [
            models.Index(fields=['district', 'post_date']),
            models.Index(fields=['service_category', 'post_date']),
            models.Index(fields=['sentiment_score']),
        ]
    
    def __str__(self):
        return f"{self.source} - {self.sentiment_score:.2f} ({self.post_date.date()})"


class ServiceLog(models.Model):
    """Tracks actual service delivery data"""
    STATUS_CHOICES = [
        ('planned', 'Planned'),
        ('ongoing', 'Ongoing'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
        ('delayed', 'Delayed')
    ]
    
    district = models.ForeignKey(District, on_delete=models.CASCADE)
    service_category = models.ForeignKey(ServiceCategory, on_delete=models.CASCADE)
    
    # Service Details
    service_name = models.CharField(max_length=200)
    service_name_nepali = models.CharField(max_length=200, blank=True)
    description = models.TextField()
    
    # Metrics
    beneficiaries_planned = models.IntegerField(default=0)
    beneficiaries_actual = models.IntegerField(default=0)
    budget_allocated = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    budget_utilized = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    
    # Timeline
    start_date = models.DateField()
    end_date = models.DateField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='planned')
    
    # Quality Metrics
    satisfaction_score = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(5.0)]
    )
    completion_percentage = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)]
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-start_date']
        indexes = [
            models.Index(fields=['district', 'start_date']),
            models.Index(fields=['service_category', 'start_date']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"{self.service_name} - {self.district.name}"
    
    @property
    def efficiency_score(self):
        """Calculate service efficiency (0-100)"""
        if self.beneficiaries_planned == 0 or self.budget_allocated == 0:
            return 0
        
        beneficiary_ratio = min(self.beneficiaries_actual / self.beneficiaries_planned, 1.0)
        budget_efficiency = 1.0 - min(float(self.budget_utilized / self.budget_allocated), 1.0)
        
        return (beneficiary_ratio * 0.6 + budget_efficiency * 0.4) * 100


class DailyScore(models.Model):
    """Daily SSEI (Social Service Effectiveness Index) for each district"""
    district = models.ForeignKey(District, on_delete=models.CASCADE)
    date = models.DateField()
    
    # Core Metrics (0-100 scale)
    sentiment_score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)])
    service_delivery_score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)])
    demand_fulfillment_score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)])
    resource_efficiency_score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)])
    
    # Composite SSEI Score
    ssei_score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)])
    
    # Bias Mitigation
    raw_ssei_score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)])
    bias_adjustment = models.FloatField(default=0.0)
    
    # Forecasting Data
    predicted_demand = models.FloatField(default=0.0)
    actual_demand = models.FloatField(default=0.0)
    demand_gap_percentage = models.FloatField(default=0.0)
    
    # Metadata
    total_posts_analyzed = models.IntegerField(default=0)
    total_services_tracked = models.IntegerField(default=0)
    calculation_timestamp = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['district', 'date']
        ordering = ['-date', 'district__name']
        indexes = [
            models.Index(fields=['date', 'ssei_score']),
            models.Index(fields=['district', 'date']),
        ]
    
    def __str__(self):
        return f"{self.district.name} - {self.date} (SSEI: {self.ssei_score:.1f})"
    
    def save(self, *args, **kwargs):
        """Auto-calculate SSEI score before saving"""
        if not self.ssei_score:
            self.calculate_ssei()
        super().save(*args, **kwargs)
    
    def calculate_ssei(self):
        """Calculate composite SSEI score with weights"""
        weights = {
            'sentiment': 0.25,
            'delivery': 0.30,
            'demand': 0.25,
            'efficiency': 0.20
        }
        
        self.raw_ssei_score = (
            self.sentiment_score * weights['sentiment'] +
            self.service_delivery_score * weights['delivery'] +
            self.demand_fulfillment_score * weights['demand'] +
            self.resource_efficiency_score * weights['efficiency']
        )
        
        # Apply bias adjustment
        self.ssei_score = max(0, min(100, self.raw_ssei_score + self.bias_adjustment))


class MLModelMetadata(models.Model):
    """Tracks ML model versions and performance"""
    MODEL_TYPES = [
        ('sentiment', 'Sentiment Analysis'),
        ('forecasting', 'Demand Forecasting'),
        ('gap_detection', 'Service Gap Detection'),
        ('bias_mitigation', 'Bias Mitigation'),
        ('optimization', 'Resource Optimization')
    ]
    
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    version = models.CharField(max_length=20)
    file_path = models.CharField(max_length=500)
    
    # Performance Metrics
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    
    # Training Info
    training_data_size = models.IntegerField(default=0)
    training_date = models.DateTimeField()
    is_active = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['model_type', 'version']
        ordering = ['-training_date']
    
    def __str__(self):
        return f"{self.get_model_type_display()} v{self.version}"


class ScrapingLog(models.Model):
    """Tracks web scraping activities"""
    source = models.CharField(max_length=50)
    url = models.URLField()
    posts_scraped = models.IntegerField(default=0)
    posts_processed = models.IntegerField(default=0)
    errors_count = models.IntegerField(default=0)
    
    start_time = models.DateTimeField()
    end_time = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=[
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('partial', 'Partial Success')
    ])
    
    error_details = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-start_time']
    
    def __str__(self):
        return f"{self.source} - {self.start_time.date()} ({self.status})"
