import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import json
import os
from datetime import datetime
import pickle

# Simple sentiment analysis without external dependencies
class SimpleSentiment:
    def __init__(self):
        self.positive_words = ['good', 'great', 'excellent', 'satisfied', 'happy', 'improved', 'better', 'working', 'helpful']
        self.negative_words = ['bad', 'terrible', 'poor', 'unsatisfied', 'angry', 'worse', 'broken', 'useless', 'corrupt']
    
    def polarity(self, text):
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        if positive_count + negative_count == 0:
            return 0
        return (positive_count - negative_count) / (positive_count + negative_count)

class ServiceGapAnalyzer:
    def __init__(self):
        self.load_models()
        self.service_categories = ['health', 'education', 'infrastructure', 'employment', 'governance', 'social_welfare']
        self.regions = ['koshi', 'madhesh', 'bagmati', 'gandaki', 'lumbini', 'karnali', 'sudurpashchim']
        self.gap_patterns = self.load_gap_patterns()
        
    def load_models(self):
        """Load existing ML models"""
        try:
            with open('production_models/best_model.pkl', 'rb') as f:
                self.classifier = pickle.load(f)
            with open('production_models/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
        except:
            self.classifier = None
            self.vectorizer = None
    
    def load_gap_patterns(self):
        """Define patterns that indicate service gaps"""
        return {
            'negative_indicators': [
                'not working', 'broken', 'delayed', 'poor quality', 'unavailable',
                'corruption', 'bribery', 'unfair', 'discrimination', 'no response',
                'waiting too long', 'rejected', 'denied', 'insufficient', 'lack of'
            ],
            'urgency_indicators': [
                'emergency', 'urgent', 'critical', 'immediate', 'crisis',
                'desperate', 'dying', 'starving', 'homeless', 'helpless'
            ],
            'access_barriers': [
                'too far', 'expensive', 'no transport', 'language barrier',
                'documentation required', 'complex process', 'office closed'
            ]
        }
    
    def analyze_sentiment_advanced(self, text):
        """Advanced sentiment analysis with context"""
        sentiment_analyzer = SimpleSentiment()
        base_sentiment = sentiment_analyzer.polarity(text)
        
        # Adjust for service-specific context
        negative_count = sum(1 for pattern in self.gap_patterns['negative_indicators'] 
                           if pattern in text.lower())
        urgency_count = sum(1 for pattern in self.gap_patterns['urgency_indicators'] 
                          if pattern in text.lower())
        barrier_count = sum(1 for pattern in self.gap_patterns['access_barriers'] 
                          if pattern in text.lower())
        
        # Calculate adjusted sentiment
        adjustment = -(negative_count * 0.2 + urgency_count * 0.3 + barrier_count * 0.25)
        final_sentiment = max(-1, min(1, base_sentiment + adjustment))
        
        return {
            'sentiment_score': final_sentiment,
            'sentiment_label': 'positive' if final_sentiment > 0.1 else 'negative' if final_sentiment < -0.1 else 'neutral',
            'negative_indicators': negative_count,
            'urgency_level': urgency_count,
            'access_barriers': barrier_count,
            'confidence': abs(final_sentiment)
        }
    
    def extract_service_mentions(self, text):
        """Extract mentioned services and locations"""
        text_lower = text.lower()
        
        # Service detection
        services_found = []
        for category in self.service_categories:
            if category in text_lower or any(keyword in text_lower for keyword in self.get_service_keywords(category)):
                services_found.append(category)
        
        # Location detection
        locations_found = []
        for region in self.regions:
            if region in text_lower:
                locations_found.append(region)
        
        return services_found, locations_found
    
    def get_service_keywords(self, category):
        """Get keywords for each service category"""
        keywords = {
            'health': ['hospital', 'clinic', 'doctor', 'medicine', 'treatment', 'health post'],
            'education': ['school', 'college', 'university', 'teacher', 'student', 'education'],
            'infrastructure': ['road', 'bridge', 'water', 'electricity', 'internet', 'transport'],
            'employment': ['job', 'work', 'employment', 'unemployment', 'salary', 'wage'],
            'governance': ['government', 'office', 'service', 'bureaucracy', 'administration'],
            'social_welfare': ['pension', 'allowance', 'welfare', 'support', 'assistance']
        }
        return keywords.get(category, [])
    
    def identify_gaps(self, feedback_data):
        """Main gap identification function"""
        gaps = defaultdict(lambda: defaultdict(list))
        
        for _, row in feedback_data.iterrows():
            text = str(row.get('text', ''))
            
            # Analyze sentiment
            sentiment_analysis = self.analyze_sentiment_advanced(text)
            
            # Extract services and locations
            services, locations = self.extract_service_mentions(text)
            
            # Only process negative feedback for gap analysis
            if sentiment_analysis['sentiment_label'] == 'negative':
                gap_info = {
                    'text': text,
                    'sentiment_score': sentiment_analysis['sentiment_score'],
                    'negative_indicators': sentiment_analysis['negative_indicators'],
                    'urgency_level': sentiment_analysis['urgency_level'],
                    'access_barriers': sentiment_analysis['access_barriers'],
                    'timestamp': row.get('timestamp', datetime.now().isoformat())
                }
                
                # Categorize gaps by service and location
                for service in services or ['general']:
                    for location in locations or ['unspecified']:
                        gaps[service][location].append(gap_info)
        
        return self.process_gaps(gaps)
    
    def process_gaps(self, raw_gaps):
        """Process and prioritize identified gaps"""
        processed_gaps = {}
        
        for service, locations in raw_gaps.items():
            processed_gaps[service] = {}
            
            for location, gap_list in locations.items():
                if not gap_list:
                    continue
                
                # Calculate gap metrics
                total_complaints = len(gap_list)
                avg_sentiment = np.mean([g['sentiment_score'] for g in gap_list])
                urgency_score = np.mean([g['urgency_level'] for g in gap_list])
                barrier_score = np.mean([g['access_barriers'] for g in gap_list])
                
                # Calculate priority score
                priority_score = (
                    (total_complaints * 0.3) +
                    (abs(avg_sentiment) * 0.3) +
                    (urgency_score * 0.25) +
                    (barrier_score * 0.15)
                )
                
                processed_gaps[service][location] = {
                    'total_complaints': total_complaints,
                    'avg_sentiment': avg_sentiment,
                    'urgency_score': urgency_score,
                    'barrier_score': barrier_score,
                    'priority_score': priority_score,
                    'priority_level': self.get_priority_level(priority_score),
                    'sample_complaints': gap_list[:3],  # Top 3 examples
                    'recommendations': self.generate_recommendations(service, location, gap_list)
                }
        
        return processed_gaps
    
    def get_priority_level(self, score):
        """Determine priority level based on score"""
        if score >= 3.0:
            return 'CRITICAL'
        elif score >= 2.0:
            return 'HIGH'
        elif score >= 1.0:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def generate_recommendations(self, service, location, gap_list):
        """Generate actionable recommendations"""
        common_issues = Counter()
        for gap in gap_list:
            text = gap['text'].lower()
            for indicator in self.gap_patterns['negative_indicators']:
                if indicator in text:
                    common_issues[indicator] += 1
        
        recommendations = []
        top_issues = common_issues.most_common(3)
        
        for issue, count in top_issues:
            if issue in ['not working', 'broken']:
                recommendations.append(f"Immediate repair/maintenance of {service} infrastructure in {location}")
            elif issue in ['delayed', 'waiting too long']:
                recommendations.append(f"Improve service delivery speed for {service} in {location}")
            elif issue in ['corruption', 'bribery']:
                recommendations.append(f"Implement anti-corruption measures for {service} in {location}")
            elif issue in ['unavailable', 'lack of']:
                recommendations.append(f"Increase {service} availability/resources in {location}")
        
        return recommendations
    
    def analyze_dataset(self, dataset_path=None):
        """Analyze the main dataset for gaps"""
        if dataset_path is None:
            dataset_path = 'production_data/master_dataset.csv'
        
        try:
            df = pd.read_csv(dataset_path)
            gaps = self.identify_gaps(df)
            
            # Save results
            results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_records_analyzed': len(df),
                'gaps_identified': gaps,
                'summary': self.generate_summary(gaps)
            }
            
            # Save to file
            os.makedirs('gap_analysis_results', exist_ok=True)
            with open('gap_analysis_results/latest_gap_analysis.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            return results
            
        except Exception as e:
            return {'error': f"Analysis failed: {str(e)}"}
    
    def generate_summary(self, gaps):
        """Generate executive summary of gaps"""
        total_gaps = 0
        critical_gaps = 0
        high_gaps = 0
        
        service_priorities = {}
        location_priorities = {}
        
        for service, locations in gaps.items():
            service_total = 0
            for location, gap_data in locations.items():
                total_gaps += gap_data['total_complaints']
                service_total += gap_data['total_complaints']
                
                if gap_data['priority_level'] == 'CRITICAL':
                    critical_gaps += 1
                elif gap_data['priority_level'] == 'HIGH':
                    high_gaps += 1
                
                # Track location priorities
                if location not in location_priorities:
                    location_priorities[location] = 0
                location_priorities[location] += gap_data['priority_score']
            
            service_priorities[service] = service_total
        
        return {
            'total_gaps_identified': total_gaps,
            'critical_priority_areas': critical_gaps,
            'high_priority_areas': high_gaps,
            'most_affected_services': sorted(service_priorities.items(), key=lambda x: x[1], reverse=True)[:3],
            'most_affected_locations': sorted(location_priorities.items(), key=lambda x: x[1], reverse=True)[:3],
            'key_recommendations': [
                "Focus immediate attention on critical priority areas",
                "Implement systematic monitoring for high-priority gaps",
                "Develop targeted intervention programs for most affected services"
            ]
        }

def run_gap_analysis():
    """Main function to run gap analysis"""
    analyzer = ServiceGapAnalyzer()
    results = analyzer.analyze_dataset()
    
    print("Gap Analysis Complete!")
    print(f"Total gaps identified: {results.get('summary', {}).get('total_gaps_identified', 0)}")
    print(f"Critical areas: {results.get('summary', {}).get('critical_priority_areas', 0)}")
    
    return results

if __name__ == "__main__":
    run_gap_analysis()