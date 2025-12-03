#!/usr/bin/env python3
"""
Data Validation and Preprocessing System for Nepal Social Service AI
Validates and cleans collected training data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import re
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates and preprocesses collected training data
    """
    
    def __init__(self):
        # Validation thresholds
        self.min_text_length = 10
        self.max_text_length = 10000
        self.required_columns = ['text', 'category']
        self.quality_threshold = 0.1
        
        # Language detection patterns (basic)
        self.nepali_patterns = [
            r'क|ख|ग|घ|ङ|च|छ|ज|झ|ञ|ट|ठ|ड|ढ|ण|त|थ|द|ध|न|प|फ|ब|भ|म|य|र|ल|व|श|ष|स|ह|क्ष|त्र|ज्ञ',
            r'अ|आ|इ|ई|उ|ऊ|ए|ऐ|ओ|औ|अं|अः',
            r'ोक|नेपाल|सरकार|मन्त्रालय|विभाग|गाउँपालिका|नगरपालिका'
        ]
        
        # Quality indicators
        self.quality_indicators = {
            'service_keywords': ['स्वास्थ्य', 'शिक्षा', 'रोजगार', 'विकास', 'सरकार'],
            'government_terms': ['मन्त्रालय', 'विभाग', 'नगरपालिका', 'जिल्ला'],
            'location_terms': ['काठमाडौं', 'नेपाल', 'जिल्ला', 'गाउँ']
        }
    
    def validate_text_quality(self, text: str) -> Dict[str, float]:
        """
        Assess quality of text content
        """
        if not isinstance(text, str):
            return {'overall_score': 0, 'language_score': 0, 'service_score': 0, 'length_score': 0}
        
        text_lower = text.lower()
        
        # Language detection score
        language_score = 0
        for pattern in self.nepali_patterns:
            if re.search(pattern, text):
                language_score += 0.2
        language_score = min(language_score, 1.0)
        
        # Service relevance score
        service_score = 0
        for indicator_group in self.quality_indicators.values():
            for keyword in indicator_group:
                if keyword in text_lower:
                    service_score += 0.1
        service_score = min(service_score, 1.0)
        
        # Length appropriateness score
        length_score = 0
        text_len = len(text)
        if self.min_text_length <= text_len <= self.max_text_length:
            length_score = 1.0
        elif text_len > self.max_text_length:
            length_score = 0.5
        else:
            length_score = text_len / self.min_text_length * 0.5
        
        # Overall quality score
        overall_score = (language_score * 0.3 + service_score * 0.4 + length_score * 0.3)
        
        return {
            'overall_score': overall_score,
            'language_score': language_score,
            'service_score': service_score,
            'length_score': length_score
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Nepali characters
        text = re.sub(r'[^\w\s\-।-ॿ]', ' ', text)
        
        # Strip and normalize
        text = text.strip()
        
        return text
    
    def categorize_content(self, text: str) -> str:
        """
        Categorize content based on keywords
        """
        if not isinstance(text, str):
            return 'general'
        
        text_lower = text.lower()
        
        category_keywords = {
            'health': ['स्वास्थ्य', 'अस्पताल', 'डाक्टर', 'औषधि', 'खोप', 'चिकित्सा'],
            'education': ['शिक्षा', 'विद्यालय', 'पाठशाला', 'छात्रवृत्ति', 'परीक्षा'],
            'employment': ['रोजगार', 'काम', 'जागिर', 'श्रमिक', 'तलब'],
            'infrastructure': ['सडक', 'पुल', 'पानी', 'बिजुली', 'निर्माण'],
            'social_welfare': ['राहत', 'सहायता', 'बाढी', 'भूकम्प', 'उद्धार'],
            'agriculture': ['कृषि', 'किसान', 'बाली', 'धान', 'तरकारी']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate and clean entire dataframe
        """
        validation_report = {
            'total_samples': len(df),
            'valid_samples': 0,
            'removed_samples': 0,
            'quality_distribution': {},
            'category_distribution': {},
            'issues': []
        }
        
        logger.info(f"Starting validation of {len(df)} samples")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Step 1: Remove duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['text'] if 'text' in df_clean.columns else df_clean.columns[0])
        duplicates_removed = initial_count - len(df_clean)
        if duplicates_removed > 0:
            validation_report['issues'].append(f"Removed {duplicates_removed} duplicate samples")
        
        # Step 2: Clean text columns
        text_columns = ['text', 'content', 'description', 'title']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self.clean_text)
        
        # Step 3: Add quality scores
        if 'text' in df_clean.columns:
            quality_scores = df_clean['text'].apply(self.validate_text_quality)
            df_clean['quality_score'] = quality_scores.apply(lambda x: x['overall_score'])
            df_clean['language_score'] = quality_scores.apply(lambda x: x['language_score'])
            df_clean['service_score'] = quality_scores.apply(lambda x: x['service_score'])
        
        # Step 4: Recategorize content
        if 'text' in df_clean.columns:
            df_clean['category_cleaned'] = df_clean['text'].apply(self.categorize_content)
        
        # Step 5: Filter by quality threshold (more lenient)
        if 'quality_score' in df_clean.columns:
            before_quality_filter = len(df_clean)
            # Keep samples with any quality score > 0
            df_clean = df_clean[df_clean['quality_score'] > 0]
            quality_filtered = before_quality_filter - len(df_clean)
            if quality_filtered > 0:
                validation_report['issues'].append(f"Removed {quality_filtered} low-quality samples")
        
        # Step 6: Filter by length (more lenient)
        if 'text' in df_clean.columns:
            before_length_filter = len(df_clean)
            # Only remove completely empty or extremely long texts
            df_clean = df_clean[
                (df_clean['text'].str.len() >= 5) & (df_clean['text'].str.len() <= self.max_text_length)
            ]
            length_filtered = before_length_filter - len(df_clean)
            if length_filtered > 0:
                validation_report['issues'].append(f"Removed {length_filtered} samples with inappropriate length")
        
        # Step 7: Add metadata
        df_clean['validated_at'] = datetime.now().isoformat()
        
        # Generate final report
        validation_report['valid_samples'] = len(df_clean)
        validation_report['removed_samples'] = initial_count - len(df_clean)
        
        if 'quality_score' in df_clean.columns:
            validation_report['quality_distribution'] = {
                'mean': float(df_clean['quality_score'].mean()),
                'median': float(df_clean['quality_score'].median()),
                'std': float(df_clean['quality_score'].std())
            }
        
        if 'category_cleaned' in df_clean.columns:
            validation_report['category_distribution'] = df_clean['category_cleaned'].value_counts().to_dict()
        
        logger.info(f"Validation complete: {len(df_clean)} valid samples from {initial_count} total")
        
        return df_clean, validation_report
    
    def save_validated_data(self, df: pd.DataFrame, validation_report: Dict, filename: str = None) -> str:
        """
        Save validated data with report
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/processed/validated_training_data_{timestamp}.csv"
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned data
        df.to_csv(filename, index=False, encoding='utf-8')
        
        # Save validation report
        report_filename = filename.replace('.csv', '_validation_report.json')
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validated data saved to: {filename}")
        logger.info(f"Validation report saved to: {report_filename}")
        
        return filename


def main():
    """
    Main validation function
    """
    # Find most recent collected data file
    data_dir = Path("data/collected")
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Validating data from: {latest_file}")
            
            # Load and validate data
            df = pd.read_csv(latest_file, encoding='utf-8')
            validator = DataValidator()
            clean_df, report = validator.validate_dataframe(df)
            
            # Save validated data
            filename = validator.save_validated_data(clean_df, report)
            
            # Print summary
            print(f"\n✅ VALIDATION SUMMARY")
            print(f"Original samples: {report['total_samples']:,}")
            print(f"Valid samples: {report['valid_samples']:,}")
            print(f"Removed samples: {report['removed_samples']:,}")
            print(f"Data quality: {report['quality_distribution'].get('mean', 0):.2f}")
            
            if report['category_distribution']:
                print(f"\n📂 CATEGORY DISTRIBUTION:")
                for category, count in report['category_distribution'].items():
                    print(f"  {category}: {count:,}")
            
            print(f"\n💾 Saved to: {filename}")
            
        else:
            print("❌ No collected data files found to validate")
    else:
        print("❌ No data directory found")

if __name__ == "__main__":
    main()