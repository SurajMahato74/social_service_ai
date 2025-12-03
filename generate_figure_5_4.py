"""
Figure 5.4: Model Performance Comparison - STANDALONE VERSION
Run this script to generate and display the bar chart
"""

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for display
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Model performance data from thesis Chapter 5
models = ['Logistic\nRegression', 'Random\nForest', 'SVM\n(RBF)', 'Ensemble\n(Voting)']
accuracy = [71.88, 72.40, 71.35, 72.40]
f1_score = [71.81, 69.85, 71.98, 71.98]
cv_score = [75.80, 72.58, 75.66, 75.80]

# Set up the bar positions
x = np.arange(len(models))
width = 0.25

# Create figure with white background
fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
ax.set_facecolor('white')

# Create bars with distinct colors
bars1 = ax.bar(x - width, accuracy, width, label='Accuracy (%)', 
               color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, f1_score, width, label='F1-Score (%)', 
               color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, cv_score, width, label='Cross-Validation Score (%)', 
               color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.5)

# Customize the chart
ax.set_xlabel('Machine Learning Model', fontsize=13, fontweight='bold')
ax.set_ylabel('Performance Score (%)', fontsize=13, fontweight='bold')
ax.set_title('Figure 5.4: Model Performance Comparison\nAccuracy, F1-Score, and Cross-Validation Scores Across Models', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.set_ylim(65, 80)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

# Add value labels on bars
def add_value_labels(bars, color='black'):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)

add_value_labels(bars1, '#2c3e50')
add_value_labels(bars2, '#27ae60')
add_value_labels(bars3, '#c0392b')

# Add a horizontal reference line
ax.axhline(y=72, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
ax.text(3.6, 72.3, 'Target: 72%', fontsize=10, color='gray', style='italic', fontweight='bold')

# Highlight the best model (Ensemble) with gold border
for i in [3, 7, 11]:  # Ensemble bars
    ax.patches[i].set_edgecolor('#f39c12')
    ax.patches[i].set_linewidth(3)

# Add annotation for best model
ax.annotate('Best Overall\nPerformance', 
            xy=(3, 75.8), xytext=(3.7, 77.8),
            arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2.5),
            fontsize=11, fontweight='bold', color='#d35400',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#fff9e6', 
                     edgecolor='#f39c12', linewidth=2.5))

# Add thesis reference
fig.text(0.99, 0.01, 'Source: Thesis Chapter 5 - Results and Analysis', 
         ha='right', va='bottom', fontsize=9, style='italic', color='gray')

plt.tight_layout()

# Save the figure in high resolution
output_path = 'visualizations/figure_5_4_model_performance.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n{'='*60}")
print(f"✓ SUCCESS: Figure 5.4 generated successfully!")
print(f"{'='*60}")
print(f"📊 Chart saved to: {output_path}")
print(f"📈 Resolution: 300 DPI (publication quality)")
print(f"{'='*60}\n")

# Display the chart
print("Displaying chart... (Close the window to exit)")
plt.show()
