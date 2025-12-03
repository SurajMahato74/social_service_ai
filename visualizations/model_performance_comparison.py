"""
Figure 5.4: Model Performance Comparison
Bar chart comparing Accuracy, F1-Score, and CV scores across LR, RF, SVM, and Ensemble
"""

import matplotlib.pyplot as plt
import numpy as np

# Model performance data from thesis results
models = ['Logistic\nRegression', 'Random\nForest', 'SVM\n(RBF)', 'Ensemble\n(Voting)']
accuracy = [71.88, 72.40, 71.35, 72.40]
f1_score = [71.81, 69.85, 71.98, 71.98]
cv_score = [75.80, 72.58, 75.66, 75.80]

# Set up the bar positions
x = np.arange(len(models))
width = 0.25

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 7))

# Create bars
bars1 = ax.bar(x - width, accuracy, width, label='Accuracy (%)', color='#3498db', alpha=0.8)
bars2 = ax.bar(x, f1_score, width, label='F1-Score (%)', color='#2ecc71', alpha=0.8)
bars3 = ax.bar(x + width, cv_score, width, label='Cross-Validation Score (%)', color='#e74c3c', alpha=0.8)

# Customize the chart
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Figure 5.4: Model Performance Comparison\nAccuracy, F1-Score, and Cross-Validation Scores', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(loc='upper left', fontsize=10)
ax.set_ylim(60, 80)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

# Add a horizontal line at 72% for reference
ax.axhline(y=72, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.text(3.5, 72.5, 'Target: 72%', fontsize=9, color='gray', style='italic')

# Highlight the best model (Ensemble)
ax.patches[3].set_edgecolor('gold')
ax.patches[3].set_linewidth(2.5)
ax.patches[7].set_edgecolor('gold')
ax.patches[7].set_linewidth(2.5)
ax.patches[11].set_edgecolor('gold')
ax.patches[11].set_linewidth(2.5)

# Add annotation for best model
ax.annotate('Best Overall\nPerformance', 
            xy=(3, 75.8), xytext=(3.5, 77.5),
            arrowprops=dict(arrowstyle='->', color='gold', lw=2),
            fontsize=10, fontweight='bold', color='#f39c12',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gold', linewidth=2))

plt.tight_layout()

# Save the figure
output_path = 'visualizations/figure_5_4_model_performance.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure saved to: {output_path}")

# Display the chart
plt.show()
