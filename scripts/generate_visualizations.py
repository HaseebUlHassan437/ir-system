"""Generate visualizations for IR system comparison"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory
os.makedirs('experiments/plots', exist_ok=True)

print("="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Load results
df = pd.read_csv('experiments/results/system_comparison_with_hybrid.csv', index_col=0)

print("\nLoaded results:")
print(df)

# 1. Precision Comparison (P@5, P@10, P@20)
print("\n1. Creating Precision Comparison Chart...")
fig, ax = plt.subplots(figsize=(10, 6))

systems = df.index
x = np.arange(len(systems))
width = 0.25

p5 = df['P@5'].values
p10 = df['P@10'].values
p20 = df['P@20'].values

bars1 = ax.bar(x - width, p5, width, label='P@5', color='steelblue')
bars2 = ax.bar(x, p10, width, label='P@10', color='coral')
bars3 = ax.bar(x + width, p20, width, label='P@20', color='lightgreen')

ax.set_xlabel('System', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision Comparison Across Systems', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(systems, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/plots/precision_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: experiments/plots/precision_comparison.png")
plt.close()

# 2. Recall Comparison
print("\n2. Creating Recall Comparison Chart...")
fig, ax = plt.subplots(figsize=(10, 6))

r5 = df['R@5'].values
r10 = df['R@10'].values
r20 = df['R@20'].values

bars1 = ax.bar(x - width, r5, width, label='R@5', color='purple')
bars2 = ax.bar(x, r10, width, label='R@10', color='orange')
bars3 = ax.bar(x + width, r20, width, label='R@20', color='pink')

ax.set_xlabel('System', fontsize=12, fontweight='bold')
ax.set_ylabel('Recall', fontsize=12, fontweight='bold')
ax.set_title('Recall Comparison Across Systems', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(systems, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/plots/recall_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: experiments/plots/recall_comparison.png")
plt.close()

# 3. MAP and NDCG Comparison
print("\n3. Creating MAP and NDCG Comparison Chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MAP
axes[0].bar(systems, df['MAP'].values, color='teal', alpha=0.7)
axes[0].set_xlabel('System', fontsize=12, fontweight='bold')
axes[0].set_ylabel('MAP Score', fontsize=12, fontweight='bold')
axes[0].set_title('Mean Average Precision (MAP)', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# NDCG@10
axes[1].bar(systems, df['NDCG@10'].values, color='crimson', alpha=0.7)
axes[1].set_xlabel('System', fontsize=12, fontweight='bold')
axes[1].set_ylabel('NDCG@10 Score', fontsize=12, fontweight='bold')
axes[1].set_title('Normalized Discounted Cumulative Gain @10', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/plots/map_ndcg_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: experiments/plots/map_ndcg_comparison.png")
plt.close()

# 4. Query Time Comparison
print("\n4. Creating Query Time Comparison Chart...")
fig, ax = plt.subplots(figsize=(10, 6))

query_times = df['Avg Query Time (ms)'].values
colors = ['green' if t < 1 else 'yellow' if t < 5 else 'red' for t in query_times]

bars = ax.bar(systems, query_times, color=colors, alpha=0.7)
ax.set_xlabel('System', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Query Time (ms)', fontsize=12, fontweight='bold')
ax.set_title('Query Speed Comparison (Lower is Better)', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}ms',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('experiments/plots/query_time_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: experiments/plots/query_time_comparison.png")
plt.close()

# 5. Precision-Recall Trade-off
print("\n5. Creating Precision-Recall Trade-off Chart...")
fig, ax = plt.subplots(figsize=(10, 6))

for system in systems:
    precision = [df.loc[system, 'P@5'], df.loc[system, 'P@10'], df.loc[system, 'P@20']]
    recall = [df.loc[system, 'R@5'], df.loc[system, 'R@10'], df.loc[system, 'R@20']]
    ax.plot(recall, precision, marker='o', linewidth=2, markersize=8, label=system)

ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/plots/precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: experiments/plots/precision_recall_tradeoff.png")
plt.close()

# 6. Overall Performance Radar Chart
print("\n6. Creating Overall Performance Radar Chart...")

# Normalize metrics to 0-1 scale for radar chart
metrics = ['P@10', 'R@10', 'NDCG@10', 'MAP']

# Select top 3 systems for clarity
top_systems = ['TF-IDF', 'BM25', 'Hybrid-Fusion']

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for system in top_systems:
    if system in df.index:
        values = df.loc[system, metrics].tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=system, markersize=8)
        ax.fill(angles, values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_title('Overall Performance Comparison\n(Top 3 Systems)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig('experiments/plots/performance_radar.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: experiments/plots/performance_radar.png")
plt.close()
# 7. Speed vs Accuracy Trade-off
print("\n7. Creating Speed vs Accuracy Trade-off Chart...")
fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(df['Avg Query Time (ms)'], df['P@10'], 
                    s=200, alpha=0.6, c=range(len(systems)), cmap='viridis')

# Add labels for each point
for i, system in enumerate(systems):
    ax.annotate(system, 
                (df.loc[system, 'Avg Query Time (ms)'], df.loc[system, 'P@10']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, fontweight='bold')

ax.set_xlabel('Average Query Time (ms) - Lower is Better', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision@10 - Higher is Better', fontsize=12, fontweight='bold')
ax.set_title('Speed vs Accuracy Trade-off', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add quadrant lines
median_time = df['Avg Query Time (ms)'].median()
median_precision = df['P@10'].median()
ax.axvline(median_time, color='gray', linestyle='--', alpha=0.5)
ax.axhline(median_precision, color='gray', linestyle='--', alpha=0.5)

# Add annotations for quadrants
ax.text(0.02, 0.98, 'Fast & Accurate\n(Ideal)', 
        transform=ax.transAxes, fontsize=10, 
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig('experiments/plots/speed_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: experiments/plots/speed_accuracy_tradeoff.png")
plt.close()

# 8. Summary Table Image
print("\n8. Creating Summary Table...")
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Select key metrics for table
table_data = df[['MAP', 'P@10', 'R@10', 'NDCG@10', 'Avg Query Time (ms)']].round(4)
table_data.columns = ['MAP', 'P@10', 'R@10', 'NDCG@10', 'Query Time (ms)']

table = ax.table(cellText=table_data.values,
                rowLabels=table_data.index,
                colLabels=table_data.columns,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style header
for i in range(len(table_data.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style row labels
for i in range(len(table_data)):
    table[(i+1, -1)].set_facecolor('#D9E1F2')
    table[(i+1, -1)].set_text_props(weight='bold')

# Highlight best values in each column
for col_idx in range(len(table_data.columns)):
    col_name = table_data.columns[col_idx]
    
    if col_name == 'Query Time (ms)':
        best_idx = table_data[col_name].idxmin()  # Lower is better
    else:
        best_idx = table_data[col_name].idxmax()  # Higher is better
    
    row_idx = list(table_data.index).index(best_idx) + 1
    table[(row_idx, col_idx)].set_facecolor('#C6E0B4')
    table[(row_idx, col_idx)].set_text_props(weight='bold')

plt.title('System Performance Summary', fontsize=16, fontweight='bold', pad=20)
plt.savefig('experiments/plots/performance_summary_table.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: experiments/plots/performance_summary_table.png")
plt.close()

print("\n" + "="*60)
print("✅ All visualizations generated successfully!")
print("="*60)
print("\nGenerated plots:")
print("  1. precision_comparison.png")
print("  2. recall_comparison.png")
print("  3. map_ndcg_comparison.png")
print("  4. query_time_comparison.png")
print("  5. precision_recall_tradeoff.png")
print("  6. performance_radar.png")
print("  7. speed_accuracy_tradeoff.png")
print("  8. performance_summary_table.png")
print("\nLocation: experiments/plots/")