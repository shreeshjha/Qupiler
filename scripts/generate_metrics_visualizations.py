#!/usr/bin/env python3
"""
Academic Visualization Generator for Qupiler Optimization Metrics
Generates publication-ready graphs from quantum circuit optimization data
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set academic style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Comprehensive metrics data from all 14 test cases
data = {
    'test_cases': [
        'Add-Sub Chain', 'Basic Addition', 'AST Sample', 'Clean Addition', 
        'Clean Subtraction', 'Dynamic Addition', 'Dynamic Division', 'Dynamic Multiplication',
        'Left Shift', 'Multi-Test', 'Negation Testing', 'Realistic Division', 
        'Realistic Multiplication', 'Right Shift'
    ],
    'original_gates': [104, 46, 60, 45, 59, 45, 32, 60, 48, 61, 112, 33, 60, 47],
    'optimized_gates': [60, 26, 36, 26, 36, 26, 19, 33, 26, 33, 54, 20, 33, 22],
    'original_depth': [43, 23, 34, 23, 34, 23, 25, 37, 30, 37, 48, 26, 37, 25],
    'optimized_depth': [36, 17, 28, 17, 28, 17, 17, 26, 18, 26, 30, 18, 26, 15],
    'gate_reduction_pct': [42.3, 43.5, 40.0, 42.2, 39.0, 42.2, 40.6, 45.0, 45.8, 45.9, 51.8, 39.4, 45.0, 53.2],
    'depth_reduction_pct': [16.3, 26.1, 17.6, 26.1, 17.6, 26.1, 32.0, 29.7, 40.0, 29.7, 37.5, 30.8, 29.7, 40.0],
    'operation_type': ['Chaining', 'Arithmetic', 'Logic', 'Arithmetic', 'Arithmetic', 'Arithmetic', 
                      'Arithmetic', 'Arithmetic', 'Bitwise', 'Arithmetic', 'Logic', 'Arithmetic', 
                      'Arithmetic', 'Bitwise']
}

# Gate-specific optimization data (averages across relevant test cases)
gate_data = {
    'gate_types': ['H-Gate', 'S-Gate', 'X-Gate', 'T-Gate', 'CX-Gate', 'Toffoli'],
    'optimization_pct': [100.0, 100.0, 65.9, 27.6, 12.7, 7.6],
    'colors': ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#2ecc71', '#34495e']
}

def create_gate_count_reduction_chart():
    """Create comprehensive gate count reduction visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Before/After Gate Counts by Operation Type
    df = pd.DataFrame(data)
    
    # Group by operation type and calculate averages
    type_stats = df.groupby('operation_type').agg({
        'original_gates': 'mean',
        'optimized_gates': 'mean',
        'gate_reduction_pct': 'mean'
    }).round(1)
    
    x = np.arange(len(type_stats.index))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, type_stats['original_gates'], width, 
                   label='Original', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, type_stats['optimized_gates'], width,
                   label='Optimized', color='#2ecc71', alpha=0.8)
    
    ax1.set_xlabel('Operation Type', fontsize=12, weight='bold')
    ax1.set_ylabel('Average Gate Count', fontsize=12, weight='bold')
    ax1.set_title('Gate Count Reduction by Operation Type', fontsize=14, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(type_stats.index, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add reduction percentages on bars
    for i, (orig, opt, pct) in enumerate(zip(type_stats['original_gates'], 
                                           type_stats['optimized_gates'], 
                                           type_stats['gate_reduction_pct'])):
        ax1.text(i, max(orig, opt) + 2, f'{pct:.1f}%', ha='center', va='bottom', 
                weight='bold', fontsize=10)
    
    # Right: Distribution of optimization effectiveness
    ax2.hist(df['gate_reduction_pct'], bins=8, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.axvline(df['gate_reduction_pct'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Average: {df["gate_reduction_pct"].mean():.1f}%')
    ax2.set_xlabel('Gate Reduction Percentage (%)', fontsize=12, weight='bold')
    ax2.set_ylabel('Number of Test Cases', fontsize=12, weight='bold')
    ax2.set_title('Distribution of Optimization Effectiveness', fontsize=14, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gate_count_reduction_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

def create_gate_specific_optimization_chart():
    """Create gate-specific optimization performance radar chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Bar chart of gate-specific optimizations
    bars = ax1.bar(gate_data['gate_types'], gate_data['optimization_pct'], 
                  color=gate_data['colors'], alpha=0.8, edgecolor='black')
    
    ax1.set_ylabel('Optimization Percentage (%)', fontsize=12, weight='bold')
    ax1.set_title('Gate-Specific Optimization Performance', fontsize=14, weight='bold')
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, gate_data['optimization_pct']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%', ha='center', va='bottom', weight='bold')
    
    # Right: Circuit complexity vs optimization correlation
    df = pd.DataFrame(data)
    scatter = ax2.scatter(df['original_gates'], df['gate_reduction_pct'], 
                         s=100, alpha=0.7, c=df['original_depth'], cmap='viridis')
    
    # Add trend line
    z = np.polyfit(df['original_gates'], df['gate_reduction_pct'], 1)
    p = np.poly1d(z)
    ax2.plot(df['original_gates'], p(df['original_gates']), "r--", alpha=0.8)
    
    ax2.set_xlabel('Original Gate Count', fontsize=12, weight='bold')
    ax2.set_ylabel('Gate Reduction Percentage (%)', fontsize=12, weight='bold')
    ax2.set_title('Circuit Complexity vs Optimization Effectiveness', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Original Circuit Depth', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig('gate_specific_optimization_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

# def create_fault_tolerant_impact_chart():
#     """Create T-gate optimization impact chart for fault-tolerant quantum computing"""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
#     # T-gate specific data from multiplication/division operations
#     t_gate_operations = ['Dynamic Division', 'Realistic Division', 'Dynamic Multiplication', 
#                         'Multi-Test', 'Realistic Multiplication']
#     t_gate_original = [15, 15, 19, 19, 19]
#     t_gate_optimized = [15, 15, 11, 11, 11]
#     t_gate_reduction = [0, 0, 42.1, 42.1, 42.1]
    
#     # Left: T-gate count before/after
#     x = np.arange(len(t_gate_operations))
#     width = 0.35
    
#     bars1 = ax1.bar(x - width/2, t_gate_original, width, 
#                    label='Original T-Gates', color='#9b59b6', alpha=0.8)
#     bars2 = ax1.bar(x + width/2, t_gate_optimized, width,
#                    label='Optimized T-Gates', color='#8e44ad', alpha=0.8)
    
#     ax1.set_xlabel('Operation Type', fontsize=12, weight='bold')
#     ax1.set_ylabel('T-Gate Count', fontsize=12, weight='bold')
#     ax1.set_title('T-Gate Optimization Impact (Fault-Tolerant Focus)', fontsize=14, weight='bold')
#     ax1.set_xticks(x)
#     ax1.set_xticklabels([op.replace(' ', '\n') for op in t_gate_operations], fontsize=9)
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # Add reduction percentages
#     for i, pct in enumerate(t_gate_reduction):
#         if pct > 0:
#             ax1.text(i, max(t_gate_original[i], t_gate_optimized[i]) + 0.5, 
#                     f'{pct:.1f}%', ha='center', va='bottom', weight='bold', color='red')
    
#     # Right: Overall optimization summary
#     summary_metrics = ['Gate Count', 'Circuit Depth', 'T-Gates', 'H-Gates', 'S-Gates', 'X-Gates']
#     summary_values = [44.6, 28.3, 27.6, 100.0, 100.0, 65.9]
#     colors = ['#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#e67e22', '#2ecc71']
    
#     bars = ax2.barh(summary_metrics, summary_values, color=colors, alpha=0.8)
#     ax2.set_xlabel('Optimization Percentage (%)', fontsize=12, weight='bold')
#     ax2.set_title('Overall Qupiler Optimization Summary', fontsize=14, weight='bold')
#     ax2.grid(True, alpha=0.3)
    
#     # Add percentage labels
#     for bar, val in zip(bars, summary_values):
#         width = bar.get_width()
#         ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
#                 f'{val:.1f}%', ha='left', va='center', weight='bold')
    
#     plt.tight_layout()
#     plt.savefig('fault_tolerant_impact_analysis.png', dpi=300, bbox_inches='tight')
#     # plt.show()
def create_fault_tolerant_impact_chart():
    """Create T-gate optimization impact chart for fault-tolerant quantum computing"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # T-gate specific data from multiplication/division operations
    t_gate_operations = ['Dynamic Division', 'Realistic Division', 'Dynamic Multiplication', 
                        'Multi-Test', 'Realistic Multiplication']
    t_gate_original = [15, 15, 19, 19, 19]
    t_gate_optimized = [15, 15, 11, 11, 11]
    t_gate_reduction = [0, 0, 42.1, 42.1, 42.1]
    
    # Left: T-gate count before/after
    x = np.arange(len(t_gate_operations))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, t_gate_original, width, 
                   label='Original T-Gates', color='#9b59b6', alpha=0.8)
    bars2 = ax1.bar(x + width/2, t_gate_optimized, width,
                   label='Optimized T-Gates', color='#8e44ad', alpha=0.8)
    
    ax1.set_xlabel('Operation Type', fontsize=12, weight='bold')
    ax1.set_ylabel('T-Gate Count', fontsize=12, weight='bold')
    ax1.set_title('T-Gate Optimization Impact (Fault-Tolerant Focus)', fontsize=14, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([op.replace(' ', '\n') for op in t_gate_operations], fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add reduction percentages
    for i, pct in enumerate(t_gate_reduction):
        if pct > 0:
            ax1.text(i, max(t_gate_original[i], t_gate_optimized[i]) + 0.5, 
                    f'{pct:.1f}%', ha='center', va='bottom', weight='bold', color='red')
    
    # Right: Overall optimization summary (removed H-Gates and S-Gates)
    summary_metrics = ['Gate Count', 'Circuit Depth', 'T-Gates', 'X-Gates']
    summary_values = [44.6, 28.3, 27.6, 65.9]
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']
    
    bars = ax2.barh(summary_metrics, summary_values, color=colors, alpha=0.8)
    ax2.set_xlabel('Optimization Percentage (%)', fontsize=12, weight='bold')
    ax2.set_title('Overall Qupiler Optimization Summary', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, val in zip(bars, summary_values):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center', weight='bold')
    
    plt.tight_layout()
    plt.savefig('fault_tolerant_impact_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()


def create_multi_parameter_correlation():
    """Create multi-parameter correlation analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    df = pd.DataFrame(data)
    
    # 1. Gate Count vs Depth correlation
    ax1.scatter(df['original_gates'], df['original_depth'], 
               label='Original', alpha=0.7, s=100, color='#e74c3c')
    ax1.scatter(df['optimized_gates'], df['optimized_depth'], 
               label='Optimized', alpha=0.7, s=100, color='#2ecc71')
    ax1.set_xlabel('Gate Count', fontsize=12, weight='bold')
    ax1.set_ylabel('Circuit Depth', fontsize=12, weight='bold')
    ax1.set_title('Gate Count vs Circuit Depth Correlation', fontsize=14, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Optimization effectiveness by operation type
    type_colors = {'Arithmetic': '#3498db', 'Bitwise': '#e74c3c', 'Logic': '#f39c12', 'Chaining': '#9b59b6'}
    for op_type in df['operation_type'].unique():
        mask = df['operation_type'] == op_type
        ax2.scatter(df[mask]['original_gates'], df[mask]['gate_reduction_pct'], 
                   label=op_type, alpha=0.7, s=100, color=type_colors.get(op_type, '#34495e'))
    
    ax2.set_xlabel('Original Gate Count', fontsize=12, weight='bold')
    ax2.set_ylabel('Gate Reduction (%)', fontsize=12, weight='bold')
    ax2.set_title('Optimization by Operation Type', fontsize=14, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Reduction correlation
    ax3.scatter(df['gate_reduction_pct'], df['depth_reduction_pct'], 
               alpha=0.7, s=100, color='#8e44ad')
    
    # Add trend line
    z = np.polyfit(df['gate_reduction_pct'], df['depth_reduction_pct'], 1)
    p = np.poly1d(z)
    ax3.plot(df['gate_reduction_pct'], p(df['gate_reduction_pct']), "r--", alpha=0.8)
    
    ax3.set_xlabel('Gate Reduction (%)', fontsize=12, weight='bold')
    ax3.set_ylabel('Depth Reduction (%)', fontsize=12, weight='bold')
    ax3.set_title('Gate vs Depth Reduction Correlation', fontsize=14, weight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Optimization effectiveness distribution
    operation_stats = df.groupby('operation_type')['gate_reduction_pct'].agg(['mean', 'std']).fillna(0)
    
    ax4.bar(operation_stats.index, operation_stats['mean'], 
           yerr=operation_stats['std'], capsize=5,
           color=[type_colors.get(op, '#34495e') for op in operation_stats.index],
           alpha=0.8, edgecolor='black')
    
    ax4.set_ylabel('Average Gate Reduction (%)', fontsize=12, weight='bold')
    ax4.set_title('Optimization Performance by Operation Type', fontsize=14, weight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_parameter_correlation_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

# def create_academic_summary_figure():
#     """Create comprehensive academic summary figure"""
#     fig = plt.figure(figsize=(20, 12))
    
#     # Create a 2x3 grid
#     gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
#     # Main summary statistics
#     ax_main = fig.add_subplot(gs[0, :])
    
#     metrics = ['Average Gate\nReduction', 'Average Depth\nReduction', 'H-Gate\nElimination', 
#               'S-Gate\nElimination', 'X-Gate\nReduction', 'T-Gate\nOptimization']
#     values = [44.6, 28.3, 100.0, 100.0, 65.9, 27.6]
#     colors = ['#3498db', '#e74c3c', '#f39c12', '#e67e22', '#2ecc71', '#9b59b6']
    
#     bars = ax_main.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
#     ax_main.set_ylabel('Optimization Percentage (%)', fontsize=14, weight='bold')
#     ax_main.set_title('Qupiler Quantum Circuit Optimization Summary (14 Test Cases)', 
#                      fontsize=16, weight='bold', pad=20)
#     ax_main.set_ylim(0, 110)
#     ax_main.grid(True, alpha=0.3, axis='y')
    
#     # Add value labels on bars
#     for bar, val in zip(bars, values):
#         height = bar.get_height()
#         ax_main.text(bar.get_x() + bar.get_width()/2., height + 1,
#                     f'{val:.1f}%', ha='center', va='bottom', weight='bold', fontsize=12)
    
#     # Bottom left: Operation type breakdown
#     ax1 = fig.add_subplot(gs[1, 0])
#     df = pd.DataFrame(data)
#     type_stats = df.groupby('operation_type')['gate_reduction_pct'].mean()
    
#     wedges, texts, autotexts = ax1.pie(type_stats.values, labels=type_stats.index, autopct='%1.1f%%',
#                                       colors=['#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
#     ax1.set_title('Optimization by Operation Type', fontsize=12, weight='bold')
    
#     # Bottom middle: Gate count distribution
#     ax2 = fig.add_subplot(gs[1, 1])
#     ax2.hist([df['original_gates'], df['optimized_gates']], 
#             bins=8, alpha=0.7, label=['Original', 'Optimized'],
#             color=['#e74c3c', '#2ecc71'])
#     ax2.set_xlabel('Gate Count', fontsize=10, weight='bold')
#     ax2.set_ylabel('Frequency', fontsize=10, weight='bold')
#     ax2.set_title('Gate Count Distribution', fontsize=12, weight='bold')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     # Bottom right: Key statistics
#     ax3 = fig.add_subplot(gs[1, 2])
#     ax3.axis('off')
    
#     stats_text = f"""
#     Key Statistics:
    
#     • Total Test Cases: 14
#     • Average Gate Reduction: 44.6%
#     • Average Depth Reduction: 28.3%
#     • Circuit Width Preserved: 100%
    
#     Gate-Specific Results:
#     • H-Gates: 100% eliminated
#     • S-Gates: 100% eliminated  
#     • X-Gates: 65.9% reduced
#     • T-Gates: 27.6% reduced
#     • CX-Gates: 12.7% reduced
    
#     Operation Coverage:
#     • Arithmetic: 9 test cases
#     • Bitwise: 2 test cases
#     • Logic: 2 test cases
#     • Chaining: 1 test case
#     """
    
#     ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=11,
#             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
#     plt.savefig('qupiler_academic_summary.png', dpi=300, bbox_inches='tight')
#     # plt.show()

def create_academic_summary_figure():
    """Create comprehensive academic summary figure"""
    fig = plt.figure(figsize=(20, 12))
    
    # Create a 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Main summary statistics (removed H-Gate and S-Gate)
    ax_main = fig.add_subplot(gs[0, :])
    
    metrics = ['Average Gate\nReduction', 'Average Depth\nReduction', 
              'X-Gate\nReduction', 'T-Gate\nOptimization']
    values = [44.6, 28.3, 65.9, 27.6]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    bars = ax_main.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax_main.set_ylabel('Optimization Percentage (%)', fontsize=14, weight='bold')
    ax_main.set_title('Qupiler Quantum Circuit Optimization Summary (14 Test Cases)', 
                     fontsize=16, weight='bold', pad=20)
    ax_main.set_ylim(0, 110)
    ax_main.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax_main.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}%', ha='center', va='bottom', weight='bold', fontsize=12)
    
    # Bottom left: Operation type breakdown
    ax1 = fig.add_subplot(gs[1, 0])
    df = pd.DataFrame(data)
    type_stats = df.groupby('operation_type')['gate_reduction_pct'].mean()
    
    wedges, texts, autotexts = ax1.pie(type_stats.values, labels=type_stats.index, autopct='%1.1f%%',
                                      colors=['#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
    ax1.set_title('Optimization by Operation Type', fontsize=12, weight='bold')
    
    # Bottom middle: Gate count distribution
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.hist([df['original_gates'], df['optimized_gates']], 
            bins=8, alpha=0.7, label=['Original', 'Optimized'],
            color=['#e74c3c', '#2ecc71'])
    ax2.set_xlabel('Gate Count', fontsize=10, weight='bold')
    ax2.set_ylabel('Frequency', fontsize=10, weight='bold')
    ax2.set_title('Gate Count Distribution', fontsize=12, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bottom right: Key statistics (removed H-Gates and S-Gates mentions)
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis('off')
    
    stats_text = f"""
    Key Statistics:
    
    • Total Test Cases: 14
    • Average Gate Reduction: 44.6%
    • Average Depth Reduction: 28.3%
    • Circuit Width Preserved: 100%
    
    Gate-Specific Results:
    • X-Gates: 65.9% reduced
    • T-Gates: 27.6% reduced
    • CX-Gates: 12.7% reduced
    
    Operation Coverage:
    • Arithmetic: 9 test cases
    • Bitwise: 2 test cases
    • Logic: 2 test cases
    • Chaining: 1 test case
    """
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.savefig('qupiler_academic_summary.png', dpi=300, bbox_inches='tight')
    # plt.show()


def plot_gate_count_reduction_by_operation_type():
    """Save bar chart of average gate count before/after by operation type (single figure)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df = pd.DataFrame(data)
    type_stats = df.groupby('operation_type').agg({
        'original_gates': 'mean',
        'optimized_gates': 'mean',
        'gate_reduction_pct': 'mean'
    }).round(1)

    x = np.arange(len(type_stats.index))
    width = 0.38

    ax.bar(x - width/2, type_stats['original_gates'], width,
           label='Original', color='#e74c3c', alpha=0.85)
    ax.bar(x + width/2, type_stats['optimized_gates'], width,
           label='Optimized', color='#2ecc71', alpha=0.85)

    ax.set_xlabel('Operation Type', fontsize=12, weight='bold')
    ax.set_ylabel('Average Gate Count', fontsize=12, weight='bold')
    ax.set_title('Gate Count Reduction by Operation Type', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(type_stats.index, rotation=25, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    for i, (orig, opt, pct) in enumerate(zip(type_stats['original_gates'],
                                             type_stats['optimized_gates'],
                                             type_stats['gate_reduction_pct'])):
        ax.text(i, max(orig, opt) + 2, f'{pct:.1f}%', ha='center', va='bottom',
                weight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('gate_count_reduction_by_operation_type.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_gate_reduction_distribution_hist():
    """Save histogram of gate reduction percentages (single figure)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.DataFrame(data)

    ax.hist(df['gate_reduction_pct'], bins=8, color='#3498db', alpha=0.7, edgecolor='black')
    ax.axvline(df['gate_reduction_pct'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Average: {df["gate_reduction_pct"].mean():.1f}%')
    ax.set_xlabel('Gate Reduction Percentage (%)', fontsize=12, weight='bold')
    ax.set_ylabel('Number of Test Cases', fontsize=12, weight='bold')
    ax.set_title('Distribution of Optimization Effectiveness', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gate_reduction_distribution_hist.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_gate_reduction_distribution_kde():
    """Save KDE line plot of gate reduction percentages for a smooth, appealing view."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.DataFrame(data)

    sns.kdeplot(df['gate_reduction_pct'], ax=ax, bw_adjust=0.8, fill=False, color='#2c3e50', linewidth=2)
    ax.set_xlabel('Gate Reduction Percentage (%)', fontsize=12, weight='bold')
    ax.set_ylabel('Density', fontsize=12, weight='bold')
    ax.set_title('Gate Reduction Distribution (KDE)', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gate_reduction_distribution_kde.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_gate_specific_optimization_bar():
    """Save bar chart of gate-specific optimization percentages (single figure)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(gate_data['gate_types'], gate_data['optimization_pct'],
                  color=gate_data['colors'], alpha=0.85, edgecolor='black')
    ax.set_ylabel('Optimization Percentage (%)', fontsize=12, weight='bold')
    ax.set_title('Gate-Specific Optimization Performance', fontsize=14, weight='bold')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    for bar, pct in zip(bars, gate_data['optimization_pct']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%', ha='center', va='bottom', weight='bold')

    plt.tight_layout()
    plt.savefig('gate_specific_optimization_bar.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_circuit_complexity_vs_optimization():
    """Save scatter with trend line: original gate count vs gate reduction (%)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.DataFrame(data)

    scatter = ax.scatter(df['original_gates'], df['gate_reduction_pct'],
                         s=100, alpha=0.75, c=df['original_depth'], cmap='viridis', edgecolor='black')

    z = np.polyfit(df['original_gates'], df['gate_reduction_pct'], 1)
    p = np.poly1d(z)
    xs = np.linspace(df['original_gates'].min(), df['original_gates'].max(), 100)
    ax.plot(xs, p(xs), 'r--', alpha=0.9, linewidth=1.8, label='Trend')

    ax.set_xlabel('Original Gate Count', fontsize=12, weight='bold')
    ax.set_ylabel('Gate Reduction Percentage (%)', fontsize=12, weight='bold')
    ax.set_title('Circuit Complexity vs Optimization Effectiveness', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Original Circuit Depth', fontsize=10, weight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig('circuit_complexity_vs_optimization.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_t_gate_impact():
    """Save grouped bar chart of T-gate count before/after for key operations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    t_gate_operations = ['Dynamic Division', 'Realistic Division', 'Dynamic Multiplication',
                         'Multi-Test', 'Realistic Multiplication']
    t_gate_original = [15, 15, 19, 19, 19]
    t_gate_optimized = [15, 15, 11, 11, 11]
    t_gate_reduction = [0, 0, 42.1, 42.1, 42.1]

    x = np.arange(len(t_gate_operations))
    width = 0.38

    ax.bar(x - width/2, t_gate_original, width, label='Original T-Gates', color='#9b59b6', alpha=0.85)
    ax.bar(x + width/2, t_gate_optimized, width, label='Optimized T-Gates', color='#8e44ad', alpha=0.85)

    ax.set_xlabel('Operation Type', fontsize=12, weight='bold')
    ax.set_ylabel('T-Gate Count', fontsize=12, weight='bold')
    ax.set_title('T-Gate Optimization Impact (Fault-Tolerant Focus)', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([op.replace(' ', '\n') for op in t_gate_operations], fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    for i, pct in enumerate(t_gate_reduction):
        if pct > 0:
            ax.text(i, max(t_gate_original[i], t_gate_optimized[i]) + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', weight='bold', color='red')

    plt.tight_layout()
    plt.savefig('t_gate_impact.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_overall_optimization_summary():
    """Save horizontal bar chart summarizing overall optimization percentages."""
    fig, ax = plt.subplots(figsize=(10, 6))

    summary_metrics = ['Gate Count', 'Circuit Depth', 'T-Gates', 'X-Gates']
    summary_values = [44.6, 28.3, 27.6, 65.9]
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

    bars = ax.barh(summary_metrics, summary_values, color=colors, alpha=0.85)
    ax.set_xlabel('Optimization Percentage (%)', fontsize=12, weight='bold')
    ax.set_title('Overall Qupiler Optimization Summary', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)

    for bar, val in zip(bars, summary_values):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                ha='left', va='center', weight='bold')

    plt.tight_layout()
    plt.savefig('overall_optimization_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_gate_count_vs_depth_correlation():
    """Save scatter plot of gate count vs circuit depth for original and optimized."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.DataFrame(data)

    ax.scatter(df['original_gates'], df['original_depth'], label='Original', alpha=0.75,
               s=100, color='#e74c3c', edgecolor='black')
    ax.scatter(df['optimized_gates'], df['optimized_depth'], label='Optimized', alpha=0.75,
               s=100, color='#2ecc71', edgecolor='black')

    ax.set_xlabel('Gate Count', fontsize=12, weight='bold')
    ax.set_ylabel('Circuit Depth', fontsize=12, weight='bold')
    ax.set_title('Gate Count vs Circuit Depth Correlation', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gate_count_vs_depth_correlation.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_optimization_by_operation_type():
    """Save scatter of gate reduction vs original gates, colored by operation type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.DataFrame(data)
    type_colors = {'Arithmetic': '#3498db', 'Bitwise': '#e74c3c', 'Logic': '#f39c12', 'Chaining': '#9b59b6'}

    for op_type in df['operation_type'].unique():
        mask = df['operation_type'] == op_type
        ax.scatter(df[mask]['original_gates'], df[mask]['gate_reduction_pct'],
                   label=op_type, alpha=0.8, s=110, color=type_colors.get(op_type, '#34495e'), edgecolor='black')

    ax.set_xlabel('Original Gate Count', fontsize=12, weight='bold')
    ax.set_ylabel('Gate Reduction (%)', fontsize=12, weight='bold')
    ax.set_title('Optimization by Operation Type', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimization_by_operation_type.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_gate_vs_depth_reduction_correlation():
    """Save scatter with trend line: gate reduction vs depth reduction."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.DataFrame(data)

    ax.scatter(df['gate_reduction_pct'], df['depth_reduction_pct'], alpha=0.75, s=110, color='#8e44ad', edgecolor='black')

    z = np.polyfit(df['gate_reduction_pct'], df['depth_reduction_pct'], 1)
    p = np.poly1d(z)
    xs = np.linspace(df['gate_reduction_pct'].min(), df['gate_reduction_pct'].max(), 100)
    ax.plot(xs, p(xs), 'r--', alpha=0.9, linewidth=1.8, label='Trend')

    ax.set_xlabel('Gate Reduction (%)', fontsize=12, weight='bold')
    ax.set_ylabel('Depth Reduction (%)', fontsize=12, weight='bold')
    ax.set_title('Gate vs Depth Reduction Correlation', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('gate_vs_depth_reduction_correlation.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_optimization_performance_by_operation_type():
    """Save bar chart with error bars: avg gate reduction by operation type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.DataFrame(data)
    type_colors = {'Arithmetic': '#3498db', 'Bitwise': '#e74c3c', 'Logic': '#f39c12', 'Chaining': '#9b59b6'}

    operation_stats = df.groupby('operation_type')['gate_reduction_pct'].agg(['mean', 'std']).fillna(0)
    ax.bar(operation_stats.index, operation_stats['mean'], yerr=operation_stats['std'], capsize=5,
           color=[type_colors.get(op, '#34495e') for op in operation_stats.index], alpha=0.85,
           edgecolor='black')

    ax.set_ylabel('Average Gate Reduction (%)', fontsize=12, weight='bold')
    ax.set_title('Optimization Performance by Operation Type', fontsize=14, weight='bold')
    ax.tick_params(axis='x', rotation=25)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimization_performance_by_operation_type.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_summary_bars():
    """Save summary bars for key optimization metrics (single figure)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Average Gate\nReduction', 'Average Depth\nReduction', 'X-Gate\nReduction', 'T-Gate\nOptimization']
    values = [44.6, 28.3, 65.9, 27.6]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    bars = ax.bar(metrics, values, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Optimization Percentage (%)', fontsize=12, weight='bold')
    ax.set_title('Qupiler Optimization Summary (14 Test Cases)', fontsize=14, weight='bold')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{val:.1f}%', ha='center', va='bottom',
                weight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig('summary_bars.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_operation_type_breakdown_pie():
    """Save pie chart for optimization by operation type (single figure)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    df = pd.DataFrame(data)
    type_stats = df.groupby('operation_type')['gate_reduction_pct'].mean()

    ax.pie(type_stats.values, labels=type_stats.index, autopct='%1.1f%%',
           colors=['#3498db', '#e74c3c', '#f39c12', '#9b59b6'], textprops={'fontsize': 11})
    ax.set_title('Optimization by Operation Type', fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig('operation_type_breakdown_pie.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_gate_count_distribution_hist():
    """Save histogram comparing original vs optimized gate counts (single figure)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.DataFrame(data)

    ax.hist([df['original_gates'], df['optimized_gates']], bins=8, alpha=0.7, label=['Original', 'Optimized'],
            color=['#e74c3c', '#2ecc71'], edgecolor='black')
    ax.set_xlabel('Gate Count', fontsize=12, weight='bold')
    ax.set_ylabel('Frequency', fontsize=12, weight='bold')
    ax.set_title('Gate Count Distribution', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gate_count_distribution_hist.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_gate_count_distribution_kde():
    """Save KDE line plot comparing original vs optimized gate counts (more appealing lines)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.DataFrame(data)

    sns.kdeplot(df['original_gates'], ax=ax, bw_adjust=0.8, fill=False, color='#e74c3c', linewidth=2, label='Original')
    sns.kdeplot(df['optimized_gates'], ax=ax, bw_adjust=0.8, fill=False, color='#2ecc71', linewidth=2, label='Optimized')

    ax.set_xlabel('Gate Count', fontsize=12, weight='bold')
    ax.set_ylabel('Density', fontsize=12, weight='bold')
    ax.set_title('Gate Count Distribution (KDE)', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('gate_count_distribution_kde.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_key_statistics_card():
    """Save a simple figure with key statistics as text (single figure)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    stats_text = (
        "Key Statistics:\n\n"
        "• Total Test Cases: 14\n"
        "• Average Gate Reduction: 44.6%\n"
        "• Average Depth Reduction: 28.3%\n"
        "• Circuit Width Preserved: 100%\n\n"
        "Gate-Specific Results:\n"
        "• X-Gates: 65.9% reduced\n"
        "• T-Gates: 27.6% reduced\n"
        "• CX-Gates: 12.7% reduced\n\n"
        "Operation Coverage:\n"
        "• Arithmetic: 9 test cases\n"
        "• Bitwise: 2 test cases\n"
        "• Logic: 2 test cases\n"
        "• Chaining: 1 test case\n"
    )

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12, va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig('key_statistics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    """Generate all academic visualizations"""
    print("Generating academic visualizations for Qupiler optimization metrics...")
    
    # Create all visualizations (separate figures)
    # Gate count reduction and distributions
    plot_gate_count_reduction_by_operation_type()
    plot_gate_reduction_distribution_hist()
    plot_gate_reduction_distribution_kde()

    # Gate-specific and complexity correlation
    plot_gate_specific_optimization_bar()
    plot_circuit_complexity_vs_optimization()

    # Fault-tolerant T-gate impact and overall summary
    plot_t_gate_impact()
    plot_overall_optimization_summary()

    # Multi-parameter correlations
    plot_gate_count_vs_depth_correlation()
    plot_optimization_by_operation_type()
    plot_gate_vs_depth_reduction_correlation()
    plot_optimization_performance_by_operation_type()

    # Summary components, split
    plot_summary_bars()
    plot_operation_type_breakdown_pie()
    plot_gate_count_distribution_hist()
    plot_gate_count_distribution_kde()
    plot_key_statistics_card()
    
    print("✅ All visualizations generated successfully!")
    print("📁 Generated files:")
    print("   • gate_count_reduction_by_operation_type.png")
    print("   • gate_reduction_distribution_hist.png")
    print("   • gate_reduction_distribution_kde.png")
    print("   • gate_specific_optimization_bar.png")
    print("   • circuit_complexity_vs_optimization.png")
    print("   • t_gate_impact.png")
    print("   • overall_optimization_summary.png")
    print("   • gate_count_vs_depth_correlation.png")
    print("   • optimization_by_operation_type.png")
    print("   • gate_vs_depth_reduction_correlation.png")
    print("   • optimization_performance_by_operation_type.png")
    print("   • summary_bars.png")
    print("   • operation_type_breakdown_pie.png")
    print("   • gate_count_distribution_hist.png")
    print("   • gate_count_distribution_kde.png")
    print("   • key_statistics.png")

if __name__ == "__main__":
    main()
