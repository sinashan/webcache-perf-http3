#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import numpy as np
import matplotlib
sys.path.append('/home/ubuntu/measure')
from common import (golden_ratio, figwidth, color_pallete, kfmt,
                    legendHandleTestPad, legendColumnSpacing, 
                    legendHandleLength, legendLabelSpacing, 
                    legendBorderpadSpacing)

results_dir = "results/"
os.makedirs(results_dir, exist_ok=True)

def analyze_results(csv_file):
    # Load data
    df = pd.read_csv(csv_file)
    
    # Check if we need to rename columns for compatibility
    if 'resource_type' in df.columns and 'asset_type' not in df.columns:
        df['asset_type'] = df['resource_type']
        print("Renamed 'resource_type' column to 'asset_type' for analysis")
    
    # Check if we have 0-RTT data available
    has_zero_rtt = 'zero_rtt_used' in df.columns
    if has_zero_rtt:
        print("0-RTT resumption data available - generating additional analyses")
        
        # Convert string boolean values to actual booleans
        if df['zero_rtt_used'].dtype == 'object':
            df['zero_rtt_used'] = df['zero_rtt_used'].map({'true': True, 'false': False})
        if 'tls_resumed' in df.columns and df['tls_resumed'].dtype == 'object':
            df['tls_resumed'] = df['tls_resumed'].map({'true': True, 'false': False})
        if 'connection_reused' in df.columns and df['connection_reused'].dtype == 'object':
            df['connection_reused'] = df['connection_reused'].map({'true': True, 'false': False})
        if 'from_disk_cache' in df.columns and df['from_disk_cache'].dtype == 'object':
            df['from_disk_cache'] = df['from_disk_cache'].map({'true': True, 'false': False})
    
    # Calculate statistics grouped by protocol, cache state and asset type
    stats = df.groupby(['protocol', 'cache_state', 'asset_type'])['load_time_ms'].agg(
        ['mean', 'std', 'min', 'max', 'count']
    ).reset_index()
    
    # Print summary
    print(stats)
    
    # Save detailed stats
    stats.to_csv(f"{csv_file.replace('.csv', '_stats.csv')}", index=False)
    
    # Set plot style according to common.py standards
    plt.rcParams['figure.figsize'] = (figwidth, figwidth / golden_ratio)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['text.usetex'] = False  # Disable LaTeX to avoid requiring installation
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', color_pallete)
    
    # Adjust font sizes - reduced from the original
    title_size = 14
    label_size = 12
    tick_size = 8
    legend_font_size = 11  # Slightly smaller than in common.py
    
    # Generate comparison charts
    for cache_state in df['cache_state'].unique():
        plt.figure()
        
        subset = df[df['cache_state'] == cache_state]
        ax = sns.barplot(x='asset_type', y='load_time_ms', hue='protocol', data=subset, 
                         errorbar='sd', palette=color_pallete[:2])
        
        # Apply formatting with adjusted sizes
        ax.set_title(f'HTTP Protocol Load Times ({cache_state.title()} Cache)', fontsize=title_size)
        ax.set_xlabel('Asset Type', fontsize=label_size)
        ax.set_ylabel('Load Time (ms)', fontsize=label_size)
        ax.tick_params(axis='x', rotation=30, labelsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)
        
        # Format y-axis with kfmt if values are large
        if subset['load_time_ms'].max() > 1000:
            ax.yaxis.set_major_formatter(kfmt)
            
        # Improve legend using common.py parameters with adjusted size
        ax.legend(title='Protocol', frameon=False, 
                  fontsize=legend_font_size,
                  handletextpad=legendHandleTestPad,
                  columnspacing=legendColumnSpacing,
                  handlelength=legendHandleLength,
                  labelspacing=legendLabelSpacing,
                  borderpad=legendBorderpadSpacing)
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Save as PDF only
        base_name = os.path.basename(csv_file).replace('.csv', f'_{cache_state}_cache')
        pdf_path = os.path.join(results_dir, f"{base_name}.pdf")
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    # Generate protocol improvement chart
    improvement = stats.pivot_table(
        index=['cache_state', 'asset_type'], 
        columns='protocol', 
        values='mean'
    ).reset_index()
    
    # Calculate improvement percentage
    improvement['improvement_pct'] = ((improvement['h2'] - improvement['h3']) / improvement['h2'] * 100)
    
    plt.figure(figsize=(figwidth*1.2, figwidth*1.2 / golden_ratio))  # Slightly wider for labels
    ax = sns.barplot(x='asset_type', y='improvement_pct', hue='cache_state', 
                     data=improvement, palette=color_pallete[2:4])
    
    # Apply formatting with adjusted sizes
    ax.set_title('HTTP/3 Performance Improvement over HTTP/2', fontsize=title_size)
    ax.set_xlabel('Asset Type', fontsize=label_size)
    ax.set_ylabel('Improvement (%)', fontsize=label_size)
    ax.tick_params(axis='x', rotation=30, labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    
    # Add a horizontal line at y=0 to show the baseline
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Improve legend using common.py parameters
    ax.legend(title='Cache State', frameon=False,
              fontsize=legend_font_size,
              handletextpad=legendHandleTestPad,
              columnspacing=legendColumnSpacing,
              handlelength=legendHandleLength,
              labelspacing=legendLabelSpacing,
              borderpad=legendBorderpadSpacing)
    
    plt.tight_layout()
    
    # Save as PDF only
    base_name = os.path.basename(csv_file).replace('.csv', '_improvement')
    pdf_path = os.path.join(results_dir, f"{base_name}.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    # Generate cache efficiency comparison
    plt.figure(figsize=(figwidth*1.2, figwidth*1.2 / golden_ratio))  # Slightly wider for labels
    cache_effect = stats.pivot_table(
        index=['protocol', 'asset_type'],
        columns='cache_state',
        values='mean'
    ).reset_index()
    
    cache_effect['cache_benefit_pct'] = ((cache_effect['cold'] - cache_effect['warm']) / cache_effect['cold'] * 100)
    
    ax = sns.barplot(x='asset_type', y='cache_benefit_pct', hue='protocol', 
                     data=cache_effect, palette=color_pallete[:2])
    
    # Apply formatting with adjusted sizes
    ax.set_title('Cache Performance Benefit by Protocol', fontsize=title_size)
    ax.set_xlabel('Asset Type', fontsize=label_size)
    ax.set_ylabel('Load Time Reduction (%)', fontsize=label_size)
    ax.tick_params(axis='x', rotation=30, labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Improve legend using common.py parameters
    ax.legend(title='Protocol', frameon=False,
              fontsize=legend_font_size,
              handletextpad=legendHandleTestPad,
              columnspacing=legendColumnSpacing,
              handlelength=legendHandleLength,
              labelspacing=legendLabelSpacing,
              borderpad=legendBorderpadSpacing)
    
    plt.tight_layout()
    
    # Save as PDF only
    base_name = os.path.basename(csv_file).replace('.csv', '_cache_benefit')
    pdf_path = os.path.join(results_dir, f"{base_name}.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    # ============== 0-RTT SPECIFIC ANALYSIS ================
    if has_zero_rtt:
        # 1. Analyze HTTP/3 with 0-RTT vs without 0-RTT
        h3_data = df[df['protocol'] == 'h3'].copy()
        
        # Group by whether 0-RTT was used
        if len(h3_data[h3_data['zero_rtt_used'] == True]) > 0:
            h3_rtt_stats = h3_data.groupby(['zero_rtt_used', 'asset_type'])['load_time_ms'].agg(
                ['mean', 'std', 'min', 'max', 'count']
            ).reset_index()
            
            # Create a bar graph comparing 0-RTT vs non-0-RTT for HTTP/3
            plt.figure(figsize=(figwidth*1.2, figwidth*1.2 / golden_ratio))
            h3_rtt_subset = h3_data.copy()
            h3_rtt_subset['Zero-RTT Used'] = h3_rtt_subset['zero_rtt_used'].map({True: 'Yes', False: 'No'})
            
            ax = sns.barplot(x='asset_type', y='load_time_ms', hue='Zero-RTT Used', 
                            data=h3_rtt_subset, errorbar='sd',
                            palette=[color_pallete[0], color_pallete[2]])
            
            ax.set_title('HTTP/3 Performance: 0-RTT vs Regular Connection', fontsize=title_size)
            ax.set_xlabel('Asset Type', fontsize=label_size)
            ax.set_ylabel('Load Time (ms)', fontsize=label_size)
            ax.tick_params(axis='x', rotation=30, labelsize=tick_size)
            ax.tick_params(axis='y', labelsize=tick_size)
            
            if h3_rtt_subset['load_time_ms'].max() > 1000:
                ax.yaxis.set_major_formatter(kfmt)
                
            ax.legend(title='0-RTT Used', frameon=False,
                    fontsize=legend_font_size,
                    handletextpad=legendHandleTestPad,
                    columnspacing=legendColumnSpacing,
                    handlelength=legendHandleLength,
                    labelspacing=legendLabelSpacing,
                    borderpad=legendBorderpadSpacing)
            
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            rtt_base_name = os.path.basename(csv_file).replace('.csv', '_0rtt_comparison')
            rtt_pdf_path = os.path.join(results_dir, f"{rtt_base_name}.pdf")
            plt.savefig(rtt_pdf_path, format='pdf', bbox_inches='tight')
            
            # 2. Calculate improvement from 0-RTT
            h3_rtt_pivot = h3_rtt_stats.pivot_table(
                index='asset_type',
                columns='zero_rtt_used',
                values='mean'
            ).reset_index()
            
            # Ensure we have both True and False columns
            if True in h3_rtt_pivot.columns and False in h3_rtt_pivot.columns:
                h3_rtt_pivot['zero_rtt_benefit_pct'] = ((h3_rtt_pivot[False] - h3_rtt_pivot[True]) / h3_rtt_pivot[False] * 100)
                
                plt.figure(figsize=(figwidth*1.2, figwidth*1.2 / golden_ratio))
                ax = sns.barplot(x='asset_type', y='zero_rtt_benefit_pct', data=h3_rtt_pivot,
                              color=color_pallete[0])
                
                ax.set_title('Performance Benefit of 0-RTT in HTTP/3', fontsize=title_size)
                ax.set_xlabel('Asset Type', fontsize=label_size)
                ax.set_ylabel('Load Time Improvement (%)', fontsize=label_size)
                ax.tick_params(axis='x', rotation=30, labelsize=tick_size)
                ax.tick_params(axis='y', labelsize=tick_size)
                
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                
                plt.tight_layout()
                
                benefit_base_name = os.path.basename(csv_file).replace('.csv', '_0rtt_benefit')
                benefit_pdf_path = os.path.join(results_dir, f"{benefit_base_name}.pdf")
                plt.savefig(benefit_pdf_path, format='pdf', bbox_inches='tight')
        
        # 3. Compare H3 0-RTT vs H2 Connection Reuse vs Cold Connections
        connection_comparison = df.copy()
        
        # Create a connection type field
        conditions = [
            (connection_comparison['protocol'] == 'h3') & (connection_comparison['zero_rtt_used'] == True),
            (connection_comparison['protocol'] == 'h3') & (connection_comparison['zero_rtt_used'] == False),
            (connection_comparison['protocol'] == 'h2') & (connection_comparison['tls_resumed'] == True),
            (connection_comparison['protocol'] == 'h2') & (connection_comparison['tls_resumed'] == False)
        ]
        
        connection_types = [
            'HTTP/3 with 0-RTT',
            'HTTP/3 without 0-RTT',
            'HTTP/2 TLS resumed',
            'HTTP/2 new connection'
        ]
        
        if 'tls_resumed' in connection_comparison.columns:
            connection_comparison['connection_type'] = np.select(conditions, connection_types, default='Unknown')
            
            # Plot connection type comparison
            plt.figure(figsize=(figwidth*1.4, figwidth*1.4 / golden_ratio))  # Wider for more categories
            
            # Use only warm cache for this comparison
            warm_connections = connection_comparison[connection_comparison['cache_state'] == 'warm']
            
            # Check if we have at least 3 connection types with data
            conn_type_counts = warm_connections['connection_type'].value_counts()
            valid_conn_types = conn_type_counts[conn_type_counts > 5].index.tolist()
            
            if len(valid_conn_types) >= 2:  # At least 2 different connection types
                valid_connections = warm_connections[warm_connections['connection_type'].isin(valid_conn_types)]
                
                ax = sns.barplot(x='asset_type', y='load_time_ms', hue='connection_type', 
                                data=valid_connections, errorbar='sd',
                                palette=color_pallete[:len(valid_conn_types)])
                
                ax.set_title('Connection Resumption Performance Comparison', fontsize=title_size)
                ax.set_xlabel('Asset Type', fontsize=label_size)
                ax.set_ylabel('Load Time (ms)', fontsize=label_size)
                ax.tick_params(axis='x', rotation=30, labelsize=tick_size)
                ax.tick_params(axis='y', labelsize=tick_size)
                
                if valid_connections['load_time_ms'].max() > 1000:
                    ax.yaxis.set_major_formatter(kfmt)
                    
                ax.legend(title='Connection Type', frameon=False,
                        fontsize=legend_font_size-1,  # Slightly smaller for more items
                        handletextpad=legendHandleTestPad,
                        columnspacing=legendColumnSpacing,
                        handlelength=legendHandleLength,
                        labelspacing=legendLabelSpacing,
                        borderpad=legendBorderpadSpacing)
                
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                
                conn_base_name = os.path.basename(csv_file).replace('.csv', '_connection_types')
                conn_pdf_path = os.path.join(results_dir, f"{conn_base_name}.pdf")
                plt.savefig(conn_pdf_path, format='pdf', bbox_inches='tight')
            
        # Add to analyze_results function
        if 'cdn' in df.columns:
            # Performance by CDN provider
            cdn_stats = df.groupby(['protocol', 'cdn'])['load_time_ms'].agg(
                ['mean', 'std', 'count']
            ).reset_index()
            print("\nPerformance by CDN provider:")
            print(cdn_stats)
            
            # Create plot comparing HTTP/3 benefit by CDN
            plt.figure()
            cdn_pivot = cdn_stats.pivot_table(
                index='cdn', 
                columns='protocol', 
                values='mean'
            ).reset_index()
            cdn_pivot['improvement_pct'] = ((cdn_pivot['h2'] - cdn_pivot['h3']) / cdn_pivot['h2'] * 100)
            
            # Only include CDNs with enough samples
            valid_cdns = cdn_pivot[cdn_pivot['h3_count'] > 5]
            sns.barplot(x='cdn', y='improvement_pct', data=valid_cdns, palette=color_pallete)
            plt.title('HTTP/3 Performance Benefit by CDN')
            plt.xlabel('CDN Provider')
            plt.ylabel('Improvement (%)')
            plt.xticks(rotation=45)
            plt.axhline(y=0, color='gray', linestyle='--')
            plt.tight_layout()
            plt.savefig(f"{csv_file.replace('.csv', '_cdn_comparison.pdf')}", format='pdf')
            
        # 4. Compare Browser Cache with Connection Resumption
        if 'from_disk_cache' in df.columns:
            # Create a metric that combines protocol, cache, and 0-RTT
            cache_conditions = [
                (df['protocol'] == 'h3') & (df['zero_rtt_used'] == True) & (df['from_disk_cache'] == True),
                (df['protocol'] == 'h3') & (df['zero_rtt_used'] == True) & (df['from_disk_cache'] == False),
                (df['protocol'] == 'h3') & (df['zero_rtt_used'] == False) & (df['from_disk_cache'] == True),
                (df['protocol'] == 'h3') & (df['zero_rtt_used'] == False) & (df['from_disk_cache'] == False),
                (df['protocol'] == 'h2') & (df['from_disk_cache'] == True),
                (df['protocol'] == 'h2') & (df['from_disk_cache'] == False),
            ]
            
            cache_types = [
                'HTTP/3 + 0-RTT + Browser Cache',
                'HTTP/3 + 0-RTT only',
                'HTTP/3 + Browser Cache only',
                'HTTP/3 only',
                'HTTP/2 + Browser Cache',
                'HTTP/2 only'
            ]
            
            df['optimization_type'] = np.select(cache_conditions, cache_types, default='Unknown')
            
            # Create a narrower subset for this visualization - only look at one asset type
            # Use most common asset type that isn't too large
            asset_counts = df['asset_type'].value_counts()
            target_asset = asset_counts.index[0]  # Most common asset type
            
            subset_cache = df[df['asset_type'] == target_asset].copy()
            opt_type_counts = subset_cache['optimization_type'].value_counts()
            valid_opt_types = opt_type_counts[opt_type_counts > 3].index.tolist()
            
            if len(valid_opt_types) >= 3:  # At least 3 different optimization types
                valid_subset = subset_cache[subset_cache['optimization_type'].isin(valid_opt_types)]
                
                plt.figure(figsize=(figwidth*1.4, figwidth*1.4 / golden_ratio))
                
                # Use boxplot instead of barplot for this detailed view
                ax = sns.boxplot(x='optimization_type', y='load_time_ms', data=valid_subset,
                               palette=color_pallete[:len(valid_opt_types)], width=0.6)
                
                ax.set_title(f'Combined Effect of Protocol, 0-RTT and Browser Cache\n({target_asset})', 
                             fontsize=title_size)
                ax.set_xlabel('Optimization Combination', fontsize=label_size)
                ax.set_ylabel('Load Time (ms)', fontsize=label_size)
                ax.tick_params(axis='x', rotation=45, labelsize=tick_size)
                ax.tick_params(axis='y', labelsize=tick_size)
                
                if valid_subset['load_time_ms'].max() > 1000:
                    ax.yaxis.set_major_formatter(kfmt)
                
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                
                opt_base_name = os.path.basename(csv_file).replace('.csv', '_optimizations')
                opt_pdf_path = os.path.join(results_dir, f"{opt_base_name}.pdf")
                plt.savefig(opt_pdf_path, format='pdf', bbox_inches='tight')
                
                # Statistical summary for report
                stats_summary = valid_subset.groupby('optimization_type')['load_time_ms'].agg(
                    ['count', 'mean', 'std', 'min', 'max']).sort_values(by='mean')
                
                stats_summary_name = os.path.basename(csv_file).replace('.csv', '_optimizations_stats.csv')
                stats_summary_path = os.path.join(results_dir, stats_summary_name)
                stats_summary.to_csv(stats_summary_path)
                print(f"Optimization comparison statistics saved to {stats_summary_path}")
    
    print(f"Analysis complete. Charts saved to {results_dir} in PDF format")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_csv_file>")
        sys.exit(1)
    
    analyze_results(sys.argv[1])
