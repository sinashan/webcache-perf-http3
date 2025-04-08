#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import numpy as np
import argparse
import glob
import matplotlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

def analyze_cdn_requests(csv_file):
    """Analyze requests served by CDNs vs origin servers"""
    df = pd.read_csv(csv_file)
    
    # Check if CDN information is available
    if 'cdn' not in df.columns:
        print("No CDN information found in the dataset.")
        return
    
    # Mark CDN vs origin requests (treat "Unknown" as non-CDN)
    df['is_cdn'] = df['cdn'].apply(lambda x: False if x == "Unknown" else True)
    
    # 1. Generate basic statistics
    cdn_stats = df.groupby(['protocol', 'is_cdn']).agg(
        count=('url', 'count'),
        avg_load_time=('load_time_ms', 'mean'),
        avg_connection_time=('connection_time_ms', 'mean')
    ).reset_index()
    
    # Calculate percentage of requests served by CDNs
    total_requests = len(df)
    cdn_requests = len(df[df['is_cdn'] == True])
    cdn_percent = cdn_requests / total_requests * 100
    
    print("\n=== CDN Response Analysis ===")
    print(f"Total requests: {total_requests}")
    print(f"Served by CDNs: {cdn_requests} ({cdn_percent:.1f}%)")
    print(f"Served by origin: {total_requests - cdn_requests} ({100 - cdn_percent:.1f}%)")
    
    # 2. Generate CDN distribution pie chart
    plt.figure(figsize=(figwidth, figwidth))
    
    # Count requests by CDN
    cdn_counts = df['cdn'].value_counts()
    # Filter out small counts for readability
    threshold = total_requests * 0.02  # 2% threshold
    other_counts = cdn_counts[cdn_counts < threshold].sum()
    filtered_counts = cdn_counts[cdn_counts >= threshold]
    if other_counts > 0:
        filtered_counts['Other'] = other_counts
    
    # Create pie chart
    plt.pie(filtered_counts, labels=filtered_counts.index, autopct='%1.1f%%',
           colors=color_pallete[:len(filtered_counts)], startangle=90)
    plt.axis('equal')
    plt.title('Distribution of Requests by CDN Provider')
    
    # Save chart
    plt.tight_layout()
    cdn_dist_path = os.path.join(results_dir, f"{os.path.basename(csv_file).replace('.csv', '_cdn_distribution.pdf')}")
    plt.savefig(cdn_dist_path, format='pdf')
    
    # 3. Compare HTTP/2 vs HTTP/3 for CDN responses
    plt.figure(figsize=(figwidth*1.2, figwidth / golden_ratio))
    
    # Prepare data
    cdn_protocol = df[df['is_cdn'] == True].groupby(['cdn', 'protocol'])['load_time_ms'].mean().reset_index()
    cdn_protocol_pivot = cdn_protocol.pivot_table(index='cdn', columns='protocol', values='load_time_ms').reset_index()
    
    # Calculate improvement percentage
    if 'h2' in cdn_protocol_pivot.columns and 'h3' in cdn_protocol_pivot.columns:
        cdn_protocol_pivot['improvement'] = (cdn_protocol_pivot['h2'] - cdn_protocol_pivot['h3']) / cdn_protocol_pivot['h2'] * 100
        
        # Drop rows with missing values
        cdn_protocol_pivot = cdn_protocol_pivot.dropna(subset=['improvement'])
        
        if len(cdn_protocol_pivot) > 0:
            # Create bar chart
            ax = sns.barplot(x='cdn', y='improvement', data=cdn_protocol_pivot, palette=color_pallete)
            
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            plt.title('HTTP/3 Performance Improvement over HTTP/2 by CDN')
            plt.xlabel('CDN Provider')
            plt.ylabel('Performance Improvement (%)')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Save chart
            plt.tight_layout()
            cdn_perf_path = os.path.join(results_dir, f"{os.path.basename(csv_file).replace('.csv', '_cdn_http3_improvement.pdf')}")
            plt.savefig(cdn_perf_path, format='pdf')
    
    # 4. Compare 0-RTT Success Rate by CDN (if data available)
    if 'zero_rtt_used' in df.columns:
        # Convert string boolean to actual boolean if needed
        if df['zero_rtt_used'].dtype == 'object':
            df['zero_rtt_used'] = df['zero_rtt_used'].map({'true': True, 'false': False})
            
        h3_cdn_data = df[(df['protocol'] == 'h3') & (df['is_cdn'] == True)]
        
        if len(h3_cdn_data) > 0:
            # Group by CDN and calculate 0-RTT percentage
            zero_rtt_by_cdn = h3_cdn_data.groupby('cdn')['zero_rtt_used'].agg(
                ['count', 'sum']
            ).reset_index()
            
            # Calculate percentage
            zero_rtt_by_cdn['zero_rtt_pct'] = (zero_rtt_by_cdn['sum'] / zero_rtt_by_cdn['count']) * 100
            
            # Filter CDNs with enough data
            min_requests = 5
            zero_rtt_by_cdn = zero_rtt_by_cdn[zero_rtt_by_cdn['count'] >= min_requests]
            
            if len(zero_rtt_by_cdn) > 0:
                plt.figure(figsize=(figwidth*1.2, figwidth / golden_ratio))
                
                # Sort by percentage
                zero_rtt_by_cdn = zero_rtt_by_cdn.sort_values('zero_rtt_pct', ascending=False)
                
                # Create bar chart
                ax = sns.barplot(x='cdn', y='zero_rtt_pct', data=zero_rtt_by_cdn, palette=color_pallete)
                
                plt.title('HTTP/3 0-RTT Success Rate by CDN')
                plt.xlabel('CDN Provider')
                plt.ylabel('0-RTT Success Rate (%)')
                plt.xticks(rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                
                # Annotate with count
                for i, row in enumerate(zero_rtt_by_cdn.itertuples()):
                    ax.text(i, 5, f"n={row.count}", ha='center')
                
                # Save chart
                plt.tight_layout()
                cdn_0rtt_path = os.path.join(results_dir, f"{os.path.basename(csv_file).replace('.csv', '_cdn_0rtt_rate.pdf')}")
                plt.savefig(cdn_0rtt_path, format='pdf')
    
    return cdn_stats

def analyze_cache_validation(csv_file):
    """Analyze cache validation strategies and effectiveness"""
    df = pd.read_csv(csv_file)
    
    # Check if we need to rename columns for compatibility
    if 'resource_type' in df.columns and 'asset_type' not in df.columns:
        df['asset_type'] = df['resource_type']
        print("Renamed 'resource_type' column to 'asset_type' for analysis")
    
    # Check if we have caching data
    if 'cache_control' not in df.columns or 'etag' not in df.columns:
        print("Missing cache validation headers in dataset")
        return
    
    print("\n=== Cache Validation Analysis ===")
    
    # 1. Analyze presence of validation headers
    has_etag = ~df['etag'].isna() & (df['etag'] != '')
    has_cache_control = ~df['cache_control'].isna() & (df['cache_control'] != '')
    
    print(f"Resources with ETag: {sum(has_etag)} ({sum(has_etag)/len(df)*100:.1f}%)")
    print(f"Resources with Cache-Control: {sum(has_cache_control)} ({sum(has_cache_control)/len(df)*100:.1f}%)")
    
    # 2. Analyze Cache-Control directives
    if has_cache_control.any():
        # Extract common cache-control directives
        df['no_cache'] = df['cache_control'].str.contains('no-cache', case=False, na=False)
        df['no_store'] = df['cache_control'].str.contains('no-store', case=False, na=False)
        df['private'] = df['cache_control'].str.contains('private', case=False, na=False)
        df['public'] = df['cache_control'].str.contains('public', case=False, na=False)
        df['max_age'] = df['cache_control'].str.extract(r'max-age=(\d+)', expand=False).astype(float)
        
        # Count occurrences
        directives_count = {
            'no-cache': sum(df['no_cache']),
            'no-store': sum(df['no_store']),
            'private': sum(df['private']),
            'public': sum(df['public']),
            'with max-age': df['max_age'].notna().sum()
        }
        
        print("\nCache-Control Directives:")
        for directive, count in directives_count.items():
            if count > 0:
                print(f"  {directive}: {count} ({count/len(df)*100:.1f}%)")
        
        # Analyze max-age values
        if df['max_age'].notna().any():
            max_ages = df['max_age'].dropna()
            print(f"\nMax-age statistics:")
            print(f"  Mean: {max_ages.mean():.1f} seconds ({max_ages.mean()/3600:.1f} hours)")
            print(f"  Median: {max_ages.median():.1f} seconds ({max_ages.median()/3600:.1f} hours)")
            print(f"  Min: {max_ages.min():.1f} seconds")
            print(f"  Max: {max_ages.max():.1f} seconds ({max_ages.max()/86400:.1f} days)")
            
            # Plot max-age distribution
            plt.figure(figsize=(figwidth, figwidth / golden_ratio))
            
            # Use log scale for better visualization
            max_ages_for_plot = max_ages[max_ages > 0]  # Filter out zero values for log scale
            if len(max_ages_for_plot) > 5:
                plt.hist(max_ages_for_plot, bins=15, color=color_pallete[0], alpha=0.7)
                plt.xscale('log')
                plt.title('Distribution of max-age Values (Log Scale)')
                plt.xlabel('max-age (seconds)')
                plt.ylabel('Number of Resources')
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{csv_file.replace('.csv', '_max_age_distribution.pdf')}", format='pdf')
    
    # 3. Analyze cache effectiveness by resource type
    if 'from_disk_cache' in df.columns:
        # Convert to boolean if it's string
        if df['from_disk_cache'].dtype == 'object':
            df['from_disk_cache'] = df['from_disk_cache'].map({'true': True, 'false': False})
        
        # Only analyze warm cache hits as that's where we expect caching
        warm_cache = df[df['cache_state'] == 'warm']
        if len(warm_cache) > 0:
            # Determine which column to use (asset_type or resource_type)
            group_cols = []
            
            if 'asset_type' in warm_cache.columns:
                group_cols.append('asset_type')
            elif 'resource_type' in warm_cache.columns:
                group_cols.append('resource_type')
            else:
                print("Warning: Neither 'asset_type' nor 'resource_type' column found. Skipping type-based analysis.")
                group_cols = []
            
            if group_cols and 'protocol' in warm_cache.columns:
                group_cols.append('protocol')
                cache_by_type = warm_cache.groupby(group_cols)['from_disk_cache'].mean().reset_index()
                cache_by_type['cache_hit_pct'] = cache_by_type['from_disk_cache'] * 100
                
                # Determine which column to use for display
                type_col = 'asset_type' if 'asset_type' in cache_by_type.columns else 'resource_type'
                
                print("\nCache Hit Rate by Resource Type (Warm Cache):")
                for protocol in df['protocol'].unique():
                    protocol_data = cache_by_type[cache_by_type['protocol'] == protocol]
                    if len(protocol_data) > 0:
                        print(f"\n  {protocol.upper()}:")
                        for _, row in protocol_data.iterrows():
                            print(f"    {row[type_col]}: {row['cache_hit_pct']:.1f}% cache hit rate")
                
                # Plot cache hit rate by resource type and protocol
                plt.figure(figsize=(figwidth*1.2, figwidth / golden_ratio))
                ax = sns.barplot(x=type_col, y='cache_hit_pct', hue='protocol', 
                                data=cache_by_type, palette=color_pallete[:2])
                plt.title('Cache Hit Rate by Resource Type (Warm Cache)')
                plt.xlabel('Resource Type')
                plt.ylabel('Cache Hit Rate (%)')
                plt.xticks(rotation=30)
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{csv_file.replace('.csv', '_cache_hit_rate.pdf')}", format='pdf')
            
    # 4. Compare validation strategies with actual caching outcomes
    if 'from_disk_cache' in df.columns and has_etag.any():
        # Create validation strategy categories
        conditions = [
            (has_etag & has_cache_control),
            (has_etag & ~has_cache_control),
            (~has_etag & has_cache_control),
            (~has_etag & ~has_cache_control)
        ]
        
        strategies = [
            'ETag + Cache-Control',
            'ETag only',
            'Cache-Control only',
            'No validation'
        ]
        
        df['validation_strategy'] = np.select(conditions, strategies, default='Unknown')
        
        # Only analyze warm cache where validation matters
        warm_df = df[df['cache_state'] == 'warm']
        if len(warm_df) > 0:
            validation_effect = warm_df.groupby(['validation_strategy', 'protocol'])['from_disk_cache'].agg(
                ['mean', 'count']
            ).reset_index()
            
            # Calculate hit rate percentage
            validation_effect['hit_rate_pct'] = validation_effect['mean'] * 100
            
            print("\nCache Hit Rate by Validation Strategy (Warm Cache):")
            for protocol in df['protocol'].unique():
                protocol_data = validation_effect[validation_effect['protocol'] == protocol]
                if len(protocol_data) > 0:
                    print(f"\n  {protocol.upper()}:")
                    for _, row in protocol_data.iterrows():
                        if row['count'] >= 5:  # Only show strategies with enough samples
                            print(f"    {row['validation_strategy']}: {row['hit_rate_pct']:.1f}% hit rate (n={int(row['count'])})")
            
            # Plot validation strategy effectiveness
            # Only include strategies with sufficient data points
            valid_strategies = validation_effect[validation_effect['count'] >= 5]
            if len(valid_strategies) > 1:  # Need at least 2 strategies to compare
                plt.figure(figsize=(figwidth*1.2, figwidth / golden_ratio))
                ax = sns.barplot(x='validation_strategy', y='hit_rate_pct', hue='protocol', 
                                data=valid_strategies, palette=color_pallete[:2])
                plt.title('Cache Hit Rate by Validation Strategy (Warm Cache)')
                plt.xlabel('Validation Strategy')
                plt.ylabel('Cache Hit Rate (%)')
                plt.xticks(rotation=30)
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{csv_file.replace('.csv', '_validation_effectiveness.pdf')}", format='pdf')


    # 5. Compare CDN cache operation with validation headers
    if 'cdn' in df.columns and 'from_disk_cache' in df.columns:
        # Group data by CDN and validation presence
        cdn_etag = df.groupby(['cdn', has_etag, 'protocol'])['from_disk_cache'].mean().reset_index()
        cdn_etag.columns = ['cdn', 'has_etag', 'protocol', 'cache_hit_rate']
        cdn_etag['cache_hit_pct'] = cdn_etag['cache_hit_rate'] * 100
        
        # Only consider CDNs with enough data points
        cdn_counts = df.groupby('cdn').size()
        major_cdns = cdn_counts[cdn_counts >= 10].index
        
        if len(major_cdns) > 0:
            cdn_etag_filtered = cdn_etag[cdn_etag['cdn'].isin(major_cdns)]
            
            print("\nCDN Cache Hit Rate by ETag Usage:")
            for cdn in major_cdns:
                cdn_data = cdn_etag_filtered[cdn_etag_filtered['cdn'] == cdn]
                if len(cdn_data) > 0:
                    print(f"\n  {cdn}:")
                    for _, row in cdn_data.iterrows():
                        etag_status = "With ETag" if row['has_etag'] else "Without ETag"
                        print(f"    {row['protocol']} {etag_status}: {row['cache_hit_pct']:.1f}% hit rate")
            
            # Plot CDN validation effectiveness
            plt.figure(figsize=(figwidth*1.4, figwidth / golden_ratio))
            
            # Convert boolean to string for better labels
            cdn_etag_filtered['ETag Present'] = cdn_etag_filtered['has_etag'].map({True: 'Yes', False: 'No'})
            
            ax = sns.barplot(x='cdn', y='cache_hit_pct', hue='ETag Present', 
                            data=cdn_etag_filtered[cdn_etag_filtered['protocol'] == 'h3'],
                            palette=[color_pallete[0], color_pallete[2]])
            plt.title('HTTP/3 Cache Hit Rate by CDN and ETag Usage')
            plt.xlabel('CDN Provider')
            plt.ylabel('Cache Hit Rate (%)')
            plt.xticks(rotation=30)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{csv_file.replace('.csv', '_cdn_validation_effectiveness.pdf')}", format='pdf')

def analyze_validation_and_0rtt(csv_file):
    """Analyze validation response times and 0-RTT usage for conditional requests"""
    df = pd.read_csv(csv_file)
    
    print("\n=== Validation Response Time & 0-RTT Analysis ===")
    
    # Convert string boolean values to actual booleans if needed
    for col in ['zero_rtt_used', 'from_disk_cache', 'tls_resumed']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].map({'true': True, 'false': False})
    
    # 1. Identify conditional requests (304 Not Modified responses)
    df['is_conditional'] = df['status_code'] == 304
    
    # Also check for requests with If-None-Match or If-Modified-Since headers
    if 'request_headers' in df.columns:
        df['has_conditional_header'] = df['request_headers'].apply(
            lambda h: isinstance(h, str) and ('if-none-match' in h.lower() or 'if-modified-since' in h.lower())
        )
        df['is_conditional'] = df['is_conditional'] | df['has_conditional_header']
    
    conditional_count = df['is_conditional'].sum()
    total_warm = len(df[df['cache_state'] == 'warm'])
    
    print(f"Conditional requests detected: {conditional_count}")
    if total_warm > 0:
        print(f"Percentage of warm requests that were conditional: {conditional_count/total_warm*100:.1f}%")
    
    if conditional_count == 0:
        print("No conditional requests found for analysis.")
        return
    
    # 2. Compare response times for conditional vs non-conditional requests
    conditional_df = df[df['is_conditional']]
    non_conditional_df = df[~df['is_conditional'] & (df['cache_state'] == 'warm')]
    
    # Calculate statistics
    cond_stats = conditional_df.groupby('protocol')['load_time_ms'].agg(['mean', 'median', 'count']).reset_index()
    non_cond_stats = non_conditional_df.groupby('protocol')['load_time_ms'].agg(['mean', 'median', 'count']).reset_index()
    
    print("\nConditional Request Response Times (ms):")
    for _, row in cond_stats.iterrows():
        print(f"  {row['protocol'].upper()}: {row['mean']:.2f}ms mean, {row['median']:.2f}ms median (n={int(row['count'])})")
    
    print("\nNon-Conditional Request Response Times (ms):")
    for _, row in non_cond_stats.iterrows():
        print(f"  {row['protocol'].upper()}: {row['mean']:.2f}ms mean, {row['median']:.2f}ms median (n={int(row['count'])})")
    
    # 3. Analyze 0-RTT usage for conditional requests
    if 'zero_rtt_used' in conditional_df.columns:
        # How often 0-RTT was used for conditional requests with HTTP/3
        h3_conditional = conditional_df[conditional_df['protocol'] == 'h3']
        if len(h3_conditional) > 0:
            zero_rtt_count = h3_conditional['zero_rtt_used'].sum()
            zero_rtt_pct = zero_rtt_count / len(h3_conditional) * 100
            
            print(f"\n0-RTT Usage for Conditional HTTP/3 Requests:")
            print(f"  Used 0-RTT: {zero_rtt_count}/{len(h3_conditional)} ({zero_rtt_pct:.1f}%)")
            
            # Compare response times for 0-RTT vs non-0-RTT conditional requests
            if zero_rtt_count > 0:
                zero_rtt_time = h3_conditional[h3_conditional['zero_rtt_used']]['load_time_ms'].mean()
                non_zero_rtt_time = h3_conditional[~h3_conditional['zero_rtt_used']]['load_time_ms'].mean()
                
                print(f"  Avg response time with 0-RTT: {zero_rtt_time:.2f}ms")
                print(f"  Avg response time without 0-RTT: {non_zero_rtt_time:.2f}ms")
                print(f"  Time savings with 0-RTT: {non_zero_rtt_time - zero_rtt_time:.2f}ms ({(non_zero_rtt_time - zero_rtt_time)/non_zero_rtt_time*100:.1f}%)")
                
                # Plot comparison if we have enough samples
                if zero_rtt_count >= 5 and len(h3_conditional) - zero_rtt_count >= 5:
                    plt.figure(figsize=(figwidth, figwidth / golden_ratio))
                    h3_conditional['0-RTT Used'] = h3_conditional['zero_rtt_used'].map({True: 'Yes', False: 'No'})
                    
                    sns.boxplot(x='0-RTT Used', y='load_time_ms', data=h3_conditional, palette=[color_pallete[0], color_pallete[2]])
                    plt.title('HTTP/3 Conditional Request Response Time by 0-RTT Usage')
                    plt.xlabel('0-RTT Used')
                    plt.ylabel('Response Time (ms)')
                    plt.grid(axis='y', linestyle='--', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"{csv_file.replace('.csv', '_conditional_0rtt_boxplot.pdf')}", format='pdf')
                    
                    # Also generate CDF plot for better distribution view
                    plt.figure(figsize=(figwidth, figwidth / golden_ratio))
                    
                    for used_0rtt, group in h3_conditional.groupby('zero_rtt_used'):
                        x = np.sort(group['load_time_ms'])
                        y = np.arange(1, len(x) + 1) / len(x)
                        label = "With 0-RTT" if used_0rtt else "Without 0-RTT"
                        color = color_pallete[0] if used_0rtt else color_pallete[2]
                        plt.plot(x, y, label=label, color=color)
                    
                    plt.title('CDF of HTTP/3 Conditional Request Response Times')
                    plt.xlabel('Response Time (ms)')
                    plt.ylabel('Cumulative Probability')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"{csv_file.replace('.csv', '_conditional_0rtt_cdf.pdf')}", format='pdf')
    
    # 4. Compare validation time by asset type
    if len(conditional_df) >= 10:
        if 'asset_type' in conditional_df.columns:
            # Get validation times by asset type and protocol
            validation_by_type = conditional_df.groupby(['asset_type', 'protocol'])['load_time_ms'].agg(
                ['mean', 'count']
            ).reset_index()
            
            # Only keep asset types with at least 3 samples
            valid_validation_types = validation_by_type[validation_by_type['count'] >= 3]
            
            if len(valid_validation_types) > 0:
                print("\nConditional Request Response Times by Asset Type:")
                for _, row in valid_validation_types.iterrows():
                    print(f"  {row['protocol'].upper()} {row['asset_type']}: {row['mean']:.2f}ms (n={int(row['count'])})")
                
                # Plot comparison
                plt.figure(figsize=(figwidth*1.2, figwidth / golden_ratio))
                pivot_data = valid_validation_types.pivot_table(
                    index='asset_type', columns='protocol', values='mean'
                ).reset_index()
                
                plt.figure(figsize=(figwidth*1.2, figwidth / golden_ratio))
                pivot_data_long = valid_validation_types.rename(columns={'mean': 'load_time_ms'})
                
                ax = sns.barplot(x='asset_type', y='load_time_ms', hue='protocol', 
                                data=pivot_data_long, palette=color_pallete[:2])
                plt.title('Conditional Request Response Times by Asset Type')
                plt.xlabel('Asset Type')
                plt.ylabel('Response Time (ms)')
                plt.xticks(rotation=30)
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{csv_file.replace('.csv', '_conditional_asset_type.pdf')}", format='pdf')

def analyze_network_conditions(base_directory=results_dir, date_pattern=None):
    """Analyze HTTP/3 vs HTTP/2 performance across different network conditions"""
    # Find all result files with the same date pattern but different network conditions
    if date_pattern is None:
        # Look for the most recent date pattern
        csv_files = glob.glob(os.path.join(base_directory, "cache_perf_results_*.csv"))
        if not csv_files:
            print("No result files found in the results directory.")
            return
            
        # Extract date patterns
        date_patterns = set()
        for file in csv_files:
            match = re.search(r'cache_perf_results_(\d+_\d+)_', file)
            if match:
                date_patterns.add(match.group(1))
        
        if not date_patterns:
            print("No valid date patterns found in result files.")
            return
            
        # Use the most recent date pattern
        date_pattern = sorted(date_patterns)[-1]
        print(f"Using most recent test date: {date_pattern}")
    
    # Find all files matching the date pattern
    network_files = glob.glob(os.path.join(base_directory, f"cache_perf_results_{date_pattern}_*.csv"))
    
    if len(network_files) <= 1:
        print(f"Not enough network condition files found for date {date_pattern}.")
        return
        
    print(f"Found {len(network_files)} network condition files for analysis.")
    
    # Load all files into dataframes
    network_data = {}
    for file_path in network_files:
        # Extract network condition name from filename
        match = re.search(r'_(\w+(?:\s\w+)*)\.csv$', file_path)
        if match:
            network_name = match.group(1)
            try:
                df = pd.read_csv(file_path)
                network_data[network_name] = df
                print(f"Loaded {network_name} data: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    if not network_data:
        print("No valid data files could be loaded.")
        return
    
    # Combine dataframes with network condition column
    combined_data = []
    for network, df in network_data.items():
        df_copy = df.copy()
        df_copy['network_condition'] = network
        combined_data.append(df_copy)
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Convert string values to boolean if needed
    for col in ['zero_rtt_used', 'from_disk_cache', 'tls_resumed']:
        if col in combined_df.columns and combined_df[col].dtype == 'object':
            combined_df[col] = combined_df[col].map({'true': True, 'false': False})
    
    # Create the comparison visualizations
    
    # 1. Protocol comparison across network conditions
    plt.figure(figsize=(figwidth*1.4, figwidth / golden_ratio))
    
    # Calculate mean load times for each protocol and network condition
    agg_data = combined_df.groupby(['network_condition', 'protocol'])['load_time_ms'].mean().reset_index()
    
    # Order network conditions from fast to very slow
    network_order = ['fast', 'typical', 'slow', 'very slow']
    agg_data['network_condition'] = pd.Categorical(agg_data['network_condition'], 
                                                 categories=network_order, 
                                                 ordered=True)
    agg_data = agg_data.sort_values('network_condition')
    
    # Create the grouped bar chart
    ax = sns.barplot(x='network_condition', y='load_time_ms', hue='protocol', data=agg_data, palette=color_pallete[:2])
    
    ax.set_title('HTTP/2 vs HTTP/3 Performance Across Network Conditions', fontsize=14)
    ax.set_xlabel('Network Condition', fontsize=12)
    ax.set_ylabel('Average Load Time (ms)', fontsize=12)
    
    if agg_data['load_time_ms'].max() > 1000:
        ax.yaxis.set_major_formatter(kfmt)
    
    ax.legend(title='Protocol')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    network_comp_file = os.path.join(base_directory, f"network_comparison_{date_pattern}.pdf")
    plt.savefig(network_comp_file, format='pdf')
    print(f"Saved network comparison chart to {network_comp_file}")
    
    # 2. HTTP/3 improvement percentage across network conditions
    plt.figure(figsize=(figwidth*1.2, figwidth / golden_ratio))
    
    # Calculate improvement percentages
    improvement_data = []
    
    for network in network_order:
        if network not in network_data:
            continue
            
        network_df = network_data[network]
        
        # Group by protocol and calculate mean load times
        protocol_means = network_df.groupby('protocol')['load_time_ms'].mean()
        
        if 'h2' in protocol_means and 'h3' in protocol_means:
            h2_time = protocol_means['h2']
            h3_time = protocol_means['h3']
            improvement = ((h2_time - h3_time) / h2_time) * 100
            
            improvement_data.append({
                'network_condition': network,
                'improvement_pct': improvement
            })
    
    if improvement_data:
        improvement_df = pd.DataFrame(improvement_data)
        improvement_df['network_condition'] = pd.Categorical(
            improvement_df['network_condition'],
            categories=network_order,
            ordered=True
        )
        improvement_df = improvement_df.sort_values('network_condition')
        
        ax = sns.barplot(x='network_condition', y='improvement_pct', data=improvement_df, color=color_pallete[0])
        
        ax.set_title('HTTP/3 Performance Improvement Over HTTP/2', fontsize=14)
        ax.set_xlabel('Network Condition', fontsize=12)
        ax.set_ylabel('Improvement (%)', fontsize=12)
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        improvement_file = os.path.join(base_directory, f"http3_improvement_by_network_{date_pattern}.pdf")
        plt.savefig(improvement_file, format='pdf')
        print(f"Saved HTTP/3 improvement chart to {improvement_file}")
    
    # 3. Connection time comparison across network conditions
    plt.figure(figsize=(figwidth*1.2, figwidth / golden_ratio))
    
    conn_agg_data = combined_df.groupby(['network_condition', 'protocol'])['connection_time_ms'].mean().reset_index()
    
    conn_agg_data['network_condition'] = pd.Categorical(
        conn_agg_data['network_condition'],
        categories=network_order,
        ordered=True
    )
    conn_agg_data = conn_agg_data.sort_values('network_condition')
    
    ax = sns.barplot(x='network_condition', y='connection_time_ms', hue='protocol', data=conn_agg_data, palette=color_pallete[:2])
    
    ax.set_title('Connection Time Comparison by Network Condition', fontsize=14)
    ax.set_xlabel('Network Condition', fontsize=12)
    ax.set_ylabel('Average Connection Time (ms)', fontsize=12)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    conn_file = os.path.join(base_directory, f"connection_time_by_network_{date_pattern}.pdf")
    plt.savefig(conn_file, format='pdf')
    print(f"Saved connection time comparison to {conn_file}")
    
    # 4. 0-RTT Success Rate across network conditions (if available)
    if 'zero_rtt_used' in combined_df.columns:
        h3_data = combined_df[(combined_df['protocol'] == 'h3')]
        
        if len(h3_data) > 0:
            plt.figure(figsize=(figwidth*1.2, figwidth / golden_ratio))
            
            # Calculate 0-RTT success rate for each network
            zero_rtt_by_network = h3_data.groupby('network_condition')['zero_rtt_used'].agg(
                ['count', 'sum']
            ).reset_index()
            
            # Calculate percentage
            zero_rtt_by_network['zero_rtt_pct'] = (zero_rtt_by_network['sum'] / zero_rtt_by_network['count']) * 100
            
            # Sort by network order
            zero_rtt_by_network['network_condition'] = pd.Categorical(
                zero_rtt_by_network['network_condition'],
                categories=network_order,
                ordered=True
            )
            zero_rtt_by_network = zero_rtt_by_network.sort_values('network_condition')
            
            ax = sns.barplot(x='network_condition', y='zero_rtt_pct', data=zero_rtt_by_network, color=color_pallete[0])
            
            ax.set_title('HTTP/3 0-RTT Success Rate by Network Condition', fontsize=14)
            ax.set_xlabel('Network Condition', fontsize=12)
            ax.set_ylabel('0-RTT Success Rate (%)', fontsize=12)
            
            # Add sample size annotation
            for i, row in enumerate(zero_rtt_by_network.itertuples()):
                ax.text(i, 5, f"n={row.count}", ha='center')
            
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            rtt_file = os.path.join(base_directory, f"zero_rtt_success_by_network_{date_pattern}.pdf")
            plt.savefig(rtt_file, format='pdf')
            print(f"Saved 0-RTT success rate chart to {rtt_file}")
    
    return combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze HTTP/2 vs HTTP/3 performance results")
    parser.add_argument("file", nargs="?", help="Input CSV results file")
    parser.add_argument("--network-analysis", "-n", action="store_true", 
                      help="Perform analysis across network conditions")
    parser.add_argument("--date", "-d", help="Date pattern to use (format: YYYYMMDD_HHMMSS)")
    
    args = parser.parse_args()
    
    if args.network_analysis:
        analyze_network_conditions(date_pattern=args.date)
    elif args.file:
        analyze_results(args.file)
        analyze_cdn_requests(args.file)
        analyze_cache_validation(args.file)
        analyze_validation_and_0rtt(args.file)
    else:
        print("Please provide a CSV file to analyze or use --network-analysis")
