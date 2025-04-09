#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import numpy as np
import argparse
import glob
import re 
import matplotlib
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (golden_ratio, figwidth, color_pallete, kfmt,
                    legendHandleTestPad, legendColumnSpacing, 
                    legendHandleLength, legendLabelSpacing, 
                    legendBorderpadSpacing)
matplotlib.rcParams['text.usetex'] = False

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

def analyze_validation_distribution_across_networks(base_directory=results_dir):
    """Create a composite visualization of validation header distribution across network types"""
    # Define standard network condition directories to search
    network_directories = ['fast', 'typical', 'slow']
    
    network_data = {}
    network_order = []
    
    # Set plot style for this function
    set_plot_style()
    
    # Look for CSV files in each network directory
    for network_name in network_directories:
        network_dir = os.path.join(base_directory, network_name)
        
        # Skip if directory doesn't exist
        if not os.path.isdir(network_dir):
            print(f"Directory for {network_name} not found: {network_dir}")
            continue
            
        # Find the most recent CSV file in this directory
        csv_files = glob.glob(os.path.join(network_dir, "*.csv"))
        # Filter out stats files
        csv_files = [f for f in csv_files if not ("_stats" in f or "_optimizations" in f)]
        
        if csv_files:
            # Use the most recent file (sorted by modification time)
            csv_files.sort(key=os.path.getmtime, reverse=True)
            latest_file = csv_files[0]
            
            try:
                df = pd.read_csv(latest_file)
                if 'cache_control' in df.columns and 'etag' in df.columns:
                    network_data[network_name] = df
                    network_order.append(network_name)
                    print(f"Loaded {network_name} data: {len(df)} rows from {os.path.basename(latest_file)}")
                else:
                    print(f"Skipping {latest_file} - missing required validation header columns")
            except Exception as e:
                print(f"Error loading {latest_file}: {e}")
    
    if not network_data or len(network_data) < 2:
        print("Insufficient network condition data for cross-network comparison.")
        return
    
    # Sort network order according to standard ordering
    standard_order = ['fast', 'typical', 'slow', 'very_slow']
    network_order = sorted(network_order, key=lambda x: standard_order.index(x) if x in standard_order else 999)
    print(f"Using network ordering: {network_order}")
    
    # Create a composite figure with 1 row and up to 3 columns of pie charts
    fig, axes = plt.subplots(1, len(network_order), figsize=(figwidth*min(len(network_order), 3), figwidth*0.8))
    if len(network_order) == 1:
        axes = [axes]  # Make axes iterable if only one subplot
    
    # Process each network's validation header data
    for i, network in enumerate(network_order):
        df = network_data[network]
        
        # Prepare data
        has_etag = ~df['etag'].isna() & (df['etag'] != '')
        has_cache_control = ~df['cache_control'].isna() & (df['cache_control'] != '')
        
        # Count validation header combinations
        validation_counts = {
            'ETag + Cache-Control': sum(has_etag & has_cache_control),
            'ETag Only': sum(has_etag & ~has_cache_control),
            'Cache-Control Only': sum(~has_etag & has_cache_control),
            'No Validation': sum(~has_etag & ~has_cache_control)
        }
        
        # Filter out empty categories
        validation_counts = {k: v for k, v in validation_counts.items() if v > 0}
        
        # Create pie chart in the corresponding subplot
        ax = axes[i]
        wedges, texts, autotexts = ax.pie(
            validation_counts.values(), 
            labels=None,  # No labels, we'll use a legend
            autopct='%1.1f%%', 
            startangle=90, 
            colors=color_pallete
        )
        
        # Format percentage text to be more readable
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(f"{network.title()} Network", fontsize=20)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Create a single legend for all pie charts
    legend_labels = list(validation_counts.keys())
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color_pallete[i % len(color_pallete)]) 
        for i in range(len(legend_labels))
    ]
    
    fig.legend(
        legend_handles, 
        legend_labels, 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.05), 
        ncol=min(len(legend_labels), 2),
        frameon=False,
        fontsize=14
    )
    
    # plt.suptitle('Validation Header Distribution Across Network Conditions', fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for the legend
    
    # Generate timestamp for output file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the composite visualization
    output_file = os.path.join(base_directory, f"validation_headers_across_networks_{timestamp}.pdf")
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved cross-network validation header distribution to {output_file}")
    
    return network_data

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

def analyze_network_conditions(base_directory=results_dir):
    """Analyze HTTP/3 vs HTTP/2 performance across different network conditions"""
    # Define standard network condition directories to search
    network_directories = ['fast', 'typical', 'slow']
    
    network_data = {}
    network_order = []
    
    # Look for CSV files in each network directory
    for network_name in network_directories:
        network_dir = os.path.join(base_directory, network_name)
        
        # Skip if directory doesn't exist
        if not os.path.isdir(network_dir):
            print(f"Directory for {network_name} not found: {network_dir}")
            continue
            
        # Find the most recent CSV file in this directory
        csv_files = glob.glob(os.path.join(network_dir, "*.csv"))
        # Filter out stats files
        csv_files = [f for f in csv_files if not ("_stats" in f or "_optimizations" in f)]
        
        if csv_files:
            # Use the most recent file (sorted by modification time)
            csv_files.sort(key=os.path.getmtime, reverse=True)
            latest_file = csv_files[0]
            
            try:
                df = pd.read_csv(latest_file)
                if 'protocol' in df.columns:  # Verify this is a valid data file
                    network_data[network_name] = df
                    network_order.append(network_name)
                    print(f"Loaded {network_name} data: {len(df)} rows from {os.path.basename(latest_file)}")
                else:
                    print(f"Skipping {latest_file} - missing required columns")
            except Exception as e:
                print(f"Error loading {latest_file}: {e}")
    
    # If no network directories found, fall back to searching in the base directory
    if not network_data:
        print("No network directories found. Searching in base directory...")
        
        # Look for files with network names in their filenames
        all_csv_files = glob.glob(os.path.join(base_directory, "*.csv"))
        
        for network_name in network_directories:
            matching_files = [f for f in all_csv_files if network_name in f.lower()]
            # Filter out stats files
            matching_files = [f for f in matching_files if not ("_stats" in f or "_optimizations" in f)]
            
            if matching_files:
                # Use most recent file
                matching_files.sort(key=os.path.getmtime, reverse=True)
                latest_file = matching_files[0]
                
                try:
                    df = pd.read_csv(latest_file)
                    if 'protocol' in df.columns:
                        network_data[network_name] = df
                        if network_name not in network_order:
                            network_order.append(network_name)
                        print(f"Loaded {network_name} data: {len(df)} rows from {os.path.basename(latest_file)}")
                except Exception as e:
                    print(f"Error loading {latest_file}: {e}")
    
    if not network_data:
        print("No valid network condition files could be loaded.")
        return
    
    # Sort network order according to standard ordering
    standard_order = ['fast', 'typical', 'slow']
    network_order = sorted(network_order, key=lambda x: standard_order.index(x) if x in standard_order else 999)
    print(f"Using network ordering: {network_order}")
    
    # Combine dataframes with network condition column
    combined_data = []
    for network in network_order:
        df_copy = network_data[network].copy()
        df_copy['network_condition'] = network
        combined_data.append(df_copy)
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Convert string values to boolean if needed
    for col in ['zero_rtt_used', 'from_disk_cache', 'tls_resumed']:
        if col in combined_df.columns and combined_df[col].dtype == 'object':
            combined_df[col] = combined_df[col].map({'true': True, 'false': False})
    
    print(f"Combined dataset has {len(combined_df)} rows across {len(network_data)} network conditions.")
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # The rest of the function remains the same...
    # Create the comparison visualizations
    
    # 1. Protocol comparison across network conditions
    plt.figure(figsize=(figwidth*1.4, figwidth / golden_ratio))
    
    # Calculate mean load times for each protocol and network condition
    agg_data = combined_df.groupby(['network_condition', 'protocol'])['load_time_ms'].mean().reset_index()
    
    # Order network conditions
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
    
    network_comp_file = os.path.join(base_directory, f"network_comparison_{timestamp}.pdf")
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
        
        improvement_file = os.path.join(base_directory, f"http3_improvement_by_network_{timestamp}.pdf")
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
    
    conn_file = os.path.join(base_directory, f"connection_time_by_network_{timestamp}.pdf")
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
            
            rtt_file = os.path.join(base_directory, f"zero_rtt_success_by_network_{timestamp}.pdf")
            plt.savefig(rtt_file, format='pdf')
            print(f"Saved 0-RTT success rate chart to {rtt_file}")
    
    print("\nNetwork condition analysis complete!")
    return combined_df

def visualize_cache_validation(csv_file):
    """Create visualizations for cache validation metrics"""
    df = pd.read_csv(csv_file)
    
    # Check if we have caching data
    if 'cache_control' not in df.columns or 'etag' not in df.columns:
        print("Missing cache validation headers in dataset")
        return
    
    # Prepare data
    has_etag = ~df['etag'].isna() & (df['etag'] != '')
    has_cache_control = ~df['cache_control'].isna() & (df['cache_control'] != '')
    
    # 1. Create pie chart showing validation header presence
    plt.figure(figsize=(figwidth*1.2, figwidth))
    
    validation_counts = {
        'ETag + Cache-Control': sum(has_etag & has_cache_control),
        'ETag Only': sum(has_etag & ~has_cache_control),
        'Cache-Control Only': sum(~has_etag & has_cache_control),
        'No Validation': sum(~has_etag & ~has_cache_control)
    }
    
    # Filter out empty categories
    validation_counts = {k: v for k, v in validation_counts.items() if v > 0}
    
    plt.pie(validation_counts.values(), labels=validation_counts.keys(), 
            autopct='%1.1f%%', startangle=90, colors=color_pallete)
    plt.axis('equal')
    # plt.title('Distribution of Validation Header Usage')
    
    # Save chart
    plt.tight_layout()
    validation_dist_path = os.path.join(results_dir, f"{os.path.basename(csv_file).replace('.csv', '_validation_distribution.pdf')}")
    plt.savefig(validation_dist_path, format='pdf')
    print(f"Saved validation header distribution chart to {validation_dist_path}")
    
    # 2. Create bar chart of Cache-Control directives
    if has_cache_control.any():
        # Extract cache control directives
        df['no_cache'] = df['cache_control'].str.contains('no-cache', case=False, na=False)
        df['no_store'] = df['cache_control'].str.contains('no-store', case=False, na=False)
        df['private'] = df['cache_control'].str.contains('private', case=False, na=False)
        df['public'] = df['cache_control'].str.contains('public', case=False, na=False)
        df['max_age'] = df['cache_control'].str.extract(r'max-age=(\d+)', expand=False).notna()
        df['immutable'] = df['cache_control'].str.contains('immutable', case=False, na=False)
        df['must_revalidate'] = df['cache_control'].str.contains('must-revalidate', case=False, na=False)
        df['no_transform'] = df['cache_control'].str.contains('no-transform', case=False, na=False)
        
        # Count directives
        directives = ['no_cache', 'no_store', 'private', 'public', 'max_age', 
                     'immutable', 'must_revalidate', 'no_transform']
        directive_counts = {d.replace('_', '-'): sum(df[d]) for d in directives}
        
        # Remove directives with zero count
        directive_counts = {k: v for k, v in directive_counts.items() if v > 0}
        
        # Sort by frequency
        directive_counts = {k: v for k, v in sorted(directive_counts.items(), key=lambda x: x[1], reverse=True)}
        
        # Create bar chart
        plt.figure(figsize=(figwidth*1.2, figwidth / golden_ratio))
        
        bars = plt.bar(directive_counts.keys(), 
                [v/len(df)*100 for v in directive_counts.values()], 
                color=color_pallete[0])
        
        plt.title('Cache-Control Directives Usage')
        plt.xlabel('Directive')
        plt.ylabel('Percentage of Resources (%)')
        plt.xticks(rotation=30)
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{height:.1f}%', ha='center', va='bottom')
        
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        directive_path = os.path.join(results_dir, f"{os.path.basename(csv_file).replace('.csv', '_cache_directives.pdf')}")
        plt.savefig(directive_path, format='pdf')
        print(f"Saved cache directives usage chart to {directive_path}")
    
    # 3. Create bar chart comparing cache hit rates by validation strategy
    if 'from_disk_cache' in df.columns:
        # Convert to boolean if needed
        if df['from_disk_cache'].dtype == 'object':
            df['from_disk_cache'] = df['from_disk_cache'].map({'true': True, 'false': False})
        
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
        
        # Only analyze warm cache
        warm_df = df[df['cache_state'] == 'warm']
        if len(warm_df) > 0:
            # Compare hit rates across resource types and validation strategies
            plt.figure(figsize=(figwidth*1.5, figwidth / golden_ratio))
            
            # Create a heatmap of cache hit rates by resource type and validation strategy
            if 'asset_type' in warm_df.columns or 'resource_type' in warm_df.columns:
                # Determine which column to use
                type_col = 'asset_type' if 'asset_type' in warm_df.columns else 'resource_type'
                
                # Calculate hit rates
                pivot_data = warm_df.pivot_table(
                    index='validation_strategy',
                    columns=type_col,
                    values='from_disk_cache',
                    aggfunc='mean'
                ) * 100  # Convert to percentage
                
                # Only include strategies with data
                pivot_data = pivot_data.dropna(how='all')
                
                # Plot heatmap
                plt.figure(figsize=(figwidth*1.5, figwidth / 1.3))
                ax = sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlGnBu',
                               linewidths=.5, cbar_kws={'label': 'Cache Hit Rate (%)'})
                
                plt.title('Cache Hit Rate by Validation Strategy and Resource Type')
                plt.tight_layout()
                
                heatmap_path = os.path.join(results_dir, f"{os.path.basename(csv_file).replace('.csv', '_validation_heatmap.pdf')}")
                plt.savefig(heatmap_path, format='pdf')
                print(f"Saved validation strategy heatmap to {heatmap_path}")
    
    # 4. Visualize max-age distribution with more detail
    if has_cache_control.any():
        df['max_age_value'] = df['cache_control'].str.extract(r'max-age=(\d+)', expand=False).astype(float)
        
        max_ages = df['max_age_value'].dropna()
        if len(max_ages) > 0:
            # Create age buckets for better visualization
            age_buckets = [
                (0, 60, '< 1 min'),
                (60, 300, '1-5 min'),
                (300, 900, '5-15 min'),
                (900, 1800, '15-30 min'),
                (1800, 3600, '30-60 min'),
                (3600, 86400, '1-24 hours'),
                (86400, 604800, '1-7 days'),
                (604800, 2592000, '1-30 days'),
                (2592000, float('inf'), '> 30 days')
            ]
            
            # Categorize each max-age value
            def categorize_age(age):
                for low, high, label in age_buckets:
                    if low <= age < high:
                        return label
                return '> 30 days'
            
            df['max_age_bucket'] = df['max_age_value'].apply(lambda x: categorize_age(x) if pd.notnull(x) else None)
            
            # Count occurrences in each bucket
            bucket_counts = df['max_age_bucket'].value_counts().sort_index()
            
            # Only include buckets with data
            bucket_counts = bucket_counts[bucket_counts > 0]
            
            # Get ordered labels based on the age buckets
            ordered_labels = [label for _, _, label in age_buckets]
            ordered_labels = [l for l in ordered_labels if l in bucket_counts.index]
            
            plt.figure(figsize=(figwidth*1.2, figwidth / golden_ratio))
            
            # Create horizontal bar chart for better readability with many categories
            bars = plt.barh(ordered_labels, 
                           [bucket_counts[label]/len(max_ages)*100 for label in ordered_labels],
                           color=color_pallete[0])
            
            plt.title('Distribution of max-age Values')
            plt.xlabel('Percentage of Resources with max-age (%)')
            plt.ylabel('max-age Duration')
            
            # Add percentage labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 1, bar.get_y() + bar.get_height()/2.,
                         f'{width:.1f}%', ha='left', va='center')
            
            plt.grid(axis='x', linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            max_age_path = os.path.join(results_dir, f"{os.path.basename(csv_file).replace('.csv', '_max_age_buckets.pdf')}")
            plt.savefig(max_age_path, format='pdf')
            print(f"Saved max-age distribution chart to {max_age_path}")
    
    # 5. Compare cache hit rate effectiveness by cache directive
    if 'from_disk_cache' in df.columns and has_cache_control.any():
        warm_df = df[df['cache_state'] == 'warm']
        
        if len(warm_df) > 0:
            # Calculate cache hit rate with different directives
            directive_hit_rates = []
            directives = ['no_cache', 'no_store', 'private', 'public', 'max_age', 'immutable']
            
            for directive in directives:
                if directive in warm_df.columns:
                    with_directive = warm_df[warm_df[directive]]['from_disk_cache'].mean() * 100
                    without_directive = warm_df[~warm_df[directive]]['from_disk_cache'].mean() * 100
                    
                    directive_hit_rates.append({
                        'directive': directive.replace('_', '-'),
                        'with_directive': with_directive,
                        'without_directive': without_directive,
                        'with_count': sum(warm_df[directive]),
                        'without_count': sum(~warm_df[directive])
                    })
            
            # Only include directives with sufficient data
            directive_hit_rates = [d for d in directive_hit_rates 
                                  if d['with_count'] >= 5 and d['without_count'] >= 5]
            
            if directive_hit_rates:
                # Convert to DataFrame for easier plotting
                hit_rate_df = pd.DataFrame(directive_hit_rates)
                
                # Reshape for seaborn
                hit_rate_melted = pd.melt(
                    hit_rate_df, 
                    id_vars=['directive'], 
                    value_vars=['with_directive', 'without_directive'],
                    var_name='presence', 
                    value_name='hit_rate'
                )
                
                hit_rate_melted['presence'] = hit_rate_melted['presence'].map({
                    'with_directive': 'With Directive', 
                    'without_directive': 'Without Directive'
                })
                
                plt.figure(figsize=(figwidth*1.3, figwidth / golden_ratio))
                ax = sns.barplot(
                    x='directive', 
                    y='hit_rate', 
                    hue='presence', 
                    data=hit_rate_melted,
                    palette=[color_pallete[0], color_pallete[2]]
                )
                
                plt.title('Cache Hit Rate by Cache-Control Directive')
                plt.xlabel('Directive')
                plt.ylabel('Cache Hit Rate (%)')
                plt.xticks(rotation=30)
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                plt.legend(title='')
                plt.tight_layout()
                
                directive_effect_path = os.path.join(
                    results_dir, 
                    f"{os.path.basename(csv_file).replace('.csv', '_directive_effect.pdf')}"
                )
                plt.savefig(directive_effect_path, format='pdf')
                print(f"Saved directive effectiveness chart to {directive_effect_path}")
    
    print(f"Cache validation visualization complete. Charts saved to {results_dir}")
    return

def compare_validation_across_networks(base_directory=results_dir):
    """Compare cache validation strategies across network conditions focusing on HTTP/3"""
    # Define standard network condition directories to search
    network_directories = ['fast', 'typical', 'slow', 'very_slow']
    
    network_data = {}
    network_order = []
    
    # Look for CSV files in each network directory
    for network_name in network_directories:
        network_dir = os.path.join(base_directory, network_name)
        
        # Skip if directory doesn't exist
        if not os.path.isdir(network_dir):
            print(f"Directory for {network_name} not found: {network_dir}")
            continue
            
        # Find the most recent CSV file in this directory
        csv_files = glob.glob(os.path.join(network_dir, "*.csv"))
        # Filter out stats files
        csv_files = [f for f in csv_files if not ("_stats" in f or "_optimizations" in f)]
        
        if csv_files:
            # Use the most recent file (sorted by modification time)
            csv_files.sort(key=os.path.getmtime, reverse=True)
            latest_file = csv_files[0]
            
            try:
                df = pd.read_csv(latest_file)
                if 'protocol' in df.columns and 'cache_control' in df.columns and 'etag' in df.columns:
                    network_data[network_name] = df
                    network_order.append(network_name)
                    print(f"Loaded {network_name} data: {len(df)} rows from {os.path.basename(latest_file)}")
                else:
                    print(f"Skipping {latest_file} - missing required columns")
            except Exception as e:
                print(f"Error loading {latest_file}: {e}")
    
    if not network_data:
        print("No valid network condition files could be loaded.")
        return
    
    # Sort network order according to standard ordering
    standard_order = ['fast', 'typical', 'slow', 'very_slow']
    network_order = sorted(network_order, key=lambda x: standard_order.index(x) if x in standard_order else 999)
    print(f"Using network ordering: {network_order}")
    
    # Process each network's data to extract validation strategy information
    validation_results = []
    
    for network in network_order:
        df = network_data[network]
        
        # Convert validation-related columns to appropriate types
        if 'from_disk_cache' in df.columns and df['from_disk_cache'].dtype == 'object':
            df['from_disk_cache'] = df['from_disk_cache'].map({'true': True, 'false': False})
        
        # Extract validation headers
        has_etag = ~df['etag'].isna() & (df['etag'] != '')
        has_cache_control = ~df['cache_control'].isna() & (df['cache_control'] != '')
        
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
            # Calculate hit rates by validation strategy and protocol
            validation_effect = warm_df.groupby(['validation_strategy', 'protocol'])['from_disk_cache'].agg(
                ['mean', 'count']
            ).reset_index()
            
            # Add network condition information
            validation_effect['network_condition'] = network
            
            # Calculate hit rate percentage
            validation_effect['hit_rate_pct'] = validation_effect['mean'] * 100
            
            validation_results.append(validation_effect)
    
    if not validation_results:
        print("No validation strategy data could be extracted.")
        return
    
    # Combine all networks' results
    combined_validation = pd.concat(validation_results, ignore_index=True)
    
    # Filter for HTTP/3 data only
    h3_data = combined_validation[combined_validation['protocol'] == 'h3']
    
    # Only include strategies with sufficient data points
    valid_strategies = h3_data[h3_data['count'] >= 5]
    if len(valid_strategies) == 0:
        print("Insufficient HTTP/3 data points for visualization.")
        return
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create a single visualization for HTTP/3 only
    plt.figure(figsize=(figwidth*1.4, figwidth / golden_ratio))
    
    # Order the data for visualization
    valid_strategies['network_condition'] = pd.Categorical(
        valid_strategies['network_condition'],
        categories=network_order,
        ordered=True
    )
    
    # Create the HTTP/3 bar plot
    ax = sns.barplot(x='validation_strategy', y='hit_rate_pct', hue='network_condition', 
                   data=valid_strategies, palette=color_pallete)
    
    ax.set_title('HTTP/3 Cache Hit Rate by Validation Strategy (Warm Cache)', fontsize=14)
    ax.set_xlabel('Validation Strategy', fontsize=12)
    ax.set_ylabel('Cache Hit Rate (%)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(title='Network Condition')
    
    # Add count annotations
    for i, strategy in enumerate(valid_strategies['validation_strategy'].unique()):
        strategy_data = valid_strategies[valid_strategies['validation_strategy'] == strategy]
        for j, (_, row) in enumerate(strategy_data.iterrows()):
            ax.text(i - 0.2 + (j * 0.2), row['hit_rate_pct'] + 2, 
                   f"n={int(row['count'])}", ha='center', va='bottom', 
                   fontsize=9, rotation=0)
    
    # Improve x-axis label readability
    plt.xticks(rotation=30, ha='right')
    
    plt.tight_layout()
    validation_network_file = os.path.join(base_directory, f"http3_validation_strategy_by_network_{timestamp}.pdf")
    plt.savefig(validation_network_file, format='pdf')
    print(f"Saved HTTP/3 validation strategy comparison to {validation_network_file}")
    
    # Also create a heatmap version for better visualization of patterns
    pivot_h3 = valid_strategies.pivot_table(
        index='validation_strategy', 
        columns='network_condition',
        values='hit_rate_pct',
        aggfunc='mean'
    )
    
    if not pivot_h3.empty:
        plt.figure(figsize=(figwidth*1.3, figwidth / golden_ratio))
        ax = sns.heatmap(pivot_h3, annot=True, fmt=".1f", 
                        cmap="YlGnBu", cbar_kws={'label': 'Hit Rate (%)'})
        plt.title('HTTP/3 Cache Hit Rate by Validation Strategy and Network Condition', fontsize=14)
        
        plt.tight_layout()
        heatmap_file = os.path.join(base_directory, f"http3_validation_heatmap_{timestamp}.pdf")
        plt.savefig(heatmap_file, format='pdf')
        print(f"Saved HTTP/3 validation strategy heatmap to {heatmap_file}")
    
    return valid_strategies

def set_plot_style():
    """Set consistent, visually appealing plot style"""
    # Use seaborn style for better aesthetics
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})
    
    # Custom style settings
    plt.rcParams.update({
        # Font sizes - reduced for better proportions
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        
        # Legend appearance
        'legend.frameon': False,
        'legend.handletextpad': 0.3,
        'legend.columnspacing': 0.5,
        'legend.handlelength': 0.8,
        
        # Margins and spacing
        'figure.constrained_layout.use': True,
        
        # Remove top and right spines
        'axes.spines.right': False,
        'axes.spines.top': False
    })

def analyze_cache_by_asset_type(csv_file=None, base_directory=results_dir, cache_state="both"):
    """Compare HTTP/2 vs HTTP/3 load times by asset type under cold and/or warm cache conditions"""
    # Set plot style for this function
    set_plot_style()
    
    # If a specific CSV file is provided, analyze just that one
    if csv_file:
        df = pd.read_csv(csv_file)
        # Create a simple descriptive filename for the output
        output_prefix = os.path.basename(csv_file).replace('.csv', f'_cache_by_asset')
        source_label = os.path.basename(csv_file)
    else:
        # Otherwise, gather data from all network conditions
        network_directories = ['fast', 'typical', 'slow', 'very_slow']
        
        all_dfs = []
        network_order = []
        
        for network_name in network_directories:
            network_dir = os.path.join(base_directory, network_name)
            
            # Skip if directory doesn't exist
            if not os.path.isdir(network_dir):
                continue
                
            # Find the most recent CSV file in this directory
            csv_files = glob.glob(os.path.join(network_dir, "*.csv"))
            # Filter out stats files
            csv_files = [f for f in csv_files if not ("_stats" in f or "_optimizations" in f)]
            
            if csv_files:
                # Use the most recent file
                csv_files.sort(key=os.path.getmtime, reverse=True)
                latest_file = csv_files[0]
                
                try:
                    network_df = pd.read_csv(latest_file)
                    network_df['network_condition'] = network_name
                    all_dfs.append(network_df)
                    network_order.append(network_name)
                except Exception as e:
                    print(f"Error loading {latest_file}: {e}")
        
        if not all_dfs:
            print("No valid data files found for cache analysis.")
            return
            
        # Combine all data
        df = pd.concat(all_dfs, ignore_index=True)
        output_prefix = f"combined_cache_by_asset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        source_label = f"Combined data from {', '.join(network_order)} networks"
    
    # Check if we need to rename columns for compatibility
    if 'resource_type' in df.columns and 'asset_type' not in df.columns:
        df['asset_type'] = df['resource_type']
        print("Renamed 'resource_type' column to 'asset_type' for analysis")
    
    # Filter by cache state if specified
    if cache_state == "cold":
        filtered_df = df[df['cache_state'] == 'cold']
        state_suffix = "Cold Cache"
    elif cache_state == "warm":
        filtered_df = df[df['cache_state'] == 'warm']
        state_suffix = "Warm Cache"
    else:  # Analyze both cache states
        filtered_df = df
        state_suffix = "Cold vs Warm Cache"
    
    if len(filtered_df) == 0:
        print(f"No {cache_state} cache data found for analysis.")
        return
    
    # Group by protocol, cache state, and asset type to get mean load times
    asset_perf = filtered_df.groupby(['protocol', 'cache_state', 'asset_type'])['load_time_ms'].agg(
        ['mean', 'std', 'count']
    ).reset_index()
    
    # Only include asset types with sufficient data points
    min_samples = 5
    valid_assets = asset_perf.groupby(['asset_type', 'cache_state'])['count'].min() >= min_samples
    
    # Convert to list of valid combinations
    valid_combinations = valid_assets[valid_assets].index.tolist()
    valid_asset_types = set([asset for asset, _ in valid_combinations])
    
    if not valid_asset_types:
        print("No asset types have sufficient data points for analysis.")
        return
    
    # Filter to valid asset types
    asset_perf = asset_perf[asset_perf['asset_type'].isin(valid_asset_types)]
    
    # Create a figure for each cache state 
    for current_cache_state in asset_perf['cache_state'].unique():
        # Filter to the current cache state
        state_df = asset_perf[asset_perf['cache_state'] == current_cache_state]
        
        # Create the bar chart - reduced figure size for better proportions
        plt.figure(figsize=(figwidth*1.3, figwidth / golden_ratio * 0.9))
        
        # Prepare data for grouped bar chart
        state_pivot = state_df.pivot_table(
            index='asset_type', 
            columns='protocol', 
            values='mean'
        )
        
        # Sort asset types by average load time (descending)
        state_pivot = state_pivot.reindex(state_pivot.mean(axis=1).sort_values(ascending=False).index)
        
        # Create bar chart with nicer styling
        ax = state_pivot.plot(kind='bar', color=color_pallete[:2])
        
        ax.set_title(f'HTTP/2 vs HTTP/3 Load Time by Asset Type ({current_cache_state.title()} Cache)')
        ax.set_xlabel('Asset Type')
        ax.set_ylabel('Average Load Time (ms)')
        
        # Format y-axis if values are large
        if state_df['mean'].max() > 1000:
            ax.yaxis.set_major_formatter(kfmt)
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Small source label in bottom
        plt.figtext(0.5, 0.01, f"Source: {source_label}", ha="center", fontsize=7, alpha=0.7)
        
        # Improve legend position and sizing
        leg = ax.legend(title=None, loc='upper right', ncol=2, frameon=False)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=30, ha='right')
        
        plt.tight_layout(pad=1.1)
        
        # Save chart at higher DPI for better quality
        state_path = os.path.join(base_directory, f"{output_prefix}_{current_cache_state}.pdf")
        plt.savefig(state_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Saved {current_cache_state} cache asset type performance chart to {state_path}")
        
        # Create improvement chart for this cache state
        plt.figure(figsize=(figwidth*1.1, figwidth / golden_ratio * 0.9))
        
        # Calculate percentage improvement of HTTP/3 over HTTP/2
        improvement = pd.DataFrame(index=state_pivot.index)
        if 'h2' in state_pivot.columns and 'h3' in state_pivot.columns:
            improvement['improvement_pct'] = (
                (state_pivot['h2'] - state_pivot['h3']) / 
                state_pivot['h2'] * 100
            )
        
            # Sort by improvement percentage
            improvement = improvement.sort_values('improvement_pct', ascending=False)
            
            # Create the bar chart
            ax = sns.barplot(x=improvement.index, y='improvement_pct', data=improvement, color=color_pallete[0])
            
            ax.set_title(f'HTTP/3 Performance Improvement by Asset Type ({current_cache_state.title()} Cache)')
            ax.set_xlabel('Asset Type')
            ax.set_ylabel('Improvement (%)')
            
            # Add a horizontal line at y=0
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=30, ha='right')
            
            # Add grid for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            plt.tight_layout(pad=1.1)
            
            # Save improvement chart
            improvement_path = os.path.join(base_directory, f"{output_prefix}_{current_cache_state}_improvement.pdf")
            plt.savefig(improvement_path, format='pdf', dpi=300, bbox_inches='tight')
            print(f"Saved HTTP/3 improvement chart ({current_cache_state} cache) to {improvement_path}")
    
    # If analyzing both cache states, also create a comparison showing cache benefit for both protocols
    if cache_state == "both" and "cold" in asset_perf['cache_state'].values and "warm" in asset_perf['cache_state'].values:
        # Create a chart showing cache benefit by protocol and asset type
        plt.figure(figsize=(figwidth*1.4, figwidth / golden_ratio))
        
        # Calculate cache benefit (how much faster is warm vs cold)
        cache_benefit = pd.DataFrame()
        
        for protocol in ['h2', 'h3']:
            # Get protocol data
            proto_data = asset_perf[asset_perf['protocol'] == protocol]
            
            # Pivot to wide format with cache states as columns
            proto_pivot = proto_data.pivot_table(
                index='asset_type', 
                columns='cache_state', 
                values='mean'
            )
            
            # Only process if we have both cold and warm data
            if 'cold' in proto_pivot.columns and 'warm' in proto_pivot.columns:
                # Calculate percentage improvement from cold to warm
                benefit = ((proto_pivot['cold'] - proto_pivot['warm']) / proto_pivot['cold'] * 100)
                
                # Add to the result dataframe
                cache_benefit[protocol] = benefit
        
        # Only continue if we have data
        if not cache_benefit.empty:
            # Convert to long format for seaborn
            cache_benefit_long = cache_benefit.reset_index()
            cache_benefit_long = pd.melt(
                cache_benefit_long, 
                id_vars=['asset_type'],
                var_name='protocol', 
                value_name='cache_benefit_pct'
            )
            
            # Sort by average benefit
            asset_order = cache_benefit.mean(axis=1).sort_values(ascending=False).index
            cache_benefit_long['asset_type'] = pd.Categorical(
                cache_benefit_long['asset_type'], 
                categories=asset_order, 
                ordered=True
            )
            
            # Create the plot
            ax = sns.barplot(
                x='asset_type', 
                y='cache_benefit_pct', 
                hue='protocol', 
                data=cache_benefit_long, 
                palette=color_pallete[:2]
            )
            
            ax.set_title('Cache Benefit by Asset Type and Protocol')
            ax.set_xlabel('Asset Type')
            ax.set_ylabel('Cache Load Time Improvement (%)')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=30, ha='right')
            
            # Add grid for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            plt.tight_layout(pad=1.1)
            
            # Save the chart
            cache_benefit_path = os.path.join(base_directory, f"{output_prefix}_cache_benefit.pdf")
            plt.savefig(cache_benefit_path, format='pdf', dpi=300, bbox_inches='tight')
            print(f"Saved cache benefit comparison chart to {cache_benefit_path}")
    
    return asset_perf

def analyze_cold_cache_by_asset_type(csv_file=None, base_directory=results_dir):
    """Compare HTTP/2 vs HTTP/3 load times by asset type under cold cache conditions"""
    # Call the new function with cache_state="cold"
    return analyze_cache_by_asset_type(csv_file, base_directory, cache_state="cold")

def analyze_warm_cache_by_asset_type(csv_file=None, base_directory=results_dir):
    """Compare HTTP/2 vs HTTP/3 load times by asset type under warm cache conditions"""
    # Call the new function with cache_state="warm"
    return analyze_cache_by_asset_type(csv_file, base_directory, cache_state="warm")

def analyze_cache_directives_across_networks(base_directory=results_dir):
    """Create a visualization showing cache directive distribution across different network conditions"""
    # Define standard network condition directories to search
    network_directories = ['fast', 'typical', 'slow']
    
    network_data = {}
    network_order = []
    
    # Set plot style for better visualization
    set_plot_style()
    
    # Look for CSV files in each network directory
    for network_name in network_directories:
        network_dir = os.path.join(base_directory, network_name)
        
        # Skip if directory doesn't exist
        if not os.path.isdir(network_dir):
            print(f"Directory for {network_name} not found: {network_dir}")
            continue
            
        # Find the most recent CSV file in this directory
        csv_files = glob.glob(os.path.join(network_dir, "*.csv"))
        # Filter out stats files
        csv_files = [f for f in csv_files if not ("_stats" in f or "_optimizations" in f)]
        
        if csv_files:
            # Use the most recent file (sorted by modification time)
            csv_files.sort(key=os.path.getmtime, reverse=True)
            latest_file = csv_files[0]
            
            try:
                df = pd.read_csv(latest_file)
                if 'cache_control' in df.columns:
                    network_data[network_name] = df
                    network_order.append(network_name)
                    print(f"Loaded {network_name} data: {len(df)} rows from {os.path.basename(latest_file)}")
                else:
                    print(f"Skipping {latest_file} - missing required cache_control column")
            except Exception as e:
                print(f"Error loading {latest_file}: {e}")
    
    if not network_data:
        print("No valid data found for cache directive analysis")
        return
        
    # Sort network order according to standard ordering
    standard_order = ['fast', 'typical', 'slow', 'very_slow']
    network_order = sorted(network_order, key=lambda x: standard_order.index(x) if x in standard_order else 999)
    print(f"Analyzing cache directives across: {', '.join(network_order)} networks")
    
    # Extract and aggregate directive data across networks
    all_directives = {
        'no-cache': [],
        'no-store': [],
        'private': [],
        'public': [],
        'max-age': [],
        'immutable': [],
        'must-revalidate': [],
        'no-transform': [],
        's-maxage': [],
        'proxy-revalidate': []
    }
    
    # Process each network's data
    for network in network_order:
        df = network_data[network]
        total_resources = len(df)
        
        # Get resources with cache-control headers
        has_cache_control = ~df['cache_control'].isna() & (df['cache_control'] != '')
        cache_resources = sum(has_cache_control)
        
        # Extract each directive
        for directive in all_directives.keys():
            count = sum(df['cache_control'].str.contains(directive, case=False, na=False))
            percentage = (count / total_resources) * 100 if total_resources > 0 else 0
            all_directives[directive].append({
                'network': network,
                'count': count,
                'percentage': percentage,
                'total': total_resources
            })
    
    # Calculate averages across networks
    average_percentages = {}
    for directive, data in all_directives.items():
        if data:  # Check if we have data for this directive
            avg = sum(item['percentage'] for item in data) / len(data)
            average_percentages[directive] = avg
    
    # Only include directives that appear at least once
    average_percentages = {k: v for k, v in average_percentages.items() if v > 0}
    
    # Sort directives by average usage
    sorted_directives = sorted(average_percentages.items(), key=lambda x: x[1], reverse=True)
    directive_names = [item[0] for item in sorted_directives]
    directive_values = [item[1] for item in sorted_directives]
    
    # Create a visually appealing horizontal bar chart
    plt.figure(figsize=(figwidth*1.3, figwidth / golden_ratio * 1.2))
    
    # Generate a gradient color for bars that smoothly transitions across the spectrum
    num_directives = len(directive_names)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_directives))
    
    # Plot horizontal bars
    bars = plt.barh(directive_names, directive_values, color=colors, alpha=0.85)
    
    # Add a subtle grid
    plt.grid(axis='x', linestyle='--', alpha=0.2, zorder=0)
    
    # Add percentage labels on the bars
    for bar in bars:
        width = bar.get_width()
        if width > 2:  # Only add text if there's enough space
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                     f'{width:.1f}%', va='center', fontsize=9)
    
    # Title and axis labels with improved styling
    # plt.title('Cache-Control Directive Usage (Average Across Network Conditions)', 
    #           fontsize=12, pad=15)
    plt.xlabel('Percentage of Resources (%)', fontsize=14, labelpad=10)
    
    # Remove y-axis label as the directive names are self-explanatory
    plt.ylabel('')
    
    # Clean up the plot - remove top and right spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    # Add a subtle background color for better visual appeal
    plt.gca().set_facecolor('#f9f9f9')
    
    # Add a subtle annotation about the data source
    plt.figtext(0.02, 0.01, f"Source: Data averaged across {len(network_order)} network conditions", 
                fontsize=7, alpha=0.6)
    
    plt.tight_layout(pad=1.2)
    
    # Generate timestamp for output file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the visualization
    output_file = os.path.join(base_directory, f"cache_directives_across_networks_{timestamp}.pdf")
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved cache directives visualization to {output_file}")
    
    # Also create a comparison plot showing directive usage across different networks
    if len(network_order) >= 2:
        plt.figure(figsize=(figwidth*1.4, figwidth / golden_ratio * 1.5))
        
        # Prepare data for grouped bar chart
        comparison_data = []
        for directive in directive_names[:min(6, len(directive_names))]:  # Limit to top 6 directives
            for network_data in all_directives[directive]:
                comparison_data.append({
                    'directive': directive,
                    'network': network_data['network'],
                    'percentage': network_data['percentage']
                })
        
        if comparison_data:
            df_comp = pd.DataFrame(comparison_data)
            
            # Create a categorical type with our desired order
            df_comp['network'] = pd.Categorical(df_comp['network'], categories=network_order, ordered=True)
            df_comp['directive'] = pd.Categorical(df_comp['directive'], categories=directive_names, ordered=True)
            
            # Create grouped bar chart
            ax = sns.barplot(x='directive', y='percentage', hue='network', data=df_comp, 
                           palette=sns.color_palette("viridis", len(network_order)))
            
            plt.title('Cache-Control Directive Usage by Network Condition', fontsize=12)
            plt.xlabel('Directive', fontsize=10)
            plt.ylabel('Percentage of Resources (%)', fontsize=10)
            
            # Improve legend
            plt.legend(title='Network Condition', frameon=False, fontsize=9)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=20, ha='right', fontsize=9)
            
            plt.grid(axis='y', linestyle='--', alpha=0.2)
            plt.tight_layout()
            
            # Save comparison chart
            comparison_file = os.path.join(base_directory, f"cache_directives_by_network_{timestamp}.pdf")
            plt.savefig(comparison_file, format='pdf', dpi=300, bbox_inches='tight')
            print(f"Saved network comparison chart to {comparison_file}")
    
    return average_percentages

def analyze_validation_hit_rates_across_networks(base_directory=results_dir):
    """Create a unified visualization showing validation hit rate effectiveness across network conditions"""
    # Define standard network condition directories to search
    network_directories = ['fast', 'typical', 'slow']
    
    network_data = {}
    network_order = []
    
    # Set plot style for better visualization
    set_plot_style()
    
    # Look for CSV files in each network directory
    for network_name in network_directories:
        network_dir = os.path.join(base_directory, network_name)
        
        # Skip if directory doesn't exist
        if not os.path.isdir(network_dir):
            print(f"Directory for {network_name} not found: {network_dir}")
            continue
            
        # Find the most recent CSV file in this directory
        csv_files = glob.glob(os.path.join(network_dir, "*.csv"))
        # Filter out stats files
        csv_files = [f for f in csv_files if not ("_stats" in f or "_optimizations" in f)]
        
        if csv_files:
            # Use the most recent file (sorted by modification time)
            csv_files.sort(key=os.path.getmtime, reverse=True)
            latest_file = csv_files[0]
            
            try:
                df = pd.read_csv(latest_file)
                if 'cache_control' in df.columns and 'etag' in df.columns and 'from_disk_cache' in df.columns:
                    # Convert string boolean values to actual booleans if needed
                    if df['from_disk_cache'].dtype == 'object':
                        df['from_disk_cache'] = df['from_disk_cache'].map({'true': True, 'false': False})
                    
                    network_data[network_name] = df
                    network_order.append(network_name)
                    print(f"Loaded {network_name} data: {len(df)} rows from {os.path.basename(latest_file)}")
                else:
                    print(f"Skipping {latest_file} - missing required columns for validation analysis")
            except Exception as e:
                print(f"Error loading {latest_file}: {e}")
    
    if not network_data or len(network_data) < 2:
        print("Insufficient network condition data for cross-network validation comparison.")
        return
    
    # Sort network order according to standard ordering
    standard_order = ['fast', 'typical', 'slow', 'very_slow']
    network_order = sorted(network_order, key=lambda x: standard_order.index(x) if x in standard_order else 999)
    print(f"Analyzing validation effectiveness across: {', '.join(network_order)} networks")
    
    # Process each network's validation data
    validation_results = []
    
    for network in network_order:
        df = network_data[network]
        
        # Extract validation headers
        has_etag = ~df['etag'].isna() & (df['etag'] != '')
        has_cache_control = ~df['cache_control'].isna() & (df['cache_control'] != '')
        
        # Create validation strategy categories
        conditions = [
            (has_etag & has_cache_control),
            (has_etag & ~has_cache_control),
            (~has_etag & has_cache_control),
        ]
        
        strategies = [
            'ETag + Cache-Control',
            'ETag only',
            'Cache-Control only',
        ]
        
        df['validation_strategy'] = np.select(conditions, strategies, default='Unknown')
        
        # Only analyze warm cache where validation matters
        warm_df = df[df['cache_state'] == 'warm']
        
        if len(warm_df) > 0:
            # Calculate hit rates by validation strategy and protocol
            for protocol in warm_df['protocol'].unique():
                protocol_df = warm_df[warm_df['protocol'] == protocol]
                
                # Group by validation strategy to calculate hit rates
                for strategy in strategies:
                    strategy_df = protocol_df[protocol_df['validation_strategy'] == strategy]
                    
                    if len(strategy_df) >= 5:  # Only include if we have sufficient data points
                        hit_rate = strategy_df['from_disk_cache'].mean() * 100
                        count = len(strategy_df)
                        
                        validation_results.append({
                            'network': network,
                            'protocol': protocol,
                            'validation_strategy': strategy,
                            'hit_rate': hit_rate,
                            'count': count
                        })
    
    if not validation_results:
        print("No validation hit rate data could be extracted.")
        return
        
    # Convert to DataFrame for easier manipulation
    results_df = pd.DataFrame(validation_results)
    
    # Generate timestamp for output file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create a combined figure showing validation effectiveness across networks
    plt.figure(figsize=(figwidth*1.6, figwidth / golden_ratio * 1.3))
    
    # Set categorical order for better visualization
    results_df['network'] = pd.Categorical(
        results_df['network'], 
        categories=network_order,
        ordered=True
    )
    
    results_df['protocol'] = pd.Categorical(
        results_df['protocol'],
        categories=['h2', 'h3'],
        ordered=True
    )
    
    # Sort strategies by average hit rate
    strategy_avg = results_df.groupby('validation_strategy')['hit_rate'].mean().sort_values(ascending=False)
    strategy_order = strategy_avg.index.tolist()
    
    results_df['validation_strategy'] = pd.Categorical(
        results_df['validation_strategy'],
        categories=strategy_order,
        ordered=True
    )
    
    # Create the combined visualization
    g = sns.catplot(
        data=results_df,
        x='validation_strategy',
        y='hit_rate',
        hue='network',
        col='protocol',
        kind='bar',
        palette=color_pallete[:len(network_order)],
        legend=False,
        height=figwidth / golden_ratio,
        aspect=1.0,
        sharey=True
    )
    
    # Customize the plot
    g.set_axis_labels('Validation Strategy', 'Cache Hit Rate (%)')
    g.set_titles(col_template='{col_name}')
    
    # Rotate x-axis labels for better readability
    for ax in g.axes.flat:
        for idx, label in enumerate(ax.get_xticklabels()):
            ax.text(idx, -5, label.get_text(), rotation=30, ha='right', fontsize=8)
        
        ax.set_xticklabels([])  # Remove the original labels
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add data point counts as annotations
        for i, strategy in enumerate(strategy_order):
            for j, network in enumerate(network_order):
                data = results_df[
                    (results_df['validation_strategy'] == strategy) & 
                    (results_df['network'] == network) &
                    (results_df['protocol'] == ax.get_title())
                ]
                
                if not data.empty:
                    # Calculate position for the text
                    x_pos = i - 0.3 + (j * 0.15)
                    y_pos = data['hit_rate'].values[0]
                    count = data['count'].values[0]
                    
                    # Add count annotation if there's enough space
                    if y_pos > 5:
                        ax.text(x_pos, y_pos - 5, f"n={count}", 
                                ha='center', va='top', fontsize=7, 
                                rotation=90, alpha=0.7)
    
    # Add a single legend for the entire figure
    handles, labels = g.axes[0, 0].get_legend_handles_labels()
    g.fig.legend(
        handles, 
        labels, 
        title='Network Condition', 
        loc='upper center', 
        ncol=len(network_order),
        frameon=False,
        bbox_to_anchor=(0.5, 1.05)
    )
    
    # Don't use tight_layout with suptitle
    g.fig.subplots_adjust(top=0.85, bottom=0.15, wspace=0.1)
    
    # Add a descriptive suptitle
    g.fig.suptitle('Cache Validation Effectiveness Across Network Conditions', 
                  fontsize=14, y=0.98)
    
    # Save the visualization
    output_file = os.path.join(base_directory, f"validation_hit_rates_combined_{timestamp}.pdf")
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved combined validation hit rates visualization to {output_file}")
    
    # Also create a single unified heatmap showing protocol and network effects
    plt.figure(figsize=(figwidth*1.5, figwidth / golden_ratio * 1.2))
    
    try:
        # Create a pivot table for the heatmap - use observed=True to avoid FutureWarning
        pivot_data = results_df.pivot_table(
            index='validation_strategy',
            columns=['protocol', 'network'],
            values='hit_rate',
            observed=True
        )
        
        # Plot the heatmap
        ax = sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.1f',
            cmap='YlGnBu',
            linewidths=.5,
            cbar_kws={'label': 'Cache Hit Rate (%)'}
        )

        # Adjust the font size of labels
        ax.tick_params(axis='x', labelsize=12)  # Change x-axis label size
        ax.tick_params(axis='y', labelsize=6)  # Change y-axis label size
        
        # Change annotation (text in cells) font size
        for text in ax.texts:
            text.set_fontsize(10)  # Smaller font for the numbers inside cells
        
        # Change colorbar label size
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)  # Colorbar tick label size
        cbar.set_label('Cache Hit Rate (%)', size=9)  # Colorbar title size
   
        
        # plt.title('Combined Validation Strategy Effectiveness', fontsize=14)
        # Use manual adjustment instead of tight_layout() to avoid colorbar issues
        plt.gcf().subplots_adjust(bottom=0.15, right=0.9, top=0.9, left=0.15)
        
        # Save heatmap
        heatmap_file = os.path.join(base_directory, f"validation_hit_rates_heatmap_{timestamp}.pdf")
        plt.savefig(heatmap_file, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Saved validation hit rates heatmap to {heatmap_file}")
    except Exception as e:
        print(f"Error creating heatmap visualization: {e}")
    
    return results_df

def analyze_zero_rtt_comprehensive(csv_file=None, base_directory=results_dir):
    """
    Create comprehensive analysis and visualizations for HTTP/3 0-RTT performance
    
    This function combines several aspects of 0-RTT analysis:
    1. Connection time distribution with and without 0-RTT
    2. Success rates across different network conditions
    3. Performance impact by resource size
    4. Sequential request patterns and session resumption behavior
    5. CDN-specific 0-RTT adoption
    """
    print("\n=== Comprehensive 0-RTT Performance Analysis ===")
    
    # Set plot style for better visualization
    set_plot_style()
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Analyze either a specific file or collect data across network directories
    if csv_file:
        df = pd.read_csv(csv_file)
        # Create a simple descriptive filename for the output
        output_prefix = os.path.basename(csv_file).replace('.csv', f'_0rtt_analysis')
        network_name = "custom"
        if "network_latency" in df.columns:
            latencies = df["network_latency"].unique()
            if len(latencies) == 1:
                network_latency = latencies[0]
                if network_latency <= 5:
                    network_name = "fast"
                elif network_latency <= 25:
                    network_name = "typical"
                elif network_latency <= 100:
                    network_name = "slow"

    else:
        # Otherwise, gather data from all network conditions
        network_directories = ['fast', 'typical', 'slow']
        
        all_dfs = []
        network_order = []
        
        for network_name in network_directories:
            network_dir = os.path.join(base_directory, network_name)
            
            # Skip if directory doesn't exist
            if not os.path.isdir(network_dir):
                continue
                
            # Find the most recent CSV file in this directory
            csv_files = glob.glob(os.path.join(network_dir, "*.csv"))
            # Filter out stats files
            csv_files = [f for f in csv_files if not ("_stats" in f or "_optimizations" in f)]
            
            if csv_files:
                # Use the most recent file
                csv_files.sort(key=os.path.getmtime, reverse=True)
                latest_file = csv_files[0]
                
                try:
                    network_df = pd.read_csv(latest_file)
                    network_df['network_condition'] = network_name
                    all_dfs.append(network_df)
                    network_order.append(network_name)
                except Exception as e:
                    print(f"Error loading {latest_file}: {e}")
        
        if not all_dfs:
            print("No valid data files found for 0-RTT analysis.")
            return
            
        # Combine all data
        df = pd.concat(all_dfs, ignore_index=True)
        output_prefix = f"combined_0rtt_analysis_{timestamp}"
    
    # Check if we have 0-RTT data available
    if 'zero_rtt_used' not in df.columns:
        print("0-RTT data not available in this dataset")
        return
    
    # Convert string boolean values to actual booleans if needed
    if df['zero_rtt_used'].dtype == 'object':
        df['zero_rtt_used'] = df['zero_rtt_used'].map({'true': True, 'false': False})
        
    if 'tls_resumed' in df.columns and df['tls_resumed'].dtype == 'object':
        df['tls_resumed'] = df['tls_resumed'].map({'true': True, 'false': False})
        
    if 'connection_reused' in df.columns and df['connection_reused'].dtype == 'object':
        df['connection_reused'] = df['connection_reused'].map({'true': True, 'false': False})
        
    if 'from_disk_cache' in df.columns and df['from_disk_cache'].dtype == 'object':
        df['from_disk_cache'] = df['from_disk_cache'].map({'true': True, 'false': False})
    
    # Filter to only HTTP/3 data - 0-RTT is specific to HTTP/3
    h3_data = df[df['protocol'] == 'h3'].copy()
    
    if len(h3_data) == 0:
        print("No HTTP/3 data available for analysis")
        return
    
    # Print basic stats about 0-RTT usage
    zero_rtt_count = h3_data['zero_rtt_used'].sum()
    total_h3_requests = len(h3_data)
    zero_rtt_percentage = (zero_rtt_count / total_h3_requests * 100) if total_h3_requests > 0 else 0
    
    print(f"Total HTTP/3 requests: {total_h3_requests}")
    print(f"0-RTT requests: {zero_rtt_count} ({zero_rtt_percentage:.1f}%)")
    
    # Only continue if we have some 0-RTT requests
    if zero_rtt_count < 5:
        print("Insufficient 0-RTT requests for meaningful analysis")
        return
    
    # 1. CONNECTION TIME DISTRIBUTION ANALYSIS
    # -----------------------------------------
    plt.figure(figsize=(figwidth*1.2, figwidth / golden_ratio))
    
    # Group by 0-RTT usage
    groups = h3_data.groupby('zero_rtt_used')
    
    for name, group in groups:
        # Sort connection times
        x = np.sort(group['connection_time_ms'])
        if len(x) == 0:
            continue
            
        # Create CDF (cumulative distribution function)
        y = np.arange(1, len(x) + 1) / len(x)
        
        label = "With 0-RTT" if name else "Without 0-RTT"
        color = color_pallete[0] if name else color_pallete[2]
        plt.plot(x, y, label=label, color=color, linewidth=2)
    
    # plt.title('Connection Time Distribution (HTTP/3)', fontsize=12)
    plt.xlabel('Connection Establishment Time (ms)', fontsize=10)
    plt.ylabel('Cumulative Probability', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistical markers
    for name, group in groups:
        if len(group) == 0:
            continue
            
        # Add vertical lines for medians
        median = np.median(group['connection_time_ms'])
        color = color_pallete[0] if name else color_pallete[2]
        plt.axvline(x=median, color=color, linestyle='--', alpha=0.7)
        plt.text(median, 0.5, f'Median: {median:.1f}ms', 
                 color=color, ha='right' if name else 'left', va='center', 
                 rotation=90, alpha=0.8)
    
    plt.tight_layout()
    connection_cdf_path = os.path.join(base_directory, f"{output_prefix}_connection_cdf.pdf")
    plt.savefig(connection_cdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved 0-RTT connection time distribution to {connection_cdf_path}")
    
    # 2. NETWORK CONDITION IMPACT ANALYSIS
    # ------------------------------------
    if 'network_condition' in df.columns or 'network_latency' in df.columns:
        # Use explicit network_condition if available, otherwise derive from latency
        if 'network_condition' not in df.columns and 'network_latency' in df.columns:
            # Create network condition categories based on latency
            conditions = [
                (df['network_latency'] <= 5),
                (df['network_latency'] > 5) & (df['network_latency'] <= 25),
                (df['network_latency'] > 25) & (df['network_latency'] <= 100),
                (df['network_latency'] > 100)
            ]
            values = ['fast', 'typical', 'slow']
            df['network_condition'] = np.select(conditions, values, default='unknown')
            h3_data['network_condition'] = df.loc[h3_data.index, 'network_condition']
        
        plt.figure(figsize=(figwidth*1.2, figwidth / golden_ratio))
        
        # Calculate 0-RTT success rate by network condition
        rtt_by_network = h3_data.groupby('network_condition')['zero_rtt_used'].agg(
            ['mean', 'count', 'sum']
        ).reset_index()
        
        rtt_by_network['success_rate'] = rtt_by_network['mean'] * 100
        
        # Sort by standard network order
        standard_order = ['fast', 'typical', 'slow', 'very_slow']
        rtt_by_network['order'] = rtt_by_network['network_condition'].apply(
            lambda x: standard_order.index(x) if x in standard_order else 999
        )
        rtt_by_network = rtt_by_network.sort_values('order')
        
        # Create bar chart
        bars = plt.bar(rtt_by_network['network_condition'], rtt_by_network['success_rate'], 
                      color=color_pallete[0])
        
        # Add attempt counts as annotations
        # for i, bar in enumerate(bars):
        #     # Change from inside bar (value 5) to above bar
        #     height = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width()/2, height + 2, 
        #             f"n={int(rtt_by_network.iloc[i]['count'])}", 
        #             ha='center', va='bottom', fontsize=9)
        
        # plt.title('HTTP/3 0-RTT Success Rate by Network Condition', fontsize=12)
        plt.xlabel('Network Condition (n=1500)', fontsize=10)
        plt.ylabel('0-RTT Success Rate (%)', fontsize=10)
        
        max_success_rate = rtt_by_network['success_rate'].max()
        y_max = min(105, max(max_success_rate * 1.1, max_success_rate + 5))
        plt.ylim(0, y_max)  # Dynamic y-limit with padding for annotations
        
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        network_file = os.path.join(base_directory, f"{output_prefix}_network_success.pdf")
        plt.savefig(network_file, format='pdf', bbox_inches='tight')
        print(f"Saved 0-RTT success rate by network to {network_file}")
    
    # 3. RESOURCE SIZE IMPACT ANALYSIS
    # --------------------------------
    if 'size_bytes' in h3_data.columns:
        plt.figure(figsize=(figwidth*1.2, figwidth / golden_ratio))
        
        # Create size buckets
        size_breaks = [0, 5*1024, 50*1024, 500*1024, float('inf')]
        size_labels = ['< 5KB', '5-50KB', '50-500KB', '> 500KB']
        
        h3_data['size_bucket'] = pd.cut(h3_data['size_bytes'], bins=size_breaks, labels=size_labels)
        
        # Group by size bucket and 0-RTT usage
        size_stats = h3_data.groupby(['size_bucket', 'zero_rtt_used'])['load_time_ms'].agg(
            ['mean', 'median', 'count']
        ).reset_index()
        
        # Only keep buckets with enough samples
        min_samples = 5
        valid_buckets = size_stats.groupby('size_bucket')['count'].min() >= min_samples
        valid_buckets = valid_buckets[valid_buckets].index.tolist()
        
        if valid_buckets:
            # Filter to valid size buckets
            size_stats = size_stats[size_stats['size_bucket'].isin(valid_buckets)]
            
            # Create pivot table for plotting
            size_pivot = size_stats.pivot(index='size_bucket', columns='zero_rtt_used', values='mean')
            
            # Calculate improvement percentage
            if True in size_pivot.columns and False in size_pivot.columns:
                size_pivot['improvement'] = ((size_pivot[False] - size_pivot[True]) / size_pivot[False] * 100)
                
                # Create bar chart of improvement percentage
                ax = size_pivot['improvement'].plot(kind='bar', color=color_pallete[0])
                
                plt.title('0-RTT Performance Impact by Resource Size', fontsize=12)
                plt.xlabel('Resource Size', fontsize=10)
                plt.ylabel('Load Time Improvement (%)', fontsize=10)
                
                # Add horizontal line at 0
                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                
                # Add annotations with sample sizes
                for i, size_bucket in enumerate(size_pivot.index):
                    bucket_data = size_stats[size_stats['size_bucket'] == size_bucket]
                    count_0rtt = bucket_data[bucket_data['zero_rtt_used']]['count'].values[0]
                    count_no_rtt = bucket_data[~bucket_data['zero_rtt_used']]['count'].values[0]
                    plt.annotate(f"n={count_0rtt}/{count_no_rtt}", 
                                 (i, 5), textcoords="offset points", 
                                 xytext=(0,5), ha='center')
                
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                
                size_benefit_path = os.path.join(base_directory, f"{output_prefix}_size_impact.pdf")
                plt.savefig(size_benefit_path, format='pdf', bbox_inches='tight')
                print(f"Saved 0-RTT size impact analysis to {size_benefit_path}")
    
    # 4. SEQUENTIAL REQUEST PATTERN ANALYSIS
    # --------------------------------------
    if 'iteration' in h3_data.columns:
        plt.figure(figsize=(figwidth*1.2, figwidth / golden_ratio))
        
        # Filter to warm cache for more relevant 0-RTT analysis
        h3_warm = h3_data[h3_data['cache_state'] == 'warm']
        
        if len(h3_warm) >= 10:  # Only proceed if we have enough warm cache data
            # Calculate 0-RTT success rate by iteration
            rtt_by_iteration = h3_warm.groupby('iteration')['zero_rtt_used'].agg(
                ['mean', 'count', 'sum']
            ).reset_index()
            
            rtt_by_iteration['success_rate'] = rtt_by_iteration['mean'] * 100
            
            # Plot success rate vs iteration
            plt.plot(rtt_by_iteration['iteration'], rtt_by_iteration['success_rate'], 
                     'o-', color=color_pallete[0], markersize=5)
            
            plt.title('HTTP/3 0-RTT Success Rate by Connection Sequence', fontsize=12)
            plt.xlabel('Request Iteration (Warm Cache)', fontsize=10)
            plt.ylabel('0-RTT Success Rate (%)', fontsize=10)
            
            plt.ylim(0, 105)  # Make percentage scale clear
            
            # Add trend line if we have enough data
            if len(rtt_by_iteration) >= 3:
                z = np.polyfit(rtt_by_iteration['iteration'], rtt_by_iteration['success_rate'], 1)
                p = np.poly1d(z)
                plt.plot(rtt_by_iteration['iteration'], p(rtt_by_iteration['iteration']),
                         "--", color='gray', alpha=0.7)
                # Add slope annotation
                slope = z[0]
                plt.figtext(0.7, 0.15, f"Trend: {slope:.2f}%/iteration", 
                           ha="center", fontsize=9, 
                           bbox={"facecolor":"white", "alpha":0.8, "pad":5})
            
            # Add annotations with sample counts
            for i, row in rtt_by_iteration.iterrows():
                plt.annotate(f"n={int(row['count'])}", 
                             (row['iteration'], row['success_rate'] + 3),
                             fontsize=8, ha='center')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            sequential_path = os.path.join(base_directory, f"{output_prefix}_sequential.pdf")
            plt.savefig(sequential_path, format='pdf', bbox_inches='tight')
            print(f"Saved 0-RTT sequential analysis to {sequential_path}")
    
    # 5. CDN-SPECIFIC 0-RTT ANALYSIS
    # ------------------------------
    if 'cdn' in h3_data.columns:
        plt.figure(figsize=(figwidth*1.3, figwidth / golden_ratio))
        
        # Group by CDN provider and calculate 0-RTT percentage
        cdn_zero_rtt = h3_data.groupby('cdn')['zero_rtt_used'].agg(
            ['mean', 'count', 'sum']
        ).reset_index()
        
        cdn_zero_rtt['success_rate'] = cdn_zero_rtt['mean'] * 100
        
        # Filter to CDNs with sufficient data
        min_cdn_samples = 10
        cdn_zero_rtt = cdn_zero_rtt[cdn_zero_rtt['count'] >= min_cdn_samples]
        
        if len(cdn_zero_rtt) > 0:
            # Sort by success rate
            cdn_zero_rtt = cdn_zero_rtt.sort_values('success_rate', ascending=False)
            
            # Create bar chart
            bars = plt.barh(cdn_zero_rtt['cdn'], cdn_zero_rtt['success_rate'], color=color_pallete[0])
            
            # Add sample counts
            for i, bar in enumerate(bars):
                plt.text(5, bar.get_y() + bar.get_height()/2, 
                        f"n={int(cdn_zero_rtt.iloc[i]['count'])}", 
                        va='center', fontsize=9)
            
            plt.title('0-RTT Success Rate by CDN Provider', fontsize=12)
            plt.xlabel('Success Rate (%)', fontsize=10)
            plt.ylabel('CDN Provider', fontsize=10)
            plt.xlim(0, 105)
            
            plt.grid(axis='x', linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            cdn_path = os.path.join(base_directory, f"{output_prefix}_cdn.pdf")
            plt.savefig(cdn_path, format='pdf', bbox_inches='tight')
            print(f"Saved 0-RTT CDN analysis to {cdn_path}")
    
    # 6. COMBINED PERFORMANCE COMPARISON
    # ----------------------------------
    # Compare H3 0-RTT vs H3 normal vs H2 resumed vs H2 new
    plt.figure(figsize=(figwidth*1.4, figwidth / golden_ratio))
    
    # Create a connection type field
    connection_df = df.copy()
    
    # Create a connection type field
    conditions = [
        (connection_df['protocol'] == 'h3') & (connection_df['zero_rtt_used'] == True),
        (connection_df['protocol'] == 'h3') & (connection_df['zero_rtt_used'] == False),
        (connection_df['protocol'] == 'h2') & (connection_df.get('tls_resumed', False) == True),
        (connection_df['protocol'] == 'h2') & (connection_df.get('tls_resumed', False) == False)
    ]
    
    connection_types = [
        'HTTP/3 with 0-RTT',
        'HTTP/3 without 0-RTT',
        'HTTP/2 TLS resumed',
        'HTTP/2 new connection'
    ]
    
    connection_df['connection_type'] = np.select(conditions, connection_types, default='Other')
    
    # Group by connection type and calculate stats
    conn_stats = connection_df.groupby('connection_type')['load_time_ms'].agg(
        ['mean', 'median', 'std', 'count']
    ).reset_index()
    
    # Only include types with enough samples
    conn_stats = conn_stats[conn_stats['count'] >= 5]
    
    if len(conn_stats) > 1:  # Only create plot if we have at least two types
        # Sort by mean load time
        conn_stats = conn_stats.sort_values('mean')
        
        # Create color mapping
        color_map = {
            'HTTP/3 with 0-RTT': color_pallete[0],
            'HTTP/3 without 0-RTT': color_pallete[1],
            'HTTP/2 TLS resumed': color_pallete[2],
            'HTTP/2 new connection': color_pallete[3]
        }
        
        # Get colors in the right order
        colors = [color_map.get(conn_type, 'gray') for conn_type in conn_stats['connection_type']]
        
        # Create bar chart
        bars = plt.bar(conn_stats['connection_type'], conn_stats['mean'], color=colors)
        
        # Add error bars
        plt.errorbar(conn_stats['connection_type'], conn_stats['mean'], 
                    yerr=conn_stats['std'], fmt='none', capsize=5, color='black', alpha=0.5)
        
        # Add sample counts
        for i, bar in enumerate(bars):
            height = bar.get_height()
            conn_type = conn_stats.iloc[i]['connection_type']
            count = int(conn_stats.iloc[i]['count'])
            plt.text(bar.get_x() + bar.get_width()/2, height + 5,
                    f'n={count}', ha='center', fontsize=9)
        
        plt.title('Load Time by Connection Type', fontsize=12)
        plt.ylabel('Mean Load Time (ms)', fontsize=10)
        plt.xlabel('Connection Type', fontsize=10)
        plt.xticks(rotation=15, ha='right')
        
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        comparison_path = os.path.join(base_directory, f"{output_prefix}_comparison.pdf")
        plt.savefig(comparison_path, format='pdf', bbox_inches='tight')
        print(f"Saved connection type performance comparison to {comparison_path}")
    
    # Return combined statistics
    stats_summary = {
        "total_h3_requests": int(total_h3_requests),
        "zero_rtt_count": int(zero_rtt_count),
        "zero_rtt_percentage": float(zero_rtt_percentage),
        "connection_time": {
            "with_0rtt": float(h3_data[h3_data['zero_rtt_used']]['connection_time_ms'].mean()),
            "without_0rtt": float(h3_data[~h3_data['zero_rtt_used']]['connection_time_ms'].mean())
        },
        "load_time": {
            "with_0rtt": float(h3_data[h3_data['zero_rtt_used']]['load_time_ms'].mean()),
            "without_0rtt": float(h3_data[~h3_data['zero_rtt_used']]['load_time_ms'].mean())
        }
    }
    
    print("\nSummary Statistics:")
    print(f"  Average connection time with 0-RTT: {stats_summary['connection_time']['with_0rtt']:.2f} ms")
    print(f"  Average connection time without 0-RTT: {stats_summary['connection_time']['without_0rtt']:.2f} ms")
    print(f"  Connection time improvement: {(1 - stats_summary['connection_time']['with_0rtt'] / stats_summary['connection_time']['without_0rtt']) * 100:.1f}%")
    print(f"  Average load time with 0-RTT: {stats_summary['load_time']['with_0rtt']:.2f} ms")
    print(f"  Average load time without 0-RTT: {stats_summary['load_time']['without_0rtt']:.2f} ms")
    print(f"  Load time improvement: {(1 - stats_summary['load_time']['with_0rtt'] / stats_summary['load_time']['without_0rtt']) * 100:.1f}%")
    
    # Save summary statistics as JSON
    stats_path = os.path.join(base_directory, f"{output_prefix}_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats_summary, f, indent=2)
    
    print(f"\n0-RTT comprehensive analysis complete! All visualizations saved to {base_directory}")
    return stats_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze HTTP/2 vs HTTP/3 performance results")
    parser.add_argument("file", nargs="?", help="Input CSV results file")
    parser.add_argument("--network-analysis", "-n", action="store_true", 
                      help="Perform analysis across network conditions")
    parser.add_argument("--cache-analysis", "-c", action="store_true",
                      help="Generate detailed cache validation visualizations")
    parser.add_argument("--validation-across-networks", "-v", action="store_true",
                      help="Compare validation strategies across network conditions")
    parser.add_argument("--validation-distribution", "-vd", action="store_true",
                      help="Compare validation header distribution across network conditions")
    parser.add_argument("--validation-hit-rates", "-vhr", action="store_true",
                      help="Compare validation hit rates across network conditions")
    parser.add_argument("--cold-cache-analysis", "-cc", action="store_true",
                      help="Analyze cold cache performance by asset type")
    parser.add_argument("--warm-cache-analysis", "-wc", action="store_true",
                      help="Analyze warm cache performance by asset type")
    parser.add_argument("--cache-comparison", "-cmp", action="store_true",
                      help="Compare both cold and warm cache performance by asset type")
    parser.add_argument("--cache-directives", "-cd", action="store_true",
                      help="Analyze cache directives distribution across network conditions")
    parser.add_argument("--zero-rtt", "-zr", action="store_true",
                      help="Perform comprehensive analysis of 0-RTT performance")
    
    
    args = parser.parse_args()
    
    if args.network_analysis:
        analyze_network_conditions()
    elif args.validation_across_networks:
        compare_validation_across_networks()
    elif args.validation_distribution:
        analyze_validation_distribution_across_networks()
    elif args.validation_hit_rates:
        analyze_validation_hit_rates_across_networks()
    elif args.cold_cache_analysis:
        analyze_cold_cache_by_asset_type()
    elif args.warm_cache_analysis:
        analyze_warm_cache_by_asset_type()
    elif args.cache_comparison:
        analyze_cache_by_asset_type(cache_state="both")
    elif args.cache_directives:
        analyze_cache_directives_across_networks()
    elif args.zero_rtt:
        analyze_zero_rtt_comprehensive()
    elif args.file:
        analyze_results(args.file)
        analyze_cdn_requests(args.file)
        analyze_cache_validation(args.file)
        analyze_validation_and_0rtt(args.file)
    
        
        # Also include the cache analysis when analyzing a specific file
        analyze_cache_by_asset_type(args.file, cache_state="both")
        
        # Run detailed cache validation visualization if requested
        if args.cache_analysis:
            visualize_cache_validation(args.file)
    else:
        print("Please provide a CSV file to analyze or use one of the analysis options")