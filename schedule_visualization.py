import os

import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore


def visualize_schedule_comparison(data_dir='data/schedule_demo2_data'):
    """
    Visualize and compare planned vs actual schedule execution
    """
    # Load the data
    planned_df = pd.read_csv(os.path.join(data_dir, 'schedule.csv'))
    actual_df = pd.read_csv(os.path.join(data_dir, 'actual_schedule.csv'))
    
    # Get unique robots
    robots = planned_df['robot_id'].unique()
    
    # Create subplots for each robot
    fig, axes = plt.subplots(len(robots), 1, figsize=(15, 4*len(robots)))
    if len(robots) == 1:
        axes = [axes]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, robot in enumerate(robots):
        ax = axes[i]
        
        # Filter data for current robot
        planned_robot = planned_df[planned_df['robot_id'] == robot].reset_index(drop=True)
        actual_robot = actual_df[actual_df['robot_id'] == robot].reset_index(drop=True)
        
        # Plot planned schedule
        ax.plot(planned_robot['ETA'], range(len(planned_robot)), 
                'o--', color=colors[i%len(colors)], linewidth=2, markersize=8,
                label=f'{robot} - Planned', alpha=0.7)
        
        # Plot actual schedule
        ax.plot(actual_robot['ETA'], range(len(actual_robot)), 
                's-', color=colors[i%len(colors)], linewidth=2, markersize=8,
                label=f'{robot} - Actual', alpha=0.9)
        
        # Add node labels for planned schedule
        for j, (_, row) in enumerate(planned_robot.iterrows()):
            ax.annotate(row['node_id'], (row['ETA'], j), 
                       xytext=(5, 0), textcoords='offset points', 
                       fontsize=8, alpha=0.7)
        
        # Add node labels for actual schedule
        for j, (_, row) in enumerate(actual_robot.iterrows()):
            ax.annotate(row['node_id'], (row['ETA'], j), 
                       xytext=(-15, 0), textcoords='offset points', 
                       fontsize=8, alpha=0.9, color='red')
        
        ax.set_xlabel('Time (ETA)')
        ax.set_ylabel('Waypoint Index')
        ax.set_title(f'Robot {robot} - Schedule Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Invert y-axis so first waypoint is at top
    
    plt.tight_layout()
    plt.savefig('Demo/schedule_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return planned_df, actual_df

def create_delay_analysis(planned_df, actual_df):
    """
    Create delay analysis visualization
    """
    # Merge dataframes to compare timing
    merged_data = []
    
    for robot in planned_df['robot_id'].unique():
        planned_robot = planned_df[planned_df['robot_id'] == robot].reset_index(drop=True)
        actual_robot = actual_df[actual_df['robot_id'] == robot].reset_index(drop=True)
        
        # Find common nodes between planned and actual
        for i, planned_row in planned_robot.iterrows():
            node_id = planned_row['node_id']
            planned_time = planned_row['ETA']
            
            # Find corresponding actual time
            actual_match = actual_robot[actual_robot['node_id'] == node_id]
            if not actual_match.empty:
                actual_time = actual_match.iloc[0]['ETA']
                delay = actual_time - planned_time
                merged_data.append({
                    'robot_id': robot,
                    'node_id': node_id,
                    'planned_time': planned_time,
                    'actual_time': actual_time,
                    'delay': delay
                })
    
    if not merged_data:
        print("No common nodes found between planned and actual schedules")
        return
    
    delay_df = pd.DataFrame(merged_data)
    
    # Create delay visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Delay over time
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    for i, robot in enumerate(delay_df['robot_id'].unique()):
        robot_data = delay_df[delay_df['robot_id'] == robot]
        ax1.plot(robot_data['planned_time'], robot_data['delay'], 
                'o-', color=colors[i%len(colors)], label=robot, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Planned Time')
    ax1.set_ylabel('Delay (Actual - Planned)')
    ax1.set_title('Execution Delay Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Delay distribution
    robot_delays = [delay_df[delay_df['robot_id'] == robot]['delay'] for robot in delay_df['robot_id'].unique()]
    robot_labels = delay_df['robot_id'].unique()
    
    ax2.hist(robot_delays, bins=20, alpha=0.7, label=robot_labels)
    ax2.set_xlabel('Delay (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Delay Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('Demo/delay_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== Schedule Performance Summary ===")
    print(f"Average delay: {delay_df['delay'].mean():.2f} seconds")
    print(f"Median delay: {delay_df['delay'].median():.2f} seconds")
    print(f"Max delay: {delay_df['delay'].max():.2f} seconds")
    print(f"Min delay: {delay_df['delay'].min():.2f} seconds")
    print(f"Standard deviation: {delay_df['delay'].std():.2f} seconds")
    
    print("\n=== Per-Robot Statistics ===")
    for robot in delay_df['robot_id'].unique():
        robot_delays = delay_df[delay_df['robot_id'] == robot]['delay']
        print(f"{robot}: Avg={robot_delays.mean():.2f}s, Max={robot_delays.max():.2f}s, Min={robot_delays.min():.2f}s")
    
    return delay_df

def create_gantt_chart(planned_df, actual_df):
    """
    Create a Gantt chart showing robot schedules
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Planned schedule Gantt chart
    robots = planned_df['robot_id'].unique()
    for i, robot in enumerate(robots):
        robot_data = planned_df[planned_df['robot_id'] == robot].reset_index(drop=True)
        
        for j in range(len(robot_data) - 1):
            start_time = robot_data.iloc[j]['ETA']
            end_time = robot_data.iloc[j + 1]['ETA']
            node = robot_data.iloc[j]['node_id']
            
            ax1.barh(i, end_time - start_time, left=start_time, 
                    color=colors[i % len(colors)], alpha=0.7, 
                    label=robot if j == 0 else "")
            
            # Add node label
            ax1.text(start_time + (end_time - start_time) / 2, i, node, 
                    ha='center', va='center', fontsize=8)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Robot')
    ax1.set_title('Planned Schedule - Gantt Chart')
    ax1.set_yticks(range(len(robots)))
    ax1.set_yticklabels(robots)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Actual schedule Gantt chart
    for i, robot in enumerate(robots):
        robot_data = actual_df[actual_df['robot_id'] == robot].reset_index(drop=True)
        
        for j in range(len(robot_data) - 1):
            start_time = robot_data.iloc[j]['ETA']
            end_time = robot_data.iloc[j + 1]['ETA']
            node = robot_data.iloc[j]['node_id']
            
            ax2.barh(i, end_time - start_time, left=start_time, 
                    color=colors[i % len(colors)], alpha=0.7, 
                    label=robot if j == 0 else "")
            
            # Add node label
            ax2.text(start_time + (end_time - start_time) / 2, i, node, 
                    ha='center', va='center', fontsize=8)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Robot')
    ax2.set_title('Actual Schedule - Gantt Chart')
    ax2.set_yticks(range(len(robots)))
    ax2.set_yticklabels(robots)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('Demo/gantt_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_completion_rates(planned_df, actual_df):
    """
    Analyze completion rates and missing nodes
    """
    print("\n=== Completion Analysis ===")
    
    for robot in planned_df['robot_id'].unique():
        planned_nodes = set(planned_df[planned_df['robot_id'] == robot]['node_id'])
        actual_nodes = set(actual_df[actual_df['robot_id'] == robot]['node_id'])
        
        completed = len(actual_nodes)
        total_planned = len(planned_nodes)
        completion_rate = (completed / total_planned) * 100
        
        missing_nodes = planned_nodes - actual_nodes
        extra_nodes = actual_nodes - planned_nodes
        
        print(f"\nRobot {robot}:")
        print(f"  Completion rate: {completion_rate:.1f}% ({completed}/{total_planned} nodes)")
        if missing_nodes:
            print(f"  Missing nodes: {sorted(missing_nodes)}")
        if extra_nodes:
            print(f"  Extra nodes: {sorted(extra_nodes)}")

def main():
    """
    Main function to run all visualizations
    """
    print("Loading schedule data...")
    planned_df, actual_df = visualize_schedule_comparison()
    
    print("\nCreating delay analysis...")
    delay_df = create_delay_analysis(planned_df, actual_df)
    
    print("\nCreating Gantt chart...")
    create_gantt_chart(planned_df, actual_df)
    
    print("\nAnalyzing completion rates...")
    analyze_completion_rates(planned_df, actual_df)
    
    print("\nAll visualizations saved to Demo/ directory")

if __name__ == "__main__":
    main()
