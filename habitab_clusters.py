# Install necessary libraries:
# pip install networkx python-louvain matplotlib

import networkx as nx
import community.community_louvain as community
import matplotlib.pyplot as plt

def detect_habitat_clusters():
    """
    Simulates a habitat network using NetworkX and applies the Louvain 
    community detection algorithm to find densely connected habitat clusters.
    
    In this simulation:
    - Nodes are individual habitat patches (A, B, C...).
    - Edges are potential wildlife corridors.
    - Edge weights represent the 'connectivity score' (higher weight = better corridor).
    """
    print("--- Starting Habitat Cluster Detection ---")

    # 1. Simulate the Habitat Network Data
    # Nodes: Habitat Patches (A through K)
    # Edges: (Patch1, Patch2, Connectivity_Score)
    # Note: A higher score indicates a stronger, easier connection (e.g., a low-resistance corridor).
    habitat_edges = [
        ('A', 'B', 0.9), ('A', 'C', 0.8), ('B', 'C', 0.7),
        ('A', 'D', 0.1),  # Weak link between A/D
        
        ('D', 'E', 0.95), ('D', 'F', 0.85), ('E', 'F', 0.9),
        ('G', 'H', 0.98), ('G', 'I', 0.92), ('H', 'I', 0.88),
        
        ('C', 'E', 0.2),  # Weak link connecting Group 1 and Group 2
        ('F', 'G', 0.15), # Weak link connecting Group 2 and Group 3
        
        ('J', 'K', 0.99), # Isolated mini-cluster
        ('I', 'J', 0.1), # Very weak link to isolated cluster
    ]

    # 2. Create the Graph
    G = nx.Graph()
    # Add weighted edges
    G.add_weighted_edges_from(habitat_edges, weight='weight')
    
    print(f"Network created with {G.number_of_nodes()} habitat patches and {G.number_of_edges()} corridors.")

    # 3. Apply Louvain Community Detection
    # The Louvain method partitions the graph into communities such that the 
    # modularity within each community is maximized.
    partition = community.best_partition(G, weight='weight')

    # Calculate Modularity Score: A metric to assess the quality of the partition. 
    # Scores closer to 1 indicate a strong community structure.
    modularity = community.modularity(partition, G, weight='weight')

    print(f"\nLouvain Algorithm Results:")
    print(f"Identified {max(partition.values()) + 1} distinct habitat clusters.")
    print(f"Modularity Score: {modularity:.4f} (A high score suggests a strong clustering structure).")
    
    # Map node to its cluster ID (Community ID)
    cluster_mapping = {node: cluster_id for node, cluster_id in partition.items()}
    print("\n--- Habitat Cluster Mapping ---")
    for node, cluster_id in sorted(cluster_mapping.items()):
        print(f"Patch {node}: Cluster {cluster_id}")


    # 4. Visualization
    plt.figure(figsize=(10, 7))
    
    # Get the color map based on the partition (each cluster gets a unique color)
    cmap = plt.cm.get_cmap('viridis', max(partition.values()) + 1)
    
    # Draw the graph
    # Set the layout for consistent positioning
    pos = nx.spring_layout(G, seed=42) 
    
    # Draw nodes, colored by their assigned community (cluster)
    nx.draw_networkx_nodes(
        G, 
        pos, 
        partition.keys(), 
        node_size=1200, 
        cmap=cmap, 
        node_color=list(partition.values())
    )
    
    # Draw edges with thickness based on connectivity score (weight)
    all_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, 
        pos, 
        width=[w * 5 for w in all_weights], # Scale width for visibility
        alpha=0.6, 
        edge_color='gray'
    )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold')

    plt.title(f"Habitat Clusters Identified by Louvain Algorithm\nModularity: {modularity:.4f}", fontsize=14)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    detect_habitat_clusters()
