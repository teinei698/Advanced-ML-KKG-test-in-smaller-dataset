import os
import pickle
import pandas as pd
import networkx as nx
from datasketch import MinHash, HyperLogLog

# 1. We keep this function to find the neighbors
def get_k_hop_neighbors(graph, start_node, k):
    if start_node not in graph:
        return set()

    path_lengths = nx.single_source_shortest_path_length(graph, start_node, cutoff=k)
    neighbors = set(path_lengths.keys())

    if start_node in neighbors:
        neighbors.remove(start_node)

    return neighbors

# 2. NEW: The 128 Lotteries & Coin Flips 
def create_fingerprints(neighbor_set, num_permutations=128):
    """
    This replaces exact sets with tiny fingerprints.
    m = MinHash (The 128 winning lottery tickets)
    h = HyperLogLog (The maximum streak of zeros)
    """
    m = MinHash(num_perm=num_permutations)
    h = HyperLogLog()
    
    for neighbor in neighbor_set:
        # Hash functions require data to be in bytes
        encoded_neighbor = str(neighbor).encode('utf8')
        m.update(encoded_neighbor)
        h.update(encoded_neighbor)
        
    return m, h

# 3. NEW: Estimating 2-Way Intersections (Head & Tail)
def estimate_2way_intersection(m1, m2, h1, h2):
    # Step A: Estimate percentage overlap (Jaccard Similarity)
    jaccard_sim = m1.jaccard(m2)
    
    # Step B: Estimate total unique crowd size (Union)
    # Merging HLLs mathematically takes the MAX streak of zeros!
    h_combined = HyperLogLog()
    h_combined.merge(h1)
    h_combined.merge(h2)
    estimated_union = h_combined.count()
    
    # Step C: Math! (Percentage * Total Size = Intersection Size)
    return jaccard_sim * estimated_union

# 4. NEW: Estimating 3-Way Intersections (Head, Relation, Tail)
def estimate_3way_intersection(m_head, m_rel, m_tail, h_head, h_rel, h_tail):
    # Step A: Find the 3-way overlap by counting how many times 
    # all 3 "winning tickets" are the exact same number.
    matches = 0
    for h_ticket, r_ticket, t_ticket in zip(m_head.hashvalues, m_rel.hashvalues, m_tail.hashvalues):
        if h_ticket == r_ticket == t_ticket:
            matches += 1
            
    # 128 lotteries total
    jaccard_sim_3way = matches / 128.0 
    
    # Step B: Find the 3-way Union
    h_combined = HyperLogLog()
    h_combined.merge(h_head)
    h_combined.merge(h_rel)
    h_combined.merge(h_tail)
    estimated_union_3way = h_combined.count()
    
    return jaccard_sim_3way * estimated_union_3way


# 5. Your Main Feature Extraction (Updated)
def compute_features_for_one_triple(graph, head_entity, relation_name, tail_entity):
    relation_node = "REL_" + relation_name

    # Get neighbors
    head_1hop = get_k_hop_neighbors(graph, head_entity, 1)
    relation_1hop = get_k_hop_neighbors(graph, relation_node, 1)
    tail_1hop = get_k_hop_neighbors(graph, tail_entity, 1)

    head_2hop = get_k_hop_neighbors(graph, head_entity, 2)
    relation_2hop = get_k_hop_neighbors(graph, relation_node, 2)
    tail_2hop = get_k_hop_neighbors(graph, tail_entity, 2)

    head_3hop = get_k_hop_neighbors(graph, head_entity, 3)
    relation_3hop = get_k_hop_neighbors(graph, relation_node, 3)
    tail_3hop = get_k_hop_neighbors(graph, tail_entity, 3)

    # --- THE MAGIC HAPPENS HERE ---
    # Create the fingerprints (no more loading giant sets into memory for comparisons)
    m_h1, h_h1 = create_fingerprints(head_1hop)
    m_r1, h_r1 = create_fingerprints(relation_1hop)
    m_t1, h_t1 = create_fingerprints(tail_1hop)
    
    m_h2, h_h2 = create_fingerprints(head_2hop)
    m_r2, h_r2 = create_fingerprints(relation_2hop)
    m_t2, h_t2 = create_fingerprints(tail_2hop)
    
    m_h3, h_h3 = create_fingerprints(head_3hop)
    m_r3, h_r3 = create_fingerprints(relation_3hop)
    m_t3, h_t3 = create_fingerprints(tail_3hop)

    # Estimate features using probabilities
    one_hop_head_tail = estimate_2way_intersection(m_h1, m_t1, h_h1, h_t1)
    two_hop_head_tail = estimate_2way_intersection(m_h2, m_t2, h_h2, h_t2)
    three_hop_head_tail = estimate_2way_intersection(m_h3, m_t3, h_h3, h_t3)

    one_hop_triple = estimate_3way_intersection(m_h1, m_r1, m_t1, h_h1, h_r1, h_t1)
    two_hop_triple = estimate_3way_intersection(m_h2, m_r2, m_t2, h_h2, h_r2, h_t2)
    three_hop_triple = estimate_3way_intersection(m_h3, m_r3, m_t3, h_h3, h_r3, h_t3)

    # We use HyperLogLog to estimate total relation neighbors instantly
    total_relation_neighbors = h_r1.count()

    return {
        "one_hop_head_tail": one_hop_head_tail,
        "one_hop_triple_intersection": one_hop_triple,
        "two_hop_head_tail": two_hop_head_tail,
        "two_hop_triple_intersection": two_hop_triple,
        "three_hop_head_tail": three_hop_head_tail,
        "three_hop_triple_intersection": three_hop_triple,
        "relation_degree_1hop": total_relation_neighbors
    }

def main():
    data_folder = "data"

    triples_path = os.path.join(data_folder, "train_with_negatives.csv")
    graph_path = os.path.join(data_folder, "train_graph.pkl")
    output_path = os.path.join(data_folder, "features.csv")

    print("Loading triples...")
    triples_dataframe = pd.read_csv(triples_path)

    print("Loading graph...")
    with open(graph_path, "rb") as file:
        graph = pickle.load(file)

    print("Computing MinHash & HyperLogLog features...")
    feature_rows = []

    for index, row in triples_dataframe.iterrows():
        features = compute_features_for_one_triple(
            graph,
            row["head"],
            row["relation"],
            row["tail"]
        )

        features["label"] = row["label"]
        features["head"] = row["head"]
        features["relation"] = row["relation"]
        features["tail"] = row["tail"]

        feature_rows.append(features)

        if index % 1000 == 0: 
            print("Finished rows:", index)

    final_dataframe = pd.DataFrame(feature_rows)
    final_dataframe.to_csv(output_path, index=False)

    print("Saved features:", output_path)
    print("Success, Final shape:", final_dataframe.shape)

if __name__ == "__main__":
    main()