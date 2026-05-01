# The target of this file is to transform the triple to features so that algorithms can understand and use them

import os
import pickle
import pandas as pd
import networkx as nx

# We need to find all neighbors in k steps
def get_k_hop_neighbors(graph, start_node, k):
    if start_node not in graph:
        return set()

    # this gives all nodes in k steps
    path_lengths = nx.single_source_shortest_path_length(graph, start_node, cutoff=k)
    neighbors = set(path_lengths.keys())

    if start_node in neighbors:
        neighbors.remove(start_node)

    return neighbors

# This function compute features for one triple
def compute_features_for_one_triple(graph, head_entity, relation_name, tail_entity):
    relation_node = "REL_" + relation_name

    head_1hop = get_k_hop_neighbors(graph, head_entity, 1)
    relation_1hop = get_k_hop_neighbors(graph, relation_node, 1)
    tail_1hop = get_k_hop_neighbors(graph, tail_entity, 1)

    head_2hop = get_k_hop_neighbors(graph, head_entity, 2)
    relation_2hop = get_k_hop_neighbors(graph, relation_node, 2)
    tail_2hop = get_k_hop_neighbors(graph, tail_entity, 2)

    # 2 hop may be not enough so i add 3 hop as a parallel group
    head_3hop = get_k_hop_neighbors(graph, head_entity, 3)
    relation_3hop = get_k_hop_neighbors(graph, relation_node, 3)
    tail_3hop = get_k_hop_neighbors(graph, tail_entity, 3)

    one_hop_head_tail = len(head_1hop.intersection(tail_1hop))
    two_hop_head_tail = len(head_2hop.intersection(tail_2hop))
    three_hop_head_tail = len(head_3hop.intersection(tail_3hop))

    # Only head tail
    one_hop_triple_intersection = len(head_1hop.intersection(relation_1hop).intersection(tail_1hop))
    two_hop_triple_intersection = len(head_2hop.intersection(relation_2hop).intersection(tail_2hop))
    three_hop_triple_intersection = len(head_3hop.intersection(relation_3hop).intersection(tail_3hop))

    total_relation_neighbors = len(relation_1hop)

    return {
        "one_hop_head_tail": one_hop_head_tail,
        "one_hop_triple_intersection": one_hop_triple_intersection,
        "two_hop_head_tail": two_hop_head_tail,
        "two_hop_triple_intersection": two_hop_triple_intersection,
        "three_hop_head_tail": three_hop_head_tail,
        "three_hop_triple_intersection": three_hop_triple_intersection,
        "relation_degree_1hop": total_relation_neighbors
    }

# Define the path and load and test
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

    print("Computing features...")
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

        if index % 1000 == 0: #Track every 1000 rows
            print("Mark4, Finished rows:", index)

    final_dataframe = pd.DataFrame(feature_rows)
    final_dataframe.to_csv(output_path, index=False)

    print("Saved features:", output_path)
    print("Success, continue to step5, Final shape:", final_dataframe.shape)


if __name__ == "__main__":
    main()