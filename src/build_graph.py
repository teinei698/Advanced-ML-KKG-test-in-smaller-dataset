import os
import pickle
import pandas as pd
import networkx as nx

# This function is working on transforming words to graph
def build_graph_from_triples(triples_dataframe):
    graph = nx.Graph()

    # prepare if we want to check node types, but not core function
    entity_nodes = set()
    relation_nodes = set()

    for _, row in triples_dataframe.iterrows():
        head_entity = row["head"]
        relation_name = "REL_" + row["relation"]
        tail_entity = row["tail"]

        entity_nodes.add(head_entity)
        entity_nodes.add(tail_entity)
        relation_nodes.add(relation_name)

        # paper idea: relations are also treated like nodes
        graph.add_node(head_entity, node_type="entity")
        graph.add_node(tail_entity, node_type="entity")
        graph.add_node(relation_name, node_type="relation")

        # simple version: h -- r -- t
        graph.add_edge(head_entity, relation_name)
        graph.add_edge(relation_name, tail_entity)

    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())
    print("Entity nodes:", len(entity_nodes))
    print("Relation nodes:", len(relation_nodes))

    return graph


def main():
    data_folder = "data"
    output_folder = "data"

    train_path = os.path.join(data_folder, "train_triples.csv")
    output_path = os.path.join(output_folder, "train_graph.pkl")

    print("\Reading train triples...")
    train_dataframe = pd.read_csv(train_path)

    print("Building graph...")
    graph = build_graph_from_triples(train_dataframe)

    # save graph so other files can load it directly
    with open(output_path, "wb") as file:
        pickle.dump(graph, file)

    print("Finished, Saved graph to:", output_path)


if __name__ == "__main__":
    main()