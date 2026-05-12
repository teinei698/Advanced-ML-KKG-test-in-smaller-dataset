# Knowledge Graph Link Prediction via Topological Overlap

This repository contains a complete pipeline for reproducing the core efficiency and predictive claims of the $k$KG algorithm. By treating Knowledge Graph link prediction as a Set Theory problem rather than a traditional Graph Neural Network (GNN) matrix operation, we calculate the topological overlap between entities and relations using probabilistic data structures.

## 🧠 Methodology Highlights
* **Undirected Graph Architecture:** We transformed standard $(h, r, t)$ triples into undirected paths ($h$ --- $r$ --- $t$) using `networkx`. By treating relations as independent nodes, they accrue their own structural neighborhoods, allowing for richer multi-hop overlap calculations.
* **Probabilistic Efficiency:** Instead of loading massive sets into memory for intersection math, we utilized the `datasketch` library. We used **MinHash** (128 permutations) to estimate Jaccard Similarity and **HyperLogLog** to estimate the Union size, effectively bypassing the severe memory bottlenecks of massive graph traversal.
* **Lightweight Classification:** Because the extracted 2-hop and 3-hop topological features are highly separable, we achieved strong predictive performance (AUC 0.98) using only lightweight Scikit-Learn models (Logistic Regression and a 2-layer MLP).

## 📊 Dataset
We utilized the standard **FB15k-237** benchmark, natively loaded via `pykeen`. 
* **Note on Computational Constraints:** Traversing 3-hop neighborhoods on a highly dense, undirected graph results in exponential path branching. To accommodate standard hardware/RAM limits while proving the algorithm's validity, this repository isolates a localized subset (5,000 positive triples, balanced with 5,000 strictly filtered negative samples).

## 🛠️ Requirements
To run this pipeline, install the following Python libraries:
```bash
pip install pandas scikit-learn matplotlib networkx datasketch pykeen