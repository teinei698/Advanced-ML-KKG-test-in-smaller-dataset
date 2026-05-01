Here is the guild line for our report, preferrable we can use Latex format but here i temporary start with a markdown to make guideline.

Overall structure:

1. Introduction                 0.5 page
2. Background & Paper Summary   1 page
3. Method                       1.5 pages
4. Experimental Setup           1 page
5. Results                      2 pages
6. Discussion                   1 page
7. Conclusion                   0.5 page

As for graph:
The insertion of graph are optional so i give the priority and brief explanation of each graph.(ofc you are very welcomed to edit the code to make graph you need)
   
Highest — Feature Mean Comparison — Shows the average difference between valid and invalid triples across all features.

Highest — Model AUC Comparison — Compares model performance using different feature groups.

Highest — Two-hop Distribution — Visualizes how well two-hop intersection separates valid and invalid triples.

High — k-hop Trend — Illustrates how feature values change as hop distance increases.

High — Boxplot — Displays the distribution and spread of key features for both classes.

Medium — Accuracy and F1 — Provides additional evaluation metrics beyond AUC.

Medium — Scatter Plot — Shows the feature space distribution of valid and invalid triples.

Low — Heatmap — Highlights correlations between different features.


## 1. Introduction (0.5 page)
Why include this section：

To clearly define the problem, context

How to write

Start with:

What a Knowledge Graph (KG) is
What a triple (h, r, t) represents

Then:

State the problem: determining whether a triple is valid

Then:

Briefly introduce the paper idea:
using k-hop neighborhoods
using intersection-based features

Finally (bullet points):
what we implemented
what we analyzed (k-hop comparison, feature effectiveness)
what we achieved (high classification performance)


## 2.Background & Paper Summary (1 page)
Why include this section

To demonstrate understanding of the original paper and core concepts.

2.1 Knowledge Graph Basics (~0.3 page)

Write:

Entities and relations
Triples
Link prediction / triple classification

Can be short and clear.

2.2 Paper Idea (~0.7 page)

In logical steps:

Problem

KGs are incomplete

Key idea

valid triples have more structural overlap

Core concept

k-hop neighborhoods

Main feature

intersection of head, relation, tail neighborhoods

Insight

larger intersection → higher likelihood of being valid


## 3. Method (1.5 pages)
Why include this section

To explain your implementation clearly and reproducibly.

3.1 Graph Construction (~0.4 page)

Write:

How you transform triples into a graph
Entities and relations are both nodes
Edges: entity → relation → entity
3.2 Feature Design (~0.8 page)

Explain:

(1) k-hop neighborhoods
definition (1-hop, 2-hop, 3-hop)
(2) Head-tail features
intersection between head and tail
(3) Triple intersection features
intersection among head, relation, and tail
(4) The final features

List:

one_hop_head_tail
two_hop_head_tail
three_hop_head_tail
one_hop_triple_intersection
two_hop_triple_intersection
three_hop_triple_intersection
3.3 Models (~0.3 page)

Write:

Logistic Regression (baseline)
MLP (non-linear model)

Explain why:

simple but sufficient
aligned with paper (fully connected network)


## 4. Experimental Setup (1 page)
Why include this section

To make the experiment reproducible and credible.

4.1 Dataset (~0.3 page)

Write:

FB15K-237
sampled 5000 triples
generate negative samples
4.2 Feature Extraction (~0.3 page)

Write:

compute k-hop neighbors
compute intersection features
store as structured dataset
4.3 Training Setup (~0.4 page)

Write:

train/test split (75/25)
metrics:
Accuracy
F1
AUC
feature groups:
head-tail only
triple-intersection only
two-hop only
all features


## 5. Results (2 pages)
Why include this section

This is the core of your report — showing evidence.

5.1 Feature Analysis (~0.7 page)

Use:

Feature Mean Comparison
Boxplot

Explain:

valid > invalid
2-hop strongest
1-hop triple = 0
5.2 Distribution Analysis (~0.5 page)

Use:

Histogram (two-hop triple)

Explain:

invalid concentrated near 0
valid spread to higher values
clear separability
5.3 k-hop Analysis (~0.4 page)

Use:

k-hop trend

Explain:

1-hop insufficient
2-hop captures structure
3-hop adds context but noise
5.4 Model Performance (~0.4 page)

Use:

AUC plot
(optional Accuracy/F1)

Explain:

all_features best
MLP best
AUC ≈ 0.98

## 6. Discussion (1 page)
Why include this section

To show deeper thinking beyond results.

Three points:
1. Why 2-hop works best
captures local structure
balanced information vs noise
2. Why 3-hop is weaker than expected
introduces noisy global context
3. Limitations
small dataset
simplified graph
negative sampling 
4. Feature redundancy
correlation heatmap
some features highly correlated


## 7. Conclusion (0.5 page)