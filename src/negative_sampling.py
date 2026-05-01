# The target of negative sampling is to shuffle the triple to create fake triples,
# For example label=1, we create label=0 for the fake triples and for future test

import os
import random
import pandas as pd



def make_negative_triples(positive_dataframe, number_of_negative_samples):
    # We collect all entities and relations from real triples
    all_entities = list(set(positive_dataframe["head"]).union(set(positive_dataframe["tail"])))

    # Then we save real triples, which is in case fake triples happened to be "real"
    existing_triples = set()
    for _, row in positive_dataframe.iterrows():
        existing_triples.add((row["head"], row["relation"], row["tail"]))

    negative_rows = []
    tries = 0

    # Start to create fake triples
    while len(negative_rows) < number_of_negative_samples:
        random_row = positive_dataframe.sample(1).iloc[0]

        head_entity = random_row["head"]
        relation_name = random_row["relation"]
        tail_entity = random_row["tail"]

        # randomly corrupt head or tail
        if random.random() < 0.5:
            new_head = random.choice(all_entities)
            new_tail = tail_entity
        else:
            new_head = head_entity
            new_tail = random.choice(all_entities)

        new_triple = (new_head, relation_name, new_tail)

        # do not accidentally create a real triple
        if new_triple not in existing_triples:
            negative_rows.append({
                "head": new_head,
                "relation": relation_name,
                "tail": new_tail,
                "label": 0
            })

        tries += 1


    return pd.DataFrame(negative_rows)


# Define path and initialization
def main():
    data_folder = "data"

    train_path = os.path.join(data_folder, "train_triples.csv")
    output_path = os.path.join(data_folder, "train_with_negatives.csv")

    print("Reading positive triples...")
    positive_dataframe = pd.read_csv(train_path)
    positive_dataframe["label"] = 1

    number_of_negative_samples = len(positive_dataframe)

    print("Making negative triples...")
    negative_dataframe = make_negative_triples(
        positive_dataframe,
        number_of_negative_samples
    )

    final_dataframe = pd.concat(
        [positive_dataframe, negative_dataframe],
        ignore_index=True
    )

    final_dataframe = final_dataframe.sample(frac=1, random_state=42).reset_index(drop=True)

    final_dataframe.to_csv(output_path, index=False)

    print("Positive samples:", len(positive_dataframe))
    print("Negative samples:", len(negative_dataframe))
    print("SUCCESS, free to move to step4, Saved:", output_path)


if __name__ == "__main__":
    main()