import os
import traceback #It didn't report feedback
import pandas as pd
from pykeen.datasets import FB15k237

# Download data and save to csv in the format we want, id-name-relation-tail
def save_triples_to_csv(triples, entity_id_to_label, relation_id_to_label, output_path, sample_size=None):
    print(f"[DEBUG] Processing {output_path}...")
    rows = []

    if sample_size is not None:
        triples = triples[:sample_size]

    for head_id, relation_id, tail_id in triples:
        head_name = entity_id_to_label[int(head_id)]
        relation_name = relation_id_to_label[int(relation_id)]
        tail_name = entity_id_to_label[int(tail_id)]

        rows.append({
            "head": head_name,
            "relation": relation_name,
            "tail": tail_name
        })

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_path, index=False)
    print(f"Download and save successfully: {output_path}, rows: {len(dataframe)}")


def main():
    try:
        print("Mark2 of debug, Starting...")

        data_folder = "data"
        os.makedirs(data_folder, exist_ok=True)
        print(f"Folder isready: {data_folder}")

        print("Mark3,Loading dataset...")
        dataset = FB15k237()
        print("[Loaded")

        # PyKEEN gives label -> id, but mapped_triples are id -> id -> id
        entity_id_to_label = {value: key for key, value in dataset.training.entity_to_id.items()}
        relation_id_to_label = {value: key for key, value in dataset.training.relation_to_id.items()}

        print(f"Entities: {len(entity_id_to_label)}")
        print(f"Relations: {len(relation_id_to_label)}")

        train_triples = dataset.training.mapped_triples.numpy()
        valid_triples = dataset.validation.mapped_triples.numpy()
        test_triples = dataset.testing.mapped_triples.numpy()

        print(f"Triples: train={len(train_triples)}, valid={len(valid_triples)}, test={len(test_triples)}")

        # Here i set the size of samples for testing, can increase if want bigger datset
        # But please run this file again and editing


        small_sample_size = 5000

        save_triples_to_csv(
            train_triples,
            entity_id_to_label,
            relation_id_to_label,
            os.path.join(data_folder, "train_triples.csv"),
            sample_size=small_sample_size
        )

        save_triples_to_csv(
            valid_triples,
            entity_id_to_label,
            relation_id_to_label,
            os.path.join(data_folder, "valid_triples.csv"),
            sample_size=1000
        )

        save_triples_to_csv(
            test_triples,
            entity_id_to_label,
            relation_id_to_label,
            os.path.join(data_folder, "test_triples.csv"),
            sample_size=1000
        )

        print("Load suceessfully! Continue with step2")
    
    # Track the failure 
    except Exception as error:
        print(f"Failed, check load file again:{type(error).__name__}: {error}")
        traceback.print_exc()


if __name__ == "__main__":
    main()