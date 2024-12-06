import argparse
import json
import sys
import os
sys.path.append(os.getcwd())

def process_conll_data(data_path):
    data = []
    entities_map = {
        'PER': 1,  # 人名
        'LOC': 2,  # 地名
        'ORG': 3,  # 组织名
        'MISC': 4  # 杂项
    }

    with open(data_path, 'r', encoding='utf-8') as f:
        current_sentence = ""
        current_entities = []
        current_index = 0
        for line in f:
            line = line.strip()
            if line:
                object = line.split()
                word = object[0]
                entity = object[3]
                current_sentence = current_sentence + word + " " 
                if entity != "O":
                    for entity_name in entities_map:
                        if entity.find(entity_name)  != -1:
                            current_entities.append(
                                {
                                    "start": current_index, 
                                    "end": current_index + len(word), 
                                    "label": entities_map[entity_name]
                                }
                            )
                            break
                current_index = current_index + len(word) + 1
            else:   
                data.append({
                    "text": current_sentence,
                    "entities":current_entities
                })
                current_sentence = ""
                current_entities = []
                current_index = 0
    return data

def main(args):
    train_data = process_conll_data(args.dataset_dir + "train.txt")
    test_data = process_conll_data(args.dataset_dir + "test.txt")
    valid_data = process_conll_data(args.dataset_dir + "valid.txt")
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        
    with open(args.output_dir + "train.jsonl", 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(args.output_dir + "test.jsonl", 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(args.output_dir + "valid.jsonl", 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",type=str,default = "dataset/conll2003/")
    args = parser.parse_args("--output_dir",type=str,default = "dataset/processed_data/")
    main(args)