import argparse

from src import models
from utils import *

text = "South Korea made virtually certain of an Asian Cup quarter-final spot with a 4-2 win over Indonesia in a Group A match on Saturday . "

def main(args):
    config = load_config("cof/information_extraction_base_bert.yml")
    """ get net struction"""
    net = models[config["model_type"]](**config["model"])
    net.load_pretrained(config["logging"]["save_dir"])
    entities = net.infernece(text)
    for i in range(config["model"]["num_class"] - 1):
        if len(entities[i]):
            print("label = "+ str(i + 1) + ":", entities[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path",type=str,default = "config/entity_extraction_base_bert.yml")
    args = parser.parse_args()
    main(args)

