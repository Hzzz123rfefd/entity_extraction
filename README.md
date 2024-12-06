# entity_extraction
a code for entity extraction by PyTorch, use bert base model


## Installation
Install transformers and the packages required for development.
```bash
conda create -n entity python=3.10
conda activate entity
git clone https://github.com/Hzzz123rfefd/entity_extraction.git
cd entity_extraction
pip install -r requirements.txt
```

## Usage
### Dataset


### Trainning
An examplary training script with a Cross Entropy loss is provided in train.py.
You can adjust the model parameters in config/entity_extraction_base_bert.yml
```bash
python train.py --model_config_path config/entity_extraction_base_bert.yml
```

### Inference
Once you have trained your model, you can use the following script to perform entity recognition on the data
you can set your text in inference.py
```bash
python inference.py --model_config_path config/entity_extraction_base_bert.yml
```
You can see the following input and output:
```text
text = "South Korea made virtually certain of an Asian Cup quarter-final spot with a 4-2 win over Indonesia in a Group A match on Saturday . "

label = 2: ['south korea', 'indonesia']
label = 4: ['asian cup']
```
