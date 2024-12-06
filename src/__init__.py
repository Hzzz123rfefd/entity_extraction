from src.model import ModelPretrainForInformationExtractionBaseBert
from src.dataset import DatasetForInformationExtract


datasets = {
   "information_extraction": DatasetForInformationExtract
}

models = {
    "information_extraction_base_bert": ModelPretrainForInformationExtractionBaseBert,
}