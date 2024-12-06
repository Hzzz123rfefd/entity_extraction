import math
import torch.nn as nn
import torch
from tqdm import tqdm
import os
from torch import optim
import torch
from transformers import AutoModel,AutoTokenizer
from torch.utils.data import DataLoader

from src.utils import *

class ModelBert(nn.Module):
    def __init__(
            self,
            model_name_or_path: str = 'bert-base-uncased',
            device: str = "cpu",
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.device = device if torch.cuda.is_available() else "cpu"
        if model_name_or_path == None:
            self.tokenizer = None
            self.model = None
        else:
            self.model = AutoModel.from_pretrained(self.model_name_or_path, local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=True)
            self.model = self.model.to(self.device)

    def forward(self,batch_data):
        output = self.model(batch_data['input_ids'].to(self.device),batch_data['attention_mask'].to(self.device))
        return output

    def load_pretrained(self,save_model_dir:str):
        if os.path.isdir(save_model_dir) and os.listdir(save_model_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(save_model_dir, local_files_only = True)
            self.model = AutoModel.from_pretrained(save_model_dir, local_files_only = True)
            self.model = self.model.to(self.device)

    def save_pretrained(self,save_model_dir:str):
        self.model.save_pretrained(save_model_dir)
        self.tokenizer.save_pretrained(save_model_dir)

class ModelPretrainForInformationExtractionBaseBert(nn.Module):
    def __init__(
        self, 
        pretrain_model_name_or_path = "bert-base-uncased",
        num_class = 2,
        device = "cpu"
    ):
        super().__init__()
        self.model_name_or_path = pretrain_model_name_or_path
        self.num_class = num_class
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model = ModelBert(
            model_name_or_path = self.model_name_or_path,
            device = self.device
        )
        self.tokenizer = self.model.tokenizer
        self.predict_head = nn.Linear(self.model.model.config.hidden_size, self.num_class).to(self.device)


    def trainning(
        self,
        train_dataloader:DataLoader = None,
        test_dataloader:DataLoader = None,
        optimizer_name:str = "Adam",
        weight_decay:float = 1e-4,
        clip_max_norm:float = 0.5,
        factor:float = 0.3,
        patience:int = 15,
        lr:float = 1e-4,
        total_epoch:int = 1000,
        save_checkpoint_step:str = 10,
        save_model_dir:str = "models"
    ):
        ## 1 trainning log path 
        first_trainning = True
        check_point_path = save_model_dir  + "/checkpoint.pth"
        log_path = save_model_dir + "/train.log"

        ## 2 get net pretrain parameters if need 
        """
            If there is  training history record, load pretrain parameters
        """
        if  os.path.isdir(save_model_dir) and os.path.exists(check_point_path) and os.path.exists(log_path):
            self.load_pretrained(save_model_dir)  
            first_trainning = False

        else:
            if not os.path.isdir(save_model_dir):
                os.makedirs(save_model_dir)
            with open(log_path, "w") as file:
                pass


        ##  3 get optimizer
        if optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(self.parameters(),lr,weight_decay = weight_decay)
        else:
            optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer, 
            mode = "min", 
            factor = factor, 
            patience = patience
        )

        ## init trainng log
        if first_trainning:
            best_loss = float("inf")
            last_epoch = 0
        else:
            checkpoint = torch.load(check_point_path, map_location=self.device)
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            best_loss = checkpoint["loss"]
            last_epoch = checkpoint["epoch"] + 1

        try:
            for epoch in range(last_epoch,total_epoch):
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                train_loss = self.train_one_epoch(epoch,train_dataloader, optimizer,clip_max_norm,log_path)
                test_loss = self.test_epoch(epoch,test_dataloader,log_path)
                loss = train_loss + test_loss
                lr_scheduler.step(loss)
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
                check_point_path = save_model_dir  + "/checkpoint.pth"
                torch.save(                
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "optimizer": None,
                        "lr_scheduler": None
                    },
                    check_point_path
                )

                if epoch % save_checkpoint_step == 0:
                    os.makedirs(save_model_dir + "/" + "chaeckpoint-"+str(epoch))
                    torch.save(
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict()
                        },
                        save_model_dir + "/" + "chaeckpoint-"+str(epoch)+"/checkpoint.pth"
                    )
                if is_best:
                    self.save_pretrained(save_model_dir)

        # interrupt trianning
        except KeyboardInterrupt:
                torch.save(                
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict()
                    },
                    check_point_path
                )

    def forward(self,input:dict):
        batch_data = {
            "input_ids":input["input_ids"].reshape(-1,input["input_ids"].shape[2]).to(self.device),
            "attention_mask":input["attention_mask"].reshape(-1,input["attention_mask"].shape[2]).to(self.device)
        }
        label = input["label"].reshape(-1).to(self.device)  # shape:[batch_size * seq_len]
        outputs = self.model(batch_data)
        sequence_output = outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_size]
        predict = self.predict_head(sequence_output)  # shape: [batch_size, seq_len, num_class]
        _, _, num_class = predict.shape
        predict = predict.reshape(-1,num_class)
        mask = batch_data["attention_mask"].reshape(-1)
        class_weights = torch.ones(self.num_class).to(self.device)
        class_weights[0] = 0.1
        output = {
            "predict":predict,
            "label":label,
            "mask":mask,
            "class_weights":class_weights
        }
        return output

    def compute_loss(self, input:dict):
        output = {}
        if "class_weights" not in input:
            input["class_weights"] = None
        if "mask" not in input:
            self.criterion = nn.CrossEntropyLoss(weight = input["class_weights"])
            output["total_loss"] = self.criterion(input["predict"],input["label"])
        else:
            self.criterion = nn.CrossEntropyLoss(weight = input["class_weights"], reduction = 'none')
            loss = self.criterion(input["predict"],input["label"])
            masked_loss = loss * input["mask"]
            output["total_loss"] = masked_loss.sum() / input["mask"].sum()
        return output

    def train_one_epoch(self, epoch,train_dataloader, optimizer, clip_max_norm, log_path = None):
        self.train()
        self.to(self.device)
        pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
        total_loss = AverageMeter()
        average_hit_rate = AverageMeter()
        for batch_id, inputs in enumerate(train_dataloader):
            """ grad zeroing """
            optimizer.zero_grad()

            """ forward """
            used_memory = 0 if self.device == "cpu" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
            output = self.forward(inputs)

            """ calculate loss """
            out_criterion = self.compute_loss(output)
            out_criterion["total_loss"].backward()
            total_loss.update(out_criterion["total_loss"].item())
            average_hit_rate.update(math.exp(-total_loss.avg))

            """ grad clip """
            if clip_max_norm > 0:
                clip_gradient(optimizer,clip_max_norm)

            """ modify parameters """
            optimizer.step()
            after_used_memory = 0 if self.device == "cpu" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
            postfix_str = "total_loss: {:.4f},average_hit_rate:{:.4f},use_memory: {:.1f}G".format(
                total_loss.avg, 
                average_hit_rate.avg,
                after_used_memory - used_memory
            )
            pbar.set_postfix_str(postfix_str)
            pbar.update()
        with open(log_path, "a") as file:
            file.write(postfix_str+"\n")
        return total_loss.avg

    def test_epoch(self,epoch, test_dataloader,trainning_log_path = None):
        total_loss = AverageMeter()
        average_hit_rate = AverageMeter()
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            for batch_id, inputs in enumerate(test_dataloader):
                """ forward """
                output = self.forward(inputs)

                """ calculate loss """
                out_criterion = self.compute_loss(output)
                total_loss.update(out_criterion["total_loss"])

            average_hit_rate.update(math.exp(-total_loss.avg))
            str = "Test Epoch: {:d}, total_loss: {:.4f},average_hit_rate:{:.4f}".format(
                epoch,
                total_loss.avg, 
                average_hit_rate.avg,
            )
        print(str)
        with open(trainning_log_path, "a") as file:
            file.write(str+"\n")
        return total_loss.avg



    def load_pretrained(self, save_model_dir):
        self.model.load_pretrained(save_model_dir)
        self.predict_head.load_state_dict(torch.load(save_model_dir + "/predict_head.pth"))

    def save_pretrained(self,  save_model_dir):
        self.model.save_pretrained(save_model_dir) 
        torch.save(self.predict_head.state_dict(), save_model_dir + "/predict_head.pth")

    def infernece(self, text:str):
        entities = []
        for i in range(self.num_class - 1):
            entities.append([])
        
        """ spilt word """
        input = self.tokenizer(
            text, 
            truncation = True, 
            padding = False,
            max_length = 512,
            return_tensors="pt",
            return_offsets_mapping = False
        )
        with torch.no_grad():
            outputs = self.model(input)
            sequence_output = outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_size]
            predict = self.predict_head(sequence_output)  # shape: [batch_size, seq_len, num_class]
        predict_class = torch.argmax(predict, dim = -1)
        b, seq = predict_class.shape
        index = 0
        while index != seq:
            if predict_class[0,index] != 0:
                entity_ids = []
                entity_label = predict_class[0, index]
                while predict_class[0,index] == entity_label and index != seq:
                    entity_ids.append(input["input_ids"][0,index])
                    index = index + 1
                t = self.tokenizer.decode(entity_ids,skip_special_tokens = True)
                entities[entity_label - 1].append(t)
                print("label:",entity_label.item(),"  enetity:",t)
            index = index + 1
        return entities
    