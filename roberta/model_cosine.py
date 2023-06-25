#coding = utf-8
from pyexpat import features
import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertTokenizer,
    BertModel,
    BertPreTrainedModel
)
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

from transformers.modeling_outputs import MultipleChoiceModelOutput, TokenClassifierOutput
from config import ModelArguments
from preprocess_trainBert_cosine import unpack_instance
import os
import json
import numpy as np


def flat(t: torch.Tensor):
    return t.view(-1, t.shape[-1]) if t is not None else None


def avg_pooling(embs, input_ids, start_tok: int, end_tok: int):
    start = torch.nonzero(input_ids == start_tok, as_tuple=True)[1]
    end = torch.nonzero(input_ids == end_tok, as_tuple=True)[1]
    result = []
    for i, (s, e, item) in enumerate(zip(start + 1, end, embs)):
        result.append(torch.mean(item[s: e], dim=0))
    return torch.vstack(result)




def attention_visualization(attention, input_ids):
    f = open('attention.txt', 'a')
    print("input_ids:", input_ids, file = f)    
    print("len:", len(attention), file = f)
    print("shape:", attention[0].shape, file = f)
    #attention: 12层[i]*batch_size*12[j]*sequence_lengths*sequnce_lengths
    #[12, 10, 12, 7, 7]
    m = len(attention[0])
    n = len(attention)
    k = len(attention[0][0][0])
    p = np.empty((m, n, k, k))
    print("shape_p:", p.shape, file = f)
    for i in range(0, len(attention)):
        for j in range(0, len(attention[i])):
            result = attention[i][j][0]
            for k in range(1, 12):
                result = result + attention[i][j][k]
                #print(i, j, k, attention[i][j][k], file = f)
            result = result / 12
            p[j][i] = result.cpu().numpy()
    # for k in range(0, 12):
    #     print(k, attention[0][0][k], file = f)
    
    # print(p[0][0], file = f)
    
            
    name = 'save_attention_2'
    np.save(name, p)
    f.close()

class MultipleChoices_network(BertPreTrainedModel):
    # the init model is from huggface network api call
    _keys_to_ignore_on_load_missing = [r'position_ids']
    
    def __init__(self, model_args, config):
    # def __init__(self, plm, config):
        super().__init__(config)

        #lc
        if model_args.choose_model == 'roberta':
            self.model = RobertaModel.from_pretrained(
                model_args.model_name_or_path,
                config = config,
                cache_dir = model_args.cache_dir,
                revision  = model_args.model_revision
            )
        elif model_args.choose_model == 'distilbert':
            self.model = DistilBertModel.from_pretrained(
                model_args.model_name_or_path,
                config = config,
                cache_dir = model_args.cache_dir,
                revision  = model_args.model_revision
            )
        else:
            self.model = BertModel.from_pretrained(
                model_args.model_name_or_path,
                config = config,
                cache_dir = model_args.cache_dir,
                revision  = model_args.model_revision
            )

        hid = 700
        input = 768*4

        # 改roberta
        #BertPreTrainedModel->nn.Module
        self.choose_model = model_args.choose_model
        self.layer1 = nn.Linear(input,hid)   # 782 = 768 + 14,降低维度
        self.layer2 = nn.Linear(hid,hid)
    
    def forward(
            self,
            input_ids=None, #[400,2,*] pair靠在一起
            token_type_ids=None,
            attention_mask=None,
            other_features = None, #增加
            labels=None, #[200]
            nl_feature_tokens = None, #增加

    ):

        #lc
        
        if self.choose_model == 'distilbert':
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,  # 这里如果是 True, 那么返回的是 dict 而非一个 tuple
                output_hidden_states = True, # optional for hidden_states
                output_attentions=True,
            )
            hidden_states = outputs.hidden_states # get output_hidden_states
            hidden_states_size = 4 # count of the last states 
            hiddes_states_ind = list(range(-hidden_states_size, 0, 1))
            selected_hiddes_states = torch.cat(tuple([hidden_states[i] for i in hiddes_states_ind]), dim = 2)
            selected_hiddes_states = selected_hiddes_states[:,0,:]
            #print("hidden_states:", selected_hiddes_states.shape)
            selected_hiddes_states = selected_hiddes_states.view(-1,2, selected_hiddes_states.shape[1])
            selected_hiddes_states_first = selected_hiddes_states[:,0,:]
            selected_hiddes_states_second = selected_hiddes_states[:,1,:]
            #print("selected_hiddes_states_first:", selected_hiddes_states_first.shape, selected_hiddes_states_first)
            #print("selected_hiddes_states_second:", selected_hiddes_states_second.shape, selected_hiddes_states_second)
            features_first = selected_hiddes_states_first
            features_second = selected_hiddes_states_second
        else:
            outputs = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                return_dict=True,  # 这里如果是 True, 那么返回的是 dict 而非一个 tuple
                output_hidden_states = True, # optional for hidden_states
                output_attentions=True,
            )
            #attention_visualization(outputs.attentions, input_ids)
        
            pooler_outs = outputs.pooler_output #[400,768] pair紧靠在一起
            pooler_shape = pooler_outs.shape
            #print("pooler_shape:", pooler_shape)
            if pooler_shape[0]==43:
                print("here")
            pooler_outs = pooler_outs.view(-1,2,pooler_shape[1])
            pooler_first = pooler_outs[:,0,:]
            pooler_second = pooler_outs[:,1,:]

            #print("pooler_first:", pooler_first.shape, pooler_first)
            #print("pooler_second:", pooler_second.shape, pooler_second)
        
            features_first = pooler_first
            features_second = pooler_second
   

        x1 = torch.relu(self.layer1(features_first.float()))
        x2 = torch.relu(self.layer1(features_second.float()))


        x1 = torch.tanh(self.layer2(x1))
        x2 = torch.tanh(self.layer2(x2))
        


        logits = torch.nn.functional.cosine_similarity(x1,x2,-1)


        
        

        loss = torch.mean((logits - labels) ** 2) if labels is not None else None
        # 直接用官方提供的返回类进行包装
        # return MultipleChoiceModelOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        # 需要用单分类输出，否则多分类输出报错
        return TokenClassifierOutput(
            loss = loss,
            logits = logits,
            # attentions = features,
        )
        # return logits, loss, features
    






        
    

# # # for huggface bert

# def init_model_network(model_args: ModelArguments):
#     """
#     初始化模型
#     :param model_args: 模型参数
#     :return:
#     """
#     config = BertConfig.from_pretrained(
#         model_args.config_name if model_args.config_name else model_args.model_name_or_path,
#         cache_dir=model_args.cache_dir,
#         revision=model_args.model_revision
#     )

#     model = MultipleChoices(model_args, config)

#     # tokenizer 也可以自己定义，写在这个文件里面就好了
#     tokenizer = BertTokenizer.from_pretrained(
#         model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
#         cache_dir=model_args.cache_dir,
#         use_fast=model_args.use_fast_tokenizer,
#         revision=model_args.model_revision
#     )

#     return model, tokenizer

# # for download plm

def load_plm(cache_dir,logger):
    logger.info('start loading model and tokenizer ...')
    print('start loading model and tokenizer...')
    model: BertModel = torch.load(os.path.join(cache_dir, 'model.pth'))
    tokenizer: BertTokenizer = torch.load(os.path.join(cache_dir, 'tokenizer.pth'))

    logger.info('done loading model and tokenizer ...')
    print('done loading model and tokenizer ... ')
    return model, tokenizer

def load_config(cache_dir, logger):
    logger.info("start loading config")
    print("start loading config")
    with open(os.path.join(cache_dir,'config.json')) as f:
        config_json= json.load(f)
    logger.info('don loading config')
    print('done loading config')
    return config_json


def init_model(logger,model_args:ModelArguments):
    print('choose_model1:', model_args.choose_model)
    if model_args.choose_model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.load_plm_path,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision)
        config_json = load_config(model_args.load_cfg_path, logger)
        config = RobertaConfig(
                attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
                hidden_act=config_json["hidden_act"],
                hidden_dropout_prob=config_json["hidden_dropout_prob"],
                hidden_size=config_json["hidden_size"],
                initializer_range=config_json["initializer_range"],
                intermediate_size=config_json["intermediate_size"],
                max_position_embeddings=config_json["max_position_embeddings"],
                num_attention_heads=config_json["num_attention_heads"],
                num_hidden_layers=config_json["num_hidden_layers"],
                type_vocab_size=config_json["type_vocab_size"],
                vocab_size=config_json["vocab_size"])
        model = MultipleChoices_network(model_args,config)
        return model, tokenizer
    elif model_args.choose_model == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.load_plm_path,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision)
        #config_json = load_config(model_args.load_cfg_path, logger)
        config = DistilBertConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision
        )
        model = MultipleChoices_network(model_args,config)
        return model, tokenizer
    else:
        if model_args.load_plm_path == None:
            if model_args.load_cfg_path == None:
                config = BertConfig.from_pretrained(
                    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision
                )
                model = MultipleChoices_network(model_args,config)
                tokenizer = BertTokenizer.from_pretrained(
                    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                    cache_dir=model_args.cache_dir,
                    use_fast=model_args.use_fast_tokenizer,
                    revision=model_args.model_revision
                )
                return model, tokenizer
            else:
                config_json = load_config(model_args.load_cfg_path, logger)
                config = BertConfig(
                    attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
                    hidden_act=config_json["hidden_act"],
                    hidden_dropout_prob=config_json["hidden_dropout_prob"],
                    hidden_size=config_json["hidden_size"],
                    initializer_range=config_json["initializer_range"],
                    intermediate_size=config_json["intermediate_size"],
                    max_position_embeddings=config_json["max_position_embeddings"],
                    num_attention_heads=config_json["num_attention_heads"],
                    num_hidden_layers=config_json["num_hidden_layers"],
                    type_vocab_size=config_json["type_vocab_size"],
                    vocab_size=config_json["vocab_size"])
                model = MultipleChoices_network(model_args,config)

                # tokenizer 也可以自己定义，写在这个文件里面就好了
                tokenizer = BertTokenizer.from_pretrained(
                    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                    cache_dir=model_args.cache_dir,
                    use_fast=model_args.use_fast_tokenizer,
                    revision=model_args.model_revision
                )
                return model, tokenizer
        else:
            plm, tokenizer = load_plm(model_args.load_plm_path,logger)
            config = BertConfig(
                attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
                hidden_act=config_json["hidden_act"],
                hidden_dropout_prob=config_json["hidden_dropout_prob"],
                hidden_size=config_json["hidden_size"],
                initializer_range=config_json["initializer_range"],
                intermediate_size=config_json["intermediate_size"],
                max_position_embeddings=config_json["max_position_embeddings"],
                num_attention_heads=config_json["num_attention_heads"],
                num_hidden_layers=config_json["num_hidden_layers"],
                type_vocab_size=config_json["type_vocab_size"],
                vocab_size=config_json["vocab_size"])
            model = MultipleChoices_network(model_args,config)
            # model = MultipleChoices_network(plm,config)
            return model, tokenizer
