from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import ListField, LabelField, ArrayField
from allennlp.nn import util
from allennlp.data.instance import Instance

import argparse
import numpy as np
import os
import re
import torch
import pickle as pkl
from tqdm import tqdm
import codecs
import json
import uuid

from extractor_model import TriggerExtractor, ArgumentExtractor
from dueereader import CustomSpanField, DataMeta, TriggerReader, RoleReader, TextReader
from allennlp.data.iterators import BucketIterator
from extractormetric import ExtractorMetric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_num = -1
if torch.cuda.is_available():
    device_num = torch.cuda.current_device()

print(device)

def trigger_extractor_deal(pre_dataset, iterator, trigger_model_path, dataset_meta):
    def get_instance(sentence_data, t_list):
        instances = []
        for trigger in t_list:
            trigger_span_start = trigger[0]
            trigger_span_end = trigger[1]
            et_id = trigger[2]

            sentence_field = sentence_data['sentence']
            sentence_id_field = sentence_data['sentence_id']

            wordpiece_tokenizer = sentence_field._token_indexers['tokens'].wordpiece_tokenizer
            tokens_len = len(sentence_field)
            type_ids = [0]
            for idx in range(tokens_len):
                word_pieces = wordpiece_tokenizer(sentence_field[idx].text)
                if idx >= trigger_span_start and idx <= trigger_span_end:
                    type_ids.extend([1]*len(word_pieces))
                else:
                    type_ids.extend([0]*len(word_pieces))
            type_ids.append(0)
            type_ids = np.array(type_ids)
            type_ids_field = ArrayField(type_ids)
            event_type_field = LabelField(label=et_id, skip_indexing=True)
            trigger_span_field = CustomSpanField(trigger_span_start, trigger_span_end, et_id, -1)
            role_field_list = [CustomSpanField(-1, -1, -1, -1)]
            roles_field = ListField(role_field_list)
            fields = {'sentence': sentence_field}
            fields['sentence_id'] = sentence_id_field
            fields['type_ids'] = type_ids_field
            fields['event_type'] = event_type_field
            fields['trigger'] = trigger_span_field
            fields['roles'] = roles_field
            instances.append(Instance(fields))
        return instances
    
    pretrained_bert = PretrainedBertEmbedder(
        pretrained_model='models/chinese_roberta_wwm_ext',
        # pretrained_model='hfl/chinese-roberta-wwm-ext',
        requires_grad=True,
        top_layer_only=True)

    trigger_extractor = TriggerExtractor(
        vocab=Vocabulary(),
        embedder=pretrained_bert,
        et_num=dataset_meta.get_et_num())

    model_state = torch.load(trigger_model_path, map_location=device)
    trigger_extractor.load_state_dict(model_state)
    trigger_extractor.to(device)
    trigger_extractor.eval()


    trigger_results = {}
    batch_idx = 0
    id_data = {} 
    # print(pre_dataset)
    for instance in pre_dataset:
        tmp_id = instance['sentence_id'].metadata
        if tmp_id in id_data:
            print(tmp_id)
        id_data[tmp_id] = instance

    for data in tqdm(iterator(pre_dataset, num_epochs=1)):
        # print(batch_idx)
        batch_idx += 1
        # print(device_num)
        # print(data)
        data = util.move_to_device(data, cuda_device=device_num) 
        # print(data)
        sentences = data['sentence']
        # print(type(sentences))
        # exit(0)
        sentence_id = data['sentence_id']
        # sentences.to(device)
        output = trigger_extractor(sentences, sentence_id)
        logits = output['logits']
        pred_span = trigger_extractor.metric.get_span(logits)
        for idx, sid in enumerate(sentence_id):
            trigger_results[sid] = pred_span[idx]
    
    instances = []
    for sid, trigger_spans in trigger_results.items():
        if len(trigger_spans) > 0:
            sentence_instances = get_instance(id_data[sid], trigger_spans)
            instances.extend(sentence_instances)
    return instances


def argument_extractor_deal(instances, iterator, argument_model_path, dataset_meta):
    pretrained_bert = PretrainedBertEmbedder(
        pretrained_model='models/chinese_roberta_wwm_ext',
        # pretrained_model='hfl/chinese-roberta-wwm-ext',
        requires_grad=True,
        top_layer_only=True)

    argument_extractor = ArgumentExtractor(
        vocab=Vocabulary(),
        embedder=pretrained_bert,
        role_num=dataset_meta.get_role_num(),
        event_roles=dataset_meta.event_roles,
        prob_threshold=0.5,
        af=0,
        ief=0)

    # model_state = torch.load(argument_model_path, map_location=util.device_mapping(-1))
    model_state = torch.load(argument_model_path, map_location=device)
    argument_extractor.load_state_dict(model_state)
    argument_extractor.to(device)
    argument_extractor.eval()
    
    batch_idx = 0
    pred_spans = {}
    for data in iterator(instances, num_epochs=1):
        batch_idx += 1
        data = util.move_to_device(data, cuda_device=device_num)
        sentence = data['sentence']
        sentence_id = data['sentence_id']
        type_ids = data['type_ids']
        event_type = data['event_type']
        trigger = data['trigger']
        roles = data['roles']
        output = argument_extractor(sentence, sentence_id, type_ids, event_type, trigger)
        batch_spans = argument_extractor.metric.get_span(output['start_logits'], output['end_logits'], event_type)

        for idb, batch_span in enumerate(batch_spans):
            s_id = sentence_id[idb]
            if s_id not in pred_spans:
                pred_spans[s_id] = []
            pred_spans[s_id].extend(batch_span)
    # print(pred_spans)
    return pred_spans


def transform(input_sentence):
    # ==== indexer and reader =====
    bert_indexer = {'tokens': PretrainedBertIndexer(
        pretrained_model='models/chinese_roberta_wwm_ext/vocab.txt',
        # pretrained_model='hfl/chinese-roberta-wwm-ext/vocab.txt',
        use_starting_offsets=True,
        do_lowercase=False)}
    data_meta = DataMeta(event_id_file='data/DuEE/DuEE_events.id', role_id_file='data/DuEE/DuEE_roles.id')
    
    trigger_reader = TriggerReader(data_meta=data_meta, token_indexer=bert_indexer)
    role_reader = RoleReader(data_meta=data_meta, token_indexer=bert_indexer)

    # ==== dataset =====
    role_train_dataset = role_reader.read('data/DuEE/train.json')
    data_meta.compute_AF_IEF(role_train_dataset)


    # ==== iterator =====
    vocab = Vocabulary()
    iterator = BucketIterator(
        sorting_keys=[('sentence', 'num_tokens')],
        batch_size=1)
    iterator.index_with(vocab)

    trigger_model_path = 'models/PLMEE/trigger_model.th'
    argument_model_path = 'models/PLMEE/role_model.th'
    

    text_reader = TextReader(data_meta=data_meta, token_indexer=bert_indexer)
    pre_dataset = text_reader.read(input_sentence)
    
    print('=====> Extracting triggers...')

    instances = trigger_extractor_deal(pre_dataset=pre_dataset, iterator=iterator, trigger_model_path=trigger_model_path, dataset_meta=data_meta)

    print('=====> Extracting arguments...')
    pred_spans = argument_extractor_deal(instances=instances, iterator=iterator, argument_model_path=argument_model_path, dataset_meta=data_meta)
    # exit(0)
    id_sentence = {} 
    for data in pre_dataset:
        id_sentence[data['sentence_id'].metadata] = data['origin_text'].metadata
        for sid, pred_span in pred_spans.items():
            text = id_sentence[sid]
            # print(text)
            tmp = {}
            tmp['id'] = sid
            tmp_elist = []
            for ids, span in enumerate(pred_span):
                e_dict = {}
                e_dict['event_type'] = data_meta.get_event_type_name(span[3])
                e_dict['arguments'] = [
                    {
                        "role": data_meta.get_role_name(span[2]),
                        "argument": text[span[0]: span[1] + 1]
                    }
                ]

                tmp_elist.append(e_dict)
            output_sentence = tmp_elist
            print(output_sentence)
            print('=====finish!=====')
            return output_sentence  
  

if __name__ == "__main__":
    input = {}
    input['text'] = "中新网石家庄7月18日电 (黄歆尧)18日，据河北省应急管理厅厅长李国华介绍，2019年上半年，河北共发生各类生产安全事故486起、死亡385人，分别同比下降17.6%、13.9%。"
    uid = uuid.uuid1()
    id = uid.hex
    input['id'] = id
    inputJson = json.dumps(input)

    transform(inputJson)