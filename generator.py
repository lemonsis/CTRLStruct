from time import clock_settime
import numpy as np
from sklearn import cluster
from transformers.models.bart.modeling_bart import shift_tokens_right, BartEncoder, BartDecoder, \
BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, _expand_mask, BaseModelOutput, Seq2SeqModelOutput
from transformers import BartForConditionalGeneration, BartModel, BartConfig, BartTokenizer
import torch
import torch.nn as nn  
import torch.nn.functional as F
# from Model import Model
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import argparse
import logging
from tqdm import tqdm
from collections import OrderedDict
from transformers.file_utils import ModelOutput
import random
from bc import evaluation, inference_cluster
# from transformers.modeling_utils import PreTrainedModel

Coefficient = 1.2

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_batch_size", type=int, default=256, help="training batch size")
    parser.add_argument("--predict_batch_size", type=int, default=256, help="predict batch size")
    parser.add_argument("--epochs", type=int, default=7, help="epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--device", type=str, default='cuda:0', help="device")
    parser.add_argument("--display_interval", type=int, default=100, help="display interval")
    parser.add_argument("--num_beams", type=int, default=5, help="beam num")
    # parser.add_argument("--modelparams", type=str, default='generator/0.2/best_model_50_kl_10.pth', help="model params")
    # parser.add_argument("--checkpoisdnt", type=str, default='model/0.2/best_model_40_50.pth', help="dialogue_file path")
    parser.add_argument("--best_model_path", type=str, default='generator/0.2/dd_best_model_bart_kl_12.pth', help="best_model_path")
    parser.add_argument("--testpath", type=str, default='test/speaker.txt', help="test file path")
    parser.add_argument("--output_testpath", type=str, default='test/dd_model12kl_answer_beam1_bart.txt', help="output test file path")
    # parser.add_argument("--lambda", type=float, default=0.4, help="coefficient of cluster center")
    args = parser.parse_args()
    return args

class decode(BartDecoder):
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        head_mask = None,
        cross_attn_head_mask = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        cluster_center = None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        # print(inputs_embeds.shape)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if cluster_center != None:
            cluster_center = cluster_center.unsqueeze(dim=1)
            cluster_center = cluster_center.repeat(1,64,1)
            hidden_states += cluster_center * Coefficient
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        "The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

class bartmodel(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = decode(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        cluster_center = None
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cluster_center = cluster_center
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class dialog_generation(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = bartmodel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None,head_mask=None,
            decoder_head_mask=None,cross_attn_head_mask=None,past_key_values=None,
            inputs_embeds=None,decoder_inputs_embeds=None,labels=None,output_attentions=None,
            output_hidden_states=None,return_dict=None,use_cache=False, is_training=False,
            cluster_center=None):

        if is_training:
            _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)
        else:
            _decoder_input_ids = decoder_input_ids

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=_decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cluster_center=cluster_center
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)

        loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.config.pad_token_id)
        loss_ = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                              decoder_input_ids.view(-1))
        
        # return (lm_logits, ) + outputs[1:]
        if is_training:
            return loss_
        # return outputs
        return Seq2SeqLMOutput(
            loss = loss_,
            logits = lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions
        )

class dataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 in_metadata=None, out_metadata=None,
                 is_training=False, cluster_center=None):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training
        self.cluster_center = cluster_center

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        if self.cluster_center != None:
            return self.input_ids[in_idx], self.attention_mask[in_idx], \
                self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx], self.cluster_center[out_idx]
        else:
            return self.input_ids[in_idx], self.attention_mask[in_idx], \
                self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]

class dataloader(DataLoader):
    def __init__(self, args, dataset, is_training):
        if is_training:
            sampler=RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(dataloader, self).__init__(dataset, sampler=sampler, batch_size=batch_size, 
                                        num_workers=16, pin_memory=True, drop_last=True)

def load_data(args, tokenizer, filepath1, filepath2, filepath3, pth, is_training, is_idea, cluster_center):
    # CLUSTER_CENTER = 'kmeans/0.2_cluster_40_50/cluster_centers_60.pth'
    cluster_id = evaluation(filepath3, False, pth)
    with open (filepath1, 'r') as f, open (filepath2, 'r') as t:
        speaker = f.readlines()
        answer = t.readlines()
    speaker = tokenizer.batch_encode_plus(speaker, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
    answer = tokenizer.batch_encode_plus(answer, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
    input_ids, attention_mask = speaker["input_ids"], speaker["attention_mask"]
    decoder_input_ids, decoder_attention_mask = answer["input_ids"], answer["attention_mask"]
    if is_idea:
        cluster_center = torch.load(cluster_center)
        cluster = [x for x in cluster_center[cluster_id]]
        # print(decoder_input_ids, decoder_attention_mask)
        Dataset = dataset(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, is_training=is_training, cluster_center=cluster)
        Dataloader = dataloader(args, Dataset, is_training)
    else:
        Dataset = dataset(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, is_training=is_training, cluster_center=None)
        Dataloader = dataloader(args, Dataset, is_training)
    return Dataloader

def train(args):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    dl = load_data(args, tokenizer, 'data/speaker.txt', 'data/answer.txt', 'data/answer.txt', 
                    "dd_eva_bart.pth", is_training=True, is_idea=True, cluster_center='kmeans/bart/cluster_centers_60.pth')
    # checkpoint = torch.load(args.checkpoint)
    model = dialog_generation.from_pretrained('facebook/bart-large')
    # model_dict = model.state_dict()
    # od = OrderedDict()
    # for k, v in checkpoint['model_state_dict'].items():
    #     new = k.replace('module.encoder', 'model.encoder')
    #     od[new] = v
    # state_dict = {k: v for k, v in od.items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)
    ## model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(args.device)
    model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # load checkpoint
    # c = torch.load(args.modelparams)
    # model.load_state_dict(c['model_state_dict'])
    # model.to(args.device)
  
    model.train()
    logging.getLogger().setLevel(logging.INFO)
    logging.info("start training...")
    batch_idx = 0
    stop_training = False
    min_loss = 100000
    for epoch_idx in range(args.epochs):
        train_losses = []
        for batch in tqdm(dl):
            batch_idx += 1
            batch = [b.to(args.device) for b in batch]
            loss = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         is_training=True) #, cluster_center=batch[4]
            output = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         is_training=False).last_hidden_state
            final = torch.empty([0, 1024], device=args.device)
            for bs in output:
                sum = torch.zeros(1, 1024)
                sum = sum.to(args.device)
                for j in bs:
                    sum += j
                sum /= 64
                final = torch.cat((final, sum))
            kl_loss = F.kl_div(final.softmax(dim=-1).log(), batch[4].softmax(dim=-1), reduction='sum')
            loss = loss.mean()
            loss = loss + Coefficient * kl_loss
            if torch.isnan(loss).data:
                logging.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            train_losses.append(loss)
            if batch_idx % args.display_interval == 0:
                logging.getLogger().setLevel(logging.INFO)
                logging.info(f"epoch: {epoch_idx}, batch_idx: {batch_idx}, loss: {loss:>10f}")
        avg_loss = np.mean(train_losses)
        validloss = valid_loss(args, model)
        print(validloss)
        if validloss < min_loss:
            print(validloss,'<', min_loss," new model saved")
            min_loss = validloss
            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss
            }, args.best_model_path)
            with open("record.txt", 'a') as q:
                q.write(str(epoch_idx))
        if stop_training:
            break

def valid_loss(args, model):
    logging.getLogger().setLevel(logging.INFO)
    logging.info("start validate...")
    model.eval()
    valid_losses = []
    with torch.no_grad():
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        dl = load_data(args, tokenizer, 'valid/speaker.txt', 'valid/answer.txt', 'valid/answer.txt', 
                        "dd_eva_bart_valid.pth", is_training=True, is_idea=True, cluster_center='kmeans/bart/cluster_centers_60.pth')
        for i, batch in enumerate(tqdm(dl)):
            batch = [b.to(args.device) for b in batch]
            loss = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         is_training=True, cluster_center=None) #, cluster_center=batch[4]
            output = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         is_training=False).last_hidden_state
            final = torch.empty([0, 1024], device=args.device)
            for bs in output:
                sum = torch.zeros(1, 1024)
                sum = sum.to(args.device)
                for j in bs:
                    sum += j
                sum /= 64
                final = torch.cat((final, sum))
            kl_loss = F.kl_div(final.softmax(dim=-1).log(), batch[4].softmax(dim=-1), reduction='sum')
            loss = loss.mean()
            loss = loss + Coefficient * kl_loss
            valid_losses.append(loss.detach())
        valid_losses = [loss.item() for loss in valid_losses]
        valid_loss = np.mean(valid_losses)
    return valid_loss

def inference(args, idea):
    # import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    CLUSTER_CENTER = 'kmeans/0.2_pc_cluster_60/cluster_centers_60.pth'
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = dialog_generation.from_pretrained('facebook/bart-large')
    model = nn.DataParallel(model, device_ids=[4,5,6,7])
    checkpoint = torch.load(args.best_model_path, map_location=torch.device('cuda:4'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)

    model.eval()
    print("start conversation...")
    while(True):
        speaker = input()
        tok = tokenizer(speaker, return_tensors='pt')
        cluster_center = None
        if idea:
            cluster_center = inference_cluster(tok, "center")
            cluster_center = cluster_center.unsqueeze(0)
            cluster_center = cluster_center.unsqueeze(0)
            num_tok = tok["input_ids"].shape[1]
            cluster_center = cluster_center.repeat(1, num_tok, 1)
        input_id = tok["input_ids"].to(args.device)
        input_attention_mask = tok["attention_mask"].to(args.device)
        output = model.module.generate(input_ids=input_id, attention_mask=input_attention_mask,
                                num_beams=args.num_beams, max_length=64, early_stopping=True)
        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        print(pred)

def Eval(args):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = dialog_generation.from_pretrained('facebook/bart-large')
    model = nn.DataParallel(model, device_ids=[4,5,6,7])
    checkpoint = torch.load(args.best_model_path, map_location=torch.device('cuda:4'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)

    model.eval()
    # input_id = []
    # att_mask = []
    with open (args.testpath, 'r') as f, open (args.output_testpath, 'w') as g:
        for line in f:
            # line = line[:-1]
            speaker = tokenizer(line, return_tensors='pt')
            input_id = speaker["input_ids"].to(args.device)
            att_mask = speaker["attention_mask"].to(args.device)

    # with open (args.output_testpath, 'w') as f:
    #     for t in range(len(input_id)):
            output = model.module.generate(input_ids=input_id, attention_mask=att_mask,
                                        num_beams=args.num_beams, max_length=64, early_stopping=True)
            #num_beams=args.num_beams, max_length=64, early_stopping=True
            #do_sample=True, top_k=50, top_p=0.95, max_length=64
            pred = tokenizer.decode(output[0], skip_special_tokens=True)

            
            # if pred[0] == ' ' and pred[-1] == '\n':
            #     f.write(pred[1:])
            # elif pred[0] == ' ':
            #     f.write(pred[1:])
            #     f.write('\n')

            # if pred[-1] == '\n' and pred[-2] == '\n':
            #     g.write(pred[:-1])
            if pred == '':
                g.write('\n')
            elif pred[-1] == '\n':
                g.write(pred)
            elif pred[-1] != '\n':
                g.write(pred)
                g.write('\n')
                

def main():
    args = parse_args()
    # train(args)
    inference(args, False)
    # Eval(args)

if __name__ == "__main__":
    main()