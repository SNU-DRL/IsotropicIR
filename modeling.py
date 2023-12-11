from pytools import memoize_method
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig, AutoModel, AutoTokenizer, AutoConfig
# import modeling_util
import string
import sys

sys.path.append("glow-pytorch/")
from model import Glow

sys.path.append("nice_pytorch/")
from nice.models import NICEModel

## Biencoder for ColBERTRanker and RepBERTRanker
class BertBiencoderRanker(torch.nn.Module):
    def __init__(self, without_bert=False, bert_model=None):
        super().__init__()
        self.BERT_MODEL = bert_model
        if "bert-base" in self.BERT_MODEL:
            self.CHANNELS = 12 + 1 # from bert-base-uncased
            self.BERT_SIZE = 768 # from bert-base-uncased
            ## Config
            # print("lora_attn_dim", lora_attn_dim)
            config = BertConfig(output_hidden_states=True)
        elif "bert-large" in self.BERT_MODEL:
            self.CHANNELS = 24 + 1 # from bert-base-uncased
            self.BERT_SIZE = 1024 # from bert-base-uncased
            config = BertConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096, output_hidden_states=True)
        else:
            raise BaseException("Either bert-base or bert-large should be called")

        if(without_bert): 
            self.bert = None
        else:
            if ("bert-base" in self.BERT_MODEL) or ("bert-large" in self.BERT_MODEL):
                self.bert = BertModel.from_pretrained(self.BERT_MODEL, config=config).cuda()
                self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL)
            else:
                self.bert = AutoModel.from_pretrained(self.BERT_MODEL, config=config).cuda()
                self.tokenizer = AutoTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path, without_bert=False):
        if without_bert:
            state_org = self.state_dict(keep_vars=True)
            state = {}
            for key in list(state_org):
                if ('adapter' in key) or ('bert.' not in key):
                    state[key] = state_org[key].data                
        else:
            state = self.state_dict(keep_vars=True)
            for key in list(state):
                state[key] = state[key].data

        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        print("load model : ", path)

    def freeze_bert(self):
        for n, p in self.bert.named_parameters():
            if 'adapter' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_params(self):
        params = [(k, v) for k, v in self.named_parameters() if v.requires_grad]
        bert_params = [v for k, v in params if k.startswith('bert')]
        non_bert_params = [v for k, v in params if (not k.startswith('bert')) and ('rep' not in k) and ('glow' not in k) and ('nice' not in k)]
        glow_params = [v for k, v in params if 'glow' in k]
        nice_params = [v for k, v in params if 'nice' in k]

        print("bert_params", [k for k, v in params if k.startswith('bert')])
        print("non_bert_params", [k for k, v in params if (not k.startswith('bert')) and ('rep' not in k) and ('glow' not in k) and ('nice' not in k)])
        print("glow_params", [k for k, v in params if 'glow' in k])
        print("nice_params", [k for k, v in params if 'nice' in k])
        return bert_params, non_bert_params, glow_params, nice_params

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer(text)['input_ids'][1:-1]
        # print("toks", toks)
        return toks

    def encode_colbert(self, query_tok, query_mask, doc_tok, doc_mask, device='cuda:0', p_qrepr=None, p_drepr=None, p_start=0, roberta=False):
        # encode without subbatching
        query_lengths = (query_mask > 0).sum(1)
        doc_lengths = (doc_mask > 0).sum(1)
        BATCH, QLEN = query_tok.shape
        QLEN : 20
        DIFF = 2  # = [CLS] and [SEP]
        if roberta:
            maxlen = 512
        else:
            maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - DIFF  # doc maxlen: 510

        doc_toks = F.pad(doc_tok[:, : MAX_DOC_TOK_LEN], pad=(0, 1, 0, 0), value=-1)
        doc_mask = F.pad(doc_mask[:, :MAX_DOC_TOK_LEN], pad=(0, 1, 0, 0), value=0)
        query_toks = query_tok

        query_lengths = torch.where(query_lengths > QLEN-1, torch.tensor(QLEN-1).cuda(device), query_lengths)
        if roberta:
            query_toks[torch.arange(BATCH), query_lengths] = self.tokenizer.vocab['</s>']
        else:
            query_toks[torch.arange(BATCH), query_lengths] = self.tokenizer.vocab["[SEP]"]
        query_mask[torch.arange(BATCH), query_lengths] = 1
        doc_lengths = torch.where(doc_lengths > MAX_DOC_TOK_LEN, torch.tensor(MAX_DOC_TOK_LEN).cuda(device), doc_lengths)
        if roberta:
            doc_toks[torch.arange(BATCH), doc_lengths] = self.tokenizer.vocab['</s>']
        else:
            doc_toks[torch.arange(BATCH), doc_lengths] = self.tokenizer.vocab["[SEP]"]
        doc_mask[torch.arange(BATCH), doc_lengths] = 1

        if roberta:
            CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['<s>'])
            SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['</s>'])
        else:
            CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab["[CLS]"])
            SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab["[SEP]"])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences query & doc
        q_toks = torch.cat([CLSS, query_toks], dim=1)
        q_mask = torch.cat([ONES, query_mask], dim=1)
        q_segid = torch.cat([NILS] * (1 + QLEN), dim=1)
        # 2) Query augmentation with [MASK] tokens ([MASK] = 103)
        q_toks[q_toks == -1] = torch.tensor(self.tokenizer.mask_token_id).cuda(device)

        d_toks = torch.cat([CLSS, doc_toks], dim=1)
        d_mask = torch.cat([ONES, doc_mask], dim=1)
        d_segid = torch.cat([NILS] * (1 + doc_toks.shape[1]), dim=1)
        ## padding (Euna guess)
        if roberta:
            d_toks[d_toks == -1] = 1
        else:
            d_toks[d_toks == -1] = 0

        # execute BERT model
        q_result_tuple = self.bert(q_toks, q_mask, q_segid.long())
        d_result_tuple = self.bert(d_toks, d_mask, d_segid.long())
        q_result = q_result_tuple[2]
        d_result = d_result_tuple[2]

        # extract relevant subsequences for query and doc
        query_results = [r[:, :] for r in q_result]  # missing representation for cls and sep?
        doc_results = [r[:, :] for r in d_result]

        q_cls_result = [r[:, 0] for r in q_result]
        d_cls_result = [r[:, 0] for r in d_result]

        return q_cls_result, d_cls_result, query_results, q_mask, doc_results, d_mask

    def encode_repbert(self, query_tok, query_mask, doc_tok, doc_mask, p_qrepr=None, p_drepr=None, p_start=0, roberta=False):
        BATCH, QLEN = query_tok.shape ## shape=[batch_size=4, query length=20] ## doc_tok.shape=[bs, 800]
        DIFF = 2 # = [CLS] and [SEP]

        if roberta:
            maxlen = 512
            CLS_ID = 0
            PAD_ID = 1
            SEP_ID = 2
        else: ## BERT
            maxlen = self.bert.config.max_position_embeddings
            CLS_ID = 101
            PAD_ID = 0
            SEP_ID = 102
        MAX_DOC_TOK_LEN = maxlen - DIFF
        
        doc_toks = doc_tok[:, : MAX_DOC_TOK_LEN]
        doc_mask = doc_mask[:, :MAX_DOC_TOK_LEN]
        query_toks = query_tok
        
        CLSS = torch.full_like(query_toks[:, :1], CLS_ID)
        SEPS = torch.full_like(query_toks[:, :1], SEP_ID)
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences query & doc
        q_toks = torch.cat([CLSS, query_toks, SEPS], dim=1).cuda()
        q_mask = torch.cat([ONES, query_mask, ONES], dim=1).cuda()
        q_segid = torch.cat([NILS] * (2+QLEN), dim=1).cuda()
        q_toks[q_toks == -1] = PAD_ID
        # print("q_toks, q_mask, q_segid", q_toks.shape, q_mask.shape, q_segid.shape)

        d_toks = torch.cat([CLSS, doc_toks, SEPS], dim=1).cuda()
        d_mask = torch.cat([ONES, doc_mask, ONES], dim=1).cuda()
        d_segid = torch.cat([NILS] * (2+doc_toks.shape[1]), dim=1).cuda()
        d_toks[d_toks == -1] = PAD_ID
        # print("d_toks, d_mask, d_segid", d_toks.shape, d_mask.shape, d_segid.shape)

        # execute BERT model
        q_result_tuple = self.bert(q_toks, q_mask, q_segid.long())
        d_result_tuple = self.bert(d_toks, d_mask, d_segid.long())
        q_result = q_result_tuple[2]
        d_result = d_result_tuple[2]

        return q_result, d_result, q_mask, d_mask

## ColBERTRanker
class ColBertRanker(BertBiencoderRanker):
    def __init__(self, args, without_bert=False):
        super().__init__(without_bert=without_bert, bert_model=args.model_card)
        self.wo_mask = args.wo_mask
        self.skiplist = self.tokenize(string.punctuation)
        self.model_card=args.model_card
        self.wo_clinear = args.wo_clinear
        self.whitening = args.whitening

        if self.whitening:
            self.whitening_mu_file = args.whitening_mu_file
            self.whitening_cov_file = args.whitening_cov_file
            if args.whitening_k is None:
                self.whitening_k = self.BERT_SIZE
            else:
                self.whitening_k = args.whitening_k

        if not self.wo_clinear:
            self.dim = 128  # default: dim=128
            self.clinear = torch.nn.Linear(
                self.BERT_SIZE, self.dim, bias=False
            )  # both for queries, documents

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, value_return=False, after_clinear=False, whitening_value_return=False):
        if not "roberta" in self.model_card:
            # 1) Prepend [Q] token to query, [D] token to document
            q_length = query_tok.shape[1]
            d_length = doc_tok.shape[1]
            num_batch_samples = doc_tok.shape[0]
            
            Q_tok = torch.full(
                size=(num_batch_samples, 1), fill_value=1, dtype=torch.long
            ).cuda()  # [unused0] = 1
            D_tok = torch.full(
                size=(num_batch_samples, 1), fill_value=2, dtype=torch.long
            ).cuda()  # [unused1] = 2
            one_tok = torch.full(size=(num_batch_samples, 1), fill_value=1).cuda()

            query_tok = torch.cat([Q_tok, query_tok[:, : q_length - 1]], dim=1)
            doc_tok = torch.cat([D_tok, doc_tok[:, : d_length - 1]], dim=1)
            query_mask = torch.cat([one_tok, query_mask[:, : q_length - 1]], dim=1)
            doc_mask = torch.cat([one_tok, doc_mask[:, : d_length - 1]], dim=1)

        ## Encoding through BERT
        if self.wo_mask: ## without mask
            q_reps, d_reps, q_mask, d_mask = self.encode_repbert(query_tok, query_mask, doc_tok, doc_mask, roberta=("roberta" in self.model_card))  # reps includes rep of [CLS], [SEP]

            q_reps = list(q_reps) ## len(q_reps) = 13
            d_reps = list(d_reps)
            for i in range(len(q_reps)):
                q_reps[i] = q_reps[i] * q_mask.unsqueeze(-1)
                d_reps[i] = d_reps[i] * d_mask.unsqueeze(-1)
        else:
            q_cls_reps, d_cls_reps, q_reps, query_mask, d_reps, doc_mask = self.encode_colbert(query_tok, query_mask, doc_tok, doc_mask, roberta=("roberta" in self.model_card))  # reps includes rep of [CLS], [SEP]
        
        if not self.wo_clinear:
            col_q_reps = self.clinear(q_reps[-1])
            col_d_reps = self.clinear(d_reps[-1])
        else:
            col_q_reps = q_reps[-1]
            col_d_reps = d_reps[-1]

        if self.whitening:
            bs=col_q_reps.shape[0]
            dim=col_q_reps.shape[-1]
            col_q_reps = whitening(col_q_reps.reshape(-1, dim), self.whitening_mu_file, self.whitening_cov_file, self.whitening_k).reshape(bs, -1, self.whitening_k)
            col_d_reps = whitening(col_d_reps.reshape(-1, dim), self.whitening_mu_file, self.whitening_cov_file, self.whitening_k).reshape(bs, -1, self.whitening_k)

        if (not "roberta" in self.model_card) and (not self.wo_mask):
            # 3) skip punctuations in doc tokens
            cut_doc_tok = torch.cat([one_tok.long(), doc_tok[:, :510], one_tok.long()], dim=1)
            mask = torch.ones_like(doc_mask, dtype=torch.float).cuda()
            mask = torch.where(
                ((cut_doc_tok >= 999) & (cut_doc_tok <= 1013))
                | ((cut_doc_tok >= 1024) & (cut_doc_tok <= 1036))
                | ((cut_doc_tok >= 1063) & (cut_doc_tok <= 1066))
                | (cut_doc_tok == -1),
                torch.tensor(0.0).cuda(),
                doc_mask,
            )
            col_d_reps = col_d_reps * mask.unsqueeze(2)
        
        q_rep = F.normalize(col_q_reps, p=2, dim=2)
        d_rep = F.normalize(col_d_reps, p=2, dim=2)
        score = (q_rep @ d_rep.permute(0, 2, 1)).max(2).values.sum(1)

        simmat = torch.cat([q_rep, d_rep], dim=1)  ## for distillation
        score = score.unsqueeze(1)
        if whitening_value_return:
            if after_clinear:
                return score, [self.clinear(q_reps[-1]), self.clinear(d_reps[-1])]
            else:
                return score, [q_reps[-1], d_reps[-1]]
        if value_return:
            if after_clinear:
                return score, [self.clinear(q_reps[-1]), self.clinear(d_reps[-1])]
            else:
                return score, [col_q_reps, col_d_reps]
        else:
            return score
        
    def forward_without_bert(self, cls_reps, q_reps, d_reps, query_tok, doc_tok):
        score = (q_reps @ d_reps.permute(0, 2, 1)).max(2).values.sum(1)
        simmat = torch.cat([q_reps, d_reps], dim=1)  ## for distillation
        score = score.unsqueeze(1)
        return score, simmat

## NICE-ColBERTRanker
class NICEColBertRanker(BertBiencoderRanker): ## self.aggregator -> (self.flow_dr_layer) -> self.nice
    def __init__(self, args, without_bert=False):
        super().__init__(without_bert=without_bert, bert_model=args.model_card)
        self.model_card=args.model_card
        self.wo_clinear = args.wo_clinear
        self.wo_mask = args.wo_mask

        ## aggregator
        if not self.wo_clinear:
            self.agg_dim = 128 # default: dim=128
            self.clinear = torch.nn.Linear(
            self.BERT_SIZE, self.agg_dim, bias=False
        )  # both for queries, documents
        else:
            self.agg_dim = self.BERT_SIZE # default: dim=128
        self.skiplist = self.tokenize(string.punctuation)

        ## NICE
        self.hetero_flow = args.hetero_flow
        self.flow_mf = args.flow_mf
        if args.dim_red == 0:
            self.dim_red = self.agg_dim
            self.flow_dim = self.agg_dim
        else:
            self.dim_red = args.dim_red
            self.flow_dr_layer = torch.nn.Linear(self.agg_dim, self.dim_red)
            self.flow_dim = self.dim_red
        self.nice = NICEModel(self.flow_dim, args.nice_nhidden, args.nice_nlayers).to(self.bert.device)
        if self.hetero_flow:
            print("using heterogeneous NICE models")
            self.nice2 = NICEModel(self.flow_dim, args.nice_nhidden, args.nice_nlayers).to(self.bert.device)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, value_return=False): ## self.aggregator -> (self.flow_dr_layer) -> self.nice
        if not "roberta" in self.model_card:
            # 1) Prepend [Q] token to query, [D] token to document
            q_length = query_tok.shape[1]
            d_length = doc_tok.shape[1]
            num_batch_samples = doc_tok.shape[0]

            Q_tok = torch.full(
                size=(num_batch_samples, 1), fill_value=1, dtype=torch.long
            ).cuda()  # [unused0] = 1
            D_tok = torch.full(
                size=(num_batch_samples, 1), fill_value=2, dtype=torch.long
            ).cuda()  # [unused1] = 2
            one_tok = torch.full(size=(num_batch_samples, 1), fill_value=1).cuda()

            query_tok = torch.cat([Q_tok, query_tok[:, : q_length - 1]], dim=1)
            doc_tok = torch.cat([D_tok, doc_tok[:, : d_length - 1]], dim=1)
            query_mask = torch.cat([one_tok, query_mask[:, : q_length - 1]], dim=1)
            doc_mask = torch.cat([one_tok, doc_mask[:, : d_length - 1]], dim=1)

        ## Encoding through BERT
        if self.wo_mask: ## without mask
            q_reps, d_reps, q_mask, d_mask = self.encode_repbert(query_tok, query_mask, doc_tok, doc_mask, roberta=("roberta" in self.model_card))  # reps includes rep of [CLS], [SEP]
        
            q_reps = list(q_reps) ## len(q_reps) = 13
            d_reps = list(d_reps)
            for i in range(len(q_reps)):
                q_reps[i] = q_reps[i] * q_mask.unsqueeze(-1)
                d_reps[i] = d_reps[i] * d_mask.unsqueeze(-1)
        else:
            q_cls_reps, d_cls_reps, q_reps, query_mask, d_reps, doc_mask = self.encode_colbert(query_tok, query_mask, doc_tok, doc_mask, roberta=("roberta" in self.model_card))  # reps includes rep of [CLS], [SEP]
            
        q_rep = q_reps[-1]
        d_rep = d_reps[-1]
        q_max_seqlen = q_rep.shape[1]
        d_max_seqlen = d_rep.shape[1]

        ## clinear
        if not self.wo_clinear:
            q_rep = self.clinear(q_rep) ## self.BERT_SIZE -> self.agg_dim
            d_rep = self.clinear(d_rep) ## self.BERT_SIZE -> self.agg_dim
        
        ## NICE
        if self.dim_red != self.agg_dim:
            q_rep = self.flow_dr_layer(q_rep) ## dimension reduction before NICE
            d_rep = self.flow_dr_layer(d_rep) ## d_rep.shape=[bs*2, seq_len, dim_red]

        q_x = q_rep[q_rep.nonzero(as_tuple=True)].view(-1, self.dim_red)
        d_x = d_rep[d_rep.nonzero(as_tuple=True)].view(-1, self.dim_red)
        
        q_z = self.nice(q_x.reshape(-1, self.dim_red))
        if self.hetero_flow:
            d_z = self.nice2(d_x.reshape(-1, self.dim_red))
        else:
            d_z = self.nice(d_x.reshape(-1, self.dim_red))

        ## reshape
        if self.wo_mask:
            ## split nice output
            q_z_split = list(torch.split(q_z, list(q_mask.sum(1).long()), dim=0))
            d_z_split = list(torch.split(d_z, list(d_mask.sum(1).long()), dim=0))
            q_z_split = [F.pad(input=e, pad=(0, 0, 0, q_max_seqlen - e.shape[0]), mode='constant', value=0).unsqueeze(0) for e in q_z_split]
            d_z_split = [F.pad(input=e, pad=(0, 0, 0, d_max_seqlen - e.shape[0]), mode='constant', value=0).unsqueeze(0) for e in d_z_split]

            col_q_reps = torch.cat(q_z_split, 0).view(q_rep.shape[0], -1, self.dim_red)
            col_d_reps = torch.cat(d_z_split, 0).view(d_rep.shape[0], -1, self.dim_red)
        else:
            col_q_reps = q_z.view(q_rep.shape[0], -1, self.dim_red)
            col_d_reps = d_z.view(d_rep.shape[0], -1, self.dim_red)

        if not "roberta" in self.model_card and not self.wo_mask:
            # 3) skip punctuations in doc tokens
            cut_doc_tok = torch.cat([one_tok.long(), doc_tok[:, :510], one_tok.long()], dim=1)
            mask = torch.ones_like(doc_mask, dtype=torch.float).cuda()
            mask = torch.where(
                ((cut_doc_tok >= 999) & (cut_doc_tok <= 1013))
                | ((cut_doc_tok >= 1024) & (cut_doc_tok <= 1036))
                | ((cut_doc_tok >= 1063) & (cut_doc_tok <= 1066))
                | (cut_doc_tok == -1),
                torch.tensor(0.0).cuda(),
                doc_mask,
            )
            col_d_reps = col_d_reps * mask.unsqueeze(2)
        q_rep = F.normalize(col_q_reps, p=2, dim=2)
        d_rep = F.normalize(col_d_reps, p=2, dim=2)
        score = (q_rep @ d_rep.permute(0, 2, 1)).max(2).values.sum(1)
        score = score.unsqueeze(1)
        if self.wo_mask:
            return score, (q_z, d_z), (q_mask, d_mask)
        else:     
            return score, (q_z, d_z), (0, 0)

## Glow-ColBERTRanker
class GlowColBertRanker(BertBiencoderRanker): ## self.aggregator -> (self.flow_dr_layer) -> self.nice
    def __init__(self, args, without_bert=False):
        super().__init__(without_bert=without_bert, bert_model=args.model_card)
        self.model_card=args.model_card
        self.wo_clinear = args.wo_clinear
        self.wo_mask = args.wo_mask

        self.whitening_flow = args.whitening_flow
        self.flow_whitening = args.flow_whitening

        if self.whitening_flow or self.flow_whitening:
            self.whitening_mu_file = args.whitening_mu_file
            self.whitening_cov_file = args.whitening_cov_file
            if args.whitening_k is None:
                self.whitening_k = self.BERT_SIZE
            else:
                self.whitening_k = args.whitening_k

        ## Aggregator
        if not self.wo_clinear:
            self.agg_dim = 128 # default: dim=128
            self.clinear = torch.nn.Linear(
            self.BERT_SIZE, self.agg_dim, bias=False
        )  # both for queries, documents
        else:
            self.agg_dim = self.BERT_SIZE # default: dim=128
        self.skiplist = self.tokenize(string.punctuation)

        ## Glow
        self.hetero_flow = args.hetero_flow
        self.flow_mf = args.flow_mf
        if args.dim_red == 0:
            self.dim_red = self.agg_dim
            self.flow_dim = self.agg_dim
        else:
            self.dim_red = args.dim_red
            self.flow_dr_layer = torch.nn.Linear(self.agg_dim, self.dim_red)
            self.flow_dim = self.dim_red
        self.glow = Glow(self.BERT_SIZE, args.n_flow, args.n_block, two_conv=args.glow_two_conv)
        if self.hetero_flow:
            print("using heterogeneous Glow models")
            self.glow2 = Glow(self.BERT_SIZE, args.n_flow, args.n_block, two_conv=args.glow_two_conv)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, value_return=False, whitening_value_return=False): ## self.aggregator -> (self.flow_dr_layer) -> self.nice
        if not "roberta" in self.model_card:
            # 1) Prepend [Q] token to query, [D] token to document
            q_length = query_tok.shape[1]
            d_length = doc_tok.shape[1]
            num_batch_samples = doc_tok.shape[0]

            Q_tok = torch.full(
                size=(num_batch_samples, 1), fill_value=1, dtype=torch.long
            ).cuda()  # [unused0] = 1
            D_tok = torch.full(
                size=(num_batch_samples, 1), fill_value=2, dtype=torch.long
            ).cuda()  # [unused1] = 2
            one_tok = torch.full(size=(num_batch_samples, 1), fill_value=1).cuda()

            query_tok = torch.cat([Q_tok, query_tok[:, : q_length - 1]], dim=1)
            doc_tok = torch.cat([D_tok, doc_tok[:, : d_length - 1]], dim=1)
            query_mask = torch.cat([one_tok, query_mask[:, : q_length - 1]], dim=1)
            doc_mask = torch.cat([one_tok, doc_mask[:, : d_length - 1]], dim=1)

        ## Encoding through BERT
        if self.wo_mask: ## without mask
            q_reps, d_reps, q_mask, d_mask = self.encode_repbert(query_tok, query_mask, doc_tok, doc_mask, roberta=("roberta" in self.model_card))  # reps includes rep of [CLS], [SEP]
            
            q_reps = list(q_reps) ## len(q_reps) = 13
            d_reps = list(d_reps)
            for i in range(len(q_reps)):
                q_reps[i] = q_reps[i] * q_mask.unsqueeze(-1)
                d_reps[i] = d_reps[i] * d_mask.unsqueeze(-1)
        else:
            q_cls_reps, d_cls_reps, q_reps, query_mask, d_reps, doc_mask = self.encode_colbert(query_tok, query_mask, doc_tok, doc_mask, roberta=("roberta" in self.model_card))  # reps includes rep of [CLS], [SEP]
        
        q_rep = q_reps[-1]
        d_rep = d_reps[-1]
        q_max_seqlen = q_rep.shape[1]
        d_max_seqlen = d_rep.shape[1]

        ## clinear
        if not self.wo_clinear:
            q_rep = self.clinear(q_rep) ## self.BERT_SIZE -> self.agg_dim
            d_rep = self.clinear(d_rep) ## self.BERT_SIZE -> self.agg_dim

        ## whitening_flow
        if self.whitening_flow:
            bs=q_rep.shape[0]
            dim=q_rep.shape[-1]
            q_rep = whitening(q_rep.reshape(-1, dim), self.whitening_mu_file, self.whitening_cov_file, self.whitening_k).reshape(bs, -1, self.whitening_k)
            d_rep = whitening(d_rep.reshape(-1, dim), self.whitening_mu_file, self.whitening_cov_file, self.whitening_k).reshape(bs, -1, self.whitening_k)
            # print("q_rep.shape", q_rep.shape, "d_rep.shape", d_rep.shape) ## shape = [bs, seqlen, dim]
        
        ## Glow
        if self.dim_red != self.agg_dim:
            q_rep = self.flow_dr_layer(q_rep) ## dimension reduction before NICE
            d_rep = self.flow_dr_layer(d_rep) ## d_rep.shape=[bs*2, seq_len, dim_red]

        if not self.whitening_flow:
            q_x = q_rep[q_rep.nonzero(as_tuple=True)].view(-1, self.dim_red)
            d_x = d_rep[d_rep.nonzero(as_tuple=True)].view(-1, self.dim_red)
        else:
            q_x = q_rep.view(-1, self.dim_red)
            d_x = d_rep.view(-1, self.dim_red)

        q_log_p_sum, q_logdet, q_z_outs = self.glow(q_x.reshape(-1, self.dim_red, 1, 1))
        if self.hetero_flow:
            d_log_p_sum, d_logdet, d_z_outs = self.glow2(d_x.reshape(-1, self.dim_red, 1, 1))
        else:
            d_log_p_sum, d_logdet, d_z_outs = self.glow(d_x.reshape(-1, self.dim_red, 1, 1))
        
        q_z = q_z_outs[-1].squeeze()
        d_z = d_z_outs[-1].squeeze()

        ## flow_whitening
        if self.flow_whitening:
            q_z = whitening(q_z, self.whitening_mu_file, self.whitening_cov_file, self.whitening_k)
            d_z = whitening(d_z, self.whitening_mu_file, self.whitening_cov_file, self.whitening_k)
        
        ## reshape
        if self.wo_mask:
            q_z_split = list(torch.split(q_z, list(q_mask.sum(1).long()), dim=0))
            d_z_split = list(torch.split(d_z, list(d_mask.sum(1).long()), dim=0))
            q_z_split = [F.pad(input=e, pad=(0, 0, 0, q_max_seqlen - e.shape[0]), mode='constant', value=0).unsqueeze(0) for e in q_z_split]
            d_z_split = [F.pad(input=e, pad=(0, 0, 0, d_max_seqlen - e.shape[0]), mode='constant', value=0).unsqueeze(0) for e in d_z_split]

            col_q_reps = torch.cat(q_z_split, 0).view(q_rep.shape[0], -1, self.dim_red)
            col_d_reps = torch.cat(d_z_split, 0).view(d_rep.shape[0], -1, self.dim_red)
        else:
            col_q_reps = q_z.view(q_rep.shape[0], -1, self.dim_red)
            col_d_reps = d_z.view(d_rep.shape[0], -1, self.dim_red)

        if not "roberta" in self.model_card and not self.wo_mask:
            # 3) skip punctuations in doc tokens
            cut_doc_tok = torch.cat([one_tok.long(), doc_tok[:, :510], one_tok.long()], dim=1)
            mask = torch.ones_like(doc_mask, dtype=torch.float).cuda()
            mask = torch.where(
                ((cut_doc_tok >= 999) & (cut_doc_tok <= 1013))
                | ((cut_doc_tok >= 1024) & (cut_doc_tok <= 1036))
                | ((cut_doc_tok >= 1063) & (cut_doc_tok <= 1066))
                | (cut_doc_tok == -1),
                torch.tensor(0.0).cuda(),
                doc_mask,
            )
            col_d_reps = col_d_reps * mask.unsqueeze(2)
        q_rep = F.normalize(col_q_reps, p=2, dim=2)
        d_rep = F.normalize(col_d_reps, p=2, dim=2)
        score = (q_rep @ d_rep.permute(0, 2, 1)).max(2).values.sum(1)
        score = score.unsqueeze(1)     
        if whitening_value_return:
            return score, [q_z_outs[-1].squeeze(), d_z_outs[-1].squeeze()]
        else:
            return score, [q_log_p_sum, q_logdet, q_z, d_log_p_sum, d_logdet, d_z]


## RepBERTRanker
class RepBertRanker(BertBiencoderRanker):
    def __init__(self, args, without_bert=False):
        super().__init__(without_bert=without_bert, bert_model=args.model_card)
        self.model_card=args.model_card

        ## args
        self.rep_type = args.rep_type

        self.whitening = args.whitening
        self.whitening_tokenwise = args.whitening_tokenwise
        
        if self.whitening:
            print("applying whitening")
            self.whitening_mu_file = args.whitening_mu_file
            self.whitening_cov_file = args.whitening_cov_file
            if args.whitening_k is None:
                self.whitening_k = self.BERT_SIZE
            else:
                self.whitening_k = args.whitening_k

        ## dimension reduction
        if args.dim_red == 0:
            self.dim_red = self.BERT_SIZE
        else:
            self.dim_red = args.dim_red
            self.dim_red_layer = torch.nn.Linear(self.BERT_SIZE, self.dim_red)
        self.cos = torch.nn.CosineSimilarity()
        
    def forward(self, query_tok, query_mask, doc_tok, doc_mask, value_return=False, whitening_value_return=False):
        q_reps, d_reps, q_mask, d_mask = self.encode_repbert(query_tok, query_mask, doc_tok, doc_mask, roberta=("roberta" in self.model_card)) ## q_reps[-1].shape=[bs, seq_len, 768]

        if self.dim_red != self.BERT_SIZE: ## if dim_red=True, reduce bert rep of dim 768 into args.dim_red
            q_rep = self.dim_red_layer(q_reps[-1]) ## q_rep.shape=[bs, seq_len, 128]
            d_rep = self.dim_red_layer(d_reps[-1])
        else:
            q_rep = q_reps[-1]
            d_rep = d_reps[-1]

        if self.whitening and self.whitening_tokenwise: ## tokenwise whitening
            bs=q_rep.shape[0]
            q_rep = whitening(q_rep.reshape(-1, self.dim_red), self.whitening_mu_file, self.whitening_cov_file, self.whitening_k).reshape(bs, -1, self.dim_red)
            d_rep = whitening(d_rep.reshape(-1, self.dim_red), self.whitening_mu_file, self.whitening_cov_file, self.whitening_k).reshape(bs, -1, self.dim_red)

        if self.rep_type == "cls":
            q_rep_agg = q_rep[:, 0, :]
            d_rep_agg = d_rep[:, 0, :]
        elif self.rep_type == "last_avg": ## default
            # print("q_rep*q_mask.unsqueeze(-1)", (q_rep*(q_mask.unsqueeze(-1))).shape)
            q_rep_agg = (q_rep*(q_mask.unsqueeze(-1))).sum(dim=1) / (q_mask.sum(dim=1)).unsqueeze(-1) ## q_rep_agg.shape = [-1, BERT_SIZE]
            d_rep_agg = (d_rep*(d_mask.unsqueeze(-1))).sum(dim=1) / (d_mask.sum(dim=1)).unsqueeze(-1)
        ## TODO rep_type=last2_avg

        if self.whitening and not self.whitening_tokenwise:
            q_rep_agg = whitening(q_rep_agg, self.whitening_mu_file, self.whitening_cov_file, self.whitening_k)
            d_rep_agg = whitening(d_rep_agg, self.whitening_mu_file, self.whitening_cov_file, self.whitening_k)

        cos = self.cos(q_rep_agg, d_rep_agg)
        # print("cos.shape", cos.shape)
        if whitening_value_return:
            return cos, [q_reps[-1] * (q_mask.unsqueeze(-1)), d_reps[-1]*(d_mask.unsqueeze(-1))]
        elif value_return:
            return cos, [q_rep*(q_mask.unsqueeze(-1)), d_rep*(d_mask.unsqueeze(-1)), q_rep_agg, d_rep_agg]
        else:
            return cos

class NICERepBertRanker(BertBiencoderRanker):
    def __init__(self, args, without_bert=False):
        super().__init__(without_bert=without_bert, bert_model=args.model_card)
        self.model_card=args.model_card
        self.rep_type = args.rep_type
        self.cos = torch.nn.CosineSimilarity()

        self.whitening_flow = args.whitening_flow
        self.flow_whitening = args.flow_whitening

        if self.whitening_flow or self.flow_whitening:
            self.whitening_mu_file = args.whitening_mu_file
            self.whitening_cov_file = args.whitening_cov_file
            if args.whitening_k is None:
                self.whitening_k = self.BERT_SIZE
            else:
                self.whitening_k = args.whitening_k

        ## dim_red
        if args.dim_red == 0:
            self.dim_red = self.BERT_SIZE
        else:
            self.dim_red = args.dim_red
            self.dim_red_layer = torch.nn.Linear(self.BERT_SIZE, self.dim_red)

        ## NICE
        self.hetero_flow = args.hetero_flow
        self.nice_layernorm = args.nice_layernorm
        self.flow_mf = args.flow_mf
        
        self.nice = NICEModel(self.dim_red, args.nice_nhidden, args.nice_nlayers).to(self.bert.device)
        ## NICE layernorm
        if self.nice_layernorm:
            print("using NICE layernorm")
            self.nice_layernorm_layer = torch.nn.LayerNorm(self.BERT_SIZE)
        ## hetero NICE
        if self.hetero_flow:
            print("using heterogeneous NICE models")
            self.nice2 = NICEModel(self.dim_red, args.nice_nhidden, args.nice_nlayers).to(self.bert.device)
            if self.nice_layernorm:
                print("using NICE layernorm")
                self.nice_layernorm_layer2 = torch.nn.LayerNorm(self.BERT_SIZE)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, value_return=False, whitening_value_return=False):
        q_reps, d_reps, q_mask, d_mask = self.encode_repbert(query_tok, query_mask, doc_tok, doc_mask, roberta=("roberta" in self.model_card)) ## q_reps[-1].shape=[bs, seq_len, 768]

        if self.dim_red != self.BERT_SIZE: ## if dim_red=True, reduce bert rep of dim 768 into args.dim_red
            q_rep = self.dim_red_layer(q_reps[-1]) ## q_rep.shape=[bs, seq_len, 128]
            d_rep = self.dim_red_layer(d_reps[-1])
        else:
            q_rep = q_reps[-1]
            d_rep = d_reps[-1]

        if self.flow_mf: ## NEED TO BE DEBUGGED
            ## mean -> NICE
            # q_rep = torch.mean(q_rep, 1)
            # d_rep = torch.mean(d_rep, 1)

            masked_q = q_rep*q_mask.unsqueeze(-1) ## masked_q.shape = [bs*2, seq_len, BERT_SIZE]
            masked_d = d_rep*d_mask.unsqueeze(-1)
            # print("masked_q.shape", masked_q.shape, "masked_d.shape", masked_d.shape)

            q_nonzero = masked_q[masked_q.nonzero(as_tuple=True)].view(-1, masked_q.shape[-1]) ## q_nonzero.shape = [~, BERT_SIZE]
            d_nonzero = masked_d[masked_d.nonzero(as_tuple=True)].view(-1, masked_d.shape[-1])
            # print("q_nonzero.shape", q_nonzero.shape)

            q_nonzero_split = list(torch.split(q_nonzero, list(q_mask.sum(1).long()), dim=0))
            d_nonzero_split = list(torch.split(d_nonzero, list(d_mask.sum(1).long()), dim=0))

            q_rep = torch.stack([torch.mean(t, dim=0) for t in q_nonzero_split]) ## q_rep.shape = [bs*2, BERT_SIZE]
            d_rep = torch.stack([torch.mean(t, dim=0) for t in d_nonzero_split])
            # print("q_rep.shape", q_rep.shape)

            # if self.dim_red != self.BERT_SIZE:
            #     q_rep = self.flow_dr_layer(q_rep) ## dimension reduction before NICE
            #     d_rep = self.flow_dr_layer(d_rep)

            ## whitening_flow
            if self.whitening_flow:
                bs=q_rep.shape[0]
                dim=q_rep.shape[-1]
                q_rep = whitening(q_rep.reshape(-1, dim), self.whitening_mu_file, self.whitening_cov_file, self.whitening_k).reshape(bs, -1, self.whitening_k)
                d_rep = whitening(d_rep.reshape(-1, dim), self.whitening_mu_file, self.whitening_cov_file, self.whitening_k).reshape(bs, -1, self.whitening_k)

            q_z = self.nice(q_rep) ## q_z.shape = [bs*2, BERT_SIZE]
            # print("q_z.shape", q_z.shape)
            if self.nice_layernorm:
                q_z = self.nice_layernorm_layer(q_z)
            
            if self.hetero_flow:
                d_z = self.nice2(d_rep)
                if self.nice_layernorm:
                    d_z = self.nice_layernorm_layer2(d_z)
            else:
                d_z = self.nice(d_rep)
                if self.nice_layernorm:
                    d_z = self.nice_layernorm_layer(d_z)

            ## flow_whitening
            if self.flow_whitening:
                q_z = whitening(q_z, self.whitening_mu_file, self.whitening_cov_file, self.whitening_k)
                d_z = whitening(d_z, self.whitening_mu_file, self.whitening_cov_file, self.whitening_k)

            cos = self.cos(q_z.view(q_z.shape[0], -1), d_z.view(d_z.shape[0], -1)) ## cos.shape = [bs*2]
            # print("cos.shape", cos.shape)
        else:
            ## NICE -> mean (default)
            # if self.dim_red != self.BERT_SIZE:
            #     q_rep = self.flow_dr_layer(q_rep) ## dimension reduction before NICE
            #     d_rep = self.flow_dr_layer(d_rep) ## d_rep.shape=[bs*2, seq_len, dim_red]

            # q_z = self.nice(q_rep.reshape(-1, self.dim_red))
            masked_q = q_rep*q_mask.unsqueeze(-1)
            masked_d = d_rep*d_mask.unsqueeze(-1)
            # print("masked_q.shape", masked_q.shape, "masked_d.shape", masked_d.shape)

            ## whitening_flow
            if self.whitening_flow:
                bs=masked_q.shape[0]
                dim=masked_q.shape[-1]
                masked_q = whitening(masked_q.reshape(-1, dim), self.whitening_mu_file, self.whitening_cov_file, self.whitening_k).reshape(bs, -1, self.whitening_k)
                masked_d = whitening(masked_d.reshape(-1, dim), self.whitening_mu_file, self.whitening_cov_file, self.whitening_k).reshape(bs, -1, self.whitening_k)

            ## Aggregate
            q_x = masked_q[masked_q.nonzero(as_tuple=True)].view(-1, masked_q.shape[-1])
            d_x = masked_d[masked_d.nonzero(as_tuple=True)].view(-1, masked_d.shape[-1])
            # print("q_x.shape", q_x.shape)
            q_z = self.nice(q_x)
            if self.nice_layernorm:
                q_z = self.nice_layernorm_layer(q_z)
            if self.hetero_flow:
                # d_z = self.nice2(d_rep.reshape(-1, self.dim_red))
                d_z = self.nice2(d_x)
                if self.nice_layernorm:
                    d_z = self.nice_layernorm_layer2(d_z)
            else:
                # d_z = self.nice(d_rep.reshape(-1, self.dim_red))
                d_z = self.nice(d_x)
                if self.nice_layernorm:
                    d_z = self.nice_layernorm_layer(d_z)
            # print("q_z.shape", q_z.shape)

            ## flow_whitening
            if self.flow_whitening:
                q_z = whitening(q_z, self.whitening_mu_file, self.whitening_cov_file, self.whitening_k)
                d_z = whitening(d_z, self.whitening_mu_file, self.whitening_cov_file, self.whitening_k)

            ## split nice output
            q_z_split = list(torch.split(q_z, list(q_mask.sum(1).long()), dim=0))
            d_z_split = list(torch.split(d_z, list(d_mask.sum(1).long()), dim=0))
            if self.rep_type == "cls":
                # q_rep = q_z.reshape(q_rep.shape[0], -1, self.dim_red)[:, 0, :]
                # q_rep = d_z.reshape(d_rep.shape[0], -1, self.dim_red)[:, 0, :]
                q_rep = torch.stack([t[0] for t in q_z_split])
                d_rep = torch.stack([t[0] for t in d_z_split])
                # print("q_rep.shape", q_rep.shape, "d_rep.shape", d_rep.shape)
            elif self.rep_type == "last_avg":
                # q_rep = torch.mean(q_z.reshape(q_rep.shape[0], -1, self.dim_red), 1)
                # d_rep = torch.mean(d_z.reshape(d_rep.shape[0], -1, self.dim_red), 1)
                q_rep = torch.stack([torch.mean(t, dim=0) for t in q_z_split])
                d_rep = torch.stack([torch.mean(t, dim=0) for t in d_z_split])
                # print("q_rep.shape", q_rep.shape, "d_rep.shape", d_rep.shape)
            cos = self.cos(q_rep, d_rep)
        
        if whitening_value_return:
            return cos, [q_z, d_z]
        elif value_return:
            return cos, (q_z, d_z, q_rep, d_rep), (q_mask, d_mask)
        else:
            return cos, (q_z, d_z), (q_mask, d_mask)

class GlowRepBertRanker(BertBiencoderRanker):
    def __init__(self, args, without_bert=False):
        super().__init__(without_bert=without_bert, bert_model=args.model_card)
        self.model_card=args.model_card
        self.rep_type = args.rep_type
        self.cos = torch.nn.CosineSimilarity()

        self.whitening_flow = args.whitening_flow
        self.flow_whitening = args.flow_whitening

        if self.whitening_flow or self.flow_whitening:
            self.whitening_mu_file = args.whitening_mu_file
            self.whitening_cov_file = args.whitening_cov_file
            if args.whitening_k is None:
                self.whitening_k = self.BERT_SIZE
            else:
                self.whitening_k = args.whitening_k

        ## dim_red
        if args.dim_red == 0:
            self.dim_red = self.BERT_SIZE
        else:
            self.dim_red = args.dim_red
            self.dim_red_layer = torch.nn.Linear(self.BERT_SIZE, self.dim_red)

        ## Glow
        self.hetero_flow = args.hetero_flow
        self.flow_mf = args.flow_mf
        print("self.bert.device", self.bert.device, "flow_mf", args.flow_mf)
        
        self.glow = Glow(self.BERT_SIZE, args.n_flow, args.n_block, two_conv=args.glow_two_conv).to(self.bert.device)
        ## hetero Glow
        if self.hetero_flow:
            print("using heterogeneous Glow models")
            self.glow2 = Glow(self.BERT_SIZE, args.n_flow, args.n_block, two_conv=args.glow_two_conv).to(self.bert.device)
        
    def forward(self, query_tok, query_mask, doc_tok, doc_mask, value_return=False, whitening_value_return=False):
        q_reps, d_reps, q_mask, d_mask = self.encode_repbert(query_tok, query_mask, doc_tok, doc_mask, roberta=("roberta" in self.model_card)) ## q_reps[-1].shape=[bs, seq_len, 768]

        if self.dim_red != self.BERT_SIZE: ## if dim_red=True, reduce bert rep of dim 768 into args.dim_red
            q_rep = self.dim_red_layer(q_reps[-1]) ## q_rep.shape=[bs, seq_len, 128]
            d_rep = self.dim_red_layer(d_reps[-1])
        else:
            q_rep = q_reps[-1]
            d_rep = d_reps[-1]

        if self.flow_mf: ## NEED TO BE DEBUGGED
            ## mean -> Glow
            # q_rep = torch.mean(q_rep, 1)
            # d_rep = torch.mean(d_rep, 1)

            masked_q = q_rep*q_mask.unsqueeze(-1)
            masked_d = d_rep*d_mask.unsqueeze(-1)
            # print("masked_q.shape", masked_q.shape, "masked_d.shape", masked_d.shape)

            q_nonzero = masked_q[masked_q.nonzero(as_tuple=True)].view(-1, masked_q.shape[-1])
            d_nonzero = masked_d[masked_d.nonzero(as_tuple=True)].view(-1, masked_d.shape[-1])
            # print("q_nonzero.shape", q_nonzero.shape)

            q_nonzero_split = list(torch.split(q_nonzero, list(q_mask.sum(1).long()), dim=0))
            d_nonzero_split = list(torch.split(d_nonzero, list(d_mask.sum(1).long()), dim=0))

            q_rep = torch.stack([torch.mean(t, dim=0) for t in q_nonzero_split])
            d_rep = torch.stack([torch.mean(t, dim=0) for t in d_nonzero_split])

            # if self.dim_red != self.BERT_SIZE:
            #     q_rep = self.flow_dr_layer(q_rep) ## dimension reduction before NICE
            #     d_rep = self.flow_dr_layer(d_rep)

            ## whitening_flow
            if self.whitening_flow:
                bs=q_rep.shape[0]
                dim=q_rep.shape[-1]
                q_rep = whitening(q_rep.reshape(-1, dim), self.whitening_mu_file, self.whitening_cov_file, self.whitening_k).reshape(bs, -1, self.whitening_k)
                d_rep = whitening(d_rep.reshape(-1, dim), self.whitening_mu_file, self.whitening_cov_file, self.whitening_k).reshape(bs, -1, self.whitening_k)

            q_log_p_sum, q_logdet, q_z_outs = self.glow(q_rep.reshape(-1, self.dim_red, 1, 1))
            q_z = q_z_outs[-1].squeeze() ## q_z.shape = [bs*2, BERT_SIZE]
            # print("q_z.shape", q_z.shape)
            
            if self.hetero_flow:
                d_log_p_sum, d_logdet, d_z_outs = self.glow2(d_rep.reshape(-1, self.dim_red, 1, 1))
                d_z = d_z_outs[-1].squeeze()
            else:
                d_log_p_sum, d_logdet, d_z_outs = self.glow(d_rep.reshape(-1, self.dim_red, 1, 1))
                d_z = d_z_outs[-1].squeeze()

            ## flow_whitening
            if self.flow_whitening:
                q_z = whitening(q_z, self.whitening_mu_file, self.whitening_cov_file, self.whitening_k)
                d_z = whitening(d_z, self.whitening_mu_file, self.whitening_cov_file, self.whitening_k)

            cos = self.cos(q_z.view(q_z.shape[0], -1), d_z.view(d_z.shape[0], -1))
        else:
            ## NICE -> mean (default)
            # if self.dim_red != self.BERT_SIZE:
            #     q_rep = self.flow_dr_layer(q_rep) ## dimension reduction before NICE
            #     d_rep = self.flow_dr_layer(d_rep) ## d_rep.shape=[bs*2, seq_len, dim_red]

            # q_z = self.nice(q_rep.reshape(-1, self.dim_red))
            masked_q = q_rep*q_mask.unsqueeze(-1)
            masked_d = d_rep*d_mask.unsqueeze(-1)
            # print("masked_q.shape", masked_q.shape, "masked_d.shape", masked_d.shape)

            ## whitening_flow
            if self.whitening_flow:
                bs=masked_q.shape[0]
                dim=masked_q.shape[-1]
                whitened_q = whitening(masked_q.reshape(-1, dim), self.whitening_mu_file, self.whitening_cov_file, self.whitening_k).reshape(bs, -1, self.whitening_k)
                whitened_d = whitening(masked_d.reshape(-1, dim), self.whitening_mu_file, self.whitening_cov_file, self.whitening_k).reshape(bs, -1, self.whitening_k)

                q_x = whitened_q[masked_q.nonzero(as_tuple=True)].view(-1, masked_q.shape[-1])
                d_x = whitened_d[masked_d.nonzero(as_tuple=True)].view(-1, masked_d.shape[-1])
            else:
                ## Aggregate
                q_x = masked_q[masked_q.nonzero(as_tuple=True)].view(-1, masked_q.shape[-1])
                d_x = masked_d[masked_d.nonzero(as_tuple=True)].view(-1, masked_d.shape[-1])
            
            q_log_p_sum, q_logdet, q_z_outs = self.glow(q_x.reshape(-1, self.dim_red, 1, 1))
            q_z = q_z_outs[-1].squeeze() ## shape=[-1, BERT_SIZE]
            if self.hetero_flow:
                d_log_p_sum, d_logdet, d_z_outs = self.glow2(d_x.reshape(-1, self.dim_red, 1, 1))
                d_z = d_z_outs[-1].squeeze()
            else:
                d_log_p_sum, d_logdet, d_z_outs = self.glow(d_x.reshape(-1, self.dim_red, 1, 1))
                d_z = d_z_outs[-1].squeeze()
            # print("q_z.shape", q_z.shape)

            ## flow_whitening
            if self.flow_whitening:
                q_z = whitening(q_z, self.whitening_mu_file, self.whitening_cov_file, self.whitening_k)
                d_z = whitening(d_z, self.whitening_mu_file, self.whitening_cov_file, self.whitening_k)

            ## split nice output
            q_z_split = list(torch.split(q_z, list(q_mask.sum(1).long()), dim=0))
            d_z_split = list(torch.split(d_z, list(d_mask.sum(1).long()), dim=0))
            if self.rep_type == "cls":
                # q_rep = q_z.reshape(q_rep.shape[0], -1, self.dim_red)[:, 0, :]
                # q_rep = d_z.reshape(d_rep.shape[0], -1, self.dim_red)[:, 0, :]
                q_rep = torch.stack([t[0] for t in q_z_split])
                d_rep = torch.stack([t[0] for t in d_z_split])
                # print("q_rep.shape", q_rep.shape, "d_rep.shape", d_rep.shape)
            elif self.rep_type == "last_avg":
                # q_rep = torch.mean(q_z.reshape(q_rep.shape[0], -1, self.dim_red), 1)
                # d_rep = torch.mean(d_z.reshape(d_rep.shape[0], -1, self.dim_red), 1)
                q_rep = torch.stack([torch.mean(t, dim=0) for t in q_z_split])
                d_rep = torch.stack([torch.mean(t, dim=0) for t in d_z_split])
                # print("q_rep.shape", q_rep.shape, "d_rep.shape", d_rep.shape)
            cos = self.cos(q_rep, d_rep)
        
        if whitening_value_return:
            return cos, [q_z, d_z]
        elif value_return:
            return cos, (q_log_p_sum, q_logdet, q_z_outs[-1], d_log_p_sum, d_logdet, d_z_outs[-1], q_rep, d_rep)
        else:
            return cos, (q_log_p_sum, q_logdet, q_z_outs[-1], d_log_p_sum, d_logdet, d_z_outs[-1])


## Whitening function
def whitening(embeddings, mu_file, cov_file, k=None):
    """
    embeddings.shape = [-1, BERT_SIZE]
    """
    e_mask = embeddings.sum(-1) != 0

    mu = torch.load(mu_file).to(embeddings.device) ## shape=[BERT_SIZE]
    cov = torch.load(cov_file).to(embeddings.device) ## shape=[BERT_SIZE, BERT_SIZE]

    u, s, vt = torch.svd(cov)
    W = torch.mm(u, torch.diag(1/torch.sqrt(s)))[:, :k] ## W.shape = [BERT_SIZE, BERT_SIZE] -> [BERT_SIZE, k]
    W = W[:, :k]

    embeddings = torch.mm(embeddings - mu, W)
    embeddings *= e_mask.unsqueeze(-1)
    return embeddings


## Model map
MODEL_MAP = {
    'colbert': ColBertRanker,
    'nicecolbert': NICEColBertRanker,
    'glowcolbert': GlowColBertRanker,
    'repbert': RepBertRanker,
    'nicerepbert': NICERepBertRanker,
    'glowrepbert': GlowRepBertRanker,
    }