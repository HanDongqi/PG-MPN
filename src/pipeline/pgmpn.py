
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from src.language.foq import EFO1Query
from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.pipeline.reasoner import Reasoner
from src.griffin.griffin import Griffin, Gated_MLP_block


class PGMPNLayer(nn.Module):
    """
    data format [batch, dim]
    """
    def __init__(self, hidden_dim, nbp: NeuralBinaryPredicate, layers=1, agg_func='mean', mlp_expansion_factor=3, rnn_expansion_factor=4/3):
        super(PGMPNLayer, self).__init__()
        self.nbp = nbp
        self.feature_dim = nbp.entity_embedding.size(1)

        self.hidden_dim = hidden_dim
        self.num_entities = nbp.num_entities
        self.agg_func = agg_func

        # Griffin
        self.num_encoder_layers = layers
        self.mlp_expansion_factor = mlp_expansion_factor
        self.rnn_expansion_factor = rnn_expansion_factor

        self.griffin_layer = Griffin(
            D=self.feature_dim,
            depth=self.num_encoder_layers,
            mlp_expansion_factor=self.mlp_expansion_factor,
            rnn_expansion_factor=self.rnn_expansion_factor,
            device=nbp.device,
        )

        self.existential_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        self.universal_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        self.free_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        
        # Attention layers for hierarchical attention
        self.attention_layer_pos = nn.Linear(self.feature_dim, 1)
        self.attention_layer_neg = nn.Linear(self.feature_dim, 1)
        # Final attention layer to aggregate Griffin output
        self.final_attention_layer = nn.Sequential(
            Gated_MLP_block(
                D=self.feature_dim,
                expansion_factor=1
            ),
            nn.Linear(self.feature_dim, 1),
        )

        self.neg_tail_residual = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(), # Changed from Tanh
            nn.Linear(self.hidden_dim, self.feature_dim)
        )
        self.neg_head_residual = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(), # Changed from Tanh
            nn.Linear(self.hidden_dim, self.feature_dim)
        )

    def message_passing(self, term_emb_dict, atomic_dict, pred_emb_dict, inv_pred_emb_dict):

        term_collect_embs_dict = defaultdict(list)
        for predicate, atomic in atomic_dict.items():
            head_name, tail_name = atomic.head.name, atomic.tail.name
            head_emb = term_emb_dict[head_name]
            tail_emb = term_emb_dict[tail_name]
            sign = -1 if atomic.negated else 1

            pred_emb = pred_emb_dict[atomic.relation]
            if head_emb.size(0) == 1:
                head_emb = head_emb.expand(pred_emb.size(0), -1)
            if tail_emb.size(0) == 1:
                tail_emb = tail_emb.expand(pred_emb.size(0), -1)

            assert head_emb.size(0) == pred_emb.size(0)
            assert tail_emb.size(0) == pred_emb.size(0)

            tail_emb_transformed = self.nbp.estimate_tail_emb(head_emb, pred_emb)
            head_emb_transformed = self.nbp.estimate_head_emb(tail_emb, pred_emb)

            if sign == -1:
                # Apply reflection + residual transformation
                tail_emb_transformed = -tail_emb_transformed + self.neg_tail_residual(tail_emb_transformed)
                head_emb_transformed = -head_emb_transformed + self.neg_head_residual(head_emb_transformed)

            term_collect_embs_dict[tail_name].append(
                (tail_emb_transformed, sign)
            )

            term_collect_embs_dict[head_name].append(
                (head_emb_transformed, sign)
            )

        return term_collect_embs_dict

    def forward(self, init_term_emb_dict, predicates, pred_emb_dict, inv_pred_emb_dict):
        # message passing
        term_collect_embs_dict = self.message_passing(
            init_term_emb_dict, predicates, pred_emb_dict, inv_pred_emb_dict
        )

        # node embedding updating
        out_term_emb_dict = {}
        for t, collect_emb_list_with_signs in term_collect_embs_dict.items():
            if t not in ['f', 'e1', 'e2', 'e3'] and t in init_term_emb_dict : # Ensure t exists if it's not a variable node.
                out_term_emb_dict[t] = init_term_emb_dict[t]
                continue
            
            initial_node_emb = init_term_emb_dict[t]
            
            # Determine batch_size for potential expansion of initial_node_emb
            batch_size = -1
            if collect_emb_list_with_signs:
                batch_size = collect_emb_list_with_signs[0][0].size(0) # Batch size from first message's embedding
            elif initial_node_emb.size(0) > 1: # if initial_node_emb itself is batched
                batch_size = initial_node_emb.size(0)
            # If initial_node_emb is [1,D] and no messages, batch_size might remain -1 or be 1.
            # If batch_size is determined and > 1, and initial_node_emb is [1,D], expand it.
            if initial_node_emb.size(0) == 1 and batch_size > 1 :
                 initial_node_emb = initial_node_emb.expand(batch_size, -1)

            griffin_input_elements = [initial_node_emb.unsqueeze(1)] # Initial node emb as a sequence of length 1

            if collect_emb_list_with_signs:
                positive_messages = []
                negative_messages = []
                current_batch_size = initial_node_emb.size(0) # This is the definitive batch size for this term

                for msg_emb, sign in collect_emb_list_with_signs:
                    # Ensure message embedding has the correct batch size
                    if msg_emb.size(0) == 1 and current_batch_size > 1:
                        processed_msg_emb = msg_emb.expand(current_batch_size, -1)
                    elif msg_emb.size(0) == current_batch_size:
                        processed_msg_emb = msg_emb
                    else:
                         raise ValueError(f"Batch size mismatch for term {t}: initial_node_emb batch {current_batch_size}, message batch {msg_emb.size(0)}")

                    if sign == 1:
                        positive_messages.append(processed_msg_emb)
                    else: # sign == -1
                        negative_messages.append(processed_msg_emb)

                # Process Positive Messages
                if positive_messages:
                    positive_messages_stacked = torch.stack(positive_messages, dim=1) # [B, num_pos_messages, D]
                    positive_scores_raw = self.attention_layer_pos(positive_messages_stacked) # [B, num_pos_messages, 1]
                    positive_att_weights = F.softmax(positive_scores_raw, dim=1) # [B, num_pos_messages, 1]
                    weighted_positive_messages_seq = positive_messages_stacked * positive_att_weights # [B, num_pos_messages, D]
                    griffin_input_elements.append(weighted_positive_messages_seq)

                # Process Negative Messages
                if negative_messages:
                    negative_messages_stacked = torch.stack(negative_messages, dim=1) # [B, num_neg_messages, D]
                    negative_scores_raw = self.attention_layer_neg(negative_messages_stacked) # [B, num_neg_messages, 1]
                    negative_att_weights = F.softmax(negative_scores_raw, dim=1) # [B, num_neg_messages, 1]
                    weighted_negative_messages_seq = negative_messages_stacked * negative_att_weights # [B, num_neg_messages, D]
                    griffin_input_elements.append(weighted_negative_messages_seq)


            # Prepare input for the transformer by concatenating sequences
            # x will have shape [B, seq_len, D] where seq_len is 1 (initial_emb) + num_pos_messages + num_neg_messages
            x = torch.cat(griffin_input_elements, dim=1)

            # The Griffin model processes the sequence.
            # Griffin output shape is [B, seq_len, D]
            griffin_output = self.griffin_layer(x)

            # Apply final attention to the output of Griffin over the sequence dimension
            # griffin_output shape: [B, seq_len, D]
            final_scores_raw = self.final_attention_layer(griffin_output) # [B, seq_len, 1]
            final_att_weights = F.softmax(final_scores_raw, dim=1) # [B, seq_len, 1]
            # Weighted the sequence elements
            if self.agg_func == 'sum':
                agg_emb = (griffin_output * final_att_weights).sum(dim=1) # [B, D]
            elif self.agg_func =='mean':
                agg_emb = (griffin_output * final_att_weights).mean(dim=1) # [B, D]
            else:  # max pooling
                agg_emb = (griffin_output * final_att_weights).max(dim=1)[0] # [B, D]

            # Update the node embedding with the aggregated embedding
            out_term_emb_dict[t] = agg_emb

        # Handle terms that were not in term_collect_embs_dict (e.g. grounded entities not receiving messages)
        for t_name in init_term_emb_dict:
            if t_name not in out_term_emb_dict:
                out_term_emb_dict[t_name] = init_term_emb_dict[t_name]

        return out_term_emb_dict


class PGMPNReasoner(Reasoner):
    def __init__(self,
                 nbp: NeuralBinaryPredicate,
                 lgnn_layer: PGMPNLayer,
                 depth_shift=0):
        self.nbp = nbp
        self.lgnn_layer = lgnn_layer        # formula dependent
        self.depth_shift = depth_shift

        self.formula: EFO1Query = None
        self.term_local_emb_dict = {}

    def initialize_with_query(self, formula):
        self.formula = formula
        self.term_local_emb_dict = {term_name: None
                                    for term_name in self.formula.term_dict}

    def initialize_local_embedding(self):
        for term_name in self.formula.term_dict:
            if self.formula.has_term_grounded_entity_id_list(term_name):
                entity_id = self.formula.get_term_grounded_entity_id_list(term_name)
                emb = self.nbp.get_entity_emb(entity_id)
            elif self.formula.term_dict[term_name].is_existential:
                emb = self.lgnn_layer.existential_embedding
            elif self.formula.term_dict[term_name].is_free:
                emb = self.lgnn_layer.free_embedding
            elif self.formula.term_dict[term_name].is_universal:
                emb = self.lgnn_layer.universal_embedding
            else:
                raise KeyError(f"term name {term_name} cannot be initialized")
            self.set_local_embedding(term_name, emb)

    def estimate_variable_embeddings(self):
        self.initialize_local_embedding()
        term_emb_dict = self.term_local_emb_dict
        pred_emb_dict = {}
        inv_pred_emb_dict = {}
        for atomic_name in self.formula.atomic_dict:
            pred_name = self.formula.atomic_dict[atomic_name].relation
            if self.formula.has_pred_grounded_relation_id_list(pred_name):
                pred_emb_dict[pred_name] = self.get_rel_emb(pred_name)
                inv_pred_emb_dict[pred_name] = self.get_rel_emb(pred_name, inv=True)

        for _ in range(max(1, self.formula.quantifier_rank + self.depth_shift)):
            term_emb_dict = self.lgnn_layer(
                term_emb_dict,
                self.formula.atomic_dict,
                pred_emb_dict,
                inv_pred_emb_dict
            )

        for term_name in term_emb_dict:
            self.term_local_emb_dict[term_name] = term_emb_dict[term_name]
