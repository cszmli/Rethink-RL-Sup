import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from base_modules import BaseRNN
from utils import FLOAT, LONG, cast_type
import logging

TEACH_FORCE = "teacher_forcing"
TEACH_GEN = "teacher_gen"
GEN = "gen"
GEN_VALID = 'gen_valid'


class DecoderRNN(BaseRNN):
    def __init__(self, vocab_size, max_len, input_size, hidden_size, sos_id,
                 eos_id, n_layers=1, rnn_cell='lstm', input_dropout_p=0,
                 dropout_p=0, use_attention=False, use_gpu=True, embedding=None, output_size=None,
                 tie_output_embed=False, cat_mlp=False):

        super(DecoderRNN, self).__init__(vocab_size, input_size,
                                         hidden_size, input_dropout_p,
                                         dropout_p, n_layers, rnn_cell, False)

        self.output_size = vocab_size if output_size is None else output_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.init_input = None
        self.use_gpu = use_gpu
        self.cat_mlp = cat_mlp

        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, self.input_size)
        else:
            self.embedding = embedding
        self.project = nn.Linear(self.hidden_size, self.output_size)
        self.function = F.log_softmax

    def forward_step(self, input_var, hidden, encoder_outputs, h_0):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        
        embedded = self.embedding(input_var)
        if self.cat_mlp:
            embedded = torch.cat([embedded, h_0], -1)

        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        attn = None

        output = output.contiguous()
        logits = self.project(output.view(-1, self.hidden_size))
        predicted_softmax = self.function(logits, dim=logits.dim()-1).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, batch_size, inputs=None, init_state=None,
                attn_context=None, mode=TEACH_FORCE, gen_type='greedy',
                beam_size=4):

        # sanity checks
        ret_dict = dict()
        h_0 = init_state.squeeze(0).unsqueeze(1).repeat(1,self.max_length-1, 1)
        if mode == GEN:
            inputs = None


        if gen_type != 'beam':
            beam_size = 1

        if inputs is not None:
            decoder_input = inputs
        else:
            # prepare the BOS inputs
            with torch.no_grad():
                bos_var = Variable(torch.LongTensor([self.sos_id]))
            bos_var = cast_type(bos_var, LONG, self.use_gpu)
            decoder_input = bos_var.expand(batch_size*beam_size, 1)
            # logging.info("dec input: {}".format(decoder_input.shape))
            h_0 = init_state.squeeze(0).unsqueeze(1).repeat(beam_size, 1, 1)  # 12, 1, 100
            # logging.info("h0: {}".format(h_0.shape))

        if mode == GEN and gen_type == 'beam':
            # if beam search, repeat the initial states of the RNN
            if self.rnn_cell is nn.LSTM:
                h, c = init_state
                decoder_hidden = (self.repeat_state(h, batch_size, beam_size),
                                  self.repeat_state(c, batch_size, beam_size))
            else:
                decoder_hidden = self.repeat_state(init_state,
                                                   batch_size, beam_size)
        else:
            decoder_hidden = init_state

        decoder_outputs = [] # a list of logprob
        sequence_symbols = [] # a list word ids
        back_pointers = [] # a list of parent beam ID
        lengths = np.array([self.max_length] * batch_size * beam_size)

        def decode(step, cum_sum, step_output, step_attn):
            decoder_outputs.append(step_output)
            step_output_slice = step_output.squeeze(1)

            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

            if gen_type == 'greedy':
                symbols = step_output_slice.topk(1)[1]
            elif gen_type == 'sample':
                symbols = self.gumbel_max(step_output_slice)
            elif gen_type == 'beam':
                if step == 0:
                    seq_score = step_output_slice.view(batch_size, -1)
                    seq_score = seq_score[:, 0:self.output_size]
                else:
                    seq_score = cum_sum + step_output_slice
                    seq_score = seq_score.view(batch_size, -1)

                top_v, top_id = seq_score.topk(beam_size)

                back_ptr = top_id.div(self.output_size).view(-1, 1)
                symbols = top_id.fmod(self.output_size).view(-1, 1)
                cum_sum = top_v.view(-1, 1)
                back_pointers.append(back_ptr)
            else:
                raise ValueError("Unsupported decoding mode")

            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > di) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return cum_sum, symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability,
        # the unrolling can be done in graph
        if mode == TEACH_FORCE:
            decoder_output, decoder_hidden, attn = self.forward_step(
                decoder_input, decoder_hidden, attn_context, h_0)

            # in teach forcing mode, we don't need symbols.
            decoder_outputs = decoder_output

        else:
            # do free running here
            cum_sum = None
            for di in range(self.max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(
                    decoder_input, decoder_hidden, attn_context, h_0)

                cum_sum, symbols = decode(di, cum_sum, decoder_output, step_attn)
                decoder_input = symbols

            decoder_outputs = torch.cat(decoder_outputs, dim=1)

            if gen_type == 'beam':
                # do back tracking here to recover the 1-best according to
                # beam search.
                final_seq_symbols = []
                cum_sum = cum_sum.view(-1, beam_size)
                max_seq_id = cum_sum.topk(1)[1].data.cpu().view(-1).numpy()
                rev_seq_symbols = sequence_symbols[::-1]
                rev_back_ptrs = back_pointers[::-1]

                for symbols, back_ptrs in zip(rev_seq_symbols, rev_back_ptrs):
                    symbol2ds = symbols.view(-1, beam_size)
                    back2ds = back_ptrs.view(-1, beam_size)
                    # logging.info(symbol2ds)
                    selected_symbols = []
                    selected_parents =[]
                    for b_id in range(batch_size):
                        selected_parents.append(back2ds[b_id, max_seq_id[b_id]])
                        selected_symbols.append(symbol2ds[b_id, max_seq_id[b_id]])
                    # logging.info(selected_symbols)
                    final_seq_symbols.append(torch.stack(selected_symbols, 0).unsqueeze(1))
                    max_seq_id = torch.stack(selected_parents).data.cpu().numpy()
                sequence_symbols = final_seq_symbols[::-1]

        # save the decoded sequence symbols and sequence length
        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict




