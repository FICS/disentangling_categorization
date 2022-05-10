import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical

# EGG code: https://github.com/facebookresearch/EGG/blob/master/egg/core/gs_wrappers.py
def gumbel_softmax_sample(logits: torch.Tensor, temperature: float = 1.0, training: bool = True, straight_through: bool = False):
    size = logits.size()
    if not training:
        indexes = logits.argmax(dim=-1)
        one_hot = torch.zeros_like(logits).view(-1, size[-1])
        one_hot.scatter_(1, indexes.view(-1, 1), 1)
        one_hot = one_hot.view(*size)
        return one_hot

    sample = RelaxedOneHotCategorical(logits=logits, temperature=temperature).rsample()
    
    if straight_through:
        size = sample.size()
        indexes = sample.argmax(dim=-1)
        hard_sample = torch.zeros_like(sample).view(-1, size[-1])
        hard_sample.scatter_(1, indexes.view(-1, 1), 1)
        hard_sample = hard_sample.view(*size)

        sample = sample + (hard_sample - sample).detach()
        
    return sample
        

class RnnSenderGS(nn.Module):
    """
    Gumbel Softmax wrapper for Sender that outputs variable-length sequence of symbols.
    The user-defined `agent` takes an input and outputs an initial hidden state vector for the RNN cell;
    `RnnSenderGS` then unrolls this RNN for the `max_len` symbols. The end-of-sequence logic
    is supposed to be handled by the game implementation. Supports vanilla RNN ('rnn'), GRU ('gru'), and LSTM ('lstm')
    cells.
    """
    def __init__(
        self,
        input_size,
        vocab_size,
        hidden_size,
        max_len,
        temperature=1.0,
        embed_dim=64,
        cell='rnn',
        trainable_temperature=False,
        straight_through=False,
    ):
        super(RnnSenderGS, self).__init__()
        
        assert max_len >= 2, "Cannot have a max_len below 2"
        self.max_len = max_len
        
        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        self.agent = nn.Linear(input_size, hidden_size)
        
        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )

        self.straight_through = straight_through
        self.cell = None

        cell = cell.lower()

        if cell == "rnn":
            self.cell = nn.RNNCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "gru":
            self.cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)
        
    def choose_grad(self, mode, log=print):
        choices = ['on', 'off']
        assert mode in choices, f"Grad mode must be one of {choices}"
        if mode == 'on':
            for p in self.hidden_to_output.parameters():
                p.requires_grad = True
            for p in self.embedding.parameters():
                p.requires_grad = True
            for p in self.agent.parameters():
                p.requires_grad = True
            for p in self.cell.parameters():
                p.requires_grad = True
            self.sos_embedding.requires_grad = True
            if type(self.temperature) is torch.nn.Parameter:
                self.temperature.requires_grad = True
                
        if mode == 'off':
            for p in self.hidden_to_output.parameters():
                p.requires_grad = False
            for p in self.embedding.parameters():
                p.requires_grad = False
            for p in self.agent.parameters():
                p.requires_grad = False
            for p in self.cell.parameters():
                p.requires_grad = False
            self.sos_embedding.requires_grad = False
            if type(self.temperature) is torch.nn.Parameter:
                self.temperature.requires_grad = False
                
    
    def forward(self, sender_input):
        # representation, inner structure
        x, _ = sender_input
        prev_hidden = self.agent(x)
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
        sequence = []
        
        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = self.hidden_to_output(h_t)
            x = gumbel_softmax_sample(
                step_logits, self.temperature, self.training, self.straight_through
            )

            prev_hidden = h_t
            e_t = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0, 2)
        
        # zero out eos for experiment
        # sequence[:, :, 0] = 0

        eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
        eos[:, 0, 0] = 1
        sequence = torch.cat([sequence, eos], dim=1)

        return sequence


class FLRnnSenderGS(RnnSenderGS):
    """
    Fixed length RnnSender
    vocab_size: size without eos 
    """
    def __init__(
            self,
            input_size,
            vocab_size,
            hidden_size,
            max_len,
            temperature=1.0,
            embed_dim=64,
            cell='rnn',
            trainable_temperature=False,
            straight_through=False,
        ):
        super(FLRnnSenderGS, self).__init__(input_size,
                                                   vocab_size,
                                                   hidden_size,
                                                   max_len,
                                                   temperature,
                                                   embed_dim,
                                                   cell,
                                                   trainable_temperature,
                                                   straight_through)
        
    def forward(self, sender_input):
        # representation, inner structure
        x, _ = sender_input
        prev_hidden = self.agent(x)
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
        sequence = []
        
        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = self.hidden_to_output(h_t)
            x = gumbel_softmax_sample(
                step_logits, self.temperature, self.training, self.straight_through
            )

            prev_hidden = h_t
            e_t = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0, 2)
        
        eos_probas = torch.zeros_like(sequence[:, :, 0]).unsqueeze(2)
        # B x T x V -> B x T x V+1
        sequence = torch.cat([eos_probas, sequence], dim=2)
        
        eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
        eos[:, 0, 0] = 1
        # B x T x V+1 -> B x T+1 x V+1
        sequence = torch.cat([sequence, eos], dim=1)

        return sequence

    
class OLRnnSenderGS(RnnSenderGS):
    """
    1-length RnnSender
    vocab_size: size without eos 
    """
    def __init__(
            self,
            input_size,
            vocab_size,
            hidden_size,
            max_len,
            temperature=1.0,
            embed_dim=64,
            cell='rnn',
            trainable_temperature=False,
            straight_through=False,
        ):
        super(OLRnnSenderGS, self).__init__(input_size,
                                                   1,
                                                   hidden_size,
                                                   max_len,
                                                   temperature,
                                                   embed_dim,
                                                   cell,
                                                   trainable_temperature,
                                                   straight_through)
        
    def forward(self, sender_input):
        # representation, inner structure
        x, structure = sender_input

        sequence = []
        x = gumbel_softmax_sample(
            structure, self.temperature, self.training, self.straight_through
        )
        sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0, 2)
        
        eos_probas = torch.zeros_like(sequence[:, :, 0]).unsqueeze(2)
        # B x T x V -> B x T x V+1
        sequence = torch.cat([eos_probas, sequence], dim=2)
        
        eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
        eos[:, 0, 0] = 1
        # B x T x V+1 -> B x T+1 x V+1
        sequence = torch.cat([sequence, eos], dim=1)

        return sequence
    

class MultiHeadRnnSenderGS(RnnSenderGS):
    def __init__(
            self,
            input_size,
            structure_size,
            heads,
            vocab_size,
            hidden_size,
            max_len,
            temperature=1.0,
            embed_dim=64,
            cell='rnn',
            trainable_temperature=False,
            straight_through=False,
        ):
        super(MultiHeadRnnSenderGS, self).__init__(input_size,
                                                   vocab_size,
                                                   hidden_size,
                                                   max_len,
                                                   temperature,
                                                   embed_dim,
                                                   cell,
                                                   trainable_temperature,
                                                   straight_through)
        self.heads = heads
        self.structure_size = structure_size # + 1  # include eos
        self.toheads = torch.nn.Parameter(
            torch.rand(self.structure_size, self.structure_size * heads), requires_grad=True
        )
        # self.toheads = self.toheads.cuda()
        
    def forward(self, sender_input):
        # representation, inner structure
        x, structure = sender_input
        
        # multi head structure
        # heads = torch.matmul(structure, self.toheads.softmax(dim=0))
        # B x h x k
        # heads = heads.reshape(-1, self.heads, self.structure_size)
        
        # eos_concept = torch.zeros(structure.size(0), 1).cuda()
        # structure = torch.cat([structure, eos_concept], dim=1)
        heads = torch.matmul(structure, self.toheads.softmax(dim=0))
        heads = heads.reshape(-1, self.heads, self.structure_size)
        
        # self-attention weights
        raw_weights = torch.bmm(heads.transpose(1, 2), heads)
        # k x k
        weights = F.softmax(raw_weights, dim=2)
        
        # LM
        prev_hidden = self.agent(x)
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
        sequence = []
        
        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = self.hidden_to_output(h_t)
            x = gumbel_softmax_sample(
                step_logits, self.temperature, self.training, self.straight_through
            )

            prev_hidden = h_t
            e_t = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0, 2)

        # compute sequence weighted by self-attention
        # v = k+1
        # <b x k+1 x k+1, b x v x t>
        sequence = torch.bmm(weights, sequence.transpose(1, 2))
        # b x t x v
        sequence = sequence.transpose(2, 1)

        # address EOS
        eos_probas = torch.zeros_like(sequence[:, :, 0]).unsqueeze(2)
        # B x T x V -> B x T x V+1
        sequence = torch.cat([eos_probas, sequence], dim=2)
        
        eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
        eos[:, 0, 0] = 1
        # B x T x V+1 -> B x T+1 x V+1
        sequence = torch.cat([sequence, eos], dim=1)
        
        return sequence
    
    def choose_grad(self, mode, log=print):
        choices = ['on', 'off']
        assert mode in choices, f"Grad mode must be one of {choices}"
        if mode == 'on':
            for p in self.hidden_to_output.parameters():
                p.requires_grad = True
            for p in self.embedding.parameters():
                p.requires_grad = True
            for p in self.agent.parameters():
                p.requires_grad = True
            for p in self.cell.parameters():
                p.requires_grad = True
            self.sos_embedding.requires_grad = True
            if type(self.temperature) is torch.nn.Parameter:
                self.temperature.requires_grad = True
            
            self.toheads.requires_grad = True
                
        if mode == 'off':
            for p in self.hidden_to_output.parameters():
                p.requires_grad = False
            for p in self.embedding.parameters():
                p.requires_grad = False
            for p in self.agent.parameters():
                p.requires_grad = False
            for p in self.cell.parameters():
                p.requires_grad = False
            self.sos_embedding.requires_grad = False
            if type(self.temperature) is torch.nn.Parameter:
                self.temperature.requires_grad = False
                
            self.toheads.requires_grad = False
            
            
class MultiHeadRnnSenderGS2(MultiHeadRnnSenderGS):
    def forward(self, sender_input):
        # representation, inner structure
        x, structure = sender_input
        
        # multi head structure
        # heads = torch.matmul(structure, self.toheads.softmax(dim=0))
        # B x h x k
        # heads = heads.reshape(-1, self.heads, self.structure_size)
        
        # eos_concept = torch.zeros(structure.size(0), 1).cuda()
        # structure = torch.cat([structure, eos_concept], dim=1)
        heads = torch.matmul(structure, self.toheads.softmax(dim=0))
        heads = heads.reshape(-1, self.heads, self.structure_size)
        
        # self-attention weights
        # raw_weights = torch.bmm(heads.transpose(1, 2), heads)
        # k x k
        # weights = F.softmax(raw_weights, dim=2)
        
        # LM
        prev_hidden = self.agent(x)
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
        sequence = []
        
        for step in range(self.max_len):
            # weights = F.softmax(heads[:, step], dim=1)
            weights = heads[:, step]
            
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = self.hidden_to_output(h_t) * weights
            x = gumbel_softmax_sample(
                step_logits, self.temperature, self.training, self.straight_through
            )

            prev_hidden = h_t
            e_t = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0, 2)

        # compute sequence weighted by self-attention
        # v = k+1
        # <b x k+1 x k+1, b x v x t>
        # sequence = torch.bmm(weights, sequence.transpose(1, 2))
        # b x t x v
        # sequence = sequence.transpose(2, 1)

        # address EOS
        eos_probas = torch.zeros_like(sequence[:, :, 0]).unsqueeze(2)
        # B x T x V -> B x T x V+1
        sequence = torch.cat([eos_probas, sequence], dim=2)
        
        eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
        eos[:, 0, 0] = 1
        # B x T x V+1 -> B x T+1 x V+1
        sequence = torch.cat([sequence, eos], dim=1)
        
        return sequence
                

class ProtoSenderGS(nn.Module):
    """
    Gumbel Softmax wrapper for Sender that outputs variable-length sequence of symbols.
    The user-defined `agent` takes an input and outputs an initial hidden state vector for the RNN cell;
    `RnnSenderGS` then unrolls this RNN for the `max_len` symbols. The end-of-sequence logic
    is supposed to be handled by the game implementation. Supports vanilla RNN ('rnn'), GRU ('gru'), and LSTM ('lstm')
    cells.
    """
    def __init__(
        self,
        input_size,
        vocab_size,
        hidden_size,
        max_len,
        temperature=1.0,
        embed_dim=64,
        cell='rnn',
        trainable_temperature=False,
        straight_through=False,
    ):
        super(ProtoSenderGS, self).__init__()
        # embed_dim = vocab_size
        # hidden_size = vocab_size
        
        assert max_len >= 2, "Cannot have a max_len below 2"
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        # self.embedding = nn.Linear(vocab_size, embed_dim).cuda()
        # self.sos_embedding = nn.Parameter(torch.zeros(embed_dim)).cuda()
        self.agent = nn.Linear(input_size, hidden_size)
        
        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )

        self.straight_through = straight_through
        self.cell = None

        cell = cell.lower()

        if cell == "rnn":
            # self.cell = nn.RNNCell(input_size=vocab_size, hidden_size=vocab_size)
            self.cell = nn.RNNCell(input_size=vocab_size, hidden_size=hidden_size)
        elif cell == "gru":
            self.cell = nn.GRUCell(input_size=vocab_size, hidden_size=hidden_size)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(input_size=vocab_size, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

    def choose_grad(self, mode, log=print):
        choices = ['on', 'off']
        assert mode in choices, f"Grad mode must be one of {choices}"
        if mode == 'on':
            for p in self.hidden_to_output.parameters():
                p.requires_grad = True
            for p in self.agent.parameters():
                p.requires_grad = True
            for p in self.cell.parameters():
                p.requires_grad = True
            if type(self.temperature) is torch.nn.Parameter:
                self.temperature.requires_grad = True
                
        if mode == 'off':
            for p in self.hidden_to_output.parameters():
                p.requires_grad = False
            for p in self.agent.parameters():
                p.requires_grad = False
            for p in self.cell.parameters():
                p.requires_grad = False
            if type(self.temperature) is torch.nn.Parameter:
                self.temperature.requires_grad = False
                
    def forward(self, x):
        # B x I -> B x H
        prev_hidden = self.agent(x)
        
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM
        
        # e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
        # Treat output of proto module as learned SoS embedding
        step_logits = x
        
        x0 = gumbel_softmax_sample(
            step_logits, self.temperature, self.training, self.straight_through
        )
        # e_t = prev_hidden
        sequence = [x0]

        e_t = step_logits
        for step in range(self.max_len - 1):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)
            
            # step_logits = h_t
            # hidden (h_t) = vocab_size
            # B x H -> B x V
            step_logits = self.hidden_to_output(h_t)
            # B x V -> B x V_gs
            x = gumbel_softmax_sample(
                step_logits, self.temperature, self.training, self.straight_through
            )
            sequence.append(x)

            prev_hidden = h_t
            # gumbel(vocab_size) -> emb
            # e_t = x  # make RNN use gumbel distributed x
            e_t = step_logits  # make RNN use posterior distributed logits
            # e_t = self.embedding(x)
            # e_t = h_t

        # B x L x V
        sequence = torch.stack(sequence).permute(1, 0, 2)
        # debug('sequence', sequence.shape)
        # B x 1 x V
        eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
        eos[:, 0, 0] = 1
        
        # debug('eos', eos.shape)
        
        sequence = torch.cat([sequence, eos], dim=1)
        
        # debug('sender msg', sequence.shape) 
        # debug(sequence[0])

        return sequence
    

class ProtoSender2GS(RnnSenderGS):
    def __init__(
            self,
            input_size,
            vocab_size,
            hidden_size,
            max_len,
            temperature=1.0,
            embed_dim=64,
            cell='rnn',
            trainable_temperature=False,
            straight_through=False,
        ):
        super(ProtoSender2GS, self).__init__(input_size,
                                             vocab_size,
                                             hidden_size,
                                             max_len,
                                             temperature,
                                             embed_dim,
                                             cell,
                                             trainable_temperature,
                                             straight_through)
        self.embed_prototype = nn.Linear(input_size, embed_dim)
        self.cell = None

        cell = cell.lower()

        if cell == "rnn":
            self.cell = nn.RNNCell(input_size=embed_dim * 2, hidden_size=hidden_size)
        elif cell == "gru":
            self.cell = nn.GRUCell(input_size=embed_dim * 2, hidden_size=hidden_size)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(input_size=embed_dim * 2, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")
        
        self.reset_parameters()
        
    def forward(self, x):
        # take first prototype
        # B x topk+H x P
        prev_hidden = self.agent(x[:, 0])
        proto_embed = self.embed_prototype(x[:, 0])
        
        # B x H
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
        e_t = torch.cat([e_t, proto_embed], dim=1)
        sequence = []

        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = self.hidden_to_output(h_t)
            gs = gumbel_softmax_sample(
                step_logits, self.temperature, self.training, self.straight_through
            )

            prev_hidden = h_t
            # prev_hidden =  h_t * self.agent(x[:, step+1]).tanh()
            # prev_hidden = torch.matmul(h_t.)
            # prev_hidden = prev_hidden.tanh()
            # prev_hidden = torch.cat([h_t, x[step+1]], dim=1)
            
            proto_embed = self.embed_prototype(x[:, step+1])
            e_t = self.embedding(gs)
            e_t = torch.cat([e_t, proto_embed], dim=1)
            
            sequence.append(gs)

        sequence = torch.stack(sequence).permute(1, 0, 2)

        eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
        eos[:, 0, 0] = 1
        sequence = torch.cat([sequence, eos], dim=1)

        return sequence

    
class ProtoSender3GS(RnnSenderGS):
    def forward(self, x):
        # B x topk x P
        prototype_shape = x.shape[2]
        x = x[:, :10, ...].reshape((-1, prototype_shape*10))
        
        prev_hidden = self.agent(x)
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM

        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
        sequence = []
        
        for step in range(self.max_len):
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = self.hidden_to_output(h_t)
            x = gumbel_softmax_sample(
                step_logits, self.temperature, self.training, self.straight_through
            )

            prev_hidden = h_t
            e_t = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0, 2)

        eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
        eos[:, 0, 0] = 1
        sequence = torch.cat([sequence, eos], dim=1)

        return sequence


class Top1ReifiedRnnSenderGS(nn.Module):
    def __init__(
        self,
        input_size,
        sender_symbols,
        hidden_size,
        vocab_size,
        max_len,
        embed_dim=64,
        cell='rnn',
        temperature=1.0,
        trainable_temperature=False,
        straight_through=False,
    ):
        super(Top1ReifiedRnnSenderGS, self).__init__()
        
        # assert max_len >= 2, "Cannot have a max_len below 2"
        self.max_len = max_len
        self.matches = None
        self.recons = None
        
        self.vocab_size = vocab_size
        self.agent = nn.Linear(input_size, hidden_size//2)
        
        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.symbol_to_hidden = nn.Sequential(
            nn.Linear(64, hidden_size // 2),
            nn.ReLU(),
        )
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        self.symbols = torch.nn.Parameter(
            sender_symbols, requires_grad=False
        )
        
        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )

        self.straight_through = straight_through
        self.cell = None
        cell = cell.lower()

        if cell == "rnn":
            self.cell = nn.RNNCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "gru":
            self.cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)
        
    def choose_grad(self, mode, log=print):
        choices = ['on', 'off']
        assert mode in choices, f"Grad mode must be one of {choices}"
        if mode == 'on':
            for p in self.hidden_to_output.parameters():
                p.requires_grad = True
            for p in self.symbol_to_hidden.parameters():
                p.requires_grad = True
            for p in self.embedding.parameters():
                p.requires_grad = True
            for p in self.agent.parameters():
                p.requires_grad = True
            for p in self.cell.parameters():
                p.requires_grad = True
            self.sos_embedding.requires_grad = True
            if type(self.temperature) is torch.nn.Parameter:
                self.temperature.requires_grad = True
                
        if mode == 'off':
            for p in self.hidden_to_output.parameters():
                p.requires_grad = False
            for p in self.symbol_to_hidden.parameters():
                p.requires_grad = False
            for p in self.embedding.parameters():
                p.requires_grad = False
            for p in self.agent.parameters():
                p.requires_grad = False
            for p in self.cell.parameters():
                p.requires_grad = False
            self.sos_embedding.requires_grad = False
            if type(self.temperature) is torch.nn.Parameter:
                self.temperature.requires_grad = False
        
    def forward(self, sender_input):
        # representation, inner structure
        z, zp = sender_input
        sequence = []
        matches = []
        recons = []
        _, top_ix = torch.sort(z, dim=1, descending=True)
        
        prev_hidden = self.agent(z)
        
        # <---
        symbol_ix = top_ix[:, 0]
        matches.append(symbol_ix)
        chosen_symbol = self.symbols[symbol_ix]
        recons.append(chosen_symbol)

        h_tp = self.symbol_to_hidden(chosen_symbol.view(-1, 64))
        # h_tp = chosen_symbol.view(-1, 64)
        # prev_hidden = chosen_symbol.view(-1, 64)

        prev_hidden = torch.cat([prev_hidden, h_tp], dim=1)
        # prev_hidden = self.combiner(combined)
        # <---
            
        prev_c = torch.zeros_like(prev_hidden)  # only for LSTM
        e_t = torch.stack([self.sos_embedding] * prev_hidden.size(0))
        sequence = []
        
        for step in range(self.max_len):
            matches.append(symbol_ix)
            recons.append(chosen_symbol)
            
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c))
            else:
                h_t = self.cell(e_t, prev_hidden)

            step_logits = self.hidden_to_output(h_t)
            x = gumbel_softmax_sample(
                step_logits, self.temperature, self.training, self.straight_through
            )

            prev_hidden = h_t
            e_t = self.embedding(x)
            sequence.append(x)
        
        # add last seen symbol as eos signifier
        # recons.append(recons[-1])
        # matches.append(matches[-1])
        sequence = torch.stack(sequence).permute(1, 0, 2)
        matches = torch.stack(matches).permute(1, 0)
        recons = torch.stack(recons).permute(1, 0, 2, 3, 4)
        self.matches = matches
        self.recons = recons 
        
        eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
        eos[:, 0, 0] = 1
        sequence = torch.cat([sequence, eos], dim=1)

        return sequence
    
    

class DistractedReceiverAgent(nn.Module):
    def __init__(self, n_features, linear_units):
        super(DistractedReceiverAgent, self).__init__()
        # aux_dim x H matrix
        self.fc1 = nn.Linear(n_features, linear_units)

    def forward(self, x, recv_input):
        _input, _ = recv_input
        embedded_input = self.fc1(_input).tanh()
        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        return energies.squeeze()
    

class RnnReceiverGS(nn.Module):
    """
    Gumbel Softmax-based wrapper for Receiver agent in variable-length communication game. The user implemented logic
    is passed in `agent` and is responsible for mapping (RNN's hidden state + Receiver's optional input)
    into the output vector. Since, due to the relaxation, end-of-sequence symbol might have non-zero probability at
    each timestep of the message, `RnnReceiverGS` is applied for each timestep. The corresponding EOS logic
    is handled by `SenderReceiverRnnGS`.
    """

    def __init__(self, recv_agent, vocab_size, embed_dim, hidden_size, cell="rnn"):
        super(RnnReceiverGS, self).__init__()
        # self.agent = agent
        # self.agent = nn.Linear(hidden_size, num_distractors + 1)
        # self.agent = DistractedReceiver(aux_size, hidden_size)
        self.agent = recv_agent
        
        self.cell = None
        cell = cell.lower()
        if cell == "rnn":
            self.cell = nn.RNNCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "gru":
            self.cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.embedding = nn.Linear(vocab_size, embed_dim)

    def choose_grad(self, mode, log=print):
        choices = ['on', 'off']
        assert mode in choices, f"Grad mode must be one of {choices}"
        if mode == 'on':
            for p in self.embedding.parameters():
                p.requires_grad = True
            for p in self.agent.parameters():
                p.requires_grad = True
            for p in self.cell.parameters():
                p.requires_grad = True
                
        if mode == 'off':
            for p in self.embedding.parameters():
                p.requires_grad = False
            for p in self.agent.parameters():
                p.requires_grad = False
            for p in self.cell.parameters():
                p.requires_grad = False
                
    def forward(self, message, _input=None):
        outputs = []
        hiddens = []
        
        emb = self.embedding(message)

        prev_hidden = None
        prev_c = None

        # to get an access to the hidden states, we have to unroll the cell ourselves
        for step in range(message.size(1)):
            e_t = emb[:, step, ...]
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = (
                    self.cell(e_t, (prev_hidden, prev_c))
                    if prev_hidden is not None
                    else self.cell(e_t)
                )
            else:
                h_t = self.cell(e_t, prev_hidden)

            # outputs.append(self.agent(h_t, input))
            outputs.append(self.agent(h_t, _input))
            hiddens.append(h_t)
            
            # outputs.append(h_t)
            prev_hidden = h_t

        outputs = torch.stack(outputs).permute(1, 0, 2)
        # debug('outputs', outputs.shape)

        return outputs, hiddens

    
class FLRnnReceiverGS(RnnReceiverGS):
    """
    Gumbel Softmax-based wrapper for Receiver agent in fixed-length communication game. The user implemented logic
    is passed in `agent` and is responsible for mapping (RNN's hidden state + Receiver's optional input)
    into the output vector. Since, due to the relaxation, end-of-sequence symbol might have non-zero probability at
    each timestep of the message, `RnnReceiverGS` is applied for each timestep. The corresponding EOS logic
    is handled by `SenderReceiverRnnGS`.
    """
                
    def forward(self, message, _input=None):
        outputs = []
        hiddens = []
        
        # discard eos
        message = message[:, :, 1:]
        
        emb = self.embedding(message)

        prev_hidden = None
        prev_c = None

        # to get an access to the hidden states, we have to unroll the cell ourselves
        for step in range(message.size(1)):
            e_t = emb[:, step, ...]
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = (
                    self.cell(e_t, (prev_hidden, prev_c))
                    if prev_hidden is not None
                    else self.cell(e_t)
                )
            else:
                h_t = self.cell(e_t, prev_hidden)

            # outputs.append(self.agent(h_t, input))
            outputs.append(self.agent(h_t, _input))
            hiddens.append(h_t)
            
            # outputs.append(h_t)
            prev_hidden = h_t

        outputs = torch.stack(outputs).permute(1, 0, 2)
        # debug('outputs', outputs.shape)

        return outputs, hiddens
    
    
class ProtoReceiver2GS(RnnReceiverGS):
    def forward(self, message, _input=None):
        # _input: B x D x topk x P
        outputs = []
        hiddens = []
        
        # B x L x V -> B x L x E
        emb = self.embedding(message)

        prev_hidden = None
        prev_c = None

        # to get an access to the hidden states, we have to unroll the cell ourselves
        for step in range(message.size(1)):
            e_t = emb[:, step, ...]
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = (
                    self.cell(e_t, (prev_hidden, prev_c))
                    if prev_hidden is not None
                    else self.cell(e_t)
                )
            else:
                h_t = self.cell(e_t, prev_hidden)

            # outputs.append(self.agent(h_t, input))
            energies = self.agent(h_t, _input[:, :, step])
            outputs.append(energies)
            hiddens.append(h_t)
            
            # outputs.append(h_t)
            prev_hidden = h_t

        outputs = torch.stack(outputs).permute(1, 0, 2)
        # debug('outputs', outputs.shape)

        return outputs, hiddens
    
    
class Top1ReifiedRnnReceiverGS(nn.Module):
    def __init__(self, percept, vocab_size, signal_agent, embed_dim, hidden_size, cell="rnn"):
        super(Top1ReifiedRnnReceiverGS, self).__init__()
        self.decoder_agent = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.percept = percept
        self.num_symbols = percept.model.prototype_shape[0]
        self.symbol_channels = percept.model.prototype_shape[1]
        self.matches = None
        self.recons = None
        self.signal_agent = signal_agent
        self.apply_symbols = False
        
        self.cell = None
        cell = cell.lower()
        if cell == "rnn":
            self.cell = nn.RNNCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "gru":
            self.cell = nn.GRUCell(input_size=embed_dim, hidden_size=hidden_size)
        elif cell == "lstm":
            self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        self.embedding = nn.Linear(vocab_size, embed_dim)
    
    def choose_grad(self, mode, log=print):
        choices = ['on', 'off']
        assert mode in choices, f"Grad mode must be one of {choices}"
        if mode == 'on':
            for p in self.embedding.parameters():
                p.requires_grad = True
            for p in self.decoder_agent.parameters():
                p.requires_grad = True
            for p in self.signal_agent.parameters():
                p.requires_grad = True
            for p in self.cell.parameters():
                p.requires_grad = True
                
        if mode == 'off':
            for p in self.embedding.parameters():
                p.requires_grad = False
            for p in self.decoder_agent.parameters():
                p.requires_grad = False
            for p in self.signal_agent.parameters():
                p.requires_grad = False
            for p in self.cell.parameters():
                p.requires_grad = False

    def forward(self, message, _input=None):
        outputs = []
        hiddens = []
        matches = []
        recons = []
        
        recv_activations, _ = _input
        emb = self.embedding(message)

        prev_hidden = None
        prev_c = None

        # to get an access to the hidden states, we have to unroll the cell ourselves
        for step in range(message.size(1)):
            e_t = emb[:, step, ...]
            
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = (
                    self.cell(e_t, (prev_hidden, prev_c))
                    if prev_hidden is not None
                    else self.cell(e_t)
                )
            else:
                h_t = self.cell(e_t, prev_hidden)
            
            # <---
            recon_symbol = self.decoder_agent(h_t).view(-1, 64, 1, 1)
            recons.append(recon_symbol)
            
            min_distances = self.percept.model._l2_convolution(recon_symbol)
            min_distances = min_distances.view(-1, self.num_symbols)
            recon_activations = -min_distances  # linear activation
            matches.append(recon_activations.argmax(dim=1))
            # <---
            
            if self.apply_symbols:
                act = recv_activations * torch.unsqueeze(recon_activations, dim=1)
            else:
                act = recv_activations
            
            outputs.append(
                self.signal_agent(h_t, (act, None))
            )
            hiddens.append(h_t)
            prev_hidden = h_t
        
        outputs = torch.stack(outputs).permute(1, 0, 2)
        matches = torch.stack(matches).permute(1, 0)
        recons = torch.stack(recons).permute(1, 0, 2, 3, 4)
        
        self.matches = matches
        self.recons = recons
        
        return outputs, hiddens 