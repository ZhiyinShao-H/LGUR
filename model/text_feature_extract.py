from torch import nn
import torch
import transformers as ppb

class TextExtract(nn.Module):

    def __init__(self, opt):
        super(TextExtract, self).__init__()

        self.opt = opt
        self.last_lstm = opt.last_lstm
        self.embedding = nn.Embedding(opt.vocab_size, 512, padding_idx=0)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(512, 384, num_layers=1, bidirectional=True, bias=False)

    def forward(self, caption_id, text_length):

        text_embedding = self.embedding(caption_id)
        text_embedding = self.dropout(text_embedding)
        feature = self.calculate_different_length_lstm(text_embedding, text_length, self.lstm)

        # feature = feature.unsqueeze(2).unsqueeze(2)

        return feature

    def calculate_different_length_lstm(self, text_embedding, text_length, lstm):

        text_length = text_length.view(-1)
        _, sort_index = torch.sort(text_length, dim=0, descending=True)
        _, unsort_index = sort_index.sort()

        sortlength_text_embedding = text_embedding[sort_index, :]
        sort_text_length = text_length[sort_index]
        # print(sort_text_length)
        packed_text_embedding = nn.utils.rnn.pack_padded_sequence(sortlength_text_embedding,
                                                                  sort_text_length,
                                                                  batch_first=True)

        packed_feature, [hn, _] = lstm(packed_text_embedding)  # [hn, cn]
        sort_feature = nn.utils.rnn.pad_packed_sequence(packed_feature, batch_first=True)  # including[feature, length]
        # print(hn.size(), cn.size())

        if self.last_lstm:
            hn = torch.cat([hn[0, :, :], hn[1, :, :]], dim=1)[unsort_index, :]
            return hn
        else:
            unsort_feature = sort_feature[0][unsort_index, :]
            unsort_feature = (unsort_feature[:, :, :int(unsort_feature.size(2) / 2)]
                              + unsort_feature[:, :, int(unsort_feature.size(2) / 2):]) / 2
            # print(text_length[9])
            # print(unsort_feature[9,text_length[9]])
            # print(unsort_feature[9, text_length[9]-1])
            # feature, _ = unsort_feature.max(dim=1)
            """
            mean_feature = []
            for i in range(len(text_length)):
                mean_feature.append(torch.mean(unsort_feature[i, :text_length[i], :], dim=0).unsqueeze(0))
            mean_feature = torch.cat(mean_feature, dim=0)
            """
            return unsort_feature

class TextExtract_Bert_lstm(nn.Module):
    def __init__(self, args):
        super(TextExtract_Bert_lstm, self).__init__()

        # self.model_txt = Vit_text(768)
        self.last_lstm = args.last_lstm
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        self.text_embed = model_class.from_pretrained(pretrained_weights)
        self.text_embed.eval()
        for p in self.text_embed.parameters():
            p.requires_grad = False
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(768, 384, num_layers=1, bidirectional=True, bias=False)

    def forward(self, txt, mask):
        length = mask.sum(1)
        length = length.cpu()
        with torch.no_grad():
            txt = self.text_embed(txt, attention_mask=mask)#
            txt = txt[0]   ##64 * L * 768
            # txt = txt.unsqueeze(1)
            # txt = txt.permute(0, 3, 1, 2) ##64 * 768 * 1 * 64
        # txt = self.model_txt(txt , trans_mask)  # txt4: batch x 2048 x 1 x 64

        txt = self.calculate_different_length_lstm(txt,length,self.lstm)
        return txt

    def calculate_different_length_lstm(self, text_embedding, text_length, lstm):

        text_length = text_length.view(-1)
        _, sort_index = torch.sort(text_length, dim=0, descending=True)
        _, unsort_index = sort_index.sort()

        sortlength_text_embedding = text_embedding[sort_index, :]
        sort_text_length = text_length[sort_index]
        # print(sort_text_length)
        packed_text_embedding = nn.utils.rnn.pack_padded_sequence(sortlength_text_embedding,
                                                                  sort_text_length,
                                                                  batch_first=True)

        packed_feature, [hn, _] = lstm(packed_text_embedding)  # [hn, cn]
        sort_feature = nn.utils.rnn.pad_packed_sequence(packed_feature, batch_first=True)  # including[feature, length]
        # print(hn.size(), cn.size())

        if self.last_lstm:
            hn = torch.cat([hn[0, :, :], hn[1, :, :]], dim=1)[unsort_index, :]
            return hn
        else:
            unsort_feature = sort_feature[0][unsort_index, :]
            unsort_feature = (unsort_feature[:, :, :int(unsort_feature.size(2) / 2)]
                              + unsort_feature[:, :, int(unsort_feature.size(2) / 2):]) / 2
            # print(text_length[9])
            # print(unsort_feature[9,text_length[9]])
            # print(unsort_feature[9, text_length[9]-1])
            # feature, _ = unsort_feature.max(dim=1)
            """
            mean_feature = []
            for i in range(len(text_length)):
                mean_feature.append(torch.mean(unsort_feature[i, :text_length[i], :], dim=0).unsqueeze(0))
            mean_feature = torch.cat(mean_feature, dim=0)
            """
            return unsort_feature