
import torch
import torch.nn as nn
import torch.nn.functional as F



# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class dashboardA3C(nn.Module):

    def __init__(self, feat_len, action_num, mark_num, enc_num, field_num, agg_num):
        '''
        input shape: BATCH_SIZE * MAX_NUM * FEAT_LEN
        state: current selected charts
        action + parameters
        '''
        super(dashboardA3C, self).__init__()

        self.enc_num = enc_num
        self.field_num = field_num
        self.agg_num = agg_num

        self.dash_feature_pooling = nn.LSTM(
            feat_len, 256, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(512)
        self.fusing = nn.Linear(10, 1)


        self.l1 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.value =  nn.Linear(128, 1)

        self.act_emb = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.act_pred = nn.Linear(128, action_num)

        self.mark_emb = nn.Linear(256, 128) # N * 128
        self.bn4 = nn.BatchNorm1d(128)
        self.mark_pred = nn.Linear(128, mark_num)

        self.x_emb = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.x_agg = nn.Linear(128, agg_num)
        ## can select no field
        self.x_pred = nn.Linear(128, field_num + 1)

        self.y_emb = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.y_agg = nn.Linear(128, agg_num)
        ## can select no field
        self.y_pred = nn.Linear(128, field_num + 1)

        self.color_emb = nn.Linear(256, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.color_agg = nn.Linear(128, agg_num)
        ## can select no field
        self.color_pred = nn.Linear(128, field_num + 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).

    def forward(self, x):
        # print(x.shape)
        x = x.to(device)
        x, (ho, co) = self.dash_feature_pooling(x.to(torch.float32))

        x = self.bn1(x.transpose(1, 2)) # [5, 512, 10]
        x = F.relu(x)
        x = self.fusing(x).squeeze(2)
        x = F.relu(x)

        x = self.l1(x)
        # x = self.bn2(x)
        shared_emb = F.relu(x)
        # print(x.shape)

        value = self.value(shared_emb)

        emb = F.relu(self.act_emb(shared_emb))
        # emb = self.bn3(emb)
        act = self.act_pred(emb.view(emb.size(0), -1))
        
        emb = F.relu(self.x_emb(torch.cat([shared_emb, emb], axis = 1)))
        # emb = self.bn5(emb)
        x_topic_enc = self.x_pred(emb.view(emb.size(0), -1))

        emb = F.relu(self.mark_emb(torch.cat([shared_emb, emb], axis = 1)))
        # emb = self.bn4(emb)
        mark = self.mark_pred(emb.view(emb.size(0), -1))

        emb = F.relu(self.x_emb(torch.cat([shared_emb, emb], axis = 1)))
        # emb = self.bn5(emb)
        # x_enc = SoftmaxCategoricalHead()(self.x_pred(emb.view(emb.size(0), -1)))
        x_topic_agg = self.x_agg(emb.view(emb.size(0), -1))

        emb = F.relu(self.y_emb(torch.cat([shared_emb, emb], axis = 1)))
        # emb = self.bn6(emb)
        y_enc =self.y_pred(emb.view(emb.size(0), -1))
        y_agg = self.y_agg(emb.view(emb.size(0), -1))

        emb = F.relu(self.color_emb(torch.cat([shared_emb, emb], axis = 1)))
        # emb = self.bn7(emb)
        color_enc = self.color_pred(emb.view(emb.size(0), -1))
        color_agg = self.color_agg(emb.view(emb.size(0), -1))

        # print(act.shape, mark.shape)

        return (act, x_topic_enc, mark, y_enc, y_agg, x_topic_agg, color_enc, color_agg), value


if __name__ == '__main__':
    model = dashboardA3C(253, action_num = 2, mark_num = 4, enc_num = 3, field_num = 10, agg_num=4)

    input = torch.randn(5, 10, 253)

    res, value = model(input)
