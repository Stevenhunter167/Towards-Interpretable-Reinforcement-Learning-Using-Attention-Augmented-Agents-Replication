import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        """Initialize stateful ConvLSTM cell.
        
        Parameters
        ----------
        input_channels : ``int``
            Number of channels of input tensor.
        hidden_channels : ``int``
            Number of channels of hidden state.
        kernel_size : ``int``
            Size of the convolutional kernel.
            
        Paper
        -----
        https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf
        
        Referenced code
        ---------------
        https://github.com/automan000/Convolution_LSTM_PyTorch/blob/master/convolution_lstm.py        
        """
        super(ConvLSTMCell, self).__init__()
        
        assert hidden_channels % 2 == 0
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        
        self.padding = int((kernel_size - 1) / 2)
        
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        
        self.Wci = None
        self.Wcf = None
        self.Wco = None
        
        self.prev_hidden = None
    
    # @profile
    def forward(self, x):
        if self.prev_hidden is None:
            batch_size, _, height, width = x.size()
            h, c = self.init_hidden(batch_size, self.hidden_channels, height, width, x.device)
        else:
            h, c = self.prev_hidden
        
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        
        self.prev_hidden = ch, cc
        return ch, cc
    
    def reset(self):
        self.prev_hidden = None
    
    def init_hidden(self, batch_size, hidden, height, width, device):
        if self.Wci is None:
            self.Wci = torch.zeros(1, hidden, height, width, requires_grad=True).to(device)
            self.Wcf = torch.zeros(1, hidden, height, width, requires_grad=True).to(device)
            self.Wco = torch.zeros(1, hidden, height, width, requires_grad=True).to(device)
        return torch.zeros(batch_size, hidden, height, width, requires_grad=True).to(device), \
               torch.zeros(batch_size, hidden, height, width, requires_grad=True).to(device)


# 定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
    
    # @profile
    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class VisionNetwork(nn.Module):
    def __init__(self):
        super(VisionNetwork, self).__init__()
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(8, 8), stride=4, padding=1))
        self.conv_lstm = ConvLSTMCell(input_channels=32, hidden_channels=64, kernel_size=3)
    
    def reset(self):
        self.conv_lstm.reset()
    
    # @profile
    def forward(self, X):
        O, _ = self.conv_lstm(self.vision_cnn(X))
        return O

class ChannelWiseAttentionLayer(nn.Module):
    def __init__(self, channel, reduction, multiply=True):
        super(ChannelWiseAttentionLayer, self).__init__()
        self.multiply = multiply
        self.avgPooling = nn.AdaptiveAvgPool2d(1)
        self.maxPooling = nn.AdaptiveMaxPool2d(1)
        self.mfc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
        )
    
    def forward(self, observation):
        batch, channel, _, _ = observation.size()
        
        maxPooling = self.maxPooling(observation).view(batch, channel)
        avgPooling = self.avgPooling(observation).view(batch, channel)
        
        maxAttention = self.mfc(maxPooling).view(batch, channel, 1, 1)
        avgAttention = self.mfc(avgPooling).view(batch, channel, 1, 1)
        
        channelAttention = torch.sigmoid(maxAttention + avgAttention)
        
        if self.multiply:
            return observation * channelAttention
        else:
            return channelAttention

class SpatialWiseAttentionLayer(nn.Module):
    def __init__(self):
        super(SpatialWiseAttentionLayer, self).__init__()
        
        self.maxPooling = nn.MaxPool2d(kernel_size=(4, 4), stride=2, padding=2)
        self.avgPooling = nn.AvgPool2d(kernel_size=(4, 4), stride=2, padding=2)
        self.spatialAttention = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            # nn.Sigmoid()
        )


    def forward(self, observation):
        # in : 1 * 64 * 52 * 39
        # out : 1 *  64 *  27 * 20
        
        maxPooling = self.maxPooling(observation)
        avgPooling = self.avgPooling(observation)
        
        current_attention = torch.cat([maxPooling, avgPooling], dim=1)
        attention = self.spatialAttention(current_attention)
        
        return attention


class AttentionCore(nn.Module):
    def __init__(self):
        super(AttentionCore, self).__init__()
        
        self.channelAttention = ChannelWiseAttentionLayer(64, 2)
        self.spatialAttention = SpatialWiseAttentionLayer()
        self.attention_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    # @profile
    def forward(self, observation, last_attention):
        # in : 1 * 64 * 52 * 39
        # out : 1 * 64 *  27 * 20
        
        channelAttention = self.channelAttention(observation)
        spatialAttention = self.spatialAttention(channelAttention)
        attention = spatialAttention + last_attention
        attention = self.attention_layer(attention)
        
        return attention


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=2)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.resNet = nn.Sequential(
            ResBlock(in_channel=64, out_channel=128, stride=2),
            ResBlock(in_channel=128, out_channel=128),
            ResBlock(in_channel=128, out_channel=256, stride=2),
            ResBlock(in_channel=256, out_channel=256)
        )
        self.answer = nn.Sequential(nn.Flatten(), nn.Linear(8960, 4096), nn.Linear(4096, 2048))
    
    # @profile
    def forward(self, observation, current_attention):
        o = self.conv(observation)
        answer = self.resNet(self.conv1(o * current_attention))
        return self.answer(answer).reshape(1, 2048)


class Agent(nn.Module):
    def __init__(self, num_actions,
                 hidden_size: int = 256,
                 c_v: int = 120,
                 c_k: int = 8,
                 c_s: int = 64,
                 num_queries: int = 4,
                 ):
        """Agent implementing the attention agent.
        """
        super(Agent, self).__init__()
        self.hidden_size = hidden_size
        self.c_v, self.c_k, self.c_s, self.num_queries = c_v, c_k, c_s, num_queries
        
        self.prev_output = None
        self.prev_hidden = None
        
        self.legacy_attention = []
        
        self.vision = VisionNetwork()
        self.attention_core = AttentionCore()
        self.attention_lstm = ConvLSTMCell(input_channels=128, hidden_channels=64, kernel_size=3)
        self.resNet18 = ResNet18()
        self.policy_core = nn.LSTMCell(hidden_size, hidden_size)
        
        self.answer_processor = nn.Sequential(nn.Linear(2048, 512),nn.ReLU(),nn.Linear(512, hidden_size),)
        
        self.policy_head = nn.Sequential(nn.Linear(hidden_size, num_actions))
        self.values_head = nn.Sequential(nn.Linear(hidden_size, num_actions))
        
        self.last_attention = None
    
    def reset(self):
        self.vision.reset()
        self.attention_lstm.reset()
        self.prev_output = None
        self.prev_hidden = None
    
    # @profile
    def forward(self, X, video_recoder=None):         #X : 1, 210, 160, 3
        
        # 0. Setup.
        # ---------
        batch_size = X.size()[0]


            
        if self.prev_output is None:
            hidden_state = torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(X.device)
            self.prev_output = hidden_state
        
        #Vision
        # --------------
        X = X.transpose(1, 3).transpose(2,3)        #X: 1, 3, 210. 160
        O = self.vision(X)            #O: 1, 64, 52, 39
        
        # Attention
        if self.attention_lstm.prev_hidden is None:
            L_A = torch.ones((1,64, 27, 20), requires_grad=True).to(X.device)
        else:
            L_A, _ = self.attention_lstm.prev_hidden
        
        A = self.attention_core(O, L_A)  #A: 1, 128, 27, 20
        self.attention_lstm(A)   #L_A: 1, 64, 27, 20

        if video_recoder is not None:
            A_numpy = A.reshape(128, 27, 20).sum(0).reshape(27, 20, 1).repeat(1, 1, 3).detach().cpu().numpy()
            A_numpy = (A_numpy - np.min(A_numpy)) / (np.max(A_numpy) - np.min(A_numpy))
            A_numpy *= 255
            A_numpy = np.round(A_numpy).astype(np.uint8)
            video_recoder.record_frame(A_numpy)
        
        # answer
        answer = self.resNet18(O, A)
        answer = self.answer_processor(answer)
        
        # Policy.
        # ----------
        if self.prev_hidden is None:
            h, c = self.policy_core(answer)
        else:
            h, c = self.policy_core(answer, (self.prev_output, self.prev_hidden))
            self.prev_output, self.prev_hidden = h, c
        # (n, hidden_size)
        output = h
        
        # 4, 5. Outputs.
        # --------------
        # (n, num_actions)
        logits = self.policy_head(output)
        # (n, num_actions)
        values = self.values_head(output)
        return logits, values
