import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ResBlock(nn.Module):

    def __init__(self, channel=64):
        super(ResBlock, self).__init__()
        self.channel = channel

        self.res = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(self.channel, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(self.channel, affine=True)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.res(x)
        out += x  # self.shortcut(x)
        out = self.act(out)
        return out


class ResStrideBlock(nn.Module):

    def __init__(self, in_channel=64, out_channel=128):
        super(ResStrideBlock, self).__init__()
        self.in_dim = in_channel
        self.out_dim = out_channel

        self.res = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(self.out_dim, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(self.out_dim, affine=True)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1,
                      stride=2, bias=False),
            nn.InstanceNorm2d(self.out_dim, affine=True))
        self.act = nn.LeakyReLU()

    def forward(self, x):
        out = self.res(x)
        out += self.shortcut(x)
        out = self.act(out)
        return out


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

        self.Wxi = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whi = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxf = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whf = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxc = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whc = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxo = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Who = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )

        self.Wci = None
        self.Wcf = None
        self.Wco = None

        self.prev_hidden = None

    def forward(self, x):
        if self.prev_hidden is None:
            batch_size, _, height, width = x.size()
            h, c = self.init_hidden(
                batch_size, self.hidden_channels, height, width, x.device
            )
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
            self.Wci = torch.zeros(1, hidden, height, width, requires_grad=True).to(
                device
            )
            self.Wcf = torch.zeros(1, hidden, height, width, requires_grad=True).to(
                device
            )
            self.Wco = torch.zeros(1, hidden, height, width, requires_grad=True).to(
                device
            )
        return (
            torch.zeros(batch_size, hidden, height, width, requires_grad=True).to(
                device
            ),
            torch.zeros(batch_size, hidden, height, width, requires_grad=True).to(
                device
            ),
        )


class VisionNetwork(nn.Module):
    def __init__(self):
        super(VisionNetwork, self).__init__()
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=(8, 8),
                stride=4,
                padding=1,  # Padding s.t. the output shapes match the paper.
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(4, 4),
                stride=2,
                padding=2,  # Padding s.t. the output shapes match the paper.
            ),
        )
        self.vision_lstm = ConvLSTMCell(input_channels=128, hidden_channels=128, kernel_size=3)
        self.resnet = nn.Sequential(ResBlock(channel=64),
                                    ResBlock(channel=64),
                                    ResStrideBlock(in_channel=64, out_channel=128),
                                    ResBlock(channel=128))

    def reset(self):
        self.vision_lstm.reset()

    def forward(self, X):
        X = X.transpose(1, 3)
        O = self.vision_cnn(X)
        O = self.resnet(O)
        x, _ = self.vision_lstm(O )
        return O.transpose(1, 3)


class ResLSTM(nn.Module):
    def __init__(self):
        super(ResLSTM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.resnet = nn.Sequential(ResStrideBlock(in_channel=128, out_channel=256),
                                    ResBlock(channel=256),
                                    ResStrideBlock(in_channel=256, out_channel=512),
                                    ResBlock(channel=512))
        self.answer = nn.Sequential(nn.Flatten(), nn.Linear(6144, 4096), nn.Linear(4096, 2048))

    # def reset(self):
    #     self.vision_lstm.reset()

    def forward(self, O, A):
        # print(self.conv1(A).shape)
        x = self.resnet(O * self.conv1(A))
        # print(nn.Flatten()(x).shape)
        return self.answer(x).reshape(1, 2048)


class QueryNetwork(nn.Module):
    def __init__(self):
        super(QueryNetwork, self).__init__()
        # TODO: Add proper non-linearity.
        self.model = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 512)
        )

    def forward(self, query):
        out = self.model(query)
        return out.reshape(-1, 4, 128)


class SpatialBasis:
    # TODO: Implement Spatial.
    """
    NOTE: The `height` and `weight` depend on the inputs' size and its resulting size
    after being processed by the vision network.
    """

    def __init__(self, height=27, width=20, channels=64):
        h, w, d = height, width, channels

        p_h = torch.mul(torch.arange(1, h+1).unsqueeze(1).float(), torch.ones(1, w).float()) * (np.pi / h)
        p_w = torch.mul(torch.ones(h, 1).float(), torch.arange(1, w+1).unsqueeze(0).float()) * (np.pi / w)
        
        # NOTE: I didn't quite see how U,V = 4 made sense given that the authors form the spatial
        # basis by taking the outer product of the values. Still, I think what I have is aligned with what
        # they did, but I am less confident in this step.
        U = V = 8 # size of U, V. 
        u_basis = v_basis = torch.arange(1, U+1).unsqueeze(0).float()
        a = torch.mul(p_h.unsqueeze(2), u_basis)
        b = torch.mul(p_w.unsqueeze(2), v_basis)
        out = torch.einsum('hwu,hwv->hwuv', torch.cos(a), torch.cos(b)).reshape(h, w, d)
        self.S = out

    def __call__(self, X):
        # Stack the spatial bias (for each batch) and concat to the input.
        batch_size = X.size()[0]
        S = torch.stack([self.S] * batch_size).to(X.device)
        return torch.cat([X, S], dim=3)


def spatial_softmax(A):
    # A: batch_size x h x w x d
    b, h, w, d = A.size()
    # Flatten A s.t. softmax is applied to each grid (not over queries)
    A = A.reshape(b, h * w, d)
    A = F.softmax(A, dim=1)
    # Reshape A to original shape.
    A = A.reshape(b, h, w, d)
    return A


def apply_alpha(A, V):
    # TODO: Check this function again.
    b, h, w, c = A.size()
    A = A.reshape(b, h * w, c).transpose(1, 2)

    _, _, _, d = V.size()
    V = V.reshape(b, h * w, d)

    return torch.matmul(A, V)


class Agent(nn.Module):
    def __init__(
        self,
        num_actions,
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

        self.vision = VisionNetwork()
        self.reslstm = ResLSTM()
        self.query = QueryNetwork()
        # TODO: Implement SpatialBasis.
        self.spatial = SpatialBasis()

        self.answer_processor = nn.Sequential(
            # 1026 x 512
            nn.Linear(
                2048, 512
            ),
            nn.ReLU(),
            nn.Linear(512, hidden_size),
        )

        self.policy_core = nn.LSTMCell(hidden_size, hidden_size)
        self.prev_output = None
        self.prev_hidden = None

        self.policy_head = nn.Sequential(nn.Linear(hidden_size, num_actions))
        self.values_head = nn.Sequential(nn.Linear(hidden_size, num_actions))

    def reset(self):
        # self.reslstm.reset()
        self.vision.reset()
        self.prev_output = None
        self.prev_hidden = None

    def forward(self, X, prev_reward=None, prev_action=None, video_recoder=None):

        # 0. Setup.
        # ---------
        batch_size = X.size()[0]
        if prev_reward is None:
            # (n, 1, 1)
            prev_reward = torch.stack([torch.zeros(1, 1)] * batch_size).to(X.device)
        else:
            prev_reward = prev_reward.unsqueeze(1).unsqueeze(1)
        if prev_action is None:
            # (n, 1, 1)
            prev_action = torch.stack([torch.zeros(1, 1)] * batch_size).to(X.device)
        else:
            prev_action = prev_action.unsqueeze(1).unsqueeze(1)
        # 1 (a). Vision.
        # --------------

        # (n, h, w, c_k + c_v)
        O = self.vision(X)

        # 1 (b). Queries.
        # --------------
        if self.prev_output is None:
            hidden_state = torch.zeros(
                batch_size, self.hidden_size, requires_grad=True
            ).to(X.device)
            self.prev_output = hidden_state
        # (n, h, w, num_queries, c_k + c_s)
        Q = self.query(self.prev_output)

        # 2. Answer.
        # ----------
        # (n, h, w, num_queries)

        # print(K.shape)
        # print(Q.transpose(2, 1).unsqueeze(1).shape)
        # print(O.shape, Q.shape)
        A = torch.matmul(O, Q.transpose(2, 1).unsqueeze(1))


        # (n, h, w, num_queries)
        A = spatial_softmax(A)
        
        if video_recoder is not None:
            A_numpy = A.reshape((14, 10, 4)).sum(2).reshape(14, 10, 1).repeat(1, 1, 3).detach().cpu().numpy()
            A_numpy = (A_numpy - np.min(A_numpy)) / (np.max(A_numpy) - np.min(A_numpy))
            A_numpy *= 255
            A_numpy = np.round(A_numpy).astype(np.uint8)
            video_recoder.record_frame(A_numpy)

        # print(A.shape)
        # (n, 1, 1, num_queries)
        # print(a.shape)

        # (n, (c_v + c_s) * num_queries + (c_k + c_s) * num_queries + 1 + 1)
        answer = self.reslstm(O.transpose(1, 3), A.transpose(1, 3))
        answer = self.answer_processor(answer)

        # 3. Policy.
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
