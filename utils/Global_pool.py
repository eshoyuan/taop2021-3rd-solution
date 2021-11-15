import torch
from torch import nn


class PcamPool(nn.Module):

    def __init__(self):
        super(PcamPool, self).__init__()

    def forward(self, feat_map, logit_map):
        assert logit_map is not None

        prob_map = torch.sigmoid(logit_map)
        weight_map = prob_map / prob_map.sum(dim=2, keepdim=True)\
            .sum(dim=3, keepdim=True)
        feat = (feat_map * weight_map).sum(dim=2, keepdim=True)\
            .sum(dim=3, keepdim=True)
        weighted_feat = feat_map * weight_map
        # print('feat_map size: ', feat_map.size())
        # print('weight_map size: ', weight_map.size())
        # print('weighted_feat size', weighted_feat.size())

        return feat, weighted_feat


class LogSumExpPool(nn.Module):

    def __init__(self, gamma):
        super(LogSumExpPool, self).__init__()
        self.gamma = gamma

    def forward(self, feat_map):
        """
        Numerically stable implementation of the operation
        Arguments:
            feat_map(Tensor): tensor with shape (N, C, H, W)
            return(Tensor): tensor with shape (N, C, 1, 1)
        """
        (N, C, H, W) = feat_map.shape

        # (N, C, 1, 1) m
        m, _ = torch.max(
            feat_map, dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)

        # (N, C, H, W) value0
        value0 = feat_map - m
        area = 1.0 / (H * W)
        g = self.gamma

        # TODO: split dim=(-1, -2) for onnx.export
        return m + 1 / g * torch.log(area * torch.sum(
            torch.exp(g * value0), dim=(-1, -2), keepdim=True))


class ExpPool(nn.Module):

    def __init__(self):
        super(ExpPool, self).__init__()

    def forward(self, feat_map):
        """
        Numerically stable implementation of the operation
        Arguments:
            feat_map(Tensor): tensor with shape (N, C, H, W)
            return(Tensor): tensor with shape (N, C, 1, 1)
        """

        EPSILON = 1e-7
        (N, C, H, W) = feat_map.shape
        m, _ = torch.max(
            feat_map, dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)

        # caculate the sum of exp(xi)
        # TODO: split dim=(-1, -2) for onnx.export
        sum_exp = torch.sum(torch.exp(feat_map - m),
                            dim=(-1, -2), keepdim=True)

        # prevent from dividing by zero
        sum_exp += EPSILON

        # caculate softmax in shape of (H,W)
        exp_weight = torch.exp(feat_map - m) / sum_exp
        weighted_value = feat_map * exp_weight

        # TODO: split dim=(-1, -2) for onnx.export
        return torch.sum(weighted_value, dim=(-1, -2), keepdim=True)


class LinearPool(nn.Module):

    def __init__(self):
        super(LinearPool, self).__init__()

    def forward(self, feat_map):
        """
        Arguments:
            feat_map(Tensor): tensor with shape (N, C, H, W)
            return(Tensor): tensor with shape (N, C, 1, 1)
        """
        EPSILON = 1e-7
        (N, C, H, W) = feat_map.shape

        # sum feat_map's last two dimention into a scalar
        # so the shape of sum_input is (N,C,1,1)
        # TODO: split dim=(-1, -2) for onnx.export
        sum_input = torch.sum(feat_map, dim=(-1, -2), keepdim=True)

        # prevent from dividing by zero
        sum_input += EPSILON

        # caculate softmax in shape of (H,W)
        linear_weight = feat_map / sum_input
        weighted_value = feat_map * linear_weight

        # TODO: split dim=(-1, -2) for onnx.export
        return torch.sum(weighted_value, dim=(-1, -2), keepdim=True)


class GlobalPool(nn.Module):

    def __init__(self, pool):
        super(GlobalPool, self).__init__()
        self.cfg = pool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.exp_pool = ExpPool()
        self.pcampool = PcamPool()
        self.linear_pool = LinearPool()
        # self.lse_pool = LogSumExpPool(cfg.lse_gamma)

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, feat_map, logit_map):
        if self.cfg == 'AVG':
            return self.avgpool(feat_map)
        elif self.cfg == 'MAX':
            return self.maxpool(feat_map)
        elif self.cfg == 'PCAM':
            return self.pcampool(feat_map, logit_map)
        elif self.cfg == 'AVG_MAX':
            a = self.avgpool(feat_map)
            b = self.maxpool(feat_map)
            return torch.cat((a, b), 1)
        elif self.cfg == 'AVG_MAX_LSE':
            a = self.avgpool(feat_map)
            b = self.maxpool(feat_map)
            c = self.lse_pool(feat_map)
            return torch.cat((a, b, c), 1)
        elif self.cfg == 'EXP':
            return self.exp_pool(feat_map)
        elif self.cfg == 'LINEAR':
            return self.linear_pool(feat_map)
        elif self.cfg == 'LSE':
            return self.lse_pool(feat_map)
        else:
            raise Exception('Unknown pooling type : {}'
                            .format(self.cfg))
