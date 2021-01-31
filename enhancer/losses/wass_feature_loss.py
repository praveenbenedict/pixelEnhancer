from enhancer import *
from .loss_utils import *


class WassFeatureLoss(nn.Module):
    def __init__(
        self, layer_wgts=[5, 15, 2], wass_wgts=[3.0, 0.7, 0.01], use_input_norm=True,loss_wgts =[1e-2,1,1]
    ):

        super().__init__()
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        self.m_feat = torchvision.models.vgg16_bn(True).features.to(device).eval()
        for _, v in self.m_feat.named_parameters():
            v.requires_grad = False
        blocks = [
            i - 1
            for i, o in enumerate(children(self.m_feat))
            if isinstance(o, nn.MaxPool2d)
        ]
        layer_ids = blocks[2:5]
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = [SaveFeatures(i) for i in self.loss_features]
        self.wgts = layer_wgts
        self.wass_wgts = wass_wgts
        self.loss_wgts = loss_wgts
        self.metric_names = (
            ["pixel"]
            + [f"feat_{i}" for i in range(len(layer_ids))]
            + [f"wass_{i}" for i in range(len(layer_ids))]
        )
        self.base_loss = F.l1_loss

    def _make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.stored.clone() if clone else o.stored) for o in self.hooks]

    def _calc_2_moments(self, tensor):
        chans = tensor.shape[1]
        tensor = tensor.view(1, chans, -1)
        n = tensor.shape[2]
        mu = tensor.mean(2)
        tensor = (tensor - mu[:, :, None]).squeeze(0)
        # Prevents nasty bug that happens very occassionally- divide by zero.  Why such things happen?
        if n == 0:
            return None, None
        cov = torch.mm(tensor, tensor.t()) / float(n)
        return mu, cov

    def _get_style_vals(self, tensor):
        mean, cov = self._calc_2_moments(tensor)
        if mean is None:
            return None, None, None
        eigvals, eigvects = torch.symeig(cov, eigenvectors=True)
        eigroot_mat = torch.diag(torch.sqrt(eigvals.clamp(min=0)))
        root_cov = torch.mm(torch.mm(eigvects, eigroot_mat), eigvects.t())
        tr_cov = eigvals.clamp(min=0).sum()
        return mean, tr_cov, root_cov

    def _calc_l2wass_dist(
        self, mean_stl, tr_cov_stl, root_cov_stl, mean_synth, cov_synth
    ):
        tr_cov_synth = torch.symeig(cov_synth, eigenvectors=True)[0].clamp(min=0).sum()
        mean_diff_squared = (mean_stl - mean_synth).pow(2).sum()
        cov_prod = torch.mm(torch.mm(root_cov_stl, cov_synth), root_cov_stl)
        var_overlap = torch.sqrt(
            torch.symeig(cov_prod, eigenvectors=True)[0].clamp(min=0) + 1e-8
        ).sum()
        dist = mean_diff_squared + tr_cov_stl + tr_cov_synth - 2 * var_overlap
        return dist

    def _single_wass_loss(self, pred, targ):
        mean_test, tr_cov_test, root_cov_test = targ
        mean_synth, cov_synth = self._calc_2_moments(pred)
        loss = self._calc_l2wass_dist(
            mean_test, tr_cov_test, root_cov_test, mean_synth, cov_synth
        )
        return loss

    def forward(self, input, target):
        if self.use_input_norm:
            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std
        out_feat = self._make_features(target, clone=True)
        in_feat = self._make_features(input)
        self.feat_losses = [self.base_loss(input, target)]

        self.feat_losses += [
            self.base_loss(f_in, f_out) * w
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]

        styles = [self._get_style_vals(i) for i in out_feat]

        if styles[0][0] is not None:
            self.feat_losses += [
                self._single_wass_loss(f_pred, f_targ) * w
                for f_pred, f_targ, w in zip(in_feat, styles, self.wass_wgts)
            ]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self):
        for i in self.hooks:
            i.remove()

    def remove(self):
        for i in self.hooks:
            i.remove()
