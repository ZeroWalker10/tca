import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from ..attack import Attack


class DIFGSM(Attack):
    r"""
    DI2-FGSM in the paper 'Improving Transferability of Adversarial Examples with Input Diversity'
    [https://arxiv.org/abs/1803.06978]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 0.0)
        steps (int): number of iterations. (Default: 10)
        resize_rate (float): resize factor used in input diversity. (Default: 0.9)
        diversity_prob (float) : the probability of applying input diversity. (Default: 0.5)
        random_start (bool): using random initialization of delta. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.DIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        decay=0.0,
        resize_rate=0.9,
        diversity_prob=0.5,
        random_start=False,
        collect_logits=False,
        collect_gradients=False,
        collect_step_images=False,
        collect_outputs=False,
    ):
        super().__init__("DIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.collect_logits = collect_logits
        self.collect_gradients = collect_gradients
        self.collect_step_images = collect_step_images
        self.collect_outputs = collect_outputs

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(
            x, size=[rnd, rnd], mode="bilinear", align_corners=False
        )
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(
            rescaled,
            [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
            value=0,
        )

        return padded if torch.rand(1) < self.diversity_prob else x

    def forward(self, images, labels, *args, **kwargs):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        ret_logits = []
        ret_gradient_norms = []
        ret_gradient_cos = []
        ret_gradients = []
        ret_outputs = []
        ret_images = [adv_images.detach().cpu()]
        prev_gradients = torch.zeros_like(adv_images)
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(self.input_diversity(adv_images))

            if self.collect_outputs:
                ret_outputs.append(outputs.detach().cpu().numpy())
                continue

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)

            if self.collect_gradients:
                grad_norm = torch.linalg.vector_norm(grad.view(len(grad), -1)).detach().cpu().item()
                grad_cos = torch.cosine_similarity(grad.view(1, -1), prev_gradients.view(1, -1)).detach().cpu().item()

                ret_gradient_norms.append(grad_norm)
                ret_gradient_cos.append(grad_cos)
                ret_gradients.append(grad.detach().cpu().numpy())

                prev_gradients = grad.detach()

            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            if self.collect_logits:
                ret_logits.append(outputs.mean(0, keepdim=True).detach().cpu().numpy())
            if self.collect_step_images:
                ret_images.append(adv_images.detach().cpu())

        if self.collect_logits:
            return adv_images, ret_logits
        elif self.collect_gradients:
            return adv_images, ret_gradient_norms, ret_gradient_cos, ret_gradients
        elif self.collect_step_images:
            return adv_images, ret_images
        elif self.collect_outputs:
            return ret_outputs
        else:
            return adv_images
