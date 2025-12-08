# models/cycleGAN_model.py

import torch
import torch.nn as nn
from torch import optim

from .generator import ResnetGenerator
from .discriminator import PatchDiscriminator
from utils.image_pool import ImagePool


class GANLoss(nn.Module):
    """
    支持 lsgan / vanilla 的对抗损失，默认 lsgan
    """
    def __init__(self, gan_mode="lsgan"):
        super().__init__()
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"GAN mode {gan_mode} not implemented")

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = torch.ones_like(prediction)
        else:
            target_tensor = torch.zeros_like(prediction)
        return target_tensor

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


class CycleGANModel:
    """
    基础 CycleGAN 训练封装
    - 两个生成器 G_A: A->B, G_B: B->A
    - 两个判别器 D_A: 判别 B 域, D_B: 判别 A 域
    - 支持：
        * 对抗损失 (LSGAN)
        * 循环一致性损失 (L1)
        * Identity 损失
        * 预留感知损失接口（默认关闭）
    """

    def __init__(
        self,
        input_nc=3,
        output_nc=3,
        ngf=64,
        ndf=64,
        lambda_cycle=10.0,
        lambda_idt=0.5,
        gan_mode="lsgan",
        lr=2e-4,
        pool_size=50,
        device="cuda",
        use_perceptual_loss=False,
        perceptual_weight=0.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ---------- 网络 ----------
        self.netG_A = ResnetGenerator(input_nc, output_nc, ngf).to(self.device)
        self.netG_B = ResnetGenerator(output_nc, input_nc, ngf).to(self.device)
        self.netD_A = PatchDiscriminator(output_nc, ndf).to(self.device)
        self.netD_B = PatchDiscriminator(input_nc, ndf).to(self.device)

        # ---------- 损失函数 ----------
        self.criterionGAN = GANLoss(gan_mode).to(self.device)
        self.criterionCycle = nn.L1Loss().to(self.device)
        self.criterionIdt = nn.L1Loss().to(self.device)

        self.lambda_cycle = lambda_cycle
        self.lambda_idt = lambda_idt

        # 预留：感知损失（默认关闭，不会被调用）
        self.use_perceptual_loss = use_perceptual_loss
        self.perceptual_weight = perceptual_weight
        if use_perceptual_loss:
            from .vgg_perceptual import PerceptualLoss  # 延迟导入，避免不必要的依赖
            self.criterionPerc = PerceptualLoss().to(self.device)
        else:
            self.criterionPerc = None

        # ---------- 优化器 ----------
        self.optimizer_G = optim.Adam(
            list(self.netG_A.parameters()) + list(self.netG_B.parameters()),
            lr=lr,
            betas=(0.5, 0.999),
        )
        self.optimizer_D = optim.Adam(
            list(self.netD_A.parameters()) + list(self.netD_B.parameters()),
            lr=lr,
            betas=(0.5, 0.999),
        )

        # ---------- Image Pool ----------
        self.fake_A_pool = ImagePool(pool_size)
        self.fake_B_pool = ImagePool(pool_size)

        # 这些属性会在 set_input/forward 中被赋值
        self.real_A = None
        self.real_B = None

        # 记录损失
        self.losses = {}

    # ====== 工具函数 ======
    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_input(self, batch):
        """
        batch: {"A": tensor, "B": tensor}
        """
        self.real_A = batch["A"].to(self.device)
        self.real_B = batch["B"].to(self.device)

    # ====== 前向传播 ======
    def forward(self):
        """
        real_A: A 域图像
        real_B: B 域图像
        生成：
        fake_B = G_A(real_A)
        rec_A  = G_B(fake_B)
        fake_A = G_B(real_B)
        rec_B  = G_A(fake_A)
        """
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    # ====== 判别器反向传播 ======
    def backward_D_basic(self, netD, real, fake):
        """
        计算并反向传播单个判别器的损失
        """
        # 真样本
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        # 假样本
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D.item()

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.losses["D_A"] = loss_D_A

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.losses["D_B"] = loss_D_B

    # ====== 生成器反向传播 ======
    def backward_G(self):
        # 对抗损失
        loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # 循环一致性损失
        loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.lambda_cycle
        loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.lambda_cycle

        # Identity 损失（让 G_A(A) ≈ A, G_B(B) ≈ B）
        idt_A = self.netG_A(self.real_B)
        idt_B = self.netG_B(self.real_A)
        loss_idt_A = self.criterionIdt(idt_A, self.real_B) * self.lambda_cycle * self.lambda_idt
        loss_idt_B = self.criterionIdt(idt_B, self.real_A) * self.lambda_cycle * self.lambda_idt

        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        # 预留：感知损失
        if self.use_perceptual_loss and self.criterionPerc is not None and self.perceptual_weight > 0:
            perc_A = self.criterionPerc(self.fake_B, self.real_B)
            perc_B = self.criterionPerc(self.fake_A, self.real_A)
            loss_perc = (perc_A + perc_B) * self.perceptual_weight
            loss_G = loss_G + loss_perc
            self.losses["Perc"] = loss_perc.item()
        else:
            self.losses["Perc"] = 0.0

        loss_G.backward()

        # 记录损失
        self.losses["G_A"] = loss_G_A.item()
        self.losses["G_B"] = loss_G_B.item()
        self.losses["Cyc_A"] = loss_cycle_A.item()
        self.losses["Cyc_B"] = loss_cycle_B.item()
        self.losses["Idt_A"] = loss_idt_A.item()
        self.losses["Idt_B"] = loss_idt_B.item()

    # ====== 一次完整的优化步骤 ======
    def optimize_parameters(self):
        # 1. 前向
        self.forward()

        # 2. 更新生成器
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # 3. 更新判别器
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    # ====== 便捷接口 ======
    def get_current_losses(self):
        return dict(self.losses)

    def save(self, path, save_optim=True):
        """
        保存模型参数（以及可选优化器）
        """
        state = {
            "netG_A": self.netG_A.state_dict(),
            "netG_B": self.netG_B.state_dict(),
            "netD_A": self.netD_A.state_dict(),
            "netD_B": self.netD_B.state_dict(),
        }
        if save_optim:
            state.update(
                {
                    "optimizer_G": self.optimizer_G.state_dict(),
                    "optimizer_D": self.optimizer_D.state_dict(),
                }
            )
        torch.save(state, path)

    def load(self, path, load_optim=False):
        """
        加载模型（训练复现 / 测试）
        """
        state = torch.load(path, map_location=self.device)
        self.netG_A.load_state_dict(state["netG_A"])
        self.netG_B.load_state_dict(state["netG_B"])
        if "netD_A" in state:
            self.netD_A.load_state_dict(state["netD_A"])
        if "netD_B" in state:
            self.netD_B.load_state_dict(state["netD_B"])

        if load_optim and "optimizer_G" in state:
            self.optimizer_G.load_state_dict(state["optimizer_G"])
            self.optimizer_D.load_state_dict(state["optimizer_D"])
