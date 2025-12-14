# models/cycleGAN_model.py

import inspect
from contextlib import nullcontext

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

    def get_target_tensor(self, prediction, target_is_real: bool):
        return torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)

    def forward(self, prediction, target_is_real: bool):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


class CycleGANModel:
    """
    基础 CycleGAN 训练封装
    - 两个生成器 G_A: A->B, G_B: B->A
    - 两个判别器 D_A: 判别 B 域, D_B: 判别 A 域

    本版重点增强（面向 GPU 性能）：
    1) AMP 混合精度（fp16 / bf16，可选）
    2) channels_last（对 256x256 卷积通常更快）
    3) 可选 fused Adam（torch 新版本 + CUDA 时更快）
    4) 可选 torch.compile（训练长跑时有收益）
    5) 最关键：避免每步 .item() 触发 GPU 同步（只在打印时再转 float）
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

        # ===== 性能开关 =====
        use_amp=True,
        amp_dtype="bf16",          # "bf16" / "fp16" / "fp32"
        channels_last=True,
        use_fused_adam=True,
        use_compile=False,
        compile_mode="max-autotune",  # "default" / "reduce-overhead" / "max-autotune"

        # ===== 可选：感知损失 =====
        use_perceptual_loss=False,
        perceptual_weight=0.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ---------- AMP 配置 ----------
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.amp_dtype = self._resolve_amp_dtype(amp_dtype)
        # GradScaler：fp16 需要；bf16 通常不需要（但开着也没意义）
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.use_amp and self.amp_dtype == torch.float16)
        ) if self.device.type == "cuda" else None

        # ---------- channels_last ----------
        self.channels_last = bool(channels_last and self.device.type == "cuda")

        # ---------- 网络 ----------
        self.netG_A = ResnetGenerator(input_nc, output_nc, ngf).to(self.device)
        self.netG_B = ResnetGenerator(output_nc, input_nc, ngf).to(self.device)
        self.netD_A = PatchDiscriminator(output_nc, ndf).to(self.device)
        self.netD_B = PatchDiscriminator(input_nc, ndf).to(self.device)

        if self.channels_last:
            # 将权重/缓冲区转换成 channels_last（对 Conv2d/ConvTranspose2d 常有收益）
            self.netG_A.to(memory_format=torch.channels_last)
            self.netG_B.to(memory_format=torch.channels_last)
            self.netD_A.to(memory_format=torch.channels_last)
            self.netD_B.to(memory_format=torch.channels_last)

        # 可选：torch.compile（长训练更划算；第一次会有编译开销）
        if use_compile and hasattr(torch, "compile") and self.device.type == "cuda":
            self.netG_A = self._try_compile(self.netG_A, compile_mode, name="netG_A")
            self.netG_B = self._try_compile(self.netG_B, compile_mode, name="netG_B")
            self.netD_A = self._try_compile(self.netD_A, compile_mode, name="netD_A")
            self.netD_B = self._try_compile(self.netD_B, compile_mode, name="netD_B")

        # ---------- 损失函数 ----------
        self.criterionGAN = GANLoss(gan_mode).to(self.device)
        self.criterionCycle = nn.L1Loss().to(self.device)
        self.criterionIdt = nn.L1Loss().to(self.device)

        self.lambda_cycle = float(lambda_cycle)
        self.lambda_idt = float(lambda_idt)

        # ---------- 感知损失（默认关闭） ----------
        self.use_perceptual_loss = bool(use_perceptual_loss)
        self.perceptual_weight = float(perceptual_weight)
        if self.use_perceptual_loss:
            from .vgg_perceptual import PerceptualLoss
            self.criterionPerc = PerceptualLoss().to(self.device)
        else:
            self.criterionPerc = None

        # ---------- 优化器 ----------
        adam_kwargs = dict(lr=lr, betas=(0.5, 0.999))
        # foreach 往往更快；fused 视 torch/cuda 版本而定
        adam_sig = inspect.signature(optim.Adam).parameters
        if "foreach" in adam_sig:
            adam_kwargs["foreach"] = True
        if self.device.type == "cuda" and use_fused_adam and ("fused" in adam_sig):
            # fused=True 在某些版本上会显著提速（不支持则自动忽略）
            adam_kwargs["fused"] = True

        self.optimizer_G = optim.Adam(
            list(self.netG_A.parameters()) + list(self.netG_B.parameters()),
            **adam_kwargs
        )
        self.optimizer_D = optim.Adam(
            list(self.netD_A.parameters()) + list(self.netD_B.parameters()),
            **adam_kwargs
        )

        # ---------- Image Pool ----------
        self.fake_A_pool = ImagePool(pool_size)
        self.fake_B_pool = ImagePool(pool_size)

        # 这些属性会在 set_input/forward 中被赋值
        self.real_A = None
        self.real_B = None
        self.fake_A = None
        self.fake_B = None
        self.rec_A = None
        self.rec_B = None

        # 记录损失（注意：存 tensor.detach()，避免每步 .item() 同步）
        self.losses = {}

    # ----------------- 内部工具 -----------------
    def _resolve_amp_dtype(self, amp_dtype):
        """
        amp_dtype: "bf16"/"fp16"/"fp32" 或 torch.dtype
        """
        if not self.use_amp:
            return torch.float32

        if isinstance(amp_dtype, torch.dtype):
            return amp_dtype

        s = str(amp_dtype).lower().strip()
        if s in ("bf16", "bfloat16"):
            # bf16 需要硬件支持；不支持则回退 fp16
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if s in ("fp16", "float16", "16"):
            return torch.float16
        # fp32 等价于关闭 amp
        self.use_amp = False
        return torch.float32

    def _autocast_ctx(self):
        if self.use_amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True)
        return nullcontext()

    @staticmethod
    def _try_compile(net, mode: str, name="net"):
        try:
            return torch.compile(net, mode=mode)
        except Exception as e:
            print(f"[WARN] torch.compile({name}) failed, fallback to eager. Reason: {e}")
            return net

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def _maybe_channels_last(self, x: torch.Tensor):
        if self.channels_last and x is not None and x.ndim == 4:
            return x.contiguous(memory_format=torch.channels_last)
        return x

    def _cache_losses(self, **kwargs):
        """
        存 detach tensor，避免每步 .item() 导致 GPU 同步
        """
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                self.losses[k] = v.detach()
            else:
                # 非 tensor 也统一存 python float
                self.losses[k] = float(v)

    # ----------------- 数据输入 -----------------
    def set_input(self, batch):
        """
        batch: {"A": tensor, "B": tensor}
        配合 train.py 的 pin_memory=True，可用 non_blocking=True 提速 H2D
        """
        self.real_A = batch["A"].to(self.device, non_blocking=True)
        self.real_B = batch["B"].to(self.device, non_blocking=True)
        self.real_A = self._maybe_channels_last(self.real_A)
        self.real_B = self._maybe_channels_last(self.real_B)

    # ----------------- 前向传播 -----------------
    def forward(self):
        """
        fake_B = G_A(real_A)
        rec_A  = G_B(fake_B)
        fake_A = G_B(real_B)
        rec_B  = G_A(fake_A)
        """
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    # ----------------- Loss 计算 -----------------
    def _compute_G_loss(self):
        # 对抗损失
        loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # 循环一致性损失
        loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.lambda_cycle
        loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.lambda_cycle

        # Identity 损失：lambda_idt==0 时完全跳过（省两次 G 前向，速度提升很明显）
        loss_idt_A = 0.0
        loss_idt_B = 0.0
        if self.lambda_idt > 0:
            idt_A = self.netG_A(self.real_B)
            idt_B = self.netG_B(self.real_A)
            loss_idt_A = self.criterionIdt(idt_A, self.real_B) * self.lambda_cycle * self.lambda_idt
            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * self.lambda_cycle * self.lambda_idt

        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        # 感知损失（可选，开了会变慢但可能更清晰）
        loss_perc = 0.0
        if self.use_perceptual_loss and self.criterionPerc is not None and self.perceptual_weight > 0:
            perc_A = self.criterionPerc(self.fake_B, self.real_B)
            perc_B = self.criterionPerc(self.fake_A, self.real_A)
            loss_perc = (perc_A + perc_B) * self.perceptual_weight
            loss_G = loss_G + loss_perc

        self._cache_losses(
            G_A=loss_G_A, G_B=loss_G_B,
            Cyc_A=loss_cycle_A, Cyc_B=loss_cycle_B,
            Idt_A=loss_idt_A if torch.is_tensor(loss_idt_A) else float(loss_idt_A),
            Idt_B=loss_idt_B if torch.is_tensor(loss_idt_B) else float(loss_idt_B),
            Perc=loss_perc if torch.is_tensor(loss_perc) else float(loss_perc),
        )
        return loss_G

    def _compute_D_loss(self, netD, real, fake):
        # 真样本
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # 假样本
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        return (loss_D_real + loss_D_fake) * 0.5

    # ----------------- 一次完整优化 -----------------
    def optimize_parameters(self):
        # 0) 前向（放在 autocast 内）
        with self._autocast_ctx():
            self.forward()

        # 1) 更新生成器（需要梯度流经 D，但 D 的参数不需要梯度）
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad(set_to_none=True)

        with self._autocast_ctx():
            loss_G = self._compute_G_loss()

        if self.scaler is not None:
            self.scaler.scale(loss_G).backward()
            self.scaler.step(self.optimizer_G)
        else:
            loss_G.backward()
            self.optimizer_G.step()

        # 2) 更新判别器
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad(set_to_none=True)

        # ImagePool 查询本身不需要梯度
        with torch.no_grad():
            fake_B = self.fake_B_pool.query(self.fake_B)
            fake_A = self.fake_A_pool.query(self.fake_A)

        with self._autocast_ctx():
            loss_D_A = self._compute_D_loss(self.netD_A, self.real_B, fake_B)
            loss_D_B = self._compute_D_loss(self.netD_B, self.real_A, fake_A)
            loss_D = loss_D_A + loss_D_B

        if self.scaler is not None:
            self.scaler.scale(loss_D).backward()
            self.scaler.step(self.optimizer_D)
            # 两个 optimizer 都 step 完后再 update（更合理）
            self.scaler.update()
        else:
            loss_D.backward()
            self.optimizer_D.step()

        self._cache_losses(D_A=loss_D_A, D_B=loss_D_B)

    # ----------------- 便捷接口 -----------------
    def get_current_losses(self):
        """
        只在需要打印/记录时才 .item()，避免每步训练都触发 GPU 同步
        """
        out = {}
        for k, v in self.losses.items():
            if torch.is_tensor(v):
                out[k] = float(v.detach().cpu().item())
            else:
                out[k] = float(v)
        return out

    def save(self, path, save_optim=True):
        state = {
            "netG_A": self.netG_A.state_dict(),
            "netG_B": self.netG_B.state_dict(),
            "netD_A": self.netD_A.state_dict(),
            "netD_B": self.netD_B.state_dict(),
            "meta": {
                "use_amp": self.use_amp,
                "amp_dtype": str(self.amp_dtype),
                "channels_last": self.channels_last,
            }
        }
        if save_optim:
            state.update({
                "optimizer_G": self.optimizer_G.state_dict(),
                "optimizer_D": self.optimizer_D.state_dict(),
            })
            if self.scaler is not None:
                state["scaler"] = self.scaler.state_dict()
        torch.save(state, path)

    def load(self, path, load_optim=False):
        state = torch.load(path, map_location=self.device)
        self.netG_A.load_state_dict(state["netG_A"])
        self.netG_B.load_state_dict(state["netG_B"])
        if "netD_A" in state:
            self.netD_A.load_state_dict(state["netD_A"])
        if "netD_B" in state:
            self.netD_B.load_state_dict(state["netD_B"])

        if load_optim:
            if "optimizer_G" in state:
                self.optimizer_G.load_state_dict(state["optimizer_G"])
            if "optimizer_D" in state:
                self.optimizer_D.load_state_dict(state["optimizer_D"])
            if self.scaler is not None and "scaler" in state:
                self.scaler.load_state_dict(state["scaler"])
