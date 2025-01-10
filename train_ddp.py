# Filename: train_ddp.py

import os
import argparse
import datetime
import numpy as np
import cv2
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# -- Your existing imports for other modules you have in your project --
from functions import show_tensor
from generator import define_G
from discriminator import define_D
import losses
from replay_pool import ReplayPool
from moving_average import moving_average
from dataloader import Dataset, print_dataset, plot_batch_samples


def make_generator(device):
    """Create a generator with the specified architecture and move it to device."""
    netG = define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="global",
        norm="instance",
        n_downsample_global=3,
        n_blocks_global=9,
        n_local_enhancers=1,
        n_blocks_local=3
    )
    return netG.to(device)


def process_loss(log, losses_dict):
    """
    Process and aggregate loss values, updating the log dictionary.

    Args:
        log (dict): Dictionary to store accumulated loss values
        losses_dict (dict): Dictionary of current loss values to process

    Returns:
        torch.Tensor: Sum of all loss values
    """
    total_loss = 0.0
    for name, value in losses_dict.items():
        if isinstance(value, torch.Tensor):
            total_loss += value
            log[name] = log.get(name, 0) + value.item()
        else:
            total_loss += value
            log[name] = log.get(name, 0) + value
    return total_loss


def save_checkpoint(model, checkpoint_dir, epoch):
    """Save model checkpoint (only from rank=0 to avoid conflicts)."""
    out_file = f"epoch_{epoch}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.pt"
    out_file = os.path.join(checkpoint_dir, out_file)
    try:
        torch.save({
            "G1": model.G1_ema.state_dict(),
            "G2": model.G2_ema.state_dict(),
            "G3": model.G3_ema.state_dict(),
            "D1": model.D1.state_dict(),
            "D2": model.D2.state_dict(),
            "D3": model.D3.state_dict()
        }, out_file)
        # dist.get_rank() check is done at the caller
        print(f"Saved to {out_file}")
    except Exception as e:
        print(f"Error saving the file: {e}")


class CascadeGAN:
    """
    CascadeGAN class that holds:
      - Generators (G1, G2, G3 + their EMA versions)
      - Discriminators (D1, D2, D3)
      - Optimizers
      - Replay pools
      - Loss functions
    Provides train_step(...) and test(...) methods.
    """
    def __init__(self, device, generators, discriminators, optimizers, replay_pools, losses):
        self.device = device
        
        # Generators
        self.G1 = generators['G1']
        self.G2 = generators['G2']
        self.G3 = generators['G3']
        self.G1_ema = generators['G1_ema']
        self.G2_ema = generators['G2_ema']
        self.G3_ema = generators['G3_ema']
        
        # Discriminators
        self.D1 = discriminators['D1']
        self.D2 = discriminators['D2']
        self.D3 = discriminators['D3']
        
        # Optimizers
        self.G1_optim = optimizers['G1']
        self.G2_optim = optimizers['G2']
        self.G3_optim = optimizers['G3']
        self.D1_optim = optimizers['D1']
        self.D2_optim = optimizers['D2']
        self.D3_optim = optimizers['D3']
        
        # Replay pools
        self.replay_pool1 = replay_pools['pool1']
        self.replay_pool2 = replay_pools['pool2']
        self.replay_pool3 = replay_pools['pool3']
        
        # Loss functions
        self.criterionGAN = losses['GAN']
        self.criterionFeat = losses['Feat']
        self.criterionVGG = losses['VGG']

    def calc_G1_losses(self, img_A, img_A_mask):
        # Forward
        fake_A_mask = self.G1(img_A)
        # VGG loss
        loss_vgg = self.criterionVGG(fake_A_mask, img_A_mask)
        # Adversarial
        pred_fake = self.D1(torch.cat([img_A, fake_A_mask], dim=1))
        loss_adv = self.criterionGAN(pred_fake, 1)
        # Feature matching
        with torch.no_grad():
            pred_true = self.D1(torch.cat([img_A, img_A_mask], dim=1))
        loss_adv_feat = self.calc_feature_matching_loss(pred_fake, pred_true)

        return {
            "G1_vgg": loss_vgg,
            "G1_adv": loss_adv,
            "G1_adv_feat": 10 * loss_adv_feat
        }, fake_A_mask

    def calc_G2_losses(self, fake_A_mask, img_B_mask):
        fake_B_mask = self.G2(fake_A_mask.detach())
        # VGG loss
        loss_vgg = self.criterionVGG(fake_B_mask, img_B_mask)
        # Adversarial
        pred_fake = self.D2(torch.cat([fake_A_mask.detach(), fake_B_mask], dim=1))
        loss_adv = self.criterionGAN(pred_fake, 1)
        # Feature matching
        with torch.no_grad():
            pred_true = self.D2(torch.cat([fake_A_mask.detach(), img_B_mask], dim=1))
        loss_adv_feat = self.calc_feature_matching_loss(pred_fake, pred_true)

        return {
            "G2_vgg": loss_vgg,
            "G2_adv": loss_adv,
            "G2_adv_feat": 10 * loss_adv_feat
        }, fake_B_mask

    def calc_G3_losses(self, fake_B_mask, img_B):
        fake_B = self.G3(fake_B_mask.detach())
        # VGG loss
        loss_vgg = self.criterionVGG(fake_B, img_B)
        # Adversarial
        pred_fake = self.D3(torch.cat([fake_B_mask.detach(), fake_B], dim=1))
        loss_adv = self.criterionGAN(pred_fake, 1)
        # Feature matching
        with torch.no_grad():
            pred_true = self.D3(torch.cat([fake_B_mask.detach(), img_B], dim=1))
        loss_adv_feat = self.calc_feature_matching_loss(pred_fake, pred_true)

        return {
            "G3_vgg": loss_vgg,
            "G3_adv": loss_adv,
            "G3_adv_feat": 10 * loss_adv_feat
        }, fake_B

    def calc_feature_matching_loss(self, pred_fake, pred_true):
        loss_adv_feat = 0.0
        adv_feats_count = 0
        for d_fake_out, d_true_out in zip(pred_fake, pred_true):
            # ignoring the last output layer (which is the final prediction)
            for l_fake, l_true in zip(d_fake_out[:-1], d_true_out[:-1]):
                loss_adv_feat += self.criterionFeat(l_fake, l_true)
                adv_feats_count += 1
        return (4 / adv_feats_count) * loss_adv_feat if adv_feats_count > 0 else 0.0

    def calc_D_losses(self, D, real_input, real_target, fake_input, fake_output, name):
        # Real
        pred_true = D(torch.cat([real_input, real_target], dim=1))
        loss_true = self.criterionGAN(pred_true, 1)
        # Fake
        pred_fake = D(torch.cat([fake_input, fake_output], dim=1))
        loss_false = self.criterionGAN(pred_fake, 0)
        
        return {
            f"D{name}_true": loss_true,
            f"D{name}_false": loss_false
        }

    def train_step(self, batch):
        img_A = batch['A'].to(self.device)
        img_A_mask = batch['A_mask'].to(self.device)
        img_B_mask = batch['B_mask'].to(self.device)
        img_B = batch['B'].to(self.device)
        
        log = {}
        
        # --------------------
        #  Stage 1: A → A_mask
        # --------------------
        self.G1_optim.zero_grad()
        G1_losses, fake_A_mask = self.calc_G1_losses(img_A, img_A_mask)
        G1_loss = process_loss(log, G1_losses)
        G1_loss.backward()
        self.G1_optim.step()
        # Update EMA
        moving_average(self.G1, self.G1_ema)
        
        fake_A_mask = fake_A_mask.detach()
        
        # D1
        self.D1_optim.zero_grad()
        fake_data1 = self.replay_pool1.query({"input": img_A.detach(), "output": fake_A_mask})
        D1_losses = self.calc_D_losses(
            self.D1, img_A, img_A_mask,
            fake_data1["input"], fake_data1["output"], "1"
        )
        D1_loss = process_loss(log, D1_losses)
        D1_loss.backward()
        self.D1_optim.step()
        
        # --------------------
        #  Stage 2: A_mask → B_mask
        # --------------------
        self.G2_optim.zero_grad()
        G2_losses, fake_B_mask = self.calc_G2_losses(fake_A_mask, img_B_mask)
        G2_loss = process_loss(log, G2_losses)
        G2_loss.backward()
        self.G2_optim.step()
        moving_average(self.G2, self.G2_ema)
        
        fake_B_mask = fake_B_mask.detach()
        
        # D2
        self.D2_optim.zero_grad()
        fake_data2 = self.replay_pool2.query({"input": fake_A_mask, "output": fake_B_mask})
        D2_losses = self.calc_D_losses(
            self.D2, fake_A_mask, img_B_mask,
            fake_data2["input"], fake_data2["output"], "2"
        )
        D2_loss = process_loss(log, D2_losses)
        D2_loss.backward()
        self.D2_optim.step()
        
        # --------------------
        #  Stage 3: B_mask → B
        # --------------------
        self.G3_optim.zero_grad()
        G3_losses, fake_B = self.calc_G3_losses(fake_B_mask, img_B)
        G3_loss = process_loss(log, G3_losses)
        G3_loss.backward()
        self.G3_optim.step()
        moving_average(self.G3, self.G3_ema)
        
        # D3
        self.D3_optim.zero_grad()
        fake_data3 = self.replay_pool3.query({"input": fake_B_mask, "output": fake_B.detach()})
        D3_losses = self.calc_D_losses(
            self.D3, fake_B_mask, img_B,
            fake_data3["input"], fake_data3["output"], "3"
        )
        D3_loss = process_loss(log, D3_losses)
        D3_loss.backward()
        self.D3_optim.step()
        
        # Return a dictionary of the aggregated losses and the outputs
        return log, (fake_A_mask, fake_B_mask, fake_B)

    def test(self, test_loader, epoch, iteration, output_dir):
        """Generate test samples (usually called less frequently)."""
           # If there's no test_loader on rank != 0, just return
        if test_loader is None:
            return

        with torch.no_grad():
            # Grab a single batch each time
            try:
                batch = next(iter(test_loader))
            except StopIteration:
                # if you run out of data, re-init the iterator or just skip
                return

            img_A = batch['A'].to(self.device)
            
            self.G1_ema.eval()
            fake_A_mask = self.G1_ema(img_A)
            
            self.G2_ema.eval()
            fake_B_mask = self.G2_ema(fake_A_mask)
            
            self.G3_ema.eval()
            fake_B = self.G3_ema(fake_B_mask)
            
            # Concatenate and save
            pairs = torch.cat([
                img_A,
                fake_A_mask,
                fake_B_mask,
                fake_B,
                batch['B'].to(self.device)
            ], dim=-1)
            
            matrix = []
            for idx in range(img_A.shape[0]):
                img = 255 * (pairs[idx] + 1) / 2
                img = img.cpu().permute(1, 2, 0).clip(0, 255).numpy().astype(np.uint8)
                matrix.append(img)
            
            matrix = np.vstack(matrix)
            matrix = cv2.cvtColor(matrix, cv2.COLOR_RGB2BGR)
            out_file = os.path.join(output_dir, f"{epoch}_{iteration}.jpg")
            cv2.imwrite(out_file, matrix)
            
            # Return to train mode if needed
            self.G1_ema.train()
            self.G2_ema.train()
            self.G3_ema.train()


def train(model, train_loader, test_loader, num_epochs, checkpoint_dir):
    """Main training loop (each process)."""
    for epoch in range(num_epochs):
        current_rank = dist.get_rank()

        if current_rank == 0:
            print("This is the primary process (rank 0) -- can safely print logs or save checkpoints.")
        else:
            print(f"This is rank {current_rank}, skipping logging/checkpointing...")
        # Only rank 0 logs to console
        if dist.get_rank() == 0:
            print(f"Training epoch {epoch}...")

        # For true randomness each epoch in DDP
        train_loader.sampler.set_epoch(epoch)
        
        N = 0
        log = {}
        
        # TQDM progress bar on rank 0 only
        pbar = tqdm(train_loader, disable=(dist.get_rank() != 0))
        for batch in pbar:
            batch_log, _ = model.train_step(batch)
            
            # Accumulate metrics
            for k, v in batch_log.items():
                log[k] = log.get(k, 0) + v
            N += 1

            # Update progress bar description on rank 0
            if dist.get_rank() == 0:
                txt = " ".join([f"{k}: {v/N:.3e}" for k, v in log.items()])
                pbar.set_description(txt)

        # -- Move testing logic to happen every 10 epochs --
        if (epoch + 1) % 10 == 0 and dist.get_rank() == 0:
            # Example: run 3 test samples
            for i in range(3):
                model.test(test_loader, epoch, i, os.path.join(checkpoint_dir, "images"))

        # Save checkpoint every 20 epochs from rank 0
        if (epoch + 1) % 20 == 0 and (dist.get_rank() == 0):
            save_checkpoint(model, checkpoint_dir, epoch)



def main():
    # 1) Parse arguments
    parser = argparse.ArgumentParser(description="CascadeGAN DDP Training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/ortho2", help="Directory for checkpoints")
    parser.add_argument("--dataset_dir", type=str, default="./data/ortho", help="Base directory for dataset with train & test folders")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train")
    
    # DDP-specific argument (Populated by torchrun automatically)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # If user didn't specify --local_rank on the command line,
    # or if torchrun didn't pass it, fallback to environment:
    if args.local_rank == -1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        local_rank = args.local_rank

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}")

    # 3) Create Generators (each process has its own replica)
    generator1 = make_generator(device)  # A → A_mask
    generator2 = make_generator(device)  # A_mask → B_mask
    generator3 = make_generator(device)  # B_mask → B
    
    # EMA versions
    generator1_ema = make_generator(device)
    generator2_ema = make_generator(device)
    generator3_ema = make_generator(device)

    # Initialize EMA weights
    with torch.no_grad():
        for g, g_ema in [
            (generator1, generator1_ema),
            (generator2, generator2_ema),
            (generator3, generator3_ema)
        ]:
            for gp, ep in zip(g.parameters(), g_ema.parameters()):
                ep.data = gp.data.detach()

    # 4) Create Discriminators
    #    Each D expects channel_in = 3 + 3 for A+mask or mask+mask or mask+real
    discriminator1 = define_D(input_nc=6, ndf=64, n_layers_D=3, num_D=3, norm="instance", getIntermFeat=True).to(device)
    discriminator2 = define_D(input_nc=6, ndf=64, n_layers_D=3, num_D=3, norm="instance", getIntermFeat=True).to(device)
    discriminator3 = define_D(input_nc=6, ndf=64, n_layers_D=3, num_D=3, norm="instance", getIntermFeat=True).to(device)

    # 5) Wrap in DDP
    #    Typically you do not wrap EMA models in DDP. You only keep them on rank 0 or do manual sync if needed.
    generator1 = DDP(generator1, device_ids=[local_rank], output_device=local_rank)
    generator2 = DDP(generator2, device_ids=[local_rank], output_device=local_rank)
    generator3 = DDP(generator3, device_ids=[local_rank], output_device=local_rank)

    discriminator1 = DDP(discriminator1, device_ids=[local_rank], output_device=local_rank)
    discriminator2 = DDP(discriminator2, device_ids=[local_rank], output_device=local_rank)
    discriminator3 = DDP(discriminator3, device_ids=[local_rank], output_device=local_rank)

    # 6) Losses, Replay Pools, Optimizers
    criterionGAN = losses.GANLoss(use_lsgan=True).to(device)
    criterionFeat = torch.nn.L1Loss().to(device)
    criterionVGG = losses.VGGLoss().to(device)

    replay_pool1 = ReplayPool(10)
    replay_pool2 = ReplayPool(10)
    replay_pool3 = ReplayPool(10)

    G1_optim = torch.optim.AdamW(generator1.parameters(), lr=8e-5)
    G2_optim = torch.optim.AdamW(generator2.parameters(), lr=8e-5)
    G3_optim = torch.optim.AdamW(generator3.parameters(), lr=8e-5)

    D1_optim = torch.optim.AdamW(discriminator1.parameters(), lr=8e-5)
    D2_optim = torch.optim.AdamW(discriminator2.parameters(), lr=8e-5)
    D3_optim = torch.optim.AdamW(discriminator3.parameters(), lr=8e-5)

    # 7) DataLoaders with DistributedSampler
    train_dataset = Dataset(images_dir=os.path.join(args.dataset_dir, "train"), mode="train")
    # test_dataset = Dataset(images_dir=os.path.join(args.dataset_dir, "test"), mode="test")

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    # test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=args.batch_size,
    #     sampler=test_sampler,
    #     num_workers=4,
    #     pin_memory=True
    # )
    if dist.get_rank() == 0:
        # Non-distributed test loader
        test_dataset = Dataset(images_dir=os.path.join(args.dataset_dir, "test"), mode="test")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    else:
        test_loader = None
    # 8) (Optional) Print dataset info on rank 0 only
    if dist.get_rank() == 0:
        print_dataset(train_loader, mode="train")
        print_dataset(test_loader, mode="test")
        # You can also show some samples:
        # batch_sample = next(iter(train_loader))
        # plot_batch_samples(batch_sample)

    # 9) Create output directories (rank 0)
    checkpoint_dir = args.checkpoint_dir
    images_output_dir = os.path.join(checkpoint_dir, "images")
    if dist.get_rank() == 0:
        os.makedirs(images_output_dir, exist_ok=True)

    # 10) Create the CascadeGAN model
    model = CascadeGAN(
        device=device,
        generators={
            'G1': generator1,
            'G2': generator2,
            'G3': generator3,
            'G1_ema': generator1_ema,
            'G2_ema': generator2_ema,
            'G3_ema': generator3_ema
        },
        discriminators={
            'D1': discriminator1,
            'D2': discriminator2,
            'D3': discriminator3
        },
        optimizers={
            'G1': G1_optim,
            'G2': G2_optim,
            'G3': G3_optim,
            'D1': D1_optim,
            'D2': D2_optim,
            'D3': D3_optim
        },
        replay_pools={
            'pool1': replay_pool1,
            'pool2': replay_pool2,
            'pool3': replay_pool3
        },
        losses={
            'GAN': criterionGAN,
            'Feat': criterionFeat,
            'VGG': criterionVGG
        }
    )

    # 11) Start the training
    train(model, train_loader, test_loader, num_epochs=args.num_epochs, checkpoint_dir=checkpoint_dir)

    # 12) Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# torchrun --nproc_per_node=3 train_ddp.py --dataset_dir ./data/ortho --num_epochs 500 --batch_size 24 --checkpoint_dir ./checkpoints/ortho