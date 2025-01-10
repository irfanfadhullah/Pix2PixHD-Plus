import argparse
import os
import datetime
import numpy as np
import cv2
import torch
from tqdm import tqdm
from functions import show_tensor
from generator import define_G
from discriminator import define_D
import losses
from replay_pool import ReplayPool
from moving_average import moving_average
from dataloader import Dataset, print_dataset, plot_batch_samples

def make_generator():
    """Create a generator with the specified architecture"""
    return define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="global",
        norm="instance",
        n_downsample_global=3,
        n_blocks_global=9,
        n_local_enhancers=1,
        n_blocks_local=3
    ).to(device)

def save_checkpoint(model, checkpoint_dir, epoch):
    """Save model checkpoint"""
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
        print(f"Saved to {out_file}")
    except Exception as e:
        print(f"Error saving the file: {e}")

def process_loss(log, losses_dict):
    """
    Process and aggregate loss values, updating the log dictionary
    
    Args:
        log (dict): Dictionary to store accumulated loss values
        losses_dict (dict): Dictionary of current loss values to process
        
    Returns:
        torch.Tensor: Sum of all loss values
    """
    total_loss = 0
    
    # Add each loss component to the total and update the log
    for name, value in losses_dict.items():
        if isinstance(value, torch.Tensor):
            total_loss += value
            log[name] = log.get(name, 0) + value.item()
        else:
            total_loss += value
            log[name] = log.get(name, 0) + value
            
    return total_loss

class CascadeGAN:
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
        fake_A_mask = self.G1(img_A)
        loss_vgg = self.criterionVGG(fake_A_mask, img_A_mask)
        
        pred_fake = self.D1(torch.cat([img_A, fake_A_mask], dim=1))
        loss_adv = self.criterionGAN(pred_fake, 1)
        
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
        loss_vgg = self.criterionVGG(fake_B_mask, img_B_mask)
        
        pred_fake = self.D2(torch.cat([fake_A_mask.detach(), fake_B_mask], dim=1))
        loss_adv = self.criterionGAN(pred_fake, 1)
        
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
        loss_vgg = self.criterionVGG(fake_B, img_B)
        
        pred_fake = self.D3(torch.cat([fake_B_mask.detach(), fake_B], dim=1))
        loss_adv = self.criterionGAN(pred_fake, 1)
        
        with torch.no_grad():
            pred_true = self.D3(torch.cat([fake_B_mask.detach(), img_B], dim=1))
            
        loss_adv_feat = self.calc_feature_matching_loss(pred_fake, pred_true)
        
        return {
            "G3_vgg": loss_vgg,
            "G3_adv": loss_adv,
            "G3_adv_feat": 10 * loss_adv_feat
        }, fake_B

    def calc_feature_matching_loss(self, pred_fake, pred_true):
        loss_adv_feat = 0
        adv_feats_count = 0
        for d_fake_out, d_true_out in zip(pred_fake, pred_true):
            for l_fake, l_true in zip(d_fake_out[:-1], d_true_out[:-1]):
                loss_adv_feat += self.criterionFeat(l_fake, l_true)
                adv_feats_count += 1
        return (4/adv_feats_count) * loss_adv_feat if adv_feats_count > 0 else 0

    def calc_D_losses(self, D, real_input, real_target, fake_input, fake_output, name):
        pred_true = D(torch.cat([real_input, real_target], dim=1))
        loss_true = self.criterionGAN(pred_true, 1)
        
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
        
        # Stage 1: A → A_mask
        self.G1_optim.zero_grad()
        G1_losses, fake_A_mask = self.calc_G1_losses(img_A, img_A_mask)
        G1_loss = process_loss(log, G1_losses)
        G1_loss.backward()
        self.G1_optim.step()
        moving_average(self.G1, self.G1_ema)
        
        fake_A_mask = fake_A_mask.detach()
        
        self.D1_optim.zero_grad()
        fake_data1 = self.replay_pool1.query({
            "input": img_A.detach(),
            "output": fake_A_mask
        })
        D1_losses = self.calc_D_losses(
            self.D1, img_A, img_A_mask,
            fake_data1["input"], fake_data1["output"], "1"
        )
        D1_loss = process_loss(log, D1_losses)
        D1_loss.backward()
        self.D1_optim.step()
        
        # Stage 2: A_mask → B_mask
        self.G2_optim.zero_grad()
        G2_losses, fake_B_mask = self.calc_G2_losses(fake_A_mask, img_B_mask)
        G2_loss = process_loss(log, G2_losses)
        G2_loss.backward()
        self.G2_optim.step()
        moving_average(self.G2, self.G2_ema)
        
        fake_B_mask = fake_B_mask.detach()
        
        self.D2_optim.zero_grad()
        fake_data2 = self.replay_pool2.query({
            "input": fake_A_mask,
            "output": fake_B_mask
        })
        D2_losses = self.calc_D_losses(
            self.D2, fake_A_mask, img_B_mask,
            fake_data2["input"], fake_data2["output"], "2"
        )
        D2_loss = process_loss(log, D2_losses)
        D2_loss.backward()
        self.D2_optim.step()
        
        # Stage 3: B_mask → B
        self.G3_optim.zero_grad()
        G3_losses, fake_B = self.calc_G3_losses(fake_B_mask, img_B)
        G3_loss = process_loss(log, G3_losses)
        G3_loss.backward()
        self.G3_optim.step()
        moving_average(self.G3, self.G3_ema)
        
        self.D3_optim.zero_grad()
        fake_data3 = self.replay_pool3.query({
            "input": fake_B_mask,
            "output": fake_B.detach()
        })
        D3_losses = self.calc_D_losses(
            self.D3, fake_B_mask, img_B,
            fake_data3["input"], fake_data3["output"], "3"
        )
        D3_loss = process_loss(log, D3_losses)
        D3_loss.backward()
        self.D3_optim.step()
        
        return log, (fake_A_mask, fake_B_mask, fake_B)

    def test(self, test_loader, epoch, iteration, output_dir):
        """Generate test samples"""
        with torch.no_grad():
            batch = next(iter(test_loader))
            img_A = batch['A'].to(self.device)
            
            self.G1_ema.eval()
            fake_A_mask = self.G1_ema(img_A)
            
            self.G2_ema.eval()
            fake_B_mask = self.G2_ema(fake_A_mask)
            
            self.G3_ema.eval()
            fake_B = self.G3_ema(fake_B_mask)
            
            # Create visualization
            pairs = torch.cat([
                img_A,
                fake_A_mask,
                fake_B_mask,
                fake_B,
                batch['B'].to(self.device)
            ], -1)
            
            matrix = []
            for idx in range(img_A.shape[0]):
                img = 255 * (pairs[idx] + 1) / 2
                img = img.cpu().permute(1, 2, 0).clip(0, 255).numpy().astype(np.uint8)
                matrix.append(img)
            
            matrix = np.vstack(matrix)
            matrix = cv2.cvtColor(matrix, cv2.COLOR_RGB2BGR)
            out_file = os.path.join(output_dir, f"{epoch}_{iteration}.jpg")
            cv2.imwrite(out_file, matrix)
            
            self.G1_ema.train()
            self.G2_ema.train()
            self.G3_ema.train()

def train(model, train_loader, test_loader, num_epochs, checkpoint_dir):
    """Training loop"""
    for epoch in range(num_epochs):
        print(f"Training epoch {epoch}...")
        N = 0
        log = {}
        
        pbar = tqdm(train_loader)
        for batch in pbar:
            batch_log, _ = model.train_step(batch)
            
            for k, v in batch_log.items():
                log[k] = log.get(k, 0) + v
            
            N += 1
            if (epoch + 1) % 10 == 0:
                for i in range(3):
                    model.test(test_loader, epoch, N + i, os.path.join(checkpoint_dir, "images"))
            
            # Update progress bar description
            txt = " ".join([f"{k}: {v/N:.3e}" for k, v in log.items()])
            pbar.set_description(txt)
        
        # Save checkpoint
        if (epoch + 1) % 20 == 0:
            save_checkpoint(model, checkpoint_dir, epoch)

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="CascadeGAN Training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--checkpoint_dir", type=str, required=False, default="./checkpoints/ortho_3", help="Directory for checkpoints")
    parser.add_argument("--dataset_dir", type=str, required=False, default="./data/ortho", help="Base directory for dataset with train and test folders")
    parser.add_argument("--gpu", action="store_true", help="Flag to use GPU")
    parser.add_argument("--gpu_number", type=str, default="0", help="Comma separated GPU device numbers to use")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train")
    args = parser.parse_args()

    # Parse GPU IDs and set device
    gpu_ids = []
    if args.gpu and torch.cuda.is_available():
        gpu_ids = [int(x) for x in args.gpu_number.split(",")]
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cpu")

    # Initialize generators
    generator1 = make_generator()  # A → A_mask
    generator2 = make_generator()  # A_mask → B_mask
    generator3 = make_generator()  # B_mask → B

    generator1_ema = make_generator()
    generator2_ema = make_generator()
    generator3_ema = make_generator()

    # Initialize EMA weights
    with torch.no_grad():
        for g, g_ema in [
            (generator1, generator1_ema),
            (generator2, generator2_ema),
            (generator3, generator3_ema)
        ]:
            for gp, ep in zip(g.parameters(), g_ema.parameters()):
                ep.data = gp.data.detach()

    # Initialize discriminators
    discriminator1 = define_D(
        input_nc=3 + 3,  # A + A_mask
        ndf=64,
        n_layers_D=3,
        num_D=3,
        norm="instance",
        getIntermFeat=True
    ).to(device)

    discriminator2 = define_D(
        input_nc=3 + 3,  # A_mask + B_mask
        ndf=64,
        n_layers_D=3,
        num_D=3,
        norm="instance",
        getIntermFeat=True
    ).to(device)

    discriminator3 = define_D(
        input_nc=3 + 3,  # B_mask + B
        ndf=64,
        n_layers_D=3,
        num_D=3,
        norm="instance",
        getIntermFeat=True
    ).to(device)

    # Wrap networks with DataParallel if using multiple GPUs
    if args.gpu and len(gpu_ids) > 1:
        generator1 = torch.nn.DataParallel(generator1, device_ids=gpu_ids)
        generator2 = torch.nn.DataParallel(generator2, device_ids=gpu_ids)
        generator3 = torch.nn.DataParallel(generator3, device_ids=gpu_ids)
        generator1_ema = torch.nn.DataParallel(generator1_ema, device_ids=gpu_ids)
        generator2_ema = torch.nn.DataParallel(generator2_ema, device_ids=gpu_ids)
        generator3_ema = torch.nn.DataParallel(generator3_ema, device_ids=gpu_ids)
        discriminator1 = torch.nn.DataParallel(discriminator1, device_ids=gpu_ids)
        discriminator2 = torch.nn.DataParallel(discriminator2, device_ids=gpu_ids)
        discriminator3 = torch.nn.DataParallel(discriminator3, device_ids=gpu_ids)

    # Initialize loss functions
    criterionGAN = losses.GANLoss(use_lsgan=True).to(device)
    criterionFeat = torch.nn.L1Loss().to(device)
    criterionVGG = losses.VGGLoss().to(device)

    # Initialize replay pools
    replay_pool1 = ReplayPool(10)
    replay_pool2 = ReplayPool(10)
    replay_pool3 = ReplayPool(10)

    # Initialize optimizers
    G1_optim = torch.optim.AdamW(generator1.parameters(), lr=2e-4)
    G2_optim = torch.optim.AdamW(generator2.parameters(), lr=2e-4)
    G3_optim = torch.optim.AdamW(generator3.parameters(), lr=2e-4)

    D1_optim = torch.optim.AdamW(discriminator1.parameters(), lr=2e-4)
    D2_optim = torch.optim.AdamW(discriminator2.parameters(), lr=2e-4)
    D3_optim = torch.optim.AdamW(discriminator3.parameters(), lr=2e-4)

    # DataLoader setup using command-line arguments
    train_dataset = Dataset(images_dir=os.path.join(args.dataset_dir, "train"), mode="train")
    test_dataset = Dataset(images_dir=os.path.join(args.dataset_dir, "test"), mode="test")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Print dataset info and sample batch
    print_dataset(train_loader, mode="train")
    print_dataset(test_loader, mode="test")
    batch = next(iter(train_loader))
    # plot_batch_samples(batch)

    # Setup checkpoint directory and image output directory
    checkpoint_dir = args.checkpoint_dir
    images_output_dir = os.path.join(checkpoint_dir, "images")
    os.makedirs(images_output_dir, exist_ok=True)

    # Create the CascadeGAN model
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

    # Start training
    train(model, train_loader, test_loader, num_epochs=args.num_epochs, checkpoint_dir=checkpoint_dir)
