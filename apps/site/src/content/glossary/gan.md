---
aliases:
- GAN
- generative adversarial network
- adversarial network
- generative model
category: deep-learning
difficulty: advanced
related:
- neural-network
- deep-learning
- machine-learning
- transformer
- cnn
sources:
- author: Ian J. Goodfellow et al.
  license: cc-by
  source_title: Generative Adversarial Networks
  source_url: https://arxiv.org/abs/1406.2661
- author: Alec Radford et al.
  license: cc-by
  source_title: Unsupervised Representation Learning with Deep Convolutional Generative
    Adversarial Networks
  source_url: https://arxiv.org/abs/1511.06434
summary: A Generative Adversarial Network (GAN) is a machine learning architecture
  consisting of two neural networks—a generator and a discriminator—that compete against
  each other in a game-theoretic framework. The generator learns to create realistic
  synthetic data while the discriminator learns to distinguish between real and generated
  data, leading to increasingly sophisticated data generation capabilities across
  domains like images, text, and audio.
tags:
- deep-learning
- generative-ai
- machine-learning
- computer-vision
- algorithms
title: GAN (Generative Adversarial Network)
updated: '2025-01-15'
---

## Overview

Generative Adversarial Networks (GANs) represent a revolutionary approach to generative modeling introduced by Ian
Goodfellow in 2014. By setting up a competitive game between two neural networks, GANs have achieved remarkable success
in generating realistic images, videos, audio, and other types of data. The adversarial training process leads to a
generator that can produce increasingly convincing synthetic data, making GANs foundational to modern generative AI.

## Core Architecture

### The Adversarial Game

GANs consist of two competing neural networks:

```python
class GAN:
    def __init__(self, latent_dim, data_dim):
        # Generator: Creates fake data from random noise
        self.generator = Generator(latent_dim, data_dim)
        
        # Discriminator: Distinguishes real from fake data  
        self.discriminator = Discriminator(data_dim)
        
    def adversarial_game(self):
        """The core adversarial training process"""
        
        # Generator's goal: Fool the discriminator
        # min_G E[log(1 - D(G(z)))]
        
        # Discriminator's goal: Correctly classify real vs fake
        # max_D E[log(D(x))] + E[log(1 - D(G(z)))]
        
        # This creates a minimax game:
        # min_G max_D V(D,G) = E[log(D(x))] + E[log(1-D(G(z)))]
        
        return "Nash equilibrium when G creates perfect fakes and D can't tell difference"

```text

### Generator Network

The generator transforms random noise into synthetic data:

```python
class Generator(nn.Module):
    """Generator network for creating fake data"""
    
    def __init__(self, latent_dim=100, img_size=28, channels=1):
        super().__init__()
        
        self.img_size = img_size
        self.channels = channels
        
        # Calculate output dimensions
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2)
        )
        
        # Upsampling layers
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
    def forward(self, z):
        """Generate fake data from random noise z"""
        
        # Map noise to feature maps
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        
        # Generate image through upsampling
        img = self.conv_blocks(out)
        
        return img

# Example usage

generator = Generator(latent_dim=100, img_size=28, channels=1)

## Generate fake MNIST digits

batch_size = 64
noise = torch.randn(batch_size, 100)  # Random noise
fake_images = generator(noise)         # Generated images

```text

### Discriminator Network

The discriminator classifies data as real or fake:

```python
class Discriminator(nn.Module):
    """Discriminator network for real vs fake classification"""
    
    def __init__(self, img_size=28, channels=1):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        
        # Calculate size after convolutions
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()  # Output probability
        )
        
    def forward(self, img):
        """Classify image as real (1) or fake (0)"""
        
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        
        return validity

## Example usage

discriminator = Discriminator(img_size=28, channels=1)

## Evaluate real and fake images

real_images = torch.randn(64, 1, 28, 28)  # Real data
fake_images = generator(torch.randn(64, 100))  # Generated data

real_scores = discriminator(real_images)      # Should be close to 1
fake_scores = discriminator(fake_images)      # Should be close to 0

```text

## Training Process

### Adversarial Training Loop

```python
class GANTrainer:
    def __init__(self, generator, discriminator, lr=0.0002, b1=0.5, b2=0.999):
        self.generator = generator
        self.discriminator = discriminator
        
        # Separate optimizers for G and D
        self.optimizer_G = torch.optim.Adam(
            generator.parameters(), lr=lr, betas=(b1, b2)
        )
        self.optimizer_D = torch.optim.Adam(
            discriminator.parameters(), lr=lr, betas=(b1, b2)
        )
        
        # Loss function
        self.adversarial_loss = nn.BCELoss()
        
    def train_step(self, real_batch):
        """Single training step for both networks"""
        
        batch_size = real_batch.size(0)
        
        # Labels for real and fake data
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        self.optimizer_D.zero_grad()
        
        # Real data
        real_validity = self.discriminator(real_batch)
        real_loss = self.adversarial_loss(real_validity, real_labels)
        
        # Fake data
        z = torch.randn(batch_size, 100)  # Random noise
        fake_batch = self.generator(z).detach()  # Don't compute gradients for G
        fake_validity = self.discriminator(fake_batch)
        fake_loss = self.adversarial_loss(fake_validity, fake_labels)
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()
        
        # -----------------
        #  Train Generator  
        # -----------------
        
        self.optimizer_G.zero_grad()
        
        # Generate fake data
        z = torch.randn(batch_size, 100)
        fake_batch = self.generator(z)
        
        # Generator wants discriminator to classify fake as real
        fake_validity = self.discriminator(fake_batch)
        g_loss = self.adversarial_loss(fake_validity, real_labels)  # Use real_labels!
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return {
            'discriminator_loss': d_loss.item(),
            'generator_loss': g_loss.item(),
            'real_accuracy': (real_validity > 0.5).float().mean().item(),
            'fake_accuracy': (fake_validity < 0.5).float().mean().item()
        }
    
    def train(self, dataloader, epochs):
        """Full training loop"""
        
        for epoch in range(epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0
            
            for i, (real_batch, _) in enumerate(dataloader):
                
                # Training step
                metrics = self.train_step(real_batch)
                
                epoch_d_loss += metrics['discriminator_loss']
                epoch_g_loss += metrics['generator_loss']
                
                # Log progress
                if i % 100 == 0:
                    print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}]")
                    print(f"D_loss: {metrics['discriminator_loss']:.4f}, "
                          f"G_loss: {metrics['generator_loss']:.4f}")
            
            # Epoch summary
            avg_d_loss = epoch_d_loss / len(dataloader)
            avg_g_loss = epoch_g_loss / len(dataloader)
            
            print(f"Epoch {epoch} completed:")
            print(f"Average D_loss: {avg_d_loss:.4f}")
            print(f"Average G_loss: {avg_g_loss:.4f}")
            
            # Generate samples for visualization
            if epoch % 10 == 0:
                self.generate_samples(f"epoch_{epoch}.png")
    
    def generate_samples(self, filename, n_samples=64):
        """Generate and save sample images"""
        
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, 100)
            fake_images = self.generator(z)
            
            # Save images in a grid
            save_image(fake_images, filename, nrow=8, normalize=True)
        
        self.generator.train()

## Training example

trainer = GANTrainer(generator, discriminator)
trainer.train(dataloader, epochs=200)

```text

### Training Challenges and Solutions

```python
class GANTrainingStabilizer:
    """Techniques to stabilize GAN training"""
    
    def __init__(self):
        self.training_techniques = {
            'label_smoothing': self.apply_label_smoothing,
            'feature_matching': self.feature_matching_loss,
            'historical_averaging': self.historical_averaging,
            'minibatch_discrimination': self.minibatch_discrimination,
            'spectral_normalization': self.apply_spectral_norm
        }
    
    def apply_label_smoothing(self, labels, smoothing=0.1):
        """Apply label smoothing to prevent overconfident discriminator"""
        
        # Real labels: 1 -> 0.9, Fake labels: 0 -> 0.1
        if labels.mean() > 0.5:  # Real labels
            return labels - smoothing
        else:  # Fake labels
            return labels + smoothing
    
    def feature_matching_loss(self, real_features, fake_features):
        """Feature matching to prevent mode collapse"""
        
        # Match statistics of intermediate discriminator features
        real_mean = real_features.mean(dim=0)
        fake_mean = fake_features.mean(dim=0)
        
        feature_loss = F.mse_loss(fake_mean, real_mean)
        
        return feature_loss
    
    def spectral_normalization_conv(self, conv_layer):
        """Apply spectral normalization to convolution layer"""
        
        return nn.utils.spectral_norm(conv_layer)
    
    def wasserstein_loss(self, real_validity, fake_validity):
        """Wasserstein loss for improved training stability"""
        
        # WGAN loss: maximize real_scores - fake_scores
        wasserstein_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
        
        return wasserstein_loss
    
    def gradient_penalty(self, discriminator, real_data, fake_data, device):
        """Gradient penalty for WGAN-GP"""
        
        batch_size = real_data.size(0)
        
        # Random interpolation between real and fake data
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Discriminator output for interpolated data
        d_interpolated = discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Gradient penalty
        gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return penalty

class ImprovedGANTrainer(GANTrainer):
    """GAN trainer with stability improvements"""
    
    def __init__(self, generator, discriminator, **kwargs):
        super().__init__(generator, discriminator, **kwargs)
        self.stabilizer = GANTrainingStabilizer()
        self.lambda_gp = 10  # Gradient penalty coefficient
        
    def train_step_wgan_gp(self, real_batch):
        """Training step with WGAN-GP improvements"""
        
        batch_size = real_batch.size(0)
        
        # Train Discriminator
        for _ in range(5):  # Train D more than G
            self.optimizer_D.zero_grad()
            
            # Real data
            real_validity = self.discriminator(real_batch)
            
            # Fake data
            z = torch.randn(batch_size, 100)
            fake_batch = self.generator(z).detach()
            fake_validity = self.discriminator(fake_batch)
            
            # Wasserstein loss
            d_loss = self.stabilizer.wasserstein_loss(real_validity, fake_validity)
            
            # Gradient penalty
            gp = self.stabilizer.gradient_penalty(
                self.discriminator, real_batch, fake_batch, real_batch.device
            )
            
            # Total discriminator loss
            d_total_loss = d_loss + self.lambda_gp * gp
            d_total_loss.backward()
            self.optimizer_D.step()
        
        # Train Generator
        self.optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, 100)
        fake_batch = self.generator(z)
        fake_validity = self.discriminator(fake_batch)
        
        # Generator loss (maximize fake_validity)
        g_loss = -torch.mean(fake_validity)
        g_loss.backward()
        self.optimizer_G.step()
        
        return {
            'discriminator_loss': d_total_loss.item(),
            'generator_loss': g_loss.item(),
            'gradient_penalty': gp.item()
        }

```text

## GAN Variants

### Deep Convolutional GAN (DCGAN)

```python
class DCGANGenerator(nn.Module):
    """DCGAN Generator with architectural best practices"""
    
    def __init__(self, latent_dim=100, feature_maps=64, channels=3):
        super().__init__()
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            
            # state size: (feature_maps*8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            # state size: (feature_maps*4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # state size: (feature_maps*2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            # state size: (feature_maps) x 32 x 32
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (channels) x 64 x 64
        )
        
    def forward(self, input):
        return self.main(input)

class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator with architectural best practices"""
    
    def __init__(self, channels=3, feature_maps=64):
        super().__init__()
        
        self.main = nn.Sequential(
            # input is (channels) x 64 x 64
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (feature_maps) x 32 x 32
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (feature_maps*2) x 16 x 16
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (feature_maps*4) x 8 x 8
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (feature_maps*8) x 4 x 4
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

## DCGAN architectural guidelines

dcgan_principles = {
    "generator": [
        "Use transposed convolutions for upsampling",
        "Use batch normalization in all layers except output",
        "Use ReLU activation in all layers except output (use Tanh)",
        "No fully connected layers except first layer"
    ],
    
    "discriminator": [
        "Use strided convolutions for downsampling",
        "Use batch normalization in all layers except first",
        "Use LeakyReLU activations",
        "No fully connected layers except last layer"
    ]
}

```text

### Conditional GAN (cGAN)

```python
class ConditionalGenerator(nn.Module):
    """Generator that takes both noise and class label as input"""
    
    def __init__(self, latent_dim=100, n_classes=10, img_size=28, channels=1):
        super().__init__()
        
        self.img_size = img_size
        self.channels = channels
        
        # Embedding layer for class labels
        self.label_emb = nn.Embedding(n_classes, n_classes)
        
        # Input dimension is noise + embedded label
        input_dim = latent_dim + n_classes
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(input_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, channels * img_size * img_size),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        """Generate conditioned on class labels"""
        
        # Embed labels
        label_embedding = self.label_emb(labels)
        
        # Concatenate noise and label embedding
        gen_input = torch.cat((noise, label_embedding), -1)
        
        # Generate image
        img = self.model(gen_input)
        img = img.view(img.size(0), self.channels, self.img_size, self.img_size)
        
        return img

class ConditionalDiscriminator(nn.Module):
    """Discriminator that takes both image and class label"""
    
    def __init__(self, n_classes=10, img_size=28, channels=1):
        super().__init__()
        
        # Embedding for class labels
        self.label_emb = nn.Embedding(n_classes, n_classes)
        
        # Input is image + embedded label
        input_dim = channels * img_size * img_size + n_classes
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img, labels):
        """Discriminate based on image and label"""
        
        # Flatten image
        img_flat = img.view(img.size(0), -1)
        
        # Embed labels
        label_embedding = self.label_emb(labels)
        
        # Concatenate image and label
        d_input = torch.cat((img_flat, label_embedding), -1)
        
        # Discriminate
        validity = self.model(d_input)
        
        return validity

## Conditional GAN training

def train_conditional_gan(generator, discriminator, dataloader):
    """Training loop for conditional GAN"""
    
    for epoch in range(epochs):
        for i, (real_imgs, labels) in enumerate(dataloader):
            
            batch_size = real_imgs.size(0)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Real images
            real_validity = discriminator(real_imgs, labels)
            real_loss = adversarial_loss(real_validity, torch.ones_like(real_validity))
            
            # Fake images
            z = torch.randn(batch_size, 100)
            fake_labels = torch.randint(0, 10, (batch_size,))  # Random labels
            fake_imgs = generator(z, fake_labels)
            fake_validity = discriminator(fake_imgs.detach(), fake_labels)
            fake_loss = adversarial_loss(fake_validity, torch.zeros_like(fake_validity))
            
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, 100)
            fake_labels = torch.randint(0, 10, (batch_size,))
            fake_imgs = generator(z, fake_labels)
            fake_validity = discriminator(fake_imgs, fake_labels)
            
            g_loss = adversarial_loss(fake_validity, torch.ones_like(fake_validity))
            g_loss.backward()
            optimizer_G.step()

```text

### StyleGAN

```python
class StyleGenerator(nn.Module):
    """Simplified StyleGAN-inspired generator"""
    
    def __init__(self, latent_dim=512, n_layers=8):
        super().__init__()
        
        # Mapping network: Z -> W
        self.mapping_network = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LeakyReLU(0.2)
            ) for _ in range(8)]
        )
        
        # Synthesis network layers
        self.synthesis_layers = nn.ModuleList()
        
        for i in range(n_layers):
            # Each layer has adaptive instance normalization
            layer = StyleSynthesisLayer(
                in_channels=512 // (2 ** min(i, 4)),
                out_channels=512 // (2 ** min(i+1, 4)),
                style_dim=latent_dim
            )
            self.synthesis_layers.append(layer)
    
    def forward(self, z, inject_noise=True):
        """Generate image with style control"""
        
        # Map to style space
        w = self.mapping_network(z)
        
        # Start with learned constant
        x = torch.ones(z.size(0), 512, 4, 4).to(z.device)
        
        # Apply synthesis layers
        for layer in self.synthesis_layers:
            x = layer(x, w, inject_noise)
        
        return x

class StyleSynthesisLayer(nn.Module):
    """Single synthesis layer with style modulation"""
    
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        
        # Convolution
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Style modulation
        self.style_scale = nn.Linear(style_dim, in_channels)
        self.style_shift = nn.Linear(style_dim, in_channels)
        
        # Noise injection
        self.noise_strength = nn.Parameter(torch.zeros(1))
        
        # Activation
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x, style, inject_noise=True):
        # Style modulation (Adaptive Instance Normalization)
        style_scale = self.style_scale(style).unsqueeze(2).unsqueeze(3)
        style_shift = self.style_shift(style).unsqueeze(2).unsqueeze(3)
        
        # Normalize features
        x_norm = F.instance_norm(x)
        
        # Apply style
        x_styled = style_scale * x_norm + style_shift
        
        # Convolution
        x = self.conv(x_styled)
        
        # Add noise
        if inject_noise:
            noise = torch.randn_like(x)
            x = x + self.noise_strength * noise
        
        # Activation
        x = self.activation(x)
        
        return x

def style_mixing_regularization(generator, z1, z2, mixing_prob=0.5):
    """Style mixing regularization for StyleGAN"""
    
    if torch.rand(1) < mixing_prob:
        # Use different styles for different layers
        crossover_point = torch.randint(1, len(generator.synthesis_layers), (1,))
        
        # Generate with mixed styles
        w1 = generator.mapping_network(z1)
        w2 = generator.mapping_network(z2)
        
        # Apply style mixing at random crossover point
        mixed_w = torch.cat([w1[:crossover_point], w2[crossover_point:]])
        
        return generator.synthesis_forward(mixed_w)
    else:
        return generator(z1)

```text

## Applications and Use Cases

### Image Generation and Editing

```python
class ImageGANApplications:
    """Various applications of GANs in image processing"""
    
    def __init__(self):
        self.applications = {
            'face_generation': self.generate_faces,
            'image_super_resolution': self.super_resolution,
            'image_inpainting': self.inpaint_images,
            'domain_transfer': self.domain_transfer,
            'data_augmentation': self.augment_data
        }
    
    def generate_faces(self, generator, n_faces=100):
        """Generate realistic face images"""
        
        with torch.no_grad():
            z = torch.randn(n_faces, 100)
            fake_faces = generator(z)
            
        return fake_faces
    
    def super_resolution(self, lr_image, sr_generator):
        """Enhance image resolution using SRGAN"""
        
        with torch.no_grad():
            hr_image = sr_generator(lr_image)
            
        return hr_image
    
    def inpaint_images(self, masked_image, mask, inpainting_generator):
        """Fill missing parts of images"""
        
        with torch.no_grad():
            completed_image = inpainting_generator(masked_image, mask)
            
        return completed_image
    
    def domain_transfer(self, source_image, cyclegan):
        """Transfer image from one domain to another (e.g., day to night)"""
        
        with torch.no_grad():
            target_image = cyclegan.generator_AB(source_image)
            
        return target_image
    
    def augment_data(self, generator, n_augmentations=1000):
        """Generate synthetic training data"""
        
        augmented_data = []
        
        with torch.no_grad():
            for _ in range(n_augmentations):
                z = torch.randn(1, 100)
                synthetic_sample = generator(z)
                augmented_data.append(synthetic_sample)
        
        return torch.cat(augmented_data, dim=0)

## Text-to-Image Generation (Simplified)

class TextToImageGAN:
    """Simplified text-to-image GAN"""
    
    def __init__(self, text_encoder, generator):
        self.text_encoder = text_encoder  # E.g., CLIP text encoder
        self.generator = generator
        
    def generate_from_text(self, text_descriptions):
        """Generate images from text descriptions"""
        
        # Encode text to conditioning vector
        text_embeddings = self.text_encoder(text_descriptions)
        
        # Generate images conditioned on text
        z = torch.randn(len(text_descriptions), 100)
        generated_images = self.generator(z, text_embeddings)
        
        return generated_images
    
    def interpolate_between_texts(self, text1, text2, steps=10):
        """Generate interpolation between two text descriptions"""
        
        # Encode texts
        emb1 = self.text_encoder([text1])
        emb2 = self.text_encoder([text2])
        
        # Linear interpolation
        alphas = torch.linspace(0, 1, steps)
        interpolated_images = []
        
        for alpha in alphas:
            interp_emb = alpha * emb2 + (1 - alpha) * emb1
            z = torch.randn(1, 100)
            img = self.generator(z, interp_emb)
            interpolated_images.append(img)
        
        return torch.cat(interpolated_images, dim=0)

```text

### Audio and Music Generation

```python
class AudioGAN:
    """GAN for audio waveform generation"""
    
    def __init__(self, sample_rate=16000, window_size=1024):
        self.sample_rate = sample_rate
        self.window_size = window_size
        
        # Generator: noise -> raw audio waveform
        self.generator = self.build_audio_generator()
        
        # Discriminator: multiple scales for temporal modeling
        self.discriminators = nn.ModuleList([
            self.build_discriminator(scale) for scale in [1, 2, 4]
        ])
    
    def build_audio_generator(self):
        """Build generator for raw audio waveforms"""
        
        return nn.Sequential(
            # Transpose convolutions for upsampling
            nn.ConvTranspose1d(100, 1024, 4, stride=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(1024, 512, 4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(128, 1, 4, stride=2, padding=1),
            nn.Tanh()  # Audio samples in [-1, 1]
        )
    
    def build_discriminator(self, scale):
        """Build multi-scale discriminator"""
        
        return nn.Sequential(
            nn.Conv1d(1, 64, 15, stride=scale, padding=7),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(64, 128, 41, stride=4, padding=20),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(128, 256, 41, stride=4, padding=20),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(256, 512, 41, stride=4, padding=20),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(512, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def generate_audio(self, duration_seconds=2.0):
        """Generate audio of specified duration"""
        
        # Calculate required latent size
        target_samples = int(duration_seconds * self.sample_rate)
        
        # Generate
        with torch.no_grad():
            z = torch.randn(1, 100, 1)  # Latent noise
            audio_waveform = self.generator(z)
            
            # Trim or pad to desired length
            if audio_waveform.size(-1) > target_samples:
                audio_waveform = audio_waveform[:, :, :target_samples]
            elif audio_waveform.size(-1) < target_samples:
                padding = target_samples - audio_waveform.size(-1)
                audio_waveform = F.pad(audio_waveform, (0, padding))
        
        return audio_waveform.squeeze()

```text

## Evaluation Metrics

### Quantitative Metrics

```python
class GANEvaluator:
    """Comprehensive evaluation of GAN performance"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.inception_model = self.load_inception_model()
        
    def load_inception_model(self):
        """Load pre-trained Inception model for FID/IS computation"""
        
        from torchvision.models import inception_v3
        
        model = inception_v3(pretrained=True, transform_input=False)
        model.eval()
        model.to(self.device)
        
        return model
    
    def compute_fid(self, real_images, fake_images):
        """Compute Frechet Inception Distance"""
        
        # Extract features using Inception model
        real_features = self.extract_inception_features(real_images)
        fake_features = self.extract_inception_features(fake_images)
        
        # Compute statistics
        real_mean = np.mean(real_features, axis=0)
        fake_mean = np.mean(fake_features, axis=0)
        
        real_cov = np.cov(real_features, rowvar=False)
        fake_cov = np.cov(fake_features, rowvar=False)
        
        # FID formula
        diff = real_mean - fake_mean
        fid = np.dot(diff, diff) + np.trace(real_cov + fake_cov - 2 * sqrtm(real_cov @ fake_cov))
        
        return fid.real
    
    def compute_inception_score(self, fake_images, splits=10):
        """Compute Inception Score"""
        
        # Get predictions from Inception model
        with torch.no_grad():
            fake_images_resized = F.interpolate(fake_images, size=(299, 299), mode='bilinear')
            predictions = F.softmax(self.inception_model(fake_images_resized), dim=1)
        
        # Split into groups
        predictions = predictions.cpu().numpy()
        scores = []
        
        for i in range(splits):
            part = predictions[i * len(predictions) // splits:(i + 1) * len(predictions) // splits]
            
            # KL divergence calculation
            kl_div = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
            kl_div = np.mean(np.sum(kl_div, axis=1))
            scores.append(np.exp(kl_div))
        
        return np.mean(scores), np.std(scores)
    
    def compute_lpips(self, real_images, fake_images):
        """Compute Learned Perceptual Image Patch Similarity"""
        
        import lpips
        
        # Initialize LPIPS model
        lpips_model = lpips.LPIPS(net='alex').to(self.device)
        
        # Compute perceptual distances
        distances = []
        
        for real_img, fake_img in zip(real_images, fake_images):
            real_img = real_img.unsqueeze(0)
            fake_img = fake_img.unsqueeze(0)
            
            distance = lpips_model(real_img, fake_img)
            distances.append(distance.item())
        
        return np.mean(distances)
    
    def compute_precision_recall(self, real_features, fake_features, k=3):
        """Compute precision and recall using k-nearest neighbors"""
        
        # Compute pairwise distances
        real_to_fake = self.compute_pairwise_distances(real_features, fake_features)
        fake_to_real = self.compute_pairwise_distances(fake_features, real_features)
        
        # Find k-nearest neighbors
        real_nn_fake = np.sort(real_to_fake, axis=1)[:, :k]
        fake_nn_real = np.sort(fake_to_real, axis=1)[:, :k]
        
        # Compute precision and recall
        precision = np.mean(fake_nn_real[:, -1] < np.median(real_nn_fake[:, -1]))
        recall = np.mean(real_nn_fake[:, -1] < np.median(fake_nn_real[:, -1]))
        
        return precision, recall
    
    def comprehensive_evaluation(self, generator, real_dataloader, n_samples=10000):
        """Run comprehensive evaluation suite"""
        
        # Generate samples
        fake_images = []
        real_images = []
        
        generator.eval()
        with torch.no_grad():
            # Generate fake samples
            for _ in range(n_samples // 100):
                z = torch.randn(100, 100).to(self.device)
                fake_batch = generator(z)
                fake_images.append(fake_batch)
            
            # Collect real samples
            for i, (real_batch, _) in enumerate(real_dataloader):
                real_images.append(real_batch.to(self.device))
                if i * real_batch.size(0) >= n_samples:
                    break
        
        fake_images = torch.cat(fake_images, dim=0)[:n_samples]
        real_images = torch.cat(real_images, dim=0)[:n_samples]
        
        # Compute metrics
        results = {}
        
        print("Computing FID...")
        results['fid'] = self.compute_fid(real_images, fake_images)
        
        print("Computing Inception Score...")
        is_mean, is_std = self.compute_inception_score(fake_images)
        results['inception_score_mean'] = is_mean
        results['inception_score_std'] = is_std
        
        print("Computing LPIPS...")
        results['lpips'] = self.compute_lpips(real_images[:1000], fake_images[:1000])
        
        print("Computing Precision/Recall...")
        real_feats = self.extract_inception_features(real_images)
        fake_feats = self.extract_inception_features(fake_images)
        precision, recall = self.compute_precision_recall(real_feats, fake_feats)
        results['precision'] = precision
        results['recall'] = recall
        
        return results

## Usage example

evaluator = GANEvaluator()
metrics = evaluator.comprehensive_evaluation(generator, test_dataloader)

print("GAN Evaluation Results:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

```text

### Qualitative Assessment

```python
class QualitativeEvaluator:
    """Tools for qualitative evaluation of GANs"""
    
    def __init__(self, generator):
        self.generator = generator
        
    def generate_interpolation_grid(self, n_steps=10, save_path='interpolation.png'):
        """Generate interpolation between random points in latent space"""
        
        # Random start and end points
        z1 = torch.randn(1, 100)
        z2 = torch.randn(1, 100)
        
        # Linear interpolation
        alphas = torch.linspace(0, 1, n_steps).unsqueeze(1)
        interpolated_z = alphas * z2 + (1 - alphas) * z1
        
        # Generate images
        with torch.no_grad():
            interpolated_images = self.generator(interpolated_z)
        
        # Save as grid
        save_image(interpolated_images, save_path, nrow=n_steps, normalize=True)
        
        return interpolated_images
    
    def analyze_mode_collapse(self, n_samples=1000):
        """Check for mode collapse by analyzing diversity"""
        
        # Generate many samples
        generated_samples = []
        
        with torch.no_grad():
            for _ in range(n_samples // 100):
                z = torch.randn(100, 100)
                batch = self.generator(z)
                generated_samples.append(batch)
        
        all_samples = torch.cat(generated_samples, dim=0)
        
        # Compute pairwise similarities
        flattened = all_samples.view(n_samples, -1)
        similarities = F.cosine_similarity(
            flattened.unsqueeze(1),
            flattened.unsqueeze(0),
            dim=2
        )
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(n_samples).bool()
        similarities[mask] = 0
        
        # Analyze statistics
        mean_similarity = similarities.mean().item()
        max_similarity = similarities.max().item()
        
        # High mean similarity might indicate mode collapse
        mode_collapse_score = mean_similarity
        
        return {
            'mean_similarity': mean_similarity,
            'max_similarity': max_similarity,
            'mode_collapse_score': mode_collapse_score,
            'likely_mode_collapse': mode_collapse_score > 0.8
        }
    
    def visualize_training_progress(self, checkpoints_dir):
        """Create visualization of training progress"""
        
        import os
        from PIL import Image
        
        checkpoint_files = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')])
        
        progress_images = []
        
        for checkpoint_file in checkpoint_files:
            # Load checkpoint
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
            checkpoint = torch.load(checkpoint_path)
            self.generator.load_state_dict(checkpoint['generator'])
            
            # Generate sample
            with torch.no_grad():
                z = torch.randn(16, 100)  # Fixed noise for consistency
                sample = self.generator(z)
                
            # Convert to PIL and store
            grid = make_grid(sample, nrow=4, normalize=True)
            grid_np = grid.permute(1, 2, 0).numpy()
            grid_pil = Image.fromarray((grid_np * 255).astype(np.uint8))
            progress_images.append(grid_pil)
        
        # Create animation/video
        progress_images[0].save(
            'training_progress.gif',
            save_all=True,
            append_images=progress_images[1:],
            duration=500,
            loop=0
        )

```text

## Common Challenges and Solutions

### Training Instability

```python
class TrainingStabilizer:
    """Solutions for common GAN training problems"""
    
    def __init__(self):
        self.stabilization_techniques = {
            'mode_collapse': self.prevent_mode_collapse,
            'vanishing_gradients': self.handle_vanishing_gradients,
            'training_oscillation': self.stabilize_training,
            'discriminator_overpowering': self.balance_networks
        }
    
    def prevent_mode_collapse(self, generator, discriminator):
        """Techniques to prevent mode collapse"""
        
        solutions = {
            'unrolled_gan': self.implement_unrolled_gan,
            'minibatch_discrimination': self.add_minibatch_discrimination,
            'feature_matching': self.add_feature_matching,
            'diverse_loss': self.add_diversity_loss
        }
        
        return solutions
    
    def implement_unrolled_gan(self, generator, discriminator, unroll_steps=5):
        """Unrolled GAN to prevent mode collapse"""
        
        # Store original discriminator state
        original_state = copy.deepcopy(discriminator.state_dict())
        
        # Unroll discriminator training
        for _ in range(unroll_steps):
            # Simulate discriminator training step
            # (simplified - actual implementation more complex)
            pass
        
        # Compute generator loss with "unrolled" discriminator
        generator_loss = self.compute_generator_loss(generator, discriminator)
        
        # Restore discriminator state
        discriminator.load_state_dict(original_state)
        
        return generator_loss
    
    def add_minibatch_discrimination(self, discriminator, num_features=100):
        """Add minibatch discrimination layer"""
        
        class MinibatchDiscrimination(nn.Module):
            def __init__(self, input_features, num_features, num_kernels):
                super().__init__()
                self.num_features = num_features
                self.num_kernels = num_kernels
                self.T = nn.Parameter(torch.randn(input_features, num_features * num_kernels))
                
            def forward(self, x):
                # Compute minibatch features
                M = torch.mm(x, self.T)
                M = M.view(-1, self.num_features, self.num_kernels)
                
                # Compute L1 distances between samples
                diffs = M.unsqueeze(0) - M.unsqueeze(1)
                abs_diffs = torch.abs(diffs).sum(dim=2)
                
                # Sum exponential of negative distances
                minibatch_features = torch.exp(-abs_diffs).sum(dim=1)
                
                return torch.cat([x, minibatch_features], dim=1)
        
        # Add to discriminator architecture
        mb_disc_layer = MinibatchDiscrimination(
            input_features=discriminator.final_layer_size,
            num_features=num_features,
            num_kernels=5
        )
        
        return mb_disc_layer
    
    def balance_networks(self, d_loss, g_loss, d_lr, g_lr):
        """Dynamic learning rate balancing"""
        
        # If discriminator is too strong, reduce its learning rate
        if d_loss < 0.1 and g_loss > 2.0:
            d_lr *= 0.9
            g_lr *= 1.1
        
        # If generator is too strong, reduce its learning rate
        elif g_loss < 0.1 and d_loss > 2.0:
            g_lr *= 0.9
            d_lr *= 1.1
        
        return d_lr, g_lr
    
    def implement_progressive_growing(self, generator, discriminator, current_resolution):
        """Progressive growing of GANs for stable high-resolution training"""
        
        # Gradually increase resolution during training
        resolution_schedule = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        
        if current_resolution in resolution_schedule:
            next_idx = resolution_schedule.index(current_resolution) + 1
            
            if next_idx < len(resolution_schedule):
                next_resolution = resolution_schedule[next_idx]
                
                # Add new layers to networks
                generator = self.add_resolution_layer(generator, next_resolution)
                discriminator = self.add_resolution_layer(discriminator, next_resolution)
        
        return generator, discriminator

```text

## Future Directions

### Emerging Architectures

```python
class NextGenGANs:
    """Emerging GAN architectures and techniques"""
    
    def __init__(self):
        self.emerging_techniques = {
            'progressive_distillation': self.progressive_distillation,
            'neural_ode_gans': self.neural_ode_generator,
            'transformer_gans': self.transformer_based_gan,
            'efficient_attention': self.efficient_attention_gan
        }
    
    def progressive_distillation(self, teacher_gan, student_gan):
        """Progressive distillation for efficient GANs"""
        
        # Student learns to match teacher with fewer parameters
        distillation_loss = nn.MSELoss()
        
        def distillation_step(real_batch):
            z = torch.randn(real_batch.size(0), 100)
            
            # Teacher and student outputs
            with torch.no_grad():
                teacher_output = teacher_gan.generator(z)
            
            student_output = student_gan.generator(z)
            
            # Knowledge distillation loss
            kd_loss = distillation_loss(student_output, teacher_output)
            
            # Add discriminator loss
            d_loss = student_gan.discriminator_loss(student_output, real_batch)
            
            total_loss = kd_loss + 0.1 * d_loss
            
            return total_loss
        
        return distillation_step
    
    def transformer_based_gan(self, seq_len=64, d_model=512):
        """GAN using transformer architecture"""
        
        class TransformerGenerator(nn.Module):
            def __init__(self, seq_len, d_model, nhead=8, num_layers=6):
                super().__init__()
                
                self.seq_len = seq_len
                self.d_model = d_model
                
                # Input projection
                self.input_proj = nn.Linear(100, d_model)
                
                # Positional encoding
                self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))
                
                # Transformer layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=2048,
                    dropout=0.1
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=num_layers
                )
                
                # Output projection
                self.output_proj = nn.Linear(d_model, 3)  # RGB channels
                
            def forward(self, z):
                batch_size = z.size(0)
                
                # Project input
                x = self.input_proj(z).unsqueeze(1)  # [batch, 1, d_model]
                
                # Expand to sequence
                x = x.repeat(1, self.seq_len, 1)  # [batch, seq_len, d_model]
                
                # Add positional encoding
                x = x + self.pos_encoding.unsqueeze(0)
                
                # Transformer forward pass
                x = x.transpose(0, 1)  # [seq_len, batch, d_model]
                x = self.transformer(x)
                x = x.transpose(0, 1)  # [batch, seq_len, d_model]
                
                # Output projection
                output = self.output_proj(x)  # [batch, seq_len, 3]
                
                # Reshape to image format
                img_size = int(self.seq_len ** 0.5)
                output = output.view(batch_size, img_size, img_size, 3)
                output = output.permute(0, 3, 1, 2)  # [batch, 3, H, W]
                
                return torch.tanh(output)
        
        return TransformerGenerator(seq_len, d_model)
    
    def neural_ode_generator(self, latent_dim=100):
        """Generator using Neural ODEs for continuous generation"""
        
        try:
            from torchdiffeq import odeint
            
            class ODEFunc(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(dim, 256),
                        nn.Tanh(),
                        nn.Linear(256, 256),
                        nn.Tanh(),
                        nn.Linear(256, dim)
                    )
                
                def forward(self, t, y):
                    return self.net(y)
            
            class NeuralODEGenerator(nn.Module):
                def __init__(self, latent_dim):
                    super().__init__()
                    self.ode_func = ODEFunc(latent_dim)
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, 512),
                        nn.ReLU(),
                        nn.Linear(512, 28*28),
                        nn.Tanh()
                    )
                
                def forward(self, z, integration_time=1.0):
                    # Solve ODE
                    t = torch.tensor([0., integration_time]).to(z.device)
                    trajectory = odeint(self.ode_func, z, t)
                    
                    # Use final state
                    final_state = trajectory[-1]
                    
                    # Decode to image
                    img = self.decoder(final_state)
                    img = img.view(-1, 1, 28, 28)
                    
                    return img
            
            return NeuralODEGenerator(latent_dim)
            
        except ImportError:
            print("torchdiffeq not available, returning standard generator")
            return None

```text
Generative Adversarial Networks have fundamentally transformed the landscape of generative modeling, enabling the
creation of highly realistic synthetic data across multiple domains. From their original formulation as a minimax game
between generator and discriminator networks to sophisticated variants like StyleGAN and beyond, GANs continue to push
the boundaries of what's possible in artificial data generation. While training stability remains a challenge, ongoing
research in architecture design, training techniques, and evaluation metrics continues to expand the capabilities and
applications of this powerful framework.
