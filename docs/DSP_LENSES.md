# The 4 DSP Analytical Lenses

Modern machine learning treats deep neural networks as opaque statistical black boxes. `juicer.kn` treats them as **Cascaded Digital Filter Networks**.

---

## Lens 1: Spectral Transfer Function & Eigenvalue Attenuation ($H(\omega)$)

### The Theory
A residual Transformer block computes:
$$x_{l+1} = x_l + \Delta W_l(x_l)$$
where $\Delta W_l = W_{\text{out}} \cdot W_{\text{in}}$ represents the residual modification applied to the input latent manifold.

In linear systems and digital filter design, an audio processing block or filter stage with:
$$|H(\omega)| \approx 1 \quad \text{and} \quad \angle H(\omega) \approx 0$$
is an **All-Pass / Identity Filter**. In hardware or software audio DSP, an All-Pass filter operating at 0 dB gain with zero phase alteration is pure computational dead weight.

### The Algorithm
Using power iteration with deflation on the weight matrix product:
1. Compute top eigenvalue / spectral radius $\rho(\Delta W_l)$.
2. Compute secondary eigenvalue $\sigma_2$ and condition number $\kappa = \frac{\rho}{\sigma_2}$.
3. Delta Gain in decibels:
   $$G_{\text{dB}} = 20 \log_{10}(\rho)$$

### Verdict Rules
- If $G_{\text{dB}} < -22 \text{ dB}$ or ($G_{\text{dB}} < -15 \text{ dB}$ and $\kappa < 1.15$): The block is an **All-Pass Idling Filter** and safely prunable.

---

## Lens 2: Permutation Entropy Velocity ($dH/dl$) & LZ76 Complexity

### The Theory
Permutation Entropy ($H_{\text{PE}}$) evaluates ordinal dynamics without requiring statistical Gaussian assumptions. 

For delay embedding dimension $D=3$, each triple $(x_t, x_{t+1}, x_{t+2})$ maps to one of $3! = 6$ Lehmer permutation patterns.
The Shannon entropy across patterns is normalized:
$$H_{\text{PE}} = \frac{-\sum_{i=1}^{6} p_i \ln(p_i)}{\ln(6)} \in [0.0, 1.0]$$

### Entropy Velocity
Across model depth $l = 1 \dots L$, the rate of information entropy transformation is:
$$v_H(l) = \frac{d H_{\text{PE}}(l)}{d l} \approx H_{\text{PE}}(l) - H_{\text{PE}}(l-1)$$

- **Early Layers:** Rapid entropy collapse ($v_H \ll 0$) as macro-geometric latents lock into place.
- **Middle Layers:** $v_H \approx 0$. **Computational Idling**—weights merely recirculate latents without adding entropy or rejecting noise.
- **Late Layers:** Fine texture modulation.

---

## Lens 3: Higher-Order Spectral Analysis & Quadratic Phase Coupling (QPC)

### The Theory
In Multi-Reference and cross-attention architectures, layers act as frequency modulators and mixers.
Standard power spectrum $P(f)$ discards phase information. The **Bispectrum** retains phase:
$$B(f_1, f_2) = \mathbb{E}[X(f_1) X(f_2) X^*(f_1 + f_2)]$$

Normalized **Bicoherence** ($b^2$):
$$b^2(f_1, f_2) = \frac{|B(f_1, f_2)|^2}{P(f_1) P(f_2) P(f_1 + f_2)} \in [0.0, 1.0]$$

- High $b^2$ ($> 0.20$): True harmonic phase coupling between feature streams.
- Low $b^2$ with high out-of-band energy: **Intermodulation Distortion**—the physical cause of ghosting artifacts and hallucinations.

---

## Lens 4: Fractional Fourier Chirp Matched Filtering (FrFT)

### The Theory
Flow matching diffusion models (Wan 2.1, Flux) execute smooth trajectories over timesteps $t \in [1 \dots T]$. 

The Fractional Fourier Transform rotates signal representations in the time-frequency plane $(t, f)$ by angle $\alpha$:
$$\mathcal{F}_\alpha[x](u) = \int x(t) K_\alpha(t, u) \, dt$$
where $K_\alpha$ is a chirped kernel.

FrFT proves that non-stationary flow dynamics require full model capacity only during initial macro-geometric timesteps ($t=1$), while intermediate linear flow steps ($t=2 \dots T-1$) can bypass alternating transformer blocks with near-zero visual degradation.
