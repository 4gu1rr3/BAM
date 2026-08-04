import torch

sd = torch.load('logs/l12/bam_ssmax/version_02/model.pt', map_location='cpu')
alpha_keys = sorted([k for k in sd.keys() if 'theta_alpha' in k], key=lambda k: int(k.split('.')[1]))
beta_keys = sorted([k for k in sd.keys() if 'theta_beta' in k], key=lambda k: int(k.split('.')[1]))
mu_keys = sorted([k for k in sd.keys() if 'theta_mu' in k], key=lambda k: int(k.split('.')[1]))
eps = 1e-5

def bias(pos, theta_alpha, theta_beta, mu):
    return -( (abs(pos - mu) + eps) ** theta_beta ) * torch.exp(torch.tensor(theta_alpha)).item()

print(f"{'layer':>5} {'head':>4} {'theta_a':>8} {'theta_b':>8} | {'bias@1':>12} {'bias@100':>12} {'bias@25600':>12} {'bias@51199':>12} | long>local?")
for lk, bk, mk in zip(alpha_keys, beta_keys, mu_keys):
    layer_idx = int(lk.split('.')[1])
    a = sd[lk].flatten().tolist()
    b = sd[bk].flatten().tolist()
    mu = sd[mk].flatten().tolist()
    for h in range(len(b)):
        if b[h] < 0:
            theta_a, theta_b, m = a[h], b[h], mu[h]
            b1 = bias(1, theta_a, theta_b, m)
            b100 = bias(100, theta_a, theta_b, m)
            b25600 = bias(25600, theta_a, theta_b, m)
            b51199 = bias(51199, theta_a, theta_b, m)
            favors_long = "YES" if b51199 > b1 else "no"
            print(f"{layer_idx:>5} {h:>4} {theta_a:>8.3f} {theta_b:>8.3f} | {b1:>12.4g} {b100:>12.4g} {b25600:>12.4g} {b51199:>12.4g} | {favors_long}")
