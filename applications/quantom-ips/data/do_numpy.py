import numpy as np

CA = 3.0
CF = 4.0 / 3.0
TR = 0.5
TF = 0.5

#! well, we work in dim reg :(
euler = np.euler_gamma

#! massses
me = 0.000511
mmu = 0.105658
mtau = 1.77684
mu = 0.055
md = 0.055
ms = 0.2
mc = 1.28
mb = 4.18

mZ = 91.1876
mW = 80.398
M = 0.93891897
Mpi = 0.13803
Mk = 0.493677
Mdelta = 1.232

me2 = me**2
mmu2 = mmu**2
mtau2 = mtau**2
mu2 = mu**2
md2 = md**2
ms2 = ms**2
mc2 = mc**2
mb2 = mb**2
mZ2 = mZ**2
mW2 = mW**2
M2 = M**2
Mpi2 = Mpi**2
Mdelta2 = Mdelta**2

#! couplings
s2w = 0.23116
c2w = 1.0 - s2w
alfa = 1 / 137.036
alphaSMZ = 0.118
GF = 1.1663787e-5  # 1/GeV^2

def get_s(Q2):
    lam2 = 0.4
    Q02 = 1.0
    L = np.array(np.log(Q02 / lam2), dtype=np.float64)
    return np.log(np.log(Q2 / lam2) / L)


def get_parQ2(par, Q2):
    s = get_s(Q2)
    return par[0] + par[1] * s + par[2] * s**2


def get_pdf(x, Q2, par):
    A = get_parQ2(par[:3], Q2)
    a = get_parQ2(par[3:6], Q2)
    b = get_parQ2(par[6:9], Q2)
    c = get_parQ2(par[9:12], Q2)
    d = get_parQ2(par[12:15], Q2)
    e = get_parQ2(par[15:18], Q2)
    return A * x**a * (1 - x) ** b * (1 + c * x + d * x**2 + e * x**3)


def get_u(x, Q2, par_u):
    return get_pdf(x, Q2, par_u)


def get_d(x, Q2, par_d):
    return get_pdf(x, Q2, par_d)


def cross_section_nv(x, Q2, par_u, par_d):
    if isinstance(x, float):
        return _cross_section(x, Q2, par_u, par_d)
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = _cross_section(x[i], Q2[i], par_u, par_d)
    return out


def cross_section(x, Q2, par_u, par_d):
   return _cross_section(x, Q2, par_u, par_d)

def _cross_section(x, Q2, par_u, par_d):
    rs = 140.0
    Q2 = np.clip(Q2, a_min=mc2, a_max=rs**2 - M2, dtype=np.float64)
    x = np.clip(x, a_min=mc2 / (rs**2 - M2), a_max=1.0, dtype=np.float64)
    y = Q2 / (rs**2 - M2) / x
    Yp = 1 + (1 - y) ** 2
    Ym = 1 - (1 - y) ** 2
    K2 = Yp + 2 * x**2 * y**2 * M2 / Q2
    KL = -(y**2)
    K3 = Ym * x
    alfa = alfa = 1 / 137

    norm = 2 * np.pi * alfa**2 / x / y / Q2 * y / Q2

    xu = x * get_u(x, Q2, par_u)
    xd = x * get_d(x, Q2, par_d)

    eU2 = 4 / 9
    eD2 = 1 / 9

    F2 = eU2 * xu + eD2 * xd
    FL = 0
    F3 = 0

    xsec = norm * (K2 * F2 + KL * FL + K3 * F3)
    return xsec

def cast(x):
    return x

if __name__ == "__main__":

    par_u = np.array(
        [
            1.12995643e-01,
            2.84816039e-02,
            3.49711182e-02,
            -1.07992996e00,
            -8.67724315e-02,
            2.67006979e-02,
            -3.42002556e-01,
            2.16083133e00,
            -7.06215971e-02,
            5.99208979e01,
            -3.80095917e01,
            6.96188257e00,
            -1.93266172e02,
            1.76589359e02,
            -3.96195694e01,
            1.62782187e02,
            -1.79165065e02,
            4.60385729e01,
        ],
        dtype=np.float64,
    )

    par_d = np.array(
        [
            -6.26740092e-02,
            3.54363691e-01,
            -1.06902973e-01,
            -1.20394986e00,
            1.55339980e-01,
            -7.87939114e-02,
            2.75190189e-01,
            3.99743001e00,
            -8.40757936e-01,
            4.31272688e01,
            -3.26453695e01,
            7.87648066e00,
            -1.02351414e02,
            1.08446491e02,
            -2.81885597e01,
            2.58442965e01,
            -4.35307825e01,
            1.22755848e01,
        ],
        dtype=np.float64,
    )

    np.random.seed(42)

    TARGET_EVENTS = 204800  
    CHUNK = 200000

    collected = []
    total = 0
    while total < TARGET_EVENTS:
        x = np.random.rand(CHUNK)
        Q2 = 10.0 * np.random.rand(CHUNK)
        w = cross_section(x, Q2, par_u, par_d)
        keep = np.random.rand(CHUNK) < (w / w.max())
        accepted = np.column_stack([x[keep], Q2[keep]])
        if total + accepted.shape[0] > TARGET_EVENTS:
            accepted = accepted[: TARGET_EVENTS - total]
        collected.append(accepted)
        total += accepted.shape[0]

    result = np.vstack(collected).astype(np.float32)
    np.save("data.npy", result)
