import numpy as np
import pickle

class Gen3BodyPhaseSpace:

    '''
    The code below has been copied straight from ChatGPT. I briefly tested its functionality, logic and physics accuracy.  
    '''

    def __init__(self,masses,output_loc,mask_dimension=500,mask_ranges=[[-1.5,1.5],[-1.5,1.5]],eps=1e-14,n_grid_pts=2000,n_events=1000000):
      self.eps = eps
      self.n_grid_pts = n_grid_pts
      self.masses = masses
      self.output_loc = output_loc
      self.n_events = n_events
      self.mask_dimension = mask_dimension
      self.mask_ranges = mask_ranges

    def kaellen(self,a, b, c):
        """Källén function λ(a,b,c)."""
        return a*a + b*b + c*c - 2*(a*b + a*c + b*c)

    def two_body_momentum(self,parentM, mA, mB):
       """
       |p| for parent->A+B in the parent rest frame.
       parentM, mA, mB can be arrays or scalars (vectorized).
       """
       lam = self.kaellen(parentM**2, mA**2, mB**2)
       lam = np.maximum(lam, 0.0)
       return np.sqrt(lam) / (2.0 * parentM)

    def build_s_cdf(self, M, m1, m2, m3):
       """
       Build s_grid (m12^2) and its CDF for inverse-transform sampling.
       Returns (s_grid, cdf_grid) where cdf_grid[0]=0, cdf_grid[-1]=1.
       """
       s_min = (m1 + m2)**2
       s_max = (M - m3)**2
       s_grid = np.linspace(s_min, s_max, self.n_grid_pts)
       sqrt_s = np.sqrt(s_grid)

       # phase-space PDF in s (proportional to p_parent * q_sub / (M * sqrt(s)))
       p_parent = self.two_body_momentum(M, sqrt_s, m3)   # p*(M; m12, m3)
       q_sub   = self.two_body_momentum(sqrt_s, m1, m2)   # q*(m12; m1, m2)
       pdf = (p_parent * q_sub) / (M * sqrt_s + self.eps)
       pdf = np.maximum(pdf, 0.0)

       # integrate with trapezoid rule
       ds = np.diff(s_grid)
       mid = 0.5 * (pdf[:-1] + pdf[1:])
       cumsum = np.concatenate([[0.0], np.cumsum(mid * ds)])
       total = cumsum[-1]
       if total <= 0:
          raise RuntimeError("PDF integrated to zero; check masses or grid.")
       cdf = cumsum / total
       return s_grid, cdf

    def sample_s_from_cdf(self, u, s_grid, cdf_grid):
       """
       Inverse-transform: u in [0,1) (array-like) -> s samples via interpolation.
       """
       u = np.asarray(u)
       s = np.interp(u, cdf_grid, s_grid)
       return s

    def boost_from_rest(self, E_star, p_star, beta_vec, gamma):
      """
      Boost four-vectors (E*, p_star) from subsystem rest to lab frame.
      E_star: (N,) ; p_star: (N,3) ; beta_vec: (N,3) ; gamma: (N,)
      Returns (E_lab (N,), p_lab (N,3)).
      """
      bp = np.sum(beta_vec * p_star, axis=1)
      E_lab = gamma * (E_star + bp)
      beta2 = np.sum(beta_vec * beta_vec, axis=1)
      factor = np.where(beta2 > 1e-30, (gamma - 1.0) / (beta2 + self.eps), 0.0)
      coeff = (factor * bp + gamma * E_star)[:, None]
      p_lab = p_star + coeff * beta_vec
      return E_lab, p_lab

    def gen_3body_invCDF(self, n_events, M, m1, m2, m3, rng=None):
       """
       Generate n_events uniformly in 3-body Lorentz-invariant phase space
       using inverse-CDF sampling for the intermediate invariant mass m12.
       Returns p4 array shaped (n_events, 3, 4): (E, px, py, pz) for particles (1,2,3).
       Ordering: 0->m1, 1->m2, 2->m3 (consistent with typical Dalitz X/Y formulas).
       """
       if rng is None:
          rng = np.random
       # precompute (or compute here) the s-grid + CDF
       s_grid, cdf = self.build_s_cdf(M, m1, m2, m3)

       # sample s = m12^2 via inverse CDF
       u_s = rng.rand(n_events)
       s_samples = self.sample_s_from_cdf(u_s, s_grid, cdf)
       m12 = np.sqrt(s_samples)    # (N,)

       # parent-frame momentum magnitude of the (12) system (and of particle 3)
       p_parent = self.two_body_momentum(M, m12, m3)   # (N,)
       E3 = np.sqrt(m3**2 + p_parent**2)          # (N,)

       # isotropic direction for particle 3 in parent rest
       cost = 2.0 * rng.rand(n_events) - 1.0
       sint = np.sqrt(np.maximum(0.0, 1.0 - cost*cost))
       phi  = 2.0 * np.pi * rng.rand(n_events)
       px3 = p_parent * sint * np.cos(phi)
       py3 = p_parent * sint * np.sin(phi)
       pz3 = p_parent * cost
       p3_sp = np.stack([px3, py3, pz3], axis=1)   # (N,3)
       p3_4  = np.column_stack([E3, p3_sp])       # (N,4)

       # in m12 rest: energies and momentum for particle1 (and 2)
       E1_star = 0.5 * (s_samples + m1**2 - m2**2) / m12
       p1_star_mag = np.sqrt(np.maximum(E1_star**2 - m1**2, 0.0))

       # isotropic directions for particle1 in m12 rest frame
       cost2 = 2.0 * rng.rand(n_events) - 1.0
       sint2 = np.sqrt(np.maximum(0.0, 1.0 - cost2*cost2))
       phi2  = 2.0 * np.pi * rng.rand(n_events)
       p1x_star = p1_star_mag * sint2 * np.cos(phi2)
       p1y_star = p1_star_mag * sint2 * np.sin(phi2)
       p1z_star = p1_star_mag * cost2
       p1_star_sp = np.stack([p1x_star, p1y_star, p1z_star], axis=1)  # (N,3)
       p2_star_sp = -p1_star_sp

       p1_star_4 = np.column_stack([E1_star, p1_star_sp])
       E2_star = np.sqrt(m2**2 + p1_star_mag**2)
       p2_star_4 = np.column_stack([E2_star, p2_star_sp])

       # cluster (12) four-vector in parent frame: R = parent - p3
       E12 = M - E3
       p12_sp = -p3_sp
       # cluster velocity + gamma
       beta_vec = p12_sp / (E12[:, None] + self.eps)
       beta2 = np.sum(beta_vec * beta_vec, axis=1)
       gamma = 1.0 / np.sqrt(np.maximum(1.0 - beta2, self.eps))

       # boost particles 1 & 2 from m12 rest to parent rest
       E1_lab, p1_lab_sp = self.boost_from_rest(p1_star_4[:,0], p1_star_sp, beta_vec, gamma)
       E2_lab, p2_lab_sp = self.boost_from_rest(p2_star_4[:,0], p2_star_sp, beta_vec, gamma)

       p1_4 = np.column_stack([E1_lab, p1_lab_sp])
       p2_4 = np.column_stack([E2_lab, p2_lab_sp])
       p3_4 = p3_4

       # final shape (N, 3, 4)
       p4 = np.stack([p1_4, p2_4, p3_4], axis=1)
       return p4

    def XY_from_4vecs(self, p4, M, m1, m2, m3):
       """
       Compute Dalitz X,Y from four-vectors p4 (N,3,4).
       Follows X = sqrt(3)*(T1-T2)/Q ; Y=3*T3/Q - 1   with Q = M - (m1+m2+m3).
       """
       E = p4[:, :, 0]
       T1 = E[:, 0] - m1
       T2 = E[:, 1] - m2
       T3 = E[:, 2] - m3
       Q = M - (m1 + m2 + m3)
       X = np.sqrt(3.0) * (T1 - T2) / Q

       f= 2*((m3 + m1 + m2) / (m1+m2))
       Y = f * T3 / Q - 1.0
       return X, Y
    
    def create_phase_space_mask(self):
       H = []
       omega_flag = []

       for parent_particle, mass_set in self.masses.items():
         # Omega flag --> In case we need to switch DP parameterization:
         if parent_particle.lower() == "omega":
            omega_flag.append(True)
         else:
            omega_flag.append(False)

         # Generate phase space events:
         phase_space_events = self.gen_3body_invCDF(
          self.n_events,
          mass_set[0],
          mass_set[1],
          mass_set[2],
          mass_set[3]
         )
       
         # Get X / Y
         X, Y = self.XY_from_4vecs(
          phase_space_events,
          mass_set[0],
          mass_set[1],
          mass_set[2],
          mass_set[3]
         )

         # Create the mask:
         current_H, xb, yb = np.histogram2d(X,Y,self.mask_dimension,range=self.mask_ranges)
         xc = 0.5*(xb[:-1] + xb[1:])
         yc = 0.5*(yb[:-1] + yb[1:])
         
         H.append(current_H[None,:,:])
         
       # Write everything to file so that we can use it as many times as we want:
       with open(f"{self.output_loc}.pkl",'wb') as f:
          pickle.dump((np.concatenate(H,0),xc,yc,np.array(omega_flag,dtype=np.bool)),f)
       

       


