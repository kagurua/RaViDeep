import math
import numpy as np


# particle filter options
N_PARTICLE = 100
STATE_SIZE = 3  # State size [x,y,theta]
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling
SIM_TIME = 70.0  # simulation time [s]
show_animation = True


class Particle:

    def __init__(self):
        self.w = 1.0 / N_PARTICLE
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.dtheta = 0.0


def normalize_weight(particles):
    sum_w = sum([p.w for p in particles])
    try:
        for i in range(N_PARTICLE):
            particles[i].w /= sum_w
    except ZeroDivisionError:
        for i in range(N_PARTICLE):
            particles[i].w = 1.0 / N_PARTICLE
        return particles
    return particles


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def calc_final_state(particles):
    xEst = np.zeros((STATE_SIZE, 1))
    RT = np.zeros(3)  # coordinate transform
    particles = normalize_weight(particles)
    for i in range(N_PARTICLE):
        dx = particles[i].dx
        dy = particles[i].dy
        d2 = dx ** 2 + dy ** 2
        d = math.sqrt(d2)
        yaw = pi_2_pi(math.atan2(dy, dx) + particles[i].theta)
        particles[i].x = particles[i].x + d * math.cos(yaw)
        particles[i].y = particles[i].y + d * math.sin(yaw)
        particles[i].theta = particles[i].dtheta + particles[i].theta

        xEst[0, 0] += particles[i].w * particles[i].x
        xEst[1, 0] += particles[i].w * particles[i].y
        xEst[2, 0] += particles[i].w * particles[i].theta

        RT[0] += particles[i].w * particles[i].dx
        RT[1] += particles[i].w * particles[i].dy
        RT[2] += particles[i].w * particles[i].dtheta
    return xEst, RT


def particle_filtering(particles, u, z, Q_est, Q_reg, DT):
    particles = predict_particles(particles, u, Q_est, DT)
    particles = update_with_observation(particles, z, Q_reg)
    particles = resampling(particles)
    return particles


def predict_particles(particles, u, Q_est, DT):
    for i in range(N_PARTICLE):
        px = None
        ud = u + (np.random.randn(1, 2) @ Q_est ** 0.5).T  # add noise, sampling
        dpx = motion_model(px, ud, DT)
        particles[i].dx = dpx[0, 0]
        particles[i].dy = dpx[1, 0]
    return particles


def motion_model(x, u, DT):
    return DT * u


def update_with_observation(particles, z, Q_reg):
    for ip in range(N_PARTICLE):
        particles[ip].dtheta = z[2]
        w = compute_weight(particles[ip], z[:2], Q_reg)
        particles[ip].w = w
    return particles


def compute_weight(particle, z, Q_cov):
    zp = np.array([particle.dx, particle.dy]).reshape(2, 1)
    dz = z - zp
    try:
        invQ = np.linalg.inv(Q_cov)
    except np.linalg.LinAlgError:
        return 1.0
    num = math.exp(-0.5 * dz.T @ invQ @ dz)
    den = 2.0 * math.pi * math.sqrt(np.linalg.det(Q_cov))
    w = num / den
    return w


def resampling(particles):
    particles = normalize_weight(particles)
    pw = []
    for i in range(N_PARTICLE):
        pw.append(particles[i].w)
    pw = np.array(pw)
    n_eff = 1.0 / (pw @ pw.T)  # Effective particle number
    if n_eff < NTH:  # resampling
        w_cum = np.cumsum(pw)
        base = np.cumsum(pw * 0.0 + 1 / N_PARTICLE) - 1 / N_PARTICLE
        resample_id = base + np.random.rand(base.shape[0]) / N_PARTICLE
        inds = []
        ind = 0
        for ip in range(N_PARTICLE):
            while (ind < w_cum.shape[0] - 1) \
                    and (resample_id[ip] > w_cum[ind]):
                ind += 1
            inds.append(ind)
        tmp_particles = particles[:]
        for i in range(len(inds)):
            particles[i].x = tmp_particles[inds[i]].x
            particles[i].y = tmp_particles[inds[i]].y
            particles[i].theta = tmp_particles[inds[i]].theta
            particles[i].dx = tmp_particles[inds[i]].dx
            particles[i].dy = tmp_particles[inds[i]].dy
            particles[i].w = 1.0 / N_PARTICLE
    return particles