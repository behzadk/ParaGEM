import multiprocess as mp
import copy
import numpy as np
import time
from scipy.integrate import ode
from scipy.integrate import odeint

from loguru import logger
from bk_comms import utils

class TimeSeriesSimulation:
    def __init__(self, t_0, t_end, steps, method='vode'):
        self.t_0 = t_0
        self.t_end = t_end
        self.steps = steps
        self.method = method

    def simulate(self, particle):
        y0 = copy.deepcopy(particle.init_y)
        t_0 = self.t_0
        t_end = self.t_end
        steps = self.steps
        method = self.method

        if method == "odeint":
            t = np.linspace(t_0, t_end, steps)
            sol = odeint(particle.diff_eqs, y0, t, args=(), Dfun=particle.calculate_jacobian, col_deriv=True, mxstep=10000)

        elif method == "vode":
            sol = []
            t = []
            steps = 100
            t_points = list(np.linspace(t_0, t_end, steps))

            # Approximately set atol, rtol
            atol_list = []
            rtol_list = []
            for y in y0:
                if y > 50.0:
                    atol_list.append(1e-3)
                    rtol_list.append(1e-3)
                else:
                    atol_list.append(1e-6)
                    rtol_list.append(1e-3)
            
            start = time.time()
            solver = ode(particle.diff_eqs_vode, jac=None).set_integrator(
                "vode", method="bdf", atol=1e-4, rtol=1e-3, max_step=0.01
            )
            # print("solver initiated")

            solver.set_initial_value(y0, t=t_0)

            while solver.successful() and solver.t < t_end:
                step_out = solver.integrate(t_end, step=True)
                
                if solver.t >= t_points[0]:
                    mx = np.ma.masked_array(step_out, mask=step_out==0)
                    sol.append(step_out)
                    t.append(solver.t)
                    t_points.pop(0)

            sol = np.array(sol)
            t = np.array(t)
            end = time.time()
            logger.info(f'Simulation time: {end - start}')

        return sol, t

    def simulate_particles(self, particles, n_processes=1, sim_timeout=360.0, parallel=True):
        if parallel:
            print("running parallel")

            def wrapper(args):
                idx, args = args
                sol, t = self.simulate(args)
                return (idx, sol, t)

            pool = mp.get_context("spawn").Pool(n_processes, maxtasksperchild=1)
            futures_mp_sol = pool.imap_unordered(wrapper, enumerate(particles))

            for particle in particles:
                try:
                    idx, sol, t = futures_mp_sol.next(timeout=sim_timeout)
                    particles[idx].sol = sol
                    particles[idx].t = t

                except mp.context.TimeoutError:
                    print("TIMEOUT ERROR")
                    break

            print("Terminating pool")
            pool.terminate()
                    
            for idx, p in enumerate(particles):
                if not hasattr(p, "sol"):
                    print(f"Particle {idx} has no sol")
                    p.sol = None
                    p.t = None
        else:
            p_idx = 0
            for p in particles:
                start = time.time()
                print(f"Simulating particle idx: {p_idx}")
                output = sim_community(p)
                p.sol = output[0]
                p.t = output[1]
                p_idx += 1
                end = time.time()
                print("Sim time: ", end - start)

class GrowthRateConditionedMedia:
    def __init__(self, conditioner_particles_dir, condition_media_time_sample):
        self.conditioner_particles_dir = conditioner_particles_dir
        self.condition_media_time_sample = condition_media_time_sample
        
        repeat_prefix = "run_"
        run_dirs = [
            utils.get_experiment_repeat_directories(
                exp_dir=x, repeat_prefix=repeat_prefix
            )
            for x in self.conditioner_particles_dir
        ]
        
        self.conditoner_particles = [utils.load_all_particles(x) for x in run_dirs]

        for idx, _ in enumerate(self.input_particles):
            self.input_particles[idx] = utils.filter_particles_by_distance(
            self.input_particles[idx], epsilon=epsilon[idx]
        )

    def simulate(self, particle):
        n_populations = len(particle.populations)
        n_conditioned_medias = len(self.conditoner_particles)

        particle_growth_rates = np.zeros(shape=[n_conditioned_medias, n_populations])

        for idx, condit_part in enumerate(self.conditoner_particles):
            t_idx = utils.find_nearest(condit_part.t, self.condition_media_time_sample)
            conditioned_sol = condit_part.sol[t_idx]

            growth_rates, flux_matrix = particle.sim_step(conditioned_sol)
            particle_growth_rates[idx] = growth_rates
    
        return particle_growth_rates

    def simulate_particles(self, particles):
        for p in particles:
            p.sol = self.simulate(particle)