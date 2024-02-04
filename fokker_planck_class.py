import pde

neumann_bc = "auto_periodic_neumann"

final_bc = neumann_bc

class KuramotoSivashinskyPDE(pde.PDEBase):

    def __init__(self, diffusivity=1, bc=final_bc, bc_laplace=final_bc):
        """ initialize the class with a diffusivity and boundary conditions
        for the actual field and its second derivative """
        self.diffusivity = diffusivity
        self.bc = bc
        self.bc_laplace = bc_laplace


    def evolution_rate(self, probability, t=0):
        """ numpy implementation of the evolution equation """
        probability_lapacian = probability.laplace(bc=self.bc)
        probability_gradient = probability.gradient(bc=final_bc)
        probability_divergence = probability.divergence(bc=final_bc)
        return (- probability_lapacian.laplace(bc=self.bc_laplace)
                - probability_lapacian
                - 0.5 * self.diffusivity * (probability_gradient @ probability_gradient))


    def _make_pde_fp_numba(self, probability, diffusion, drift):
        """ the numba-accelerated evolution equation """
        # make attributes locally available
        diffusivity = self.diffusivity

        div_term = probability @ drift
        lapl_term = probability @ diffusion

        # create operators
        # laplace_u = probability.grid.make_operator("laplace", bc=self.bc)
        # gradient_u = probability.grid.make_operator("gradient", bc=self.bc)
        # divergence_u = probability.grid.make_operator("divergence", bc=self.bc)
        # laplace2_u = probability.grid.make_operator("laplace", bc=self.bc_laplace)
        # dot = pde.VectorField(probability.grid).make_dot_operator()
        
        div = div_term.grid.make_operator("divergence", bc=self.bc)
        lapl = lapl_term.grid.make_operator("laplace", bc=self.bc)

        # d_dx = probability.grid.make_operator("d_dx", bc=self.bc)
        # d_dy = probability.grid.make_operator("d_dx", bc=self.bc)
        # d_dz = probability.grid.make_operator("d_dx", bc=self.bc)
        
        # d_d2x = probability.grid.make_operator("d_d2x", bc=self.bc)
        # d_d2y = probability.grid.make_operator("d_d2x", bc=self.bc)
        # d_d2z = probability.grid.make_operator("d_d2x", bc=self.bc)
        
        # @pde.tools.numba.jit
        # def pde_fp(probability_data, t=0):
        #     """ compiled helper function evaluating right hand side """
        #     probability_lapacian = laplace_u(probability_data)
        #     probability_grad = gradient_u(probability_data)
        #     return (- laplace2_u(probability_lapacian)
        #             - probability_lapacian
        #             - diffusivity / 2 * dot(probability_grad, probability_grad))

        # return pde_fp
        
        @pde.tools.numba.jit
        def pde_fp(probability_data, t=0):
            """ compiled helper function evaluating right hand side """
            div_applied = div(div_term)
            lapl_applied = lapl(lapl_term)
            return (-div_applied + lapl_applied)

        return pde_fp