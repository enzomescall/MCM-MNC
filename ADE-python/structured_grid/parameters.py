#This module is to put the input condition for the simulation
import numpy as np
import buoyancy_better
from path_stuff.path_tester import lt, lg

runtime=1	#simulation time
dt=0.1		#time step

#put your velocity model here, it may be function or you can import from a file
#keep the dimension in mind

def velocity_model(CV_zGRID_nos, CV_yGRID_nos, CV_xGRID_nos):
	#need to difine at cv inteference points
	u = np.ones((CV_zGRID_nos,CV_yGRID_nos,CV_xGRID_nos))
	v = np.ones((CV_zGRID_nos,CV_yGRID_nos,CV_xGRID_nos))
	w = np.ones((CV_zGRID_nos,CV_yGRID_nos,CV_xGRID_nos))
 
	r0 = 0.1
	gamma0 = 1
 
	rho = r0 * np.ones((CV_zGRID_nos,CV_yGRID_nos,CV_xGRID_nos))
	D = 0.5 * (1/(gamma0**2)) * np.ones((CV_zGRID_nos,CV_yGRID_nos,CV_xGRID_nos))
 
	for i in range(CV_xGRID_nos):
		print("finished layer: ", i)
		for j in range(CV_yGRID_nos):
			for k in range(CV_zGRID_nos):
				i = lg(i)
				j = lt(j)
				k = -5000*k//CV_zGRID_nos # ideally we have a function with the height of the ionian or smth
				coord_force, rho_arr = buoyancy_better.forces(j, i, k)
				u[k, j, i] = coord_force[0]
				v[k, j, i] = coord_force[1]
				w[k, j, i] = coord_force[2]
    
				rho[k, j, i] = rho_arr
    
				sigma = np.linalg.norm(coord_force[0], coord_force[1], coord_force[2])
				D[k, j, i] = np.square(sigma)

	assert u.shape==(CV_zGRID_nos,CV_yGRID_nos,CV_xGRID_nos), "inncorect, shape of u"
	assert v.shape==(CV_zGRID_nos,CV_yGRID_nos,CV_xGRID_nos), "inncorect, shape of v"
	assert w.shape==(CV_zGRID_nos,CV_yGRID_nos,CV_xGRID_nos), "inncorect, shape of w"
	assert rho.shape==(CV_zGRID_nos,CV_yGRID_nos,CV_xGRID_nos), "inncorect, shape of rho"
	assert D.shape==(CV_zGRID_nos,CV_yGRID_nos,CV_xGRID_nos), "inncorect, shape of gamma"

	return u, v, w, rho, D


#put your diffusion model here, it may be function or you can import from a file
#keep the dimension in mind
# def diffusion_model(CV_zGRID_nos,CV_yGRID_nos,CV_xGRID_nos):
	G0=0.01
	
	#need to difine at cv inteference points
	# gamma=G0*np.ones((CV_zGRID_nos,CV_yGRID_nos,CV_xGRID_nos))		#create real diffusion coeff grid

	
	#subtract false diffusion coefficient due to oblique flow
	
	# for j in range(1, yGRID_nos):
	# 	for i in range(1, xGRID_nos):
	# 		if u[i,j]!=0 and v[i,j]!=0:
	# 			theta=atan(v[i,j]/u[i,j])
	# 			gamma[i,j]=gamma[i,j]-rho[i,j]*sqrt(u[i,j]**2+v[i,j]**2)*dxg(i)*dyg(j)*sin(2*theta)/(4*dyg(j)*sin(theta)**3+4*dxg(i)*cos(theta)**3) 
	
	# assert gamma.shape==(CV_zGRID_nos,CV_yGRID_nos,CV_xGRID_nos), "inncorect, shape of gamma"
	# return gamma


#put your density model here, it may be function or you can import from a file
#keep the dimension in mind
# def density_model(CV_zGRID_nos,CV_yGRID_nos,CV_xGRID_nos):
# 	r0=1

# 	rho = np.load("densities")

# 	#need to difine at cv inteference points
# 	rho=r0*rho

# 	assert rho.shape==(CV_zGRID_nos,CV_yGRID_nos,CV_xGRID_nos), "inncorect, shape of rho"
 
# 	return rho


#put your sourse term model here, it may be function or you can import from a file
#keep the dimension in mind
def source_model(zGRID_nos,yGRID_nos,xGRID_nos):
	sp0=0
	sc0=0

	#need to be defined at all grid points
	sp=sp0*np.ones((zGRID_nos,yGRID_nos,xGRID_nos))
	sc=sc0*np.ones((zGRID_nos,yGRID_nos,xGRID_nos))

	assert sp.shape==(zGRID_nos,yGRID_nos,xGRID_nos), "inncorect,shape of sp"
	assert sc.shape==(zGRID_nos,yGRID_nos,xGRID_nos), "inncorect,shape of sc"

	return sc, sp

