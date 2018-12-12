#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jared
"""
import numpy as np

def heat_equation(a, b, T, N_x, N_t, u_0, c_a, d_a, h_a, c_b, d_b, h_b):

    # a - float
    # b - float
    # T - positive float
    # N_x - positive integer
    # N_t - positive integer
    # u_0 - function handle for the initial function auxiliary condition
    # c_a - function handle 
    # d_a - function handle
    # h_a - function handle
    # c_b - function handle 
    # d_b - function handle
    # h_b - function handle
    nu = 1
    N_x = N_x - 1 # match Barker's expected input
    N_t = N_t - 1


    x = np.linspace(a, b, N_x+1)
    t = np.linspace(0, T, N_t+1)
    h = x[1] - x[0]
    k = t[1] - t[0]

    λ = nu*k/(2*h**2)

    # A has shape (n_x+1, n_x+1)
    # define main, lower, upper diagonals respectively
    B = np.diag((1+2*λ)*np.ones((N_x+1,)), k=0) + \
        np.diag(-λ*np.ones((N_x,)), k=-1) + \
        np.diag(-λ*np.ones((N_x,)), k=1)
    
    A = np.diag((1-2*λ)*np.ones((N_x+1,)), k=0) + \
        np.diag(λ*np.ones((N_x,)), k=-1) + \
        np.diag(λ*np.ones((N_x,)), k=1)

    

    U_0 = u_0(x)
    U = U_0
    
    print("u_0: {}".format(U_0.shape))
    Us = np.zeros((len(t), len(x)))

    # Iterate over timesteps
    for i, t_i in enumerate(t):
        Us[i,:] = U

        rhs = A @ U
        rhs[0] = h*h_a(t_i)
        rhs[-1] = h*h_b(t_i)

        B[0,0] = h*c_a(t_i) - d_a(t_i)
        B[0,1] = d_a(t_i)

        B[-1,-1] = h*c_b(t_i)+d_b(t_i)
        B[-1,-2] = -d_b(t_i)

        U = np.linalg.solve(B, rhs)
        # match h_a(t) = c_a(t)*u(a,t) + d_a(t)*u_x(a,t)
        # match h_b(t) = c_b(t)*u(b,t) + d_b(t)*u_x(b,t)
        # U[[0,-1]] = α_x, β_x
    
    return Us