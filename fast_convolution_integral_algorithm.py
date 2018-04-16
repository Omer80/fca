# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:34:17 2016

@author: Omer Tzuk <omertz@post.bgu.ac.il>
"""
import numpy as np

def calcAlpha(phi,fix_phis):
    phi_sq = phi**2
    fix_phis_sq = fix_phis**2
    F = []
    for l,phi_l_sq in enumerate(fix_phis_sq):
        F.append(2.0*phi_sq/(phi_sq-phi_l_sq))
#        print F[l]
        for j,phi_j_sq in enumerate(fix_phis_sq):
            if j != l:
                F[l]*=(phi_sq - phi_j_sq)/(phi_l_sq + phi_j_sq)
                F[l]*=(phi_l_sq + phi_j_sq)/(phi_sq + phi_j_sq)
    return np.array(F)

def calcAlpha2(phi,fix_phis):
    N = fix_phis.size
    phi_sq = phi**2
    fix_phis_sq = fix_phis**2
    alpha = np.ones((fix_phis.shape[0],phi.shape[0],phi.shape[1]))
    for l in range(N):
        alpha[l]*=(2.0*phi_sq/(phi_sq-fix_phis_sq[l]))
#        print F[l]
        for j in range(N):
            if j != l:
                alpha[l]*=(phi_sq - fix_phis_sq[j])/(fix_phis_sq[l] + fix_phis_sq[j])
                alpha[l]*=(fix_phis_sq[l] + fix_phis_sq[j])/(phi_sq + fix_phis_sq[j])
    return alpha


def calcAlpha_numpy(phi,fix_phi):
    phi=phi**2
    fix_phi=fix_phi**2
    # Get size of fix_phi
    N = fix_phi.size
    
    # Perform "(phi - fix_phi[l])/(phi + fix_phi[j]))" in a vectorized manner
    p1 = (phi - fix_phi[:,None,None,None])/(phi + fix_phi[:,None,None])
    
    # Perform "((fix_phi[l] + fix_phi[j])/(fix_phi[l] - fix_phi[j]))" in a vectorized manner
    p2 = ((fix_phi[:,None] + fix_phi)/(fix_phi[:,None] + fix_phi))
    
    # Elementwise multiplications between the previously calculated parts
    p3 = p1*p2[...,None,None]
    
    # Set the escaped portion "j != l" output as "phi/fix_phi[l]"
    p3[np.eye(N,dtype=bool)] = phi/fix_phi[:,None,None]
    Fout = p3.prod(1)
    
    p4 = (2.0*phi)/(phi - fix_phi[:,None,None])
    Fout *= p4
#    print p4

    # If you need separate arrays just like in the question, split it
#    Fout = np.array_split(Fout,N)
    return Fout
    
