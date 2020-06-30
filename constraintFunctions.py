# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:03:07 2020

@author: Calvin
"""

"""
Function for defining constraints within the skeletal rig.
"""

from scipy import integrate
from RotationFunctions import *
import numpy as np

"""
Fixed constraint for the leg. User needs to specify which leg using the boolean
Default is the left leg
"""
def FixedLegConstraint(constrMap, leftSide=True):
    
    if leftSide:
        side = 'Left '
    else:
        side = 'Right '
    
    constrMap[side+'Patella'] = [{'Type': 'quat', 
        'ind_dofs': 
            [{'Joint': side+'Shank',
             'Type': 'quat'}],
        'constrFunction': FixedConstraint,
        'params': None
        }]
    
    constrMap[side+'Fibula'] = [{'Type': 'quat', 
        'ind_dofs': 
            [{'Joint': side+'Shank',
             'Type': 'quat'}],
        'constrFunction': FixedConstraint,
        'params': None
        }]
    
    constrMap[side+'Tibia'] = [{'Type': 'quat', 
        'ind_dofs': 
            [{'Joint': side+'Shank',
             'Type': 'quat'}],
        'constrFunction': FixedConstraint,
        'params': None
        }]
    
    return constrMap

"""
Fixed constraint for the arm. User needs to specify which arm using the boolean
Default is the left arm
"""
def FixedArmConstraint(constrMap, leftSide=True):
    
    if leftSide:
        side = 'Left '
    else:
        side = 'Right '
    
    constrMap[side+'Radius'] = [{'Type': 'quat', 
        'ind_dofs': 
            [{'Joint': side+'Forearm',
             'Type': 'quat'}],
        'constrFunction': FixedConstraint,
        'params': None
        }]
    
    constrMap[side+'Ulna'] = [{'Type': 'quat', 
        'ind_dofs': 
            [{'Joint': side+'Forearm',
             'Type': 'quat'}],
        'constrFunction': FixedConstraint,
        'params': None
        }]
    
    return constrMap

"""
Ulnar-Radial Joint constraint
Proper ulnar radial joint
"""
def UlnarRadialArmConstraint(constrMap, leftSide=True):
    if leftSide:
        side = 'Left '
        sgn = -1
    else:
        side = 'Right '
        sgn = 1
        
    # Parameters of the arm that set parameters for the constraint
    # Radius' are defined as positive distal and negative proximal
    radius_r0 = 0.012
    radius_r1 = 0.02
    
    ulna_r0 = 0.0
    ulna_r1 = 0.0
    forearm_l = 0.254
    
    radius_t_params = np.array([-sgn*radius_r0,1.0,0.0,sgn*radius_r0,sgn*radius_r0,1.0,0.0,0.0,0.0,0.3343333])
    ulna_t_params = np.array([-sgn*ulna_r0,1.0,0.0,sgn*ulna_r0,-sgn*ulna_r0,1.0,0.0,0.0,0.0,0.3343333])
    
    constrMap[side+'Ulna'] = [{'Type': 'rx',
             'ind_dofs':
                 [{'Joint': side+'Forearm',
                   'Type': 'rx'}],
             'constrFunction': LinearConstraint,
             'params': [1.0,0.0]},
             {'Type': 'rz',
             'ind_dofs':
                 [{'Joint': side+'Forearm',
                   'Type': 'rz'}],
             'constrFunction': LinearConstraint,
             'params': [1.0,0.0]}]
    constrMap[side+'Ulna_trans'] = [{'Type': 'ty',
             'ind_dofs':
                 [{'Joint': side+'Forearm',
                   'Type': 'rx'},
                  {'Joint': side+'Forearm',
                   'Type': 'ry'},
                  {'Joint': side+'Forearm',
                   'Type': 'rz'}],
             'constrFunction': CustomRadioUlnarTY,
             'params': ulna_t_params},
            {'Type': 'tx',
             'ind_dofs':
                 [{'Joint': side+'Forearm',
                   'Type': 'rx'},
                  {'Joint': side+'Forearm',
                   'Type': 'ry'},
                  {'Joint': side+'Forearm',
                   'Type': 'rz'}],
             'constrFunction': CustomRadioUlnarTX,
             'params': ulna_t_params},
            {'Type': 'tz',
             'ind_dofs':
                 [{'Joint': side+'Forearm',
                   'Type': 'rx'},
                  {'Joint': side+'Forearm',
                   'Type': 'ry'},
                  {'Joint': side+'Forearm',
                   'Type': 'rz'}],
             'constrFunction': CustomRadioUlnarTZ,
             'params': ulna_t_params}]
    
    constrMap[side+'Radius'] = [{'Type': 'rx',
             'ind_dofs':
                 [{'Joint': side+'Forearm',
                   'Type': 'rx'},
                  {'Joint': side+'Forearm',
                   'Type': 'ry'}],
             'constrFunction': CustomRadioUlnarFlexion,
             'params': [-sgn*radius_r0,-sgn*radius_r1,forearm_l]},
             {'Type': 'rz',
             'ind_dofs':
                 [{'Joint': side+'Forearm',
                   'Type': 'rz'},
                  {'Joint': side+'Forearm',
                   'Type': 'ry'}],
             'constrFunction': CustomRadioUlnarAbduction,
             'params': [sgn*radius_r0,sgn*radius_r1,forearm_l]},
             {'Type': 'ry',
             'ind_dofs':
                 [{'Joint': side+'Forearm',
                   'Type': 'ry'}],
             'constrFunction': LinearConstraint,
             'params': [1.0,0.0]}]
    constrMap[side+'Radius_trans'] = [{'Type': 'ty',
             'ind_dofs':
                 [{'Joint': side+'Forearm',
                   'Type': 'rx'},
                  {'Joint': side+'Forearm',
                   'Type': 'ry'},
                  {'Joint': side+'Forearm',
                   'Type': 'rz'}],
             'constrFunction': CustomRadioUlnarTY,
             'params': radius_t_params},
            {'Type': 'tx',
             'ind_dofs':
                 [{'Joint': side+'Forearm',
                   'Type': 'rx'},
                  {'Joint': side+'Forearm',
                   'Type': 'ry'},
                  {'Joint': side+'Forearm',
                   'Type': 'rz'}],
             'constrFunction': CustomRadioUlnarTX,
             'params': radius_t_params},
            {'Type': 'tz',
             'ind_dofs':
                 [{'Joint': side+'Forearm',
                   'Type': 'rx'},
                  {'Joint': side+'Forearm',
                   'Type': 'ry'},
                  {'Joint': side+'Forearm',
                   'Type': 'rz'}],
             'constrFunction': CustomRadioUlnarTZ,
             'params': radius_t_params}]
                     
    return constrMap

"""
Cervical Spine Constraint. This applies Vasavada cervical spine bending constraint
to the cervical spine vertebrae
"""
def CervicalSpineConstraint(constrMap):
    # Weights are ordered C7, C6, C5, C4, C3, C2, C1, Skull
    rx_weights = [0.099000231505, 0.187000246301, 0.219999750512, 0.219999750512, 0.164999812884, 0.109999875256, 0.44399989235, 0.555999963268]
    ry_weights = [0.063000147321, 0.188000057654, 0.21899993916, 0.21899993916, 0.21899993916, 0.21899993916, 0.889001314925, 0.111000259566]
    rz_weights = [0.078000182398, 0.136999938394, 0.157000176148, 0.215999932144, 0.215999932144, 0.196000267347, 0.5, 0.5]
    dep_list = ['C7', 'C6', 'C5', 'C4', 'C3', 'C2', 'C1', 'Skull']
    indep_list = ['C7', 'C7', 'C7', 'C7', 'C7', 'C7', 'Skull', 'Skull']
    
    for i in range(len(dep_list)):
        constrMap[dep_list[i]] = [{'Type': 'rx',
             'ind_dofs':
                 [{'Joint': indep_list[i],
                   'Type': 'rx'}],
             'constrFunction': LinearConstraint,
             'params': [rx_weights[i],0.0]},
             {'Type': 'rz',
             'ind_dofs':
                 [{'Joint': indep_list[i],
                   'Type': 'rz'}],
             'constrFunction': LinearConstraint,
             'params': [rz_weights[i],0.0]},
             {'Type': 'ry',
             'ind_dofs':
                 [{'Joint': indep_list[i],
                   'Type': 'ry'}],
             'constrFunction': LinearConstraint,
             'params': [ry_weights[i],0.0]}]
    
    return constrMap

"""
Thoracolumbar Spine Constraint. This applies Bruno 2015 thoracolumbar spine
bending constraints. Takes in list of independent dofs as input and sets up
constraint accordingly
"""
def ThoracoLumbarSpineConstraint(constrMap, indepdofs):
    rx_weights = [0.027, 0.027, 0.027, 0.027, 0.027, 0.034, 0.04, 0.04, 0.04, 0.06, 0.08, 0.08, 0.153, 0.121, 0.107, 0.075, 0.035];
    ry_weights = [0.046, 0.062, 0.054, 0.062, 0.07, 0.074, 0.089, 0.097, 0.105, 0.1, 0.05, 0.019, 0.029, 0.031, 0.038, 0.038, 0.036];
    rz_weights = [0.057, 0.053, 0.057, 0.037, 0.033, 0.045, 0.07, 0.053, 0.066, 0.074, 0.094, 0.09, 0.051, 0.068, 0.066, 0.049, 0.037];
    depdofs = ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','L1','L2','L3','L4','L5']
    
    weight_params = np.ones(len(indepdofs))
    
    rx_list = [];
    ry_list = [];
    rz_list = [];
    
    for i in range(len(indepdofs)):
        rx_list.append({'Joint': indepdofs[i], 'Type': 'rx'})
        ry_list.append({'Joint': indepdofs[i], 'Type': 'ry'})
        rz_list.append({'Joint': indepdofs[i], 'Type': 'rz'})
        
    
    for i in range(len(depdofs)):
            
        constrMap[depdofs[i]] = [{'Type': 'rx',
                 'ind_dofs': rx_list,
                 'constrFunction': LinearConstraint,
                 'params': [rx_weights[i]*weight_params, 0.0]},
                {'Type': 'ry',
                 'ind_dofs': ry_list,
                 'constrFunction': LinearConstraint,
                 'params': [ry_weights[i]*weight_params, 0.0]},
                {'Type': 'rz',
                 'ind_dofs': rz_list,
                 'constrFunction': LinearConstraint,
                 'params': [rz_weights[i]*weight_params, 0.0]}]
    return constrMap

"""
Inverse Cervical Spine Constraint functions
"""
def InverseCervicalSpineConstraint(constrMap):
    
    indepdofs_lower = ['C7', 'C6', 'C5', 'C4', 'C3', 'C2']
    indepdofs_upper = ['C1', 'Skull']
    rx_lower = [];
    ry_lower = [];
    rz_lower = [];
    
    rx_upper = [];
    ry_upper = [];
    rz_upper = [];
    
    for i in range(len(indepdofs_lower)):
        rx_lower.append({'Joint': indepdofs_lower[i], 'Type': 'rx'})
        ry_lower.append({'Joint': indepdofs_lower[i], 'Type': 'ry'})
        rz_lower.append({'Joint': indepdofs_lower[i], 'Type': 'rz'})
    
    for i in range(len(indepdofs_upper)):
        rx_upper.append({'Joint': indepdofs_upper[i], 'Type': 'rx'})
        ry_upper.append({'Joint': indepdofs_upper[i], 'Type': 'ry'})
        rz_upper.append({'Joint': indepdofs_upper[i], 'Type': 'rz'})
    
    constrMap['C7'] = [{'Type': 'rx',
             'ind_dofs': rx_lower,
             'constrFunction': LinearConstraint,
             'params': [np.ones(len(indepdofs_lower)),0.0]},
             {'Type': 'rz',
             'ind_dofs': rz_lower,
             'constrFunction': LinearConstraint,
             'params': [np.ones(len(indepdofs_lower)),0.0]},
             {'Type': 'ry',
             'ind_dofs': ry_lower,
             'constrFunction': LinearConstraint,
             'params': [np.ones(len(indepdofs_lower)),0.0]}]
                     
    constrMap['Skull'] = [{'Type': 'rx',
             'ind_dofs': rx_upper,
             'constrFunction': LinearConstraint,
             'params': [np.ones(len(indepdofs_upper)),0.0]},
             {'Type': 'rz',
             'ind_dofs': rz_upper,
             'constrFunction': LinearConstraint,
             'params': [np.ones(len(indepdofs_upper)),0.0]},
             {'Type': 'ry',
             'ind_dofs': ry_upper,
             'constrFunction': LinearConstraint,
             'params': [np.ones(len(indepdofs_upper)),0.0]}]
    
    return constrMap

"""
Inverse Thoracolumbar spine constraint functions. Note, the subsetdofs do not 
need to be ordered properly, the code will order them from superior to inferior
automatically
"""
def InverseThoracoLumbarSpineConstraint(constrMap, subsetdofs):
    # Ordering of full spine
    depdofs = ['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','L1','L2','L3','L4','L5']

    rx_list = [];
    ry_list = [];
    rz_list = [];
    for i in range(len(depdofs)):
        
        rx_list.append({'Joint': depdofs[i], 'Type': 'rx'})
        ry_list.append({'Joint': depdofs[i], 'Type': 'ry'})
        rz_list.append({'Joint': depdofs[i], 'Type': 'rz'})
        
        if depdofs[i] in subsetdofs:
            constrMap[depdofs[i]] = [{'Type': 'rx',
                 'ind_dofs': rx_list,
                 'constrFunction': LinearConstraint,
                 'params': [np.ones(len(rx_list)), 0.0]},
                {'Type': 'ry',
                 'ind_dofs': ry_list,
                 'constrFunction': LinearConstraint,
                 'params': [np.ones(len(ry_list)), 0.0]},
                {'Type': 'rz',
                 'ind_dofs': rz_list,
                 'constrFunction': LinearConstraint,
                 'params': [np.ones(len(rz_list)), 0.0]}]
            
            # Reset the list
            rx_list = [];
            ry_list = [];
            rz_list = [];
    
    return constrMap

"""
Callback for a fixed constraint
"""
def FixedConstraint(in_dof, params):
    return in_dof[0]

"""
Callback for a linear constraint
Can have more than one independent dof. If that is the case, the scaling factor
(params[0]) can be a constant which is applied to all independent dofs, or it
can be an array with weighting factor for each degree of freedom
"""
def LinearConstraint(in_dof, params):
    return np.sum( params[0]*in_dof ) + params[1]

"""
Callback for sine constraint
"""
def SinConstraint(in_dof, params):
    return params[0]*np.sin(in_dof[0]*params[1]+params[2]) + params[3]

"""
Callback for cosine constraint
"""
def CosConstraint(in_dof, params):
    return params[0]*np.cos(in_dof[0]*params[1]+params[2]) + params[3]

"""
Custom constraint for radial-ulnar joint
"""
def CustomRadioUlnarFlexion(in_dof, params):
    p1y = params[1]*np.sin(-in_dof[1])
    return -(np.arctan2(p1y, params[2]) - in_dof[0])

def CustomRadioUlnarAbduction(in_dof, params):
    p1y = params[1]*np.sin(-in_dof[1])
    p1x = params[0] - params[1]*np.cos(-in_dof[1])
    return np.arctan2(p1x, np.sqrt(params[2]*params[2] + p1y*p1y)) + np.arctan2(params[1]-params[0], params[2]) + in_dof[0]

# Computes the translation corrections in the radio-ulnar frame prior to pronation-supination
# Rotate them back to Humerus Frame via flexion and abduction rotations
def CustomRadioUlnarThelp(in_dof, params):
    # Indof is xyz ordered rotations
    # Flexion-extension, pronation-supination, abduction-adduction
    
    tx = CosConstraint(-1*np.array(in_dof[1:2]), params[0:4])
    tz = -SinConstraint(-1*np.array(in_dof[1:2]), params[4:8])
    t_ru = np.array([[tx],[0.],[tz]])
    
    #Rabd = np.array([[np.cos(-in_dof[2]), -np.sin(-in_dof[2]), 0.],[np.sin(-in_dof[2]), np.cos(-in_dof[2]), 0.],[0., 0., 1.]])
    #Rax = np.array([[np.cos(-in_dof[1]), 0., np.sin(-in_dof[1])], [0., 1., 0.], [-np.sin(-in_dof[1]), 0., np.cos(in_dof[1])]])
    #Rflex = np.array([[1., 0., 0.],[0., np.cos(-in_dof[0]), -np.sin(-in_dof[0])],[0., np.sin(-in_dof[0]), np.cos(-in_dof[0])]])
    return t_ru #np.matmul(Rflex, np.matmul(Rabd, t_ru))

def CustomRadioUlnarTX(in_dof, params):
    return CustomRadioUlnarThelp(in_dof,params)[0,0]
    
def CustomRadioUlnarTY(in_dof, params):
    return CustomRadioUlnarThelp(in_dof,params)[1,0]+LinearConstraint(in_dof,params[8:10])
    
def CustomRadioUlnarTZ(in_dof, params):
    return CustomRadioUlnarThelp(in_dof,params)[2,0]