# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 21:39:09 2019

@author: Calvin
"""

"""
Imports
"""
import numpy as np
from DataTypes import *
from RotationFunctions import *

"""
Function for converting mvnx file into an mvn data dictionary
Takes in the raw xml data as an argument
Also takes the mvn parameter dictionary as an argument
The dictionary is filled with motion capture data, and also serves as a check
for the mvnx data (only pulls relevant data)
"""
def smpl2smpldict(smpldata, smpldict, fs, dattype):
    
    # Bookkeeping
    data_types = getDataTypes()
    nFrames = smpldata.shape[0]
    nvars = data_types[dattype]['nvars']
    
    # Create time vector based on sampling rate and number of frames
    timevec = np.linspace(0,nFrames-1,num=nFrames) / fs
    timevec.shape = [nFrames, 1]
    smpldict['Time'] = timevec
    
    # Initialize data structures for storing data
    joint_names = list( smpldict['jointMap'].keys() )
    print( joint_names )
    for i in range(len(joint_names)):
        smpldict['jointMap'][joint_names[i]]['data'] = np.zeros([nFrames,nvars])
    
    # Cycle through the frames
    for i in range(nFrames):
        for j in range(len(joint_names)):
            
            smpldict['jointMap'][joint_names[j]]['data'][i] = smpldata[i][(j*nvars):((j+1)*nvars)]
        
    return smpldict

"""
This function returns the relevant parameters for converting the mvn
dictionary into the general gltf motion dictionary
Global reference frame convention with respect to gltf convention
Mapping of mvn joint names to gltf joint names
Zero-pose of mvn joints with respect to N-pose of gltf
"""
def smplJointMap(data_type):
    
    ###
    # Zero offset rotations for the joints,
    # Since MVN is also in a neutral N-pose, no joints have a zero offset
    ###
    zero_offset = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    
    ###
    # Offsets for shoulders
    ###
    lshould_off = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    rshould_off = np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    lelbow_off = np.array([[0., 0., -1.],[0.,1.,0.],[1.,0.,0.]])
    relbow_off = np.array([[0., 0., 1.],[0.,1.,0.],[-1.,0.,0.]])
    
    ###
    # Generate Joint map with time-invariant parameters - this is the full mapping, users can select subsets
    ###
    jointMap = {}
    jointMap['L_Hip'] = {'gltf': 'Left Femur', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': data_type}
    jointMap['R_Hip'] = {'gltf': 'Right Femur', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': data_type}
    jointMap['Spine1'] = {'gltf': 'L5', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': data_type}
    jointMap['L_Knee'] = {'gltf': 'Left Shank', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': data_type}
    jointMap['R_Knee'] = {'gltf': 'Right Shank', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': data_type}
    jointMap['Spine2'] = {'gltf': 'L3', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': data_type}
    jointMap['L_Ankle'] = {'gltf': 'Left Foot', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': data_type}
    jointMap['R_Ankle'] = {'gltf': 'Right Foot', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': data_type}
    jointMap['Spine3'] = {'gltf': 'T12', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': data_type}
    jointMap['Neck'] = {'gltf': 'C7', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': data_type}
    jointMap['L_Collar'] = {'gltf': 'Left Shoulder', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': data_type}
    jointMap['R_Collar'] = {'gltf': 'Right Shoulder', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': data_type}
    jointMap['Head'] = {'gltf': 'Skull', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': data_type}
    jointMap['L_Shoulder'] = {'gltf': 'Left Upper Arm', 'Rprox': rshould_off, 'Rdist': rshould_off, 'offset': lshould_off, 'dattype': data_type}
    jointMap['R_Shoulder'] = {'gltf': 'Right Upper Arm', 'Rprox': lshould_off, 'Rdist': lshould_off, 'offset': rshould_off, 'dattype': data_type}
    jointMap['L_Elbow'] = {'gltf': 'Left Forearm', 'Rprox': rshould_off, 'Rdist': rshould_off, 'offset': zero_offset, 'dattype': data_type}
    jointMap['R_Elbow'] = {'gltf': 'Right Forearm', 'Rprox': lshould_off, 'Rdist': lshould_off, 'offset': zero_offset, 'dattype': data_type}
    jointMap['L_Wrist'] = {'gltf': 'Left Hand', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': data_type}
    jointMap['R_Wrist'] = {'gltf': 'Right Hand', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': data_type}
    jointMap['Pelvis'] = {'gltf': 'Pelvis', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': 'YXZ'}
    jointMap['Pelvis_trans'] = {'gltf': 'Pelvis_trans', 'Rprox': zero_offset, 'Rdist': zero_offset, 'offset': zero_offset, 'dattype': 'pos'}
    
    # Global orientation
    globalOrient = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    
    # Full mvn parameter map
    smplMap = {'jointMap': jointMap, 'globalRotation': globalOrient}
    return smplMap
