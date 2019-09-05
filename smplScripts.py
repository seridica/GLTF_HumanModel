# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:17:55 2019

@author: Calvin
"""

"""
This file contains various methods for processing SMPL data.
1) DIP IMU provides a subset of SMPL skeleton (no feet and hands) in a rotation matrix format
2) UBC Lower Body provides a subset of SMPL skeleton (waist down) in a euler angle format
3) SMPL is the "normal" representation in angle-axis format
"""

from smplFunctions import *
from gltfFunctions import *

def processDIP():
    # Read numpy data
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/MPI SMPL DIP/DIP_IMU_data/imu_own_validation.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/MPI SMPL DIP/DIP_IMU_data/imu_own_test.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/MPI SMPL DIP/DIP_IMU_data/imu_ubc_data.npz'
    filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/DIP Full Traces/dip_data_s01.npz'
    
    npdat = np.load(filename)
    smpl_data = npdat['smpl_pose'][0]
    
    # The data type for the smple data
    dattype = 'rotmat'
    fs = 60
    
    # Get mvn parameters and create dictionary template
    smplParams = smplJointMap(dattype)
    
    # Remove keys that are not used
    del( smplParams['jointMap']['Pelvis'] )
    del( smplParams['jointMap']['L_Wrist'] )
    del( smplParams['jointMap']['R_Wrist'] )
    del( smplParams['jointMap']['L_Ankle'] )
    del( smplParams['jointMap']['R_Ankle'] )
    
    # Parse xml data and add to the mvn dictionary
    smplDict = smpl2smpldict(smpl_data, smplParams, fs, dattype)
    
    filename_base = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/SkinCap Rig/SkeletalRig_v1_5'
    new_fname = filename_base + '_animated.gltf'
    
    filename = filename_base + '.gltf'
    
    gltfDict = mocap2gltf(smplDict)
    curr_gltf = pygltf.GLTF2().load(filename)
    new_gltf = addGltfAnimation(curr_gltf, gltfDict)
    new_gltf.save(new_fname)
    #new_gltf = None
    
    return smplDict, gltfDict, new_gltf

def processUBCLower():
    # Read numpy data
    filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Fixed Step Dataset/training_steps_top.npz'
    npdat = np.load(filename)

    smpl_data = npdat['joint_outputs'][0]
    
    # The data type for the simple data
    dattype = 'ZXY'
    fs = 400
    
    # Get parameters
    smplParams = smplJointMap(dattype)
    
    # Remove keys that are not used
    del( smplParams['jointMap']['Pelvis'] )
    del( smplParams['jointMap']['L_Wrist'] )
    del( smplParams['jointMap']['R_Wrist'] )
    del( smplParams['jointMap']['Spine1'] )
    del( smplParams['jointMap']['Spine2'] )
    del( smplParams['jointMap']['Spine3'] )
    del( smplParams['jointMap']['Neck'] )
    del( smplParams['jointMap']['Head'] )
    del( smplParams['jointMap']['L_Shoulder'] )
    del( smplParams['jointMap']['R_Shoulder'] )
    del( smplParams['jointMap']['L_Elbow'] )
    del( smplParams['jointMap']['R_Elbow'] )
    del( smplParams['jointMap']['L_Collar'] )
    del( smplParams['jointMap']['R_Collar'] )

"""
Run sample code as a script
"""
if __name__ == "__main__":
    smplDict, gltfDict, new_gltf = processDIP()