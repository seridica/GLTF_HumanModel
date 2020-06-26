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
from constraintFunctions import *
from scipy import integrate
from RotationFunctions import *
from scipy import signal
import time

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
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Datasets/Only Running/Heel 400 Input 400 Output/training_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Datasets/Activity Motions/predicted_lstm_steps_heelpelvis_lowerbody_128.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Datasets/Activity Motions/training_steps_heel_lowerbody.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Temp Dataset Folder/training_steps.npz'
    filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Temp Dataset Folder/predicted_lstm_steps.npz'
    
    npdat = np.load(filename)

    smpl_data = npdat['joint_outputs'][30]
    
    # The data type for the simple data
    dattype = 'XZY'
    fs = 400
    
    # Get parameters
    smplParams = smplJointMap(dattype)
    
    # Remove keys that are not used
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
    del( smplParams['jointMap']['Pelvis_trans'])
    
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

def processUBCComplete():
    # Read numpy data
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Working Datasets/full_angles_refined_train_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Working Datasets/full_angles_refined_lstm_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Working Datasets/full_velocities_refined_lstm_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Working Datasets/full_velocities_refined_lstm_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Working Datasets/full_angles_constraint_angles_lstm_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Working Datasets/full_angles_constraint_velocities_lstm_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Working Datasets/full_angles_fusion_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Working Datasets/full_angles_constraint_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Working Datasets/feet_angles_constraint_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Working Datasets/planar_angles_constraint_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Working Datasets/complete_angles_refined_lstm_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Complete Datasets/angles_refined_lstm_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Full Signal Datasets/Sub4Stairs/angles_constraint_validation_steps.npz'
    
    ## Example
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Complete Validation Datasets/angles_fusion_validation_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Complete Validation Datasets/angles_refined_valid_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Complete Validation Datasets/angles_constraint_validation_steps.npz'
    
    ## Running
    filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Full Signal Datasets/Sub4Run/angles_refined_valid_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Full Signal Datasets/Sub4Run/angles_refined_lstm_valid_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Full Signal Datasets/Sub4Run/velocities_refined_lstm_valid_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Full Signal Datasets/Sub4Run/angles_fusion_validation_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Full Signal Datasets/Sub4Run/angles_constraint_validation_steps.npz'
    
    ## JUMPING
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Full Signal Datasets/Sub7Jump/angles_constraint_validation_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Full Signal Datasets/Sub7Jump/angles_fusion_validation_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Full Signal Datasets/Sub7Jump/angles_refined_valid_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Full Signal Datasets/Sub7Jump/angles_refined_lstm_valid_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Full Signal Datasets/Sub7Jump/velocities_refined_lstm_valid_steps.npz'
    
    ## STAIRS
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Full Signal Datasets/Sub4Stairs/angles_refined_valid_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Full Signal Datasets/Sub4Stairs/angles_refined_lstm_valid_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Full Signal Datasets/Sub4Stairs/velocities_refined_lstm_valid_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Full Signal Datasets/Sub4Stairs/angles_fusion_validation_steps.npz'
    #filename = 'D:/UBC - Postdoc/Sensors/Motion Reconstruction/Full Signal Datasets/Sub4Stairs/angles_constraint_validation_steps.npz'
    
    ## OTHER
    #filename = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/GLTF Animation Code Working/SMPL Working/angles_refined_train_steps.npz'
    
    npdat = np.load(filename)

    smpl_data = npdat['joint_outputs'][0]
    
    ### COMMENT OR UNCOMMENT
    ### Set torso horizontal position to 0 for display and filter
    #smpl_data[:,23] = smpl_data[:,23] + np.array([4.0]) * 4
    #smpl_data[:,[21,23]] = np.array([0.5,-0.5]) * 1
    #smpl_data[:,22] = 0;
    #b, a = signal.butter(2, 0.1, 'low')
    #smpl_data = signal.filtfilt(b, a, smpl_data,axis=0)
    
    # The data type for the simple data
    dattype = 'XZY'
    fs = 50.
    
    # Get parameters
    smplParams = smplJointMap(dattype)
    
    # Remove keys that are not used
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
    # Integrate the yaw angular velocity and compute lab-fixed orientation
    yaw_av = smpl_data[:,20]
    yaw_angle = integrate.cumtrapz(yaw_av) / 400.;
    
    # SMPL data converted
    smpl_data_conv = np.copy( smpl_data )
    smpl_data_conv[1:,19] = yaw_angle
    smpl_data_conv[:,20] = smpl_data[:,19]
        
    # Cycle through body-fixed velocity and rotate to lab-fixed coordinates and integrate
    smpl_data_conv[0,21:24] = [0,0,0]
    for i in range(smpl_data.shape[0]-1):
        xyz = smpl_data_conv[i+1,18:21]
        rotmat = EulerYXZ( xyz )
        
        global_vel = np.matmul( rotmat, smpl_data[i+1,21:24] )
        smpl_data_conv[i+1,21:24] = global_vel / 400. + smpl_data_conv[i,21:24]
    
    """
    # Convert data into dictionary
    #smplDict = smpl2smpldict(smpl_data_conv, smplParams, fs, dattype)
    #smpl_data[:,:] = 0.;
    #smpl_data[:,15] = 0;
    #smpl_data[:,13] = 180.;
    #smpl_data[:,17] = 90.;
    smplDict = smpl2smpldict(smpl_data, smplParams, fs, dattype)
    
    # Create Constraints
    smplDict['constraintMap'] = BuildConstraintMap_Footsee()
    
    #filename_base = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/SkinCap Rig/SkeletalRig_v1_5_lower'
    filename_base = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/SkinCap Rig/SkeletalRig_v1_5'
    #filename_base = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/SkinCap Rig/SkeletalRig_v2_0'
    new_fname = filename_base + '_animated.gltf'
    
    filename = filename_base + '.gltf'
    
    gltfDict = mocap2gltf(smplDict)
    curr_gltf = pygltf.GLTF2().load(filename)
    new_gltf = addGltfAnimation(curr_gltf, gltfDict)
    new_gltf.save(new_fname)
    #new_gltf = None
    
    return smplDict, gltfDict, new_gltf
    

"""
Run sample code as a script
"""
if __name__ == "__main__":
    #smplDict, gltfDict, new_gltf = processDIP()
    #smplDict, gltfDict, new_gltf = processUBCLower()
    smplDict, gltfDict, new_gltf = processUBCComplete()