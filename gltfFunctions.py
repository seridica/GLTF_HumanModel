# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 23:16:21 2019

@author: Calvin
"""

"""
Imports
"""
import pygltflib as pygltf
from base64 import b64decode, b64encode
import numpy as np
from RotationFunctions import *
import struct

"""
Generate the Joint Map for gltf skeleton (same as joint map for motion capture
or modeling frameworks)
"""
def getGltfJointMap():
    eye3 = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    
    jMap = {}
    
    # Pelvis
    jMap['Pelvis'] = eye3;
    jMap['Pelvis_trans'] = eye3;
    
    # Lower body mapping from GLTF global frame to individual joints
    rFemur_rel = quat2rot( np.array([0.9999873042106628, 0, 0, -0.0050572361797094345]) / np.linalg.norm( np.array([0.9999873042106628, 0, 0, -0.0050572361797094345]) ) ).transpose()
    rFemur_abs = rFemur_rel
    jMap['Right Femur'] = rFemur_abs
    
    rShank_rel = quat2rot( np.array([2.2136126265337452e-10, 0.9999871850013733, -0.005064154975116253, -4.371083051069036e-08]) / np.linalg.norm( np.array([2.2136126265337452e-10, 0.9999871850013733, -0.005064154975116253, -4.371083051069036e-08]) ) ).transpose()
    rShank_abs = np.matmul( rFemur_abs, rShank_rel )
    jMap['Right Shank'] = rShank_abs
    jMap['Right Fibula'] = rShank_abs
    jMap['Right Tibia'] = rShank_abs
    jMap['Right Patella'] = rShank_abs
    
    rFoot_rel = quat2rot( np.array([-5.1845829851515646e-09, 0.9005414247512817, 0.43477022647857666, 6.798898510851359e-08]) / np.linalg.norm(np.array([-5.1845829851515646e-09, 0.9005414247512817, 0.43477022647857666, 6.798898510851359e-08])) ).transpose()
    rFoot_abs = np.matmul( rFoot_rel, rShank_abs )
    jMap['Right Foot'] = rFoot_abs
    
    lFemur_rel = quat2rot( np.array([0.9999873042106628, 0, 0, -0.0050572361797094345]) / np.linalg.norm( np.array([0.9999873042106628, 0, 0, -0.0050572361797094345]) ) ).transpose()
    lFemur_abs = lFemur_rel
    jMap['Left Femur'] = lFemur_abs
    
    lShank_rel = quat2rot( np.array([2.2136126265337452e-10, 0.9999871850013733, -0.005064154975116253, -4.371083051069036e-08]) / np.linalg.norm(np.array([2.2136126265337452e-10, 0.9999871850013733, -0.005064154975116253, -4.371083051069036e-08])) ).transpose()
    lShank_abs = np.matmul( lShank_rel, lFemur_abs )
    jMap['Left Shank'] = lShank_abs
    jMap['Left Fibula'] = lShank_abs
    jMap['Left Tibia'] = lShank_abs
    jMap['Left Patella'] = lShank_abs
    
    lFoot_rel = quat2rot( np.array([-5.1845829851515646e-09, 0.9005414247512817, 0.43477022647857666, 6.798898510851359e-08]) / np.linalg.norm(np.array([-5.1845829851515646e-09, 0.9005414247512817, 0.43477022647857666, 6.798898510851359e-08])) ).transpose()
    lFoot_abs = np.matmul( lFoot_rel, lShank_abs )
    jMap['Left Foot'] = lFoot_abs
    
    # Spine mapping from GLTF global frame to individual joints
    L5_rel = quat2rot( np.array([0.1822359561920166, 0, 0, 0.9832548499107361]) / np.linalg.norm(np.array([0.1822359561920166, 0, 0, 0.9832548499107361])) ).transpose()
    L5_abs = L5_rel
    jMap['L5'] = L5_abs
    
    L4_rel = quat2rot( np.array([-0.15643492341041565, 0, 0, 0.9876883625984192]) / np.linalg.norm(np.array([-0.15643492341041565, 0, 0, 0.9876883625984192])) ).transpose()
    L4_abs = np.matmul( L4_rel, L5_abs )
    jMap['L4'] = L4_abs
    
    L3_rel = quat2rot( np.array([-0.10452868044376373, 0, 0, 0.994521975517273]) / np.linalg.norm( np.array([-0.10452868044376373, 0, 0, 0.994521975517273]) ) ).transpose()
    L3_abs = np.matmul( L3_rel, L4_abs )
    jMap['L3'] = L3_abs
    
    L2_rel = quat2rot( np.array([-0.061046961694955826, 0, 0, 0.9981349110603333]) / np.linalg.norm( np.array([-0.061046961694955826, 0, 0, 0.9981349110603333]) ) ).transpose()
    L2_abs = np.matmul( L2_rel, L3_abs )
    jMap['L2'] = L2_abs
    
    L1_rel = quat2rot( np.array([-0.05233613774180412, 0, 0, 0.9986295104026794]) / np.linalg.norm( np.array([-0.05233613774180412, 0, 0, 0.9986295104026794]) ) ).transpose()
    L1_abs = np.matmul( L1_rel, L2_abs )
    jMap['L1'] = L1_abs
    
    T12_rel = quat2rot( np.array([0.017452172935009003, 0, 0, 0.9998477101325989]) / np.linalg.norm( np.array([0.017452172935009003, 0, 0, 0.9998477101325989]) ) ).transpose()
    T12_abs = np.matmul( T12_rel, L1_abs )
    jMap['T12'] = T12_abs
    
    T11_rel = quat2rot( np.array([0.03489987179636955, 0, 0, 0.9993908405303955]) / np.linalg.norm( np.array([0.03489987179636955, 0, 0, 0.9993908405303955]) ) ).transpose()
    T11_abs = np.matmul( T11_rel, T12_abs )
    jMap['T11'] = T11_abs
    
    T10_rel = quat2rot( np.array([0.043619077652692795, 0, 0, 0.9990482330322266]) / np.linalg.norm( np.array([0.043619077652692795, 0, 0, 0.9990482330322266]) ) ).transpose()
    T10_abs = np.matmul( T10_rel, T11_abs )
    jMap['T10'] = T10_abs
    
    T9_rel = quat2rot( np.array([0.03489937260746956, 0, 0, 0.9993908405303955]) / np.linalg.norm( np.array([0.03489937260746956, 0, 0, 0.9993908405303955]) ) ).transpose()
    T9_abs = np.matmul( T9_rel, T10_abs )
    jMap['T9'] = T9_abs
    
    T8_rel = quat2rot( np.array([0.043619345873594284, 0, 0, 0.9990482330322266]) / np.linalg.norm( np.array([0.043619345873594284, 0, 0, 0.9990482330322266]) ) ).transpose()
    T8_abs = np.matmul( T8_rel, T9_abs )
    jMap['T8'] = T8_abs
    
    T7_rel = quat2rot( np.array([0.05233614891767502, 0, 0, 0.9986295104026794]) / np.linalg.norm( np.array([0.05233614891767502, 0, 0, 0.9986295104026794]) ) ).transpose()
    T7_abs = np.matmul( T7_rel, T8_abs )
    jMap['T7'] = T7_abs
    
    T6_rel = quat2rot( np.array([0.0523357130587101, 0, 0, 0.9986295700073242]) / np.linalg.norm( np.array([0.0523357130587101, 0, 0, 0.9986295700073242]) ) ).transpose()
    T6_abs = np.matmul( T6_rel, T7_abs )
    jMap['T6'] = T6_abs
    
    T5_rel = quat2rot( np.array([0.05233645439147949, 0, 0, 0.9986295104026794]) / np.linalg.norm( np.array([0.05233645439147949, 0, 0, 0.9986295104026794]) ) ).transpose()
    T5_abs = np.matmul( T5_rel, T6_abs )
    jMap['T5'] = T5_abs
    
    T4_rel = quat2rot( np.array([0.05233585461974144, 0, 0, 0.9986295700073242]) / np.linalg.norm( np.array([0.05233585461974144, 0, 0, 0.9986295700073242]) ) ).transpose()
    T4_abs = np.matmul( T4_rel, T5_abs )
    jMap['T4'] = T4_abs
    
    T3_rel = quat2rot( np.array([0.03489944711327553, 0, 0, 0.9993908405303955]) / np.linalg.norm( np.array([0.03489944711327553, 0, 0, 0.9993908405303955]) ) ).transpose()
    T3_abs = np.matmul( T3_rel, T4_abs )
    jMap['T3'] = T3_abs
    
    T2_rel = quat2rot( np.array([0.026180002838373184, 0, 0, 0.9996572732925415]) / np.linalg.norm( np.array([0.026180002838373184, 0, 0, 0.9996572732925415]) ) ).transpose()
    T2_abs = np.matmul( T2_rel, T3_abs )
    jMap['T2'] = T2_abs
    
    T1_rel = quat2rot( np.array([-0.1533840447664261, 0, 0, 0.9881667494773865]) / np.linalg.norm( np.array([-0.1533840447664261, 0, 0, 0.9881667494773865]) ) ).transpose()
    T1_abs = np.matmul( T1_rel, T2_abs )
    jMap['T1'] = T1_abs
    
    C7_rel = quat2rot( np.array([0.07174857705831528, 0, 0, 0.9974227547645569]) / np.linalg.norm( np.array([0.07174857705831528, 0, 0, 0.9974227547645569]) ) ).transpose()
    C7_abs = np.matmul( C7_rel, T1_abs )
    jMap['C7'] = C7_abs
    
    C6_rel = quat2rot( np.array([-0.15467779338359833, 0, 0, 0.9879649877548218]) / np.linalg.norm( np.array([-0.15467779338359833, 0, 0, 0.9879649877548218]) ) ).transpose()
    C6_abs = np.matmul( C6_rel, C7_abs )
    jMap['C6'] = C6_abs
    
    C5_rel = quat2rot( np.array([0.07189929485321045, 0, 0, 0.9974119067192078]) / np.linalg.norm( np.array([0.07189929485321045, 0, 0, 0.9974119067192078]) ) ).transpose()
    C5_abs = np.matmul( C5_rel, C6_abs )
    jMap['C5'] = C5_abs
    
    C4_rel = quat2rot( np.array([-0.024355802685022354, 0, 0, 0.9997034668922424]) / np.linalg.norm( np.array([-0.024355802685022354, 0, 0, 0.9997034668922424]) ) ).transpose()
    C4_abs = np.matmul( C4_rel, C5_abs )
    jMap['C4'] = C4_abs
    
    C3_rel = quat2rot( np.array([-0.044016335159540176, 0, 0, 0.9990308284759521]) / np.linalg.norm( np.array([-0.044016335159540176, 0, 0, 0.9990308284759521]) ) ).transpose()
    C3_abs = np.matmul( C3_rel, C4_abs )
    jMap['C3'] = C3_abs
    
    C2_rel = quat2rot( np.array([0.0032729662489145994, 0, 0, 0.9999946355819702]) / np.linalg.norm( np.array([0.0032729662489145994, 0, 0, 0.9999946355819702]) ) ).transpose()
    C2_abs = np.matmul( C2_rel, C3_abs)
    jMap['C2'] = C2_abs
    
    C1_rel = quat2rot( np.array([-0.41500023007392883, 0, 0, 0.9098213315010071]) / np.linalg.norm( np.array([-0.41500023007392883, 0, 0, 0.9098213315010071]) ) ).transpose()
    C1_abs = np.matmul( C1_rel, C2_abs )
    jMap['C1'] = C1_abs
    
    skull_rel = quat2rot( np.array([0.3944814205169678, 0, 0, 0.9189039468765259]) / np.linalg.norm( np.array([0.3944814205169678, 0, 0, 0.9189039468765259]) ) ).transpose()
    skull_abs = np.matmul( skull_rel, C1_abs )
    jMap['Skull'] = skull_abs
    
    # Arms
    lShoulder_rel = quat2rot( np.array([-0.15792852640151978, 0.07429691404104233, -0.7473743557929993, 0.6410695314407349]) / np.linalg.norm(np.array([-0.15792852640151978, 0.07429691404104233, 0.7473743557929993, 0.6410695314407349])) ).transpose()
    lShoulder_abs = np.matmul( lShoulder_rel, T1_abs )
    jMap['Left Shoulder'] = lShoulder_abs
    
    lUpper_rel = quat2rot( np.array([-1.9840220133460207e-08, 0.09373736381530762, -0.6535478830337524, 0.7510582208633423]) / np.linalg.norm( np.array([-1.9840220133460207e-08, 0.09373736381530762, -0.6535478830337524, 0.7510582208633423]) ) ).transpose()
    lUpper_abs = np.matmul( lUpper_rel, lShoulder_abs )
    jMap['Left Upper Arm'] = lUpper_abs
    
    lFore_rel = quat2rot( np.array([7.450580152834618e-09, 1.4901160305669237e-08, -1.862645149230957e-09, 1]) / np.linalg.norm( np.array([7.450580152834618e-09, 1.4901160305669237e-08, -1.862645149230957e-09, 1]) ) ).transpose()
    lFore_abs = np.matmul( lFore_rel, lUpper_abs )
    jMap['Left Forearm'] = lFore_abs
    jMap['Left Radius'] = lFore_abs
    jMap['Left Ulna'] = lFore_abs
    
    lHand_rel = quat2rot( np.array([-1.1294776669501516e-07, -0.6272183656692505, 0.0764044001698494, 0.7750867605209351]) / np.linalg.norm( np.array([-1.1294776669501516e-07, -0.6272183656692505, 0.0764044001698494, 0.7750867605209351]) ) ).transpose()
    lHand_abs = np.matmul( lHand_rel, lFore_abs )
    jMap['Left Hand'] = lHand_abs
    
    rShoulder_rel = quat2rot( np.array([-0.15792852640151978, -0.07429691404104233, 0.7473743557929993, 0.6410695314407349]) / np.linalg.norm( np.array([-0.15792852640151978, -0.07429691404104233, -0.7473743557929993, 0.6410695314407349]) ) ).transpose()
    rShoulder_abs = np.matmul( rShoulder_rel, T1_abs )
    jMap['Right Shoulder'] = rShoulder_abs
    
    rUpper_rel = quat2rot( np.array([-1.9840220133460207e-08, -0.09373736381530762, 0.6535478830337524, 0.7510582208633423]) / np.linalg.norm( np.array([-1.9840220133460207e-08, -0.09373736381530762, 0.6535478830337524, 0.7510582208633423]) ) ).transpose()
    rUpper_abs = np.matmul( rUpper_rel, rShoulder_abs )
    jMap['Right Upper Arm'] = rUpper_abs
    
    rFore_rel = quat2rot( np.array([7.450580152834618e-09, -1.4901160305669237e-08, 1.862645149230957e-09, 1]) / np.linalg.norm( np.array([7.450580152834618e-09, -1.4901160305669237e-08, 1.862645149230957e-09, 1]) ) ).transpose()
    rFore_abs = np.matmul( rFore_rel, rUpper_abs )
    jMap['Right Forearm'] = rFore_abs
    jMap['Right Radius'] = rFore_abs
    jMap['Right Ulna'] = rFore_abs
    
    rHand_rel = quat2rot( np.array([-9.37226332098362e-08, -0.6272185444831848, 0.07640440762042999, 0.7750866413116455]) / np.linalg.norm( np.array([-9.37226332098362e-08, 0.6272185444831848, -0.07640440762042999, 0.7750866413116455]) ) )
    rHand_abs = np.matmul( rHand_rel, rFore_abs )
    jMap['Right Hand'] = rHand_abs
    
    return jMap


###
# Function does the initial mocap to gltf data dictionary given the motion
# capture parameters
###
def convertMocap(mocapDict):
    
    gltfDict = {}
    
    # Get global rotation
    globalRotation = mocapDict['globalRotation']
    
    # Get gltf joint angle mapping -> gltf joint angles are not consistent
    gltfJointMap = getGltfJointMap()
    
    # Get the time vector
    gltfDict['Time'] = mocapDict['Time']
    
    # Go through data structure and convert joint names to gltf joint names
    # and convert rotations to quaternions
    jointMap = mocapDict['jointMap']
    jointMap_keys = list( jointMap.keys() )
    for i in range( len( jointMap_keys ) ):
        
        # Time vector should not need to change
        print(jointMap_keys[i])
        if "_trans" in jointMap_keys[i]:
            jointName = jointMap_keys[i]
            
            gltf_name = jointMap[jointName]['gltf']
                
            # Get various time invariant rotation matrices
            zoff = jointMap[jointName]['offset']
            prox_mocap = jointMap[jointName]['Rprox']
            rot_gltf = gltfJointMap[gltf_name]
            
            # Compute pre- and post-rotation matric multipliers
            prox_offset = np.matmul( rot_gltf.transpose(), np.matmul(globalRotation, prox_mocap) );
                
            # Convert data
            frame_data = jointMap[jointName]['data']
            nFrames = frame_data.shape[0]
            gltfData = np.empty([nFrames, 3])
            for j in range(nFrames):
                mocap_frame = frame_data[j,:]
                
                # Euler ZXY to quaternion
                if jointMap[jointName]['dattype'] == 'pos':
                    #new_frame = np.matmul( gltfJointMap[gltf_name], np.array([mocap_frame]).transpose() )
                    pos = mocap_frame
                    
                # Compute the position
                gltf_global_rot = np.matmul( gltfJointMap[gltf_name], prox_offset )
                gltf_link_pos = np.matmul( gltf_global_rot, pos )
                gltfData[j,:] = gltf_link_pos
            
            gltfDict[gltf_name] = gltfData
        else:
            jointName = jointMap_keys[i]
            
            # Get the gltf name
            gltf_name = jointMap[jointName]['gltf']
            
            # Get the zero offset
            zoff = jointMap[jointName]['offset']
            
            # Get various time invariant rotation matrices
            zoff = jointMap[jointName]['offset']
            prox_mocap = jointMap[jointName]['Rprox']
            dist_mocap = jointMap[jointName]['Rdist']
            rot_gltf = gltfJointMap[gltf_name]
            
            # Compute pre- and post-rotation matric multipliers
            prox_offset = np.matmul( rot_gltf.transpose(), np.matmul(globalRotation, prox_mocap) );
            dist_offset = np.matmul( dist_mocap.transpose(), np.matmul(globalRotation.transpose(), rot_gltf));
            
            # Convert data
            frame_data = jointMap[jointName]['data']
            nFrames = frame_data.shape[0]
            gltfData = np.empty([nFrames, 4])
            for j in range(nFrames):
                mocap_frame = frame_data[j,:]
                
                # Euler ZXY to quaternion
                if jointMap[jointName]['dattype'] == 'ZXY':
                    #new_frame = np.matmul( gltfJointMap[gltf_name], np.array([mocap_frame]).transpose() )
                    rotmat = EulerZXY(mocap_frame)
                    #rotmat = EulerZXY(new_frame.transpose().tolist()[0])
                    #quat = rot2quat(rotmat)
                if jointMap[jointName]['dattype'] == 'YXZ':
                    #new_frame = np.matmul( gltfJointMap[gltf_name], np.array([mocap_frame]).transpose() )
                    rotmat = EulerYXZ(mocap_frame)
                    #rotmat = EulerZXY(new_frame.transpose().tolist()[0])
                    #quat = rot2quat(rotmat)
                if jointMap[jointName]['dattype'] == 'XZY':
                    #new_frame = np.matmul( gltfJointMap[gltf_name], np.array([mocap_frame]).transpose() )
                    rotmat = EulerXZY(mocap_frame)
                    #rotmat = EulerZXY(new_frame.transpose().tolist()[0])
                    #quat = rot2quat(rotmat)
                if jointMap[jointName]['dattype'] == 'LZXY':
                    #new_frame = np.matmul( gltfJointMap[gltf_name], np.array([mocap_frame]).transpose() )
                    rotmat = EulerLZXY(mocap_frame)
                    #rotmat = EulerZXY(new_frame.transpose().tolist()[0])
                    #quat = rot2quat(rotmat)
                if jointMap[jointName]['dattype'] == 'quat':
                    rotmat = quat2rot(mocap_frame)
                if jointMap[jointName]['dattype'] == 'rotmat':
                    rotmat = genRotMat(mocap_frame)
                
                # Compute the orientation\
                prox_offset_rot = np.matmul( prox_offset, zoff )
                dist_motion_rot = np.matmul( rotmat, dist_offset )
                gltf_link_rot = np.matmul( prox_offset_rot, dist_motion_rot )
                gltfData[j,:] = rot2quat( gltf_link_rot )
            
            gltfDict[gltf_name] = gltfData
            
    return gltfDict

###
# Function generates constraint gltf motion dictionary given the motion
# constraint dictionary along with the current gltf motion
###
def constraintMocap(mocapDict, indepDict, constrName='constraintMap'):
    
    depDict = {}
    nFrames = indepDict['Time'].shape[0]
        
    constrMap = mocapDict[constrName]
    constrMap_keys = list( constrMap.keys() )
    for i in range( len( constrMap_keys ) ):
        
        # Time vector should not need to change
        print(constrMap_keys[i])
        
        # What is the dependent joint
        dep_gltf_name = constrMap_keys[i]
        
        # Get list of dofs for the depndent joint
        dep_list = constrMap[dep_gltf_name]

        # Setup frame data for the dependent dof
        if '_trans' in dep_gltf_name:
            dep_gltfData = np.empty([nFrames, 3])
        else:
            dep_gltfData = np.empty([nFrames, 4])
                
        # Cycle through each frame applying the dependency
        for j in range(nFrames):
            
            # Cycle through the dependencies
            curr_xyz = np.array([0.,0.,0.])
            for k in range( len(dep_list) ):
                
                # Pull the dependency dictionary
                dep_dict = dep_list[k]
                
                # Pull the indepdenent data
                ind_data = []
                ind_list = dep_dict['ind_dofs']
                for ii in range( len(ind_list) ):
                    this_joint = ind_list[ii]['Joint']
                    if ind_list[ii]['Type'] == 'quat':
                        this_data = indepDict[this_joint][j,:]
                        ind_data.append( this_data )
                    elif ind_list[ii]['Type'] == 'rx':
                        this_data = indepDict[this_joint][j,:]
                        this_rot = quat2rot( this_data )
                        thisXZY = rot2XZY( this_rot, 0 )
                        ind_data.append( thisXZY[0] )
                    elif ind_list[ii]['Type'] == 'ry':
                        this_data = indepDict[this_joint][j,:]
                        this_rot = quat2rot( this_data )
                        thisXZY = rot2XZY( this_rot, 0 )
                        ind_data.append( thisXZY[1] )
                    elif ind_list[ii]['Type'] == 'rz':
                        this_data = indepDict[this_joint][j,:]
                        this_rot = quat2rot( this_data )
                        thisXZY = rot2XZY( this_rot, 0 )
                        ind_data.append( thisXZY[2] )
                    elif ind_list[ii]['Type'] == 'trans':
                        trans_joint = this_joint
                        this_data = indepDict[this_joint][j,:]
                        ind_data.append( this_data )
                    elif ind_list[ii]['Type'] == 'tx':
                        trans_joint = this_joint
                        this_data = indepDict[this_joint][j,0]
                        ind_data.append( this_data )
                    elif ind_list[ii]['Type'] == 'ty':
                        trans_joint = this_joint
                        this_data = indepDict[this_joint][j,1]
                        ind_data.append( this_data )
                    elif ind_list[ii]['Type'] == 'tz':
                        trans_joint = this_joint
                        this_data = indepDict[this_joint][j,2]
                        ind_data.append( this_data )
                
                # Parameters for constraint function
                dep_params = dep_dict['params']
                dep_dat = dep_dict['constrFunction'](np.array(ind_data), dep_params)
                
                # If the type is quat, then apply directly
                if dep_dict['Type'] == 'quat':
                    dep_out = dep_dat
                elif dep_dict['Type'] == 'rx':
                    curr_xyz[0] = dep_dat
                    curr_rot = EulerXZY( curr_xyz, 0 )
                    dep_out = rot2quat(curr_rot)
                elif dep_dict['Type'] == 'ry':
                    curr_xyz[1] = dep_dat
                    curr_rot = EulerXZY( curr_xyz, 0 )
                    dep_out = rot2quat(curr_rot)
                elif dep_dict['Type'] == 'rz':
                    curr_xyz[2] = dep_dat
                    curr_rot = EulerXZY( curr_xyz, 0 )
                    dep_out = rot2quat(curr_rot)
                elif dep_dict['Type'] == 'tx':
                    curr_xyz[0] = dep_dat
                    dep_out = curr_xyz
                elif dep_dict['Type'] == 'ty':
                    curr_xyz[1] = dep_dat
                    dep_out = curr_xyz
                elif dep_dict['Type'] == 'tz':
                    curr_xyz[2] = dep_dat
                    dep_out = curr_xyz
            
            dep_gltfData[j,:] = dep_out
                    
        depDict[dep_gltf_name] = dep_gltfData
        
    return depDict

###
# Function converts a gltf data dictionary to mocap data representation given
# the motion capture parameters provided in the mocapDict. Essentially the
# reverse of convertMocap
###
def convertGLTF(mocapDict, gltfDict):
    
    # Get global rotation, invert it
    globalRotation = mocapDict['globalRotation']
    
    # Get gltf joint angle mapping -> gltf joint angles are not consistent
    gltfJointMap = getGltfJointMap()
    
    # Get the time vector
    mocapDict['Time'] = gltfDict['Time']
    
    # Go through data structure and convert gltf joint names to mocap joint
    # names and convert quaternions to the mocap standard
    jointMap = mocapDict['jointMap']
    jointMap_keys = list( jointMap.keys() )
    for i in range( len( jointMap_keys ) ):
        
        # Time vector should not need to change
        print(jointMap_keys[i])
        if "_trans" in jointMap_keys[i]:
            jointName = jointMap_keys[i]
            
            gltf_name = jointMap[jointName]['gltf']
                
            # Get various time invariant rotation matrices
            zoff = jointMap[jointName]['offset']
            prox_mocap = jointMap[jointName]['Rprox']
            rot_gltf = gltfJointMap[gltf_name]
            
            # Compute pre- and post-rotation matric multipliers
            prox_offset = np.matmul( rot_gltf.transpose(), np.matmul(globalRotation, prox_mocap) );
              
            # Grab gltf data
            gltfData = gltfDict[gltf_name]
            nFrames = gltfData.shape[0]
            mocapData = np.empty([nFrames, 3])
            
            # Convert data
            for j in range(nFrames):
                gltf_frame = gltfData[j,:]
                
                # Euler ZXY to quaternion
                if jointMap[jointName]['dattype'] == 'pos':
                    #new_frame = np.matmul( gltfJointMap[gltf_name], np.array([mocap_frame]).transpose() )
                    pos = gltf_frame
                    
                # Compute the position
                gltf_global_rot = np.matmul( gltfJointMap[gltf_name], prox_offset )
                mocap_link_pos = np.matmul( gltf_global_rot.transpose(), pos )
                mocapData[j,:] = mocap_link_pos
            
            mocapDict['jointMap'][jointName]['data'] = mocapData
        else:
            jointName = jointMap_keys[i]
            
            # Get the gltf name
            gltf_name = jointMap[jointName]['gltf']
            
            # Get the zero offset
            zoff = jointMap[jointName]['offset']
            
            # Get various time invariant rotation matrices
            zoff = jointMap[jointName]['offset']
            prox_mocap = jointMap[jointName]['Rprox']
            dist_mocap = jointMap[jointName]['Rdist']
            rot_gltf = gltfJointMap[gltf_name]
            
            # Compute pre- and post-rotation matric multipliers
            prox_offset = np.matmul( rot_gltf.transpose(), np.matmul(globalRotation, prox_mocap) );
            dist_offset = np.matmul( dist_mocap.transpose(), np.matmul(globalRotation.transpose(), rot_gltf));
            
            # Grab gltf data
            gltfData = gltfDict[gltf_name]
            nFrames = gltfData.shape[0]
            
            if jointMap[jointName]['dattype'] in ['ZXY', 'YXZ', 'XZY', 'LZXY']:    
                mocapData = np.empty([nFrames, 3])
            if jointMap[jointName]['dattype'] in ['quat']:
                mocapData = np.empty([nFrames, 4])
            if jointMap[jointName]['dattype'] in ['rotmat']:
                mocapData = np.empty([nFrames, 9])
            
            # Convert data
            for j in range(nFrames):
                gltf_frame = gltfData[j,:]
                rotmat = quat2rot(gltf_frame)
                
                # Compute the orientation - Inverse of what is done for conversion from mocap to gltf
                prox_offset_rot = np.matmul( zoff.transpose(), prox_offset.transpose() )
                dist_motion_rot = np.matmul( rotmat, dist_offset.transpose() )
                mocap_link_rot = np.matmul( prox_offset_rot, dist_motion_rot )
                
                ### TODO REVERSE CONVERSIONS
                # Euler ZXY to quaternion
                if jointMap[jointName]['dattype'] == 'ZXY':
                    #new_frame = np.matmul( gltfJointMap[gltf_name], np.array([mocap_frame]).transpose() )
                    mocapData[j,:] = rot2ZXY(mocap_link_rot)
                    #rotmat = EulerZXY(new_frame.transpose().tolist()[0])
                    #quat = rot2quat(rotmat)
                if jointMap[jointName]['dattype'] == 'YXZ':
                    #new_frame = np.matmul( gltfJointMap[gltf_name], np.array([mocap_frame]).transpose() )
                    mocapData[j,:] = rot2YXZ(mocap_link_rot)
                    #rotmat = EulerZXY(new_frame.transpose().tolist()[0])
                    #quat = rot2quat(rotmat)
                if jointMap[jointName]['dattype'] == 'XZY':
                    #new_frame = np.matmul( gltfJointMap[gltf_name], np.array([mocap_frame]).transpose() )
                    mocapData[j,:] = rot2XZY(mocap_link_rot)
                    #rotmat = EulerZXY(new_frame.transpose().tolist()[0])
                    #quat = rot2quat(rotmat)
                if jointMap[jointName]['dattype'] == 'LZXY':
                    #new_frame = np.matmul( gltfJointMap[gltf_name], np.array([mocap_frame]).transpose() )
                    mocapData[j,:] = rot2LZXY(mocap_link_rot)
                    #rotmat = EulerZXY(new_frame.transpose().tolist()[0])
                    #quat = rot2quat(rotmat)
                if jointMap[jointName]['dattype'] == 'quat':
                    mocapData[j,:] = rot2quat(mocap_link_rot)
                if jointMap[jointName]['dattype'] == 'rotmat':
                    mocapData[j,:] = fromRotMat(mocap_link_rot)
            
            mocapDict['jointMap'][jointName]['data'] = mocapData
    
    return mocapDict    
    
###
# Function converts a mocap dictionary into a gltf data dictionary given the
# motion capture parameters
###
def mocap2gltf(mocapDict):
    
    ### Direct conversion of mocap to gltf
    gltfDict = convertMocap(mocapDict)
    
    ### Generate constrained motion
    constraintDict = constraintMocap(mocapDict, gltfDict)
    
    ### Add or replace constrained motion into the main gltf motion dictionary
    constr_keys = list( constraintDict.keys() )
    for i in range(len(constr_keys)):
        # What is the dependent joint
        dep_gltf_name = constr_keys[i]
        
        # Add or repalce in gltf motion dictionary
        gltfDict[dep_gltf_name] = constraintDict[dep_gltf_name]
    
    return gltfDict

###
# Function converts data in a gltf dictionary into data for a mocap definition
###
def gltf2mocap(mocapParams, gltfDict):
    
    ### Apply inverse constraints
    constraintDict = constraintMocap(mocapParams, gltfDict, 'constraintMapInv')
    
    ### Add or replace constrained motion into the main gltf motion dictionary
    constr_keys = list( constraintDict.keys () )
    for i in range(len(constr_keys)):
        # What is the dependent joint
        dep_gltf_name = constr_keys[i]
        
        # Add or replace in gltf motion dictionary
        gltfDict[dep_gltf_name] = constraintDict[dep_gltf_name]
    
    ### Direct conversion of gltf to mocap
    mocapDict = convertGLTF(mocapParams, gltfDict)
    
    return mocapDict

###
# Add animation data to existing skeletal rig
###
def addGltfAnimation(gltfData, gltfDict):
    nodeList = getGltfNodeNames( gltfData )
    print( nodeList )
     
    # Create an empty buffer to store data
    new_buffer = bytes()
    new_buffer_num = len( gltfData.buffers )
    
    ### Create time blocks first and add them to the gltf
    tvec = gltfDict['Time']
    time_buffer, time_length = convertToBytes(tvec)
    new_buffer += time_buffer
    
    # Buffer view
    t_buff_view_num = len(gltfData.bufferViews)
    t_buff_view = pygltf.BufferView()
    t_buff_view.buffer = new_buffer_num
    t_buff_view.byteOffset = 0
    t_buff_view.byteLength = time_length
    
    gltfData.bufferViews.append( t_buff_view )
    
    # Accessor
    t_access_num = len( gltfData.accessors )
    t_access = pygltf.Accessor()
    t_access.bufferView = t_buff_view_num
    t_access.byteOffset = 0
    t_access.componentType = 5126
    t_access.count = tvec.shape[0]
    t_access.type = 'SCALAR'
    t_access.max = np.amax(tvec, axis=0).tolist() # Need to add brackets for scalars...
    t_access.min = np.amin(tvec, axis=0).tolist()
    
    gltfData.accessors.append(t_access)
    
    ### Create blocks for all of the joints
    # Cycle through joints that have animation data and generate their
    # respective animation blocks
    gltf_keys = list( gltfDict.keys() )
    
    # Animations - add this as a new animation
    anim = pygltf.Animation()
    
    for i in range( len( gltf_keys ) ):
        joint_name = gltf_keys[i]
        
        # Only create blocks for joints
        if joint_name != 'Time':
            if '_trans' in joint_name:
                print(joint_name)
                
                # Apply gltf rotation offset to joint data
                joint_data_orig = gltfDict[joint_name]
                node_num = nodeList.index(joint_name.split('_trans')[0])
                gltf_quat = np.array(gltfData.nodes[node_num].rotation)
                
                # Add joint data to buffer
                joint_buffer, joint_buff_length = convertToBytes(joint_data_orig)
                byte_start = len( new_buffer )
                new_buffer += joint_buffer
                
                # Buffer View            
                jnt_buff_view_num = len(gltfData.bufferViews)
                jnt_buff_view = pygltf.BufferView()
                jnt_buff_view.buffer = new_buffer_num
                jnt_buff_view.byteOffset = byte_start
                jnt_buff_view.byteLength = joint_buff_length
                gltfData.bufferViews.append( jnt_buff_view )
                
                # Accessor
                jnt_access_num = len( gltfData.accessors )
                jnt_access = pygltf.Accessor()
                jnt_access.bufferView = jnt_buff_view_num
                jnt_access.byteOffset = 0
                jnt_access.componentType = 5126
                jnt_access.count = joint_data.shape[0]
                jnt_access.type = 'VEC3'
                jnt_access.max = np.amax(joint_data, axis=0).tolist()
                jnt_access.min = np.amin(joint_data, axis=0).tolist()
                gltfData.accessors.append(jnt_access)
                
                jnt_sampler_num = len( anim.samplers )
                jnt_sampler = pygltf.Sampler()
                jnt_sampler.input = t_access_num
                jnt_sampler.output = jnt_access_num
                jnt_sampler.interpolation = 'LINEAR'
                anim.samplers.append(jnt_sampler)
                
                jnt_channel = pygltf.Channel()
                jnt_target = pygltf.Target()
                jnt_target.node = node_num
                jnt_target.path = 'translation'
                jnt_channel.sampler = jnt_sampler_num
                jnt_channel.target = jnt_target
                anim.channels.append(jnt_channel)
            else:
                print(joint_name)
                
                # Apply gltf rotation offset to joint data
                joint_data_orig = gltfDict[joint_name]
                node_num = nodeList.index(joint_name)
                gltf_quat = np.array(gltfData.nodes[node_num].rotation)
                
                if gltf_quat.size == 0:
                    gltf_quat = [0,0,0,1]
                
                joint_data = np.zeros(joint_data_orig.shape)
                for j in range( joint_data_orig.shape[0] ):
                    joint_data[j,:] = quatmult(joint_data_orig[j,:], gltf_quat)
                    if j==0 and joint_name=='Pelvis':
                        print(joint_data[j,:])
                
                # Add joint data to buffer
                joint_buffer, joint_buff_length = convertToBytes(joint_data)
                byte_start = len( new_buffer )
                new_buffer += joint_buffer
                
                # Buffer View            
                jnt_buff_view_num = len(gltfData.bufferViews)
                jnt_buff_view = pygltf.BufferView()
                jnt_buff_view.buffer = new_buffer_num
                jnt_buff_view.byteOffset = byte_start
                jnt_buff_view.byteLength = joint_buff_length
                gltfData.bufferViews.append( jnt_buff_view )
                
                # Accessor
                jnt_access_num = len( gltfData.accessors )
                jnt_access = pygltf.Accessor()
                jnt_access.bufferView = jnt_buff_view_num
                jnt_access.byteOffset = 0
                jnt_access.componentType = 5126
                jnt_access.count = joint_data.shape[0]
                jnt_access.type = 'VEC4'
                jnt_access.max = np.amax(joint_data, axis=0).tolist()
                jnt_access.min = np.amin(joint_data, axis=0).tolist()
                gltfData.accessors.append(jnt_access)
                
                jnt_sampler_num = len( anim.samplers )
                jnt_sampler = pygltf.Sampler()
                jnt_sampler.input = t_access_num
                jnt_sampler.output = jnt_access_num
                jnt_sampler.interpolation = 'LINEAR'
                anim.samplers.append(jnt_sampler)
                
                jnt_channel = pygltf.Channel()
                jnt_target = pygltf.Target()
                jnt_target.node = node_num
                jnt_target.path = 'rotation'
                jnt_channel.sampler = jnt_sampler_num
                jnt_channel.target = jnt_target
                anim.channels.append(jnt_channel)
                
    gltfData.animations.append(anim)
    
    # Create the buffer
    buffer_header = 'data:application/octet-stream;base64,'
    data_uri = b64encode(new_buffer)
    gltf_uri = buffer_header + data_uri.decode("utf-8")
        
    # Create new buffer object
    new_gltf_buff = pygltf.Buffer()
    new_gltf_buff.uri = gltf_uri;
    new_gltf_buff.byteLength = len(new_buffer)
    
    # Add to buffer list
    gltfData.buffers.append( new_gltf_buff )

    return gltfData

###
# Get node name list from gltf
###
def getGltfNodeNames(gltfData):
    nodes = gltfData.nodes
    nodeList = []
    
    # Get node list
    for i in range( len(nodes) ):
        nodeList.append(nodes[i].name)
    
    return nodeList

###
# Convert array into byte buffer
###
def convertToBytes(arrayData):
    arrayBuffer = bytes()
    for i in range( arrayData.shape[0] ):
        for j in range( arrayData.shape[1] ):
            arrayBuffer += struct.pack( 'f', arrayData[i][j] )
        
    return arrayBuffer, len(arrayBuffer)

###
# Example script
###
def main():
    filename_base = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/SkinCap Rig/SkeletalRig_v2_0'
    new_fname = filename_base + '_animated.gltf'
    
    filename = filename_base + '.gltf'
    
    gltfDict = mocap2gltf(mvnDict)
    curr_gltf = pygltf.GLTF2().load(filename)
    new_gltf = addGltfAnimation(curr_gltf, gltfDict)
    new_gltf.save(new_fname)
    
    return gltfDict, new_gltf

if __name__ == "__main__":
    gltfDict, new_gltf = main()