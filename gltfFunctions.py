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
    jMap['Right Femur'] = np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])
    jMap['Right Shank'] = np.array([[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
    jMap['Right Foot'] = np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
    jMap['Left Femur'] = np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])
    jMap['Left Shank'] = np.array([[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
    jMap['Left Foot'] = np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
    
    # Spine mapping from GLTF global frame to individual joints
    jMap['L5'] = eye3;
    jMap['L4'] = eye3;
    jMap['L3'] = eye3;
    jMap['L2'] = eye3;
    jMap['L1'] = eye3;
    jMap['T12'] = eye3;
    jMap['T11'] = eye3;
    jMap['T10'] = eye3;
    jMap['T9'] = eye3;
    jMap['T8'] = eye3;
    jMap['T7'] = eye3;
    jMap['T6'] = eye3;
    jMap['T5'] = eye3;
    jMap['T4'] = eye3;
    jMap['T3'] = eye3;
    jMap['T2'] = eye3;
    jMap['T1'] = eye3;
    jMap['C7'] = eye3;
    jMap['C6'] = eye3;
    jMap['C5'] = eye3;
    jMap['C4'] = eye3;
    jMap['C3'] = eye3;
    jMap['C2'] = eye3;
    jMap['C1'] = eye3;
    jMap['Skull'] = eye3;
    
    # Arms
    jMap['Left Shoulder'] = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    jMap['Left Upper Arm'] = np.array([[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
    jMap['Left Forearm'] = np.array([[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
    jMap['Left Hand'] = np.array([[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
    jMap['Right Shoulder'] = np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
    jMap['Right Upper Arm'] = np.array([[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
    jMap['Right Forearm'] = np.array([[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
    jMap['Right Hand'] = np.array([[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
    
    return jMap

###
# Function converts a mocap dictionary into a gltf data dictionary given the
# motion capture parameters
###
def mocap2gltf(mocapDict):
    
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
                if jointMap[jointName]['dattype'] == 'LZXY':
                    #new_frame = np.matmul( gltfJointMap[gltf_name], np.array([mocap_frame]).transpose() )
                    rotmat = EulerLZXY(mocap_frame)
                    #rotmat = EulerZXY(new_frame.transpose().tolist()[0])
                    #quat = rot2quat(rotmat)
                if jointMap[jointName]['dattype'] == 'quat':
                    rotmat = quat2rot(mocap_frame)
                if jointMap[jointName]['dattype'] == 'rotmat':
                    rotmat = genRotMat(mocap_frame)
                
                # Compute the orientation
                prox_offset_rot = np.matmul( prox_offset, zoff )
                dist_motion_rot = np.matmul( rotmat, dist_offset )
                gltf_link_rot = np.matmul( prox_offset_rot, dist_motion_rot )
                gltfData[j,:] = rot2quat( gltf_link_rot )
            
            gltfDict[gltf_name] = gltfData
    
    return gltfDict

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
    filename_base = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/SkinCap Rig/SkeletalRig_v1_5'
    new_fname = filename_base + '_animated.gltf'
    
    filename = filename_base + '.gltf'
    
    gltfDict = mocap2gltf(mvnDict)
    curr_gltf = pygltf.GLTF2().load(filename)
    new_gltf = addGltfAnimation(curr_gltf, gltfDict)
    new_gltf.save(new_fname)
    
    return gltfDict, new_gltf

if __name__ == "__main__":
    gltfDict, new_gltf = main()