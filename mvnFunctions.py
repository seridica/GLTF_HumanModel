# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 21:39:09 2019

@author: Calvin
"""

###
# Imports
###
import xml.etree.ElementTree as etxml
import numpy as np
from RotationFunctions import *

###
# Function for converting mvnx file into an mvn data dictionary
###
def mvnx2mvndict(mvnx_data):
    
    # Setup data structure
    mvnDict = {}
    
    # First level - want the subject data
    mvnx_level_one = list(mvnx_data)
    subject_data = mvnx_level_one[2]
    
    # Second level - Want the joint names and the frame data
    mvnx_level_two = list(subject_data)
    joint_names = mvnx_level_two[3]
    frame_data = mvnx_level_two[5]
    nFrames = len(frame_data) - 3
    
    # Get joint names to start the dictionary
    for i in range(len(list(joint_names))):
        mvnDict[joint_names[i].attrib['label']] = np.zeros([nFrames,3])
    
    # Cycle through frames (starts at index 3 for some reason)
    joint_list = list(mvnDict.keys())
    
    # Also set up the pelvis root translation and rotation
    mvnDict['jPelvis'] = np.zeros([nFrames,4])
    mvnDict['jPelvis_trans'] = np.zeros([nFrames,3])
    
    # Add an entry for time
    mvnDict['Time'] = np.zeros([nFrames,1])
    
    # Find the relevant index for the joint data
    for i in range( len( list( frame_data[3] ) ) ):
        if 'jointAngle' in frame_data[3][i].tag and not 'XZY' in frame_data[3][i].tag and not 'Ergo' in frame_data[3][i].tag:
            ja_ind = i
        if 'position' in frame_data[3][i].tag:
            pos_ind = i
        if 'orientation' in frame_data[3][i].tag:
            ori_ind = i
    
    for i in range(nFrames):
        
        # Get joint data
        joint_text = frame_data[i+3][ja_ind].text
        joint_split = joint_text.split(' ')
        
        # Cycle through joints
        k = 0
        for j in range(len(joint_list)):
            if j in [18,0,19,1,2,3,4,10,5,11,12,8]:
                mvnDict[joint_list[j]][i][0] = float(joint_split[k])
            else:
                mvnDict[joint_list[j]][i][0] = -float(joint_split[k])
            k += 1
            #if j in [17,7,9,5,6,4,3,2,1,15,0,14]:
            if j in [17,8,9,5,6,4,3,2,1,15,0,14,20]:
                mvnDict[joint_list[j]][i][1] = float(joint_split[k])
            else:
                mvnDict[joint_list[j]][i][1] = -float(joint_split[k])
            k += 1
            if j in [18,14,10,6,11,7,8,9,12,17,13,16,20]:
                mvnDict[joint_list[j]][i][2] = float(joint_split[k])
            else:
                mvnDict[joint_list[j]][i][2] = -float(joint_split[k])
            k += 1
        
        # Fill in time (reported in milliseconds, should be at 60fps)
        mvnDict['Time'][i] = float( frame_data[i+3].attrib['time'] ) / 1000.
        
        # Pelvis root position and orientation
        orient_text = frame_data[i+3][ori_ind].text
        orient_split = orient_text.split(' ')
        
        pelvis_quat = [float( orient_split[1] ), float( orient_split[2] ), float( orient_split[3] ), float( orient_split[0] )]
        pelvis_rotquat = quatmult( [np.sqrt(2.)/2., 0., 0., np.sqrt(2.)/2.], quatmult( pelvis_quat, [-np.sqrt(2.)/2., 0., 0., np.sqrt(2.)/2.]) )
        
        mvnDict['jPelvis'][i][0] = float(pelvis_rotquat[0])
        mvnDict['jPelvis'][i][1] = float(pelvis_rotquat[1])
        mvnDict['jPelvis'][i][2] = float(pelvis_rotquat[2])
        mvnDict['jPelvis'][i][3] = float(pelvis_rotquat[3])
        
        pos_text = frame_data[i+3][pos_ind].text
        pos_split = pos_text.split(' ')
        pelvis_pos = [float( pos_split[0] ), float( pos_split[1] ), float( pos_split[2] )]
        pelvis_rot = quat2rot( [-np.sqrt(2.)/2., 0., 0., np.sqrt(2.)/2.] );
        pelvis_rotpos = np.matmul( pelvis_rot, pelvis_pos )
        
        if i == 0:
            pelvis_offset = pelvis_rotpos[1]
            print( pelvis_rotquat )
            print( pelvis_rotpos )
    
        mvnDict['jPelvis_trans'][i][0] = 0.0; #pelvis_rotpos[0];
        mvnDict['jPelvis_trans'][i][1] = pelvis_rotpos[1] - pelvis_offset;
        mvnDict['jPelvis_trans'][i][2] = 0.0; #pelvis_rotpos[2];
        
    return mvnDict

###
# This function returns the relevant parameters for converting the mvn
# dictionary into the general gltf motion dictionary
# Global reference frame convention with respect to gltf convention
# Mapping of mvn joint names to gltf joint names
# Zero-pose of mvn joints with respect to N-pose of gltf
###
def mvnJointMap():
    #zero_offset = np.array([0.,0.,0.,1.])
    zero_offset = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    ryp90_offset = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    ryn90_offset = np.array([[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.]])
    
    # Joint map
    jointMap = {}
    jointMap['jL5S1'] = {'gltf': 'L5', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jL4L3'] = {'gltf': 'L3', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jL1T12'] = {'gltf': 'T12', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jT9T8'] = {'gltf': 'T8', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jT1C7'] = {'gltf': 'C7', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jC1Head'] = {'gltf': 'Skull', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jRightT4Shoulder'] = {'gltf': 'Right Shoulder', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jRightShoulder'] = {'gltf': 'Right Upper Arm', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jRightElbow'] = {'gltf': 'Right Forearm', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jRightWrist'] = {'gltf': 'Right Hand', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jLeftT4Shoulder'] = {'gltf': 'Left Shoulder', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jLeftShoulder'] = {'gltf': 'Left Upper Arm', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jLeftElbow'] = {'gltf': 'Left Forearm', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jLeftWrist'] = {'gltf': 'Left Hand', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jRightHip'] = {'gltf': 'Right Femur', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jRightKnee'] = {'gltf': 'Right Shank', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jRightAnkle'] = {'gltf': 'Right Foot', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jLeftHip'] = {'gltf': 'Left Femur', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jLeftKnee'] = {'gltf': 'Left Shank', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jLeftAnkle'] = {'gltf': 'Left Foot', 'offset': zero_offset, 'rottype': 'ZXY'}
    jointMap['jPelvis'] = {'gltf': 'Pelvis', 'offset': zero_offset, 'rottype': 'quat'}
    jointMap['jPelvis_trans'] = {'gltf': 'Pelvis_trans', 'offset': zero_offset, 'rottype': 'pos'}
    
    # Global orientation
    #globalOrient = rot2quat( np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]]) )
    globalOrient = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    #globalOrient = rot2quat( np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]) )
    
    # Full mvn parameter map
    mvnMap = {'jointMap': jointMap, 'globalRotation': globalOrient}
    return mvnMap

###
# Sample code
###
def main():
    # Read xml data
    #filename = 'D:/UBC\Motion Reconstruction/Data\Animation Test/Subject01-002_withSensorData.mvnx'
    #filename = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/Xsens Files/summer_test1-001.mvnx'
    filename = 'D:/UBC - Postdoc/Sensors/Full Skeletal Rig/MVN Samples/Subject05-001-jump_full.mvnx'
    mvnx_data = etxml.parse(filename).getroot()
    
    # Parse xml data into a useful dictionary
    mvnDict = mvnx2mvndict(mvnx_data)
    
    # Get mvn parameters
    mvnParams = mvnJointMap()
    return mvnParams, mvnDict

if __name__ == "__main__":
    mvnParams, mvnDict = main()