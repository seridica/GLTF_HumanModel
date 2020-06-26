# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 21:39:09 2019

@author: Calvin
"""

"""
Imports
"""
import xml.etree.ElementTree as etxml
import numpy as np
from RotationFunctions import *
from constraintFunctions import *

"""
Function for converting mvnx file into an mvn data dictionary
Takes in the raw xml data as an argument
Also takes the mvn parameter dictionary as an argument
The dictionary is filled with motion capture data, and also serves as a check
for the mvnx data (only pulls relevant data)
"""
def mvnx2mvndict(mvnx_data, mvnDict):
    
    # First level - want the subject data
    mvnx_level_one = list(mvnx_data)
    subject_data = mvnx_level_one[2]
    
    # Second level - Want the joint names and the frame data
    mvnx_level_two = list(subject_data)
    joint_names = [a.attrib['label'] for a in list( mvnx_level_two[3] )]
    for i in range(len(list(mvnx_level_two))):
        if 'frames' in mvnx_level_two[i].tag:
            frame_ind = i
            
    frame_data = mvnx_level_two[frame_ind]
    nFrames = len(frame_data) - 3
    
    # Get joint names that were present in the motion capture session and
    # allocate memory for the data in the dictionary
    joint_list = list(mvnDict['jointMap'].keys())
    for i in range(len(joint_names)):
        curr_joint_name = joint_names[i]
        if ( curr_joint_name in joint_list ):
            mvnDict['jointMap'][curr_joint_name]['data'] = np.zeros([nFrames,3])
    
    # Also set up the pelvis root translation and rotation
    mvnDict['jointMap']['jPelvis']['data'] = np.zeros([nFrames,4])
    mvnDict['jointMap']['jPelvis_trans']['data'] = np.zeros([nFrames,3])
    
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
    
    # Cycle through the frames
    for i in range(nFrames):
        
        # Get joint data
        joint_text = frame_data[i+3][ja_ind].text
        joint_split = joint_text.split(' ')
        
        # Cycle through joints and pull their joint data
        # MVN joints specified as three euler angles in ZXY order, but refer to ISB convention for sign
        k = 0
        for j in range(len(joint_names)):
            
            # Extract only data for the joints of interest - This ensures joint data are extracted in the proper order
            if joint_names[j] in joint_list:
                mvnDict['jointMap'][joint_names[j]]['data'][i][0] = float( joint_split[k] )
                k += 1
                mvnDict['jointMap'][joint_names[j]]['data'][i][1] = float( joint_split[k] )
                k += 1
                mvnDict['jointMap'][joint_names[j]]['data'][i][2] = float( joint_split[k] )
                k += 1
            else:
                k += 3
        
        # Fill in time (reported in milliseconds, should be at 60fps)
        mvnDict['Time'][i] = float( frame_data[i+3].attrib['time'] ) / 1000.
        
        # Pelvis root position and orientation
        # Pelvis rotations are reported in quaternions, with q0 reported first
        orient_text = frame_data[i+3][ori_ind].text
        orient_split = orient_text.split(' ')
        mvnDict['jointMap']['jPelvis']['data'][i][0] = float(orient_split[1])
        mvnDict['jointMap']['jPelvis']['data'][i][1] = float(orient_split[2])
        mvnDict['jointMap']['jPelvis']['data'][i][2] = float(orient_split[3])
        mvnDict['jointMap']['jPelvis']['data'][i][3] = float(orient_split[0])
        
        # Pelvis position data
        pos_text = frame_data[i+3][pos_ind].text
        pos_split = pos_text.split(' ')
        mvnDict['jointMap']['jPelvis_trans']['data'][i][0] = float( pos_split[0] );
        mvnDict['jointMap']['jPelvis_trans']['data'][i][1] = float( pos_split[1] );
        mvnDict['jointMap']['jPelvis_trans']['data'][i][2] = float( pos_split[2] );
    
    return mvnDict

"""
This function returns the relevant parameters for converting the mvn
dictionary into the general gltf motion dictionary
Global reference frame convention with respect to gltf convention
Mapping of mvn joint names to gltf joint names
Zero-pose of mvn joints with respect to N-pose of gltf
"""
def mvnJointMap():
    
    ###
    # Zero offset rotations for the joints,
    # Since MVN is also in a neutral N-pose, no joints have a zero offset
    ###
    zero_offset = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    
    ###
    # mvnx has some common joint rotations that are defined here for convenience
    ###
    R_spine = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]])
    R_right_arm = np.array([[-1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    R_left_arm = np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
    R_right_leg = np.array([[-1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    R_left_leg = np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
    R_right_knee = np.array([[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]])
    R_left_knee = np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])
    R_pelvis = np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
    
    ###
    # Generate Joint map with time-invariant parameters
    ###
    jointMap = {}
    jointMap['jL5S1'] = {'gltf': 'L5', 'Rprox': R_spine, 'Rdist': R_spine, 'offset': zero_offset, 'dattype': 'LZXY'}
    jointMap['jL4L3'] = {'gltf': 'L3', 'Rprox': R_spine, 'Rdist': R_spine, 'offset': zero_offset, 'dattype': 'LZXY'}
    jointMap['jL1T12'] = {'gltf': 'T12', 'Rprox': R_spine, 'Rdist': R_spine, 'offset': zero_offset, 'dattype': 'LZXY'}
    jointMap['jT9T8'] = {'gltf': 'T8', 'Rprox': R_spine, 'Rdist': R_spine, 'offset': zero_offset, 'dattype': 'LZXY'}
    jointMap['jT1C7'] = {'gltf': 'C7', 'Rprox': R_spine, 'Rdist': R_spine, 'offset': zero_offset, 'dattype': 'LZXY'}
    jointMap['jC1Head'] = {'gltf': 'Skull', 'Rprox': R_spine, 'Rdist': R_spine, 'offset': zero_offset, 'dattype': 'LZXY'}
    jointMap['jRightT4Shoulder'] = {'gltf': 'Right Shoulder', 'Rprox': R_right_arm, 'Rdist': R_right_arm, 'offset': zero_offset, 'dattype': 'LZXY'}
    jointMap['jRightShoulder'] = {'gltf': 'Right Upper Arm', 'Rprox': R_right_arm, 'Rdist': R_right_arm, 'offset': zero_offset, 'dattype': 'LZXY'}    
    jointMap['jRightElbow'] = {'gltf': 'Right Forearm', 'Rprox': R_right_arm, 'Rdist': R_right_arm, 'offset': zero_offset, 'dattype': 'LZXY'}    
    jointMap['jRightWrist'] = {'gltf': 'Right Hand', 'Rprox': R_right_arm, 'Rdist': R_right_arm, 'offset': zero_offset, 'dattype': 'LZXY'}
    jointMap['jLeftT4Shoulder'] = {'gltf': 'Left Shoulder', 'Rprox': R_left_arm, 'Rdist': R_left_arm, 'offset': zero_offset, 'dattype': 'LZXY'}
    jointMap['jLeftShoulder'] = {'gltf': 'Left Upper Arm', 'Rprox': R_left_arm, 'Rdist': R_left_arm, 'offset': zero_offset, 'dattype': 'LZXY'}    
    jointMap['jLeftElbow'] = {'gltf': 'Left Forearm', 'Rprox': R_left_arm, 'Rdist': R_left_arm, 'offset': zero_offset, 'dattype': 'LZXY'}    
    jointMap['jLeftWrist'] = {'gltf': 'Left Hand', 'Rprox': R_left_arm, 'Rdist': R_left_arm, 'offset': zero_offset, 'dattype': 'LZXY'}
    jointMap['jRightHip'] = {'gltf': 'Right Femur', 'Rprox': R_right_leg, 'Rdist': R_right_leg, 'offset': zero_offset, 'dattype': 'LZXY'}    
    jointMap['jRightKnee'] = {'gltf': 'Right Shank', 'Rprox': R_right_knee, 'Rdist': R_right_knee, 'offset': zero_offset, 'dattype': 'ZXY'}    
    jointMap['jRightAnkle'] = {'gltf': 'Right Foot', 'Rprox': R_right_leg, 'Rdist': R_right_leg, 'offset': zero_offset, 'dattype': 'LZXY'}
    jointMap['jLeftHip'] = {'gltf': 'Left Femur', 'Rprox': R_left_leg, 'Rdist': R_left_leg, 'offset': zero_offset, 'dattype': 'LZXY'}    
    jointMap['jLeftKnee'] = {'gltf': 'Left Shank', 'Rprox': R_left_knee, 'Rdist': R_left_knee, 'offset': zero_offset, 'dattype': 'ZXY'}    
    jointMap['jLeftAnkle'] = {'gltf': 'Left Foot', 'Rprox': R_left_leg, 'Rdist': R_left_leg, 'offset': zero_offset, 'dattype': 'LZXY'}
    jointMap['jPelvis'] = {'gltf': 'Pelvis', 'Rprox': R_pelvis, 'Rdist': R_pelvis, 'offset': zero_offset, 'dattype': 'quat'}
    jointMap['jPelvis_trans'] = {'gltf': 'Pelvis_trans', 'Rprox': R_pelvis, 'Rdist': R_pelvis, 'offset': zero_offset, 'dattype': 'pos'}
    
    # Global orientation
    globalOrient = np.array([[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.]])
    
    # Full mvn parameter map
    mvnMap = {'jointMap': jointMap, 'globalRotation': globalOrient}
    return mvnMap

"""
Main constraint function
- Adjust this function to select what kind of constraints you want in the
skeletal rig
- It will call on the relevant helper functions to fill in the gaps
"""
def mvnConstraintMap():
    constrMap = {}
    
    constrMap = FixedLegConstraint( constrMap, True )
    constrMap = FixedLegConstraint( constrMap, False )
    
    #constrMap = FixedArmConstraint( constrMap, True )
    constrMap = UlnarRadialArmConstraint( constrMap, True )
    #constrMap = FixedArmConstraint( constrMap, False )
    constrMap = UlnarRadialArmConstraint( constrMap, False )
    
    constrMap = CervicalSpineConstraint( constrMap )
    constrMap = ThoracoLumbarSpineConstraint( constrMap, ['L5', 'L3', 'T12', 'T8'] )
    
    return constrMap

def mvnSimpleConstraintMap():
    constrMap = {};
    constrMap = FixedLegConstraint( constrMap, True )
    constrMap = FixedLegConstraint( constrMap, False )
    constrMap = FixedArmConstraint( constrMap, True )
    constrMap = FixedArmConstraint( constrMap, False )
    
    return constrMap

def mvnInverseConstraintMap():
    constrMap = {};
    constrMap = InverseCervicalSpineConstraint( constrMap )
    constrMap = InverseThoracoLumbarSpineConstraint( constrMap, ['L5', 'L3', 'T12', 'T8'])
    return constrMap

"""
Sample code
"""
def main():
    # Read xml data
    #filename = 'D:/UBC\Motion Reconstruction/Data\Animation Test/Subject01-002_withSensorData.mvnx'
    filename = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/Xsens Files/summer_test1-001.mvnx'
    #filename = 'D:/UBC - Postdoc/Sensors/Full Skeletal Rig/MVN Samples/Subject05-001-jump_full.mvnx'
    mvnx_data = etxml.parse(filename).getroot()
    
    # Get mvn parameters and create dictionary template
    mvnParams = mvnJointMap()
    mvnParams['constraintMap'] = mvnConstraintMap()
    
    # Parse xml data and add to the mvn dictionary
    mvnDict = mvnx2mvndict(mvnx_data, mvnParams)
    
    return mvnDict

"""
Run sample code as a script
"""
if __name__ == "__main__":
    mvnDict = main()