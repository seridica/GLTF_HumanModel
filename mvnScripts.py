# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:23:20 2020

@author: Calvin
"""

"""
This file contains various methods for processing MVN data and generating gltf animated skeletons.
"""

from mvnFunctions import *
from gltfFunctions import *
from scipy import integrate
from RotationFunctions import *
from scipy import signal
import time

def processMVN():
        # Read xml data
    #filename = 'D:/UBC\Motion Reconstruction/Data\Animation Test/Subject01-002_withSensorData.mvnx'
    filename = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/Xsens Files/summer_test1-001.mvnx'
    #filename = 'D:/UBC - Postdoc/Sensors/Full Skeletal Rig/MVN Samples/Subject05-001-jump_full.mvnx'
    #filename = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/Xsens Files/SkeletalRig-003.mvnx'
    #filename = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/Xsens Files/SkeletalRig-002.mvnx'
    #filename = 'D:/UBC - Postdoc/Sensors/SIGGRAPH Experiments/Subject04/Subject04-001_run.mvnx'
    mvnx_data = etxml.parse(filename).getroot()
    
    # Get mvn parameters and create dictionary template
    mvnParams = mvnJointMap()
    mvnParams['constraintMap'] = mvnConstraintMap()
    #mvnParams['constraintMap'] = BuildConstraintMap_Footsee()
    
    # Parse xml data and add to the mvn dictionary
    mvnDict = mvnx2mvndict(mvnx_data, mvnParams)
    
    #filename_base = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/SkinCap Rig/SkeletalRig_v1_5_lower'
    #filename_base = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/SkinCap Rig/SkeletalRig_v1_5'
    filename_base = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/SkinCap Rig/SkeletalRig_v2_0_material'
    new_fname = filename_base + '_animated.gltf'
    
    filename = filename_base + '.gltf'
    
    gltfDict = mocap2gltf(mvnDict)
    curr_gltf = pygltf.GLTF2().load(filename)
    new_gltf = addGltfAnimation(curr_gltf, gltfDict)
    new_gltf.save(new_fname)
    #new_gltf = None
    
    return mvnDict, gltfDict, new_gltf

def conversionTest():
    # Read xml data
    filename = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/Xsens Files/summer_test1-001.mvnx'
    mvnx_data = etxml.parse(filename).getroot()
    
    """
    ### SIMPLE MOCAP CONVERSION (NO SPINE CONSTRAINTS)
    # Get mvn parameters and create dictionary template
    mvnDict = mvnJointMap()
    mvnDict['constraintMap'] = mvnSimpleConstraintMap() # Fixed arms and legs
    
    mvnParams = mvnJointMap()
    mvnParams['constraintMapInv'] = {}
    
    # Parse xml data and add to the mvn dictionary
    mvnDict = mvnx2mvndict(mvnx_data, mvnDict)
    
    filename_base = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/SkinCap Rig/SkeletalRig_v2_0_material'
    new_fname = filename_base + '_conversion.gltf'
    
    filename = filename_base + '.gltf'
    
    # Convert to gltf and save for reference
    gltfDict = mocap2gltf(mvnDict)
    curr_gltf = pygltf.GLTF2().load(filename)
    new_gltf = addGltfAnimation(curr_gltf, gltfDict)
    new_gltf.save(new_fname)
    
    # Convert back to mvn
    mvnConvert = gltf2mocap(mvnParams, gltfDict)
    """
    ### COMP{LEX MOCAP CONVERSION (NO SPINE CONSTRAINTS)
    mvnDict2 = mvnJointMap()
    mvnDict2['constraintMap'] = mvnConstraintMap()
    
    mvnParams2 = mvnJointMap()
    mvnParams2['constraintMapInv'] = mvnInverseConstraintMap()
    
    # Parse xml data and add to the mvn dictionary
    mvnDict2 = mvnx2mvndict(mvnx_data, mvnDict2)
    
    filename_base = 'D:/Google Drive/UBC Postdoc/Full Skeletal Rig/SkinCap Rig/SkeletalRig_v2_0_material'
    new_fname = filename_base + '_conversion2.gltf'
    
    filename = filename_base + '.gltf'
    
    # Convert to gltf and save for reference
    gltfDict2 = mocap2gltf(mvnDict2)
    #new_gltf = addGltfAnimation(curr_gltf, gltfDict2)
    #new_gltf.save(new_fname)
    
    # Convert back to mvn
    mvnConvert2 = gltf2mocap(mvnParams2, gltfDict2)
    
    #return mvnDict, gltfDict, mvnConvert, mvnDict2, gltfDict2, mvnConvert2
    return mvnDict2, gltfDict2, mvnConvert2

"""
Run sample code as a script
"""
if __name__ == "__main__":
    #mvnDict, gltfDict, new_gltf = processMVN()
    #mvnDict, gltfDict, mvnParams, mvnDict2, gltfDict2, mvnParams2 = conversionTest()
    mvnDict2, gltfDict2, mvnParams2 = conversionTest()