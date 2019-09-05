# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 23:12:45 2019

@author: Calvin
"""

import numpy as np

def rot2quat(rot):
    qw = np.sqrt(1.+rot[0][0]+rot[1][1]+rot[2][2]) / 2.
    qx = (rot[2][1]-rot[1][2])/(4.*qw)
    qy = (rot[0][2]-rot[2][0])/(4.*qw)
    qz = (rot[1][0]-rot[0][1])/(4.*qw)
    return np.array([qx,qy,qz,qw])
    #return np.array([qw,qx,qy,qz])

def quatmult(quaternion0, quaternion1):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    
    quatout =  np.array([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0], dtype=np.float64)
    
    return quatout / np.linalg.norm( quatout )
    
    #w0, x0, y0, z0 = quaternion0
    #w1, x1, y1, z1 = quaternion1
    
    #return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
    #                 x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
    #                 -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
    #                 x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def EulerZXY(xyz, d=1):
    if d==0:
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]
    else:
        x = xyz[0]*np.pi/180.
        y = xyz[1]*np.pi/180.
        z = xyz[2]*np.pi/180.
    xrot = np.array([[1.,0.,0.],[0.,np.cos(x),-np.sin(x)],[0.,np.sin(x),np.cos(x)]])
    yrot = np.array([[np.cos(y),0.,np.sin(y)],[0.,1.,0.],[-np.sin(y),0.,np.cos(y)]])
    zrot = np.array([[np.cos(z),-np.sin(z),0.],[np.sin(z),np.cos(z),0.],[0.,0.,1.]])
    return np.matmul(zrot,np.matmul(xrot,yrot))

def quat2rot( quat ):
    rot = np.array([[ quat[3]*quat[3] + quat[0]*quat[0] - quat[1]*quat[1] - quat[2]*quat[2], 2*(quat[0]*quat[1]-quat[2]*quat[3]), 2*(quat[3]*quat[1]+quat[0]*quat[2])],
                    [ 2*(quat[0]*quat[1] + quat[3]*quat[2]), quat[3]*quat[3] - quat[0]*quat[0] + quat[1]*quat[1] - quat[2]*quat[2], 2*(quat[1]*quat[2] - quat[3]*quat[0])],
                    [ 2*(quat[0]*quat[2] - quat[3]*quat[1]), 2*(quat[3]*quat[0] + quat[1]*quat[2]), quat[3]*quat[3] - quat[0]*quat[0] - quat[1]*quat[1] + quat[2]*quat[2] ]])
    return rot