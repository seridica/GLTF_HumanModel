# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 23:12:45 2019

@author: Calvin
"""

import numpy as np

def rot2quat(rot):

    # Adapted from here: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    tr = rot[0][0] + rot[1][1] + rot[2][2]
    if tr> 0:
        S = np.sqrt( tr + 1.0 ) * 2.0
        qw = 0.25 * S
        qx = (rot[2][1]-rot[1][2])/S
        qy = (rot[0][2]-rot[2][0])/S
        qz = (rot[1][0]-rot[0][1])/S
    elif ((rot[0][0] > rot[1][1]) and (rot[0][0] > rot[2][2])):
        S = np.sqrt( 1.0 + rot[0][0] - rot[1][1] - rot[2][2]) * 2
        qw = (rot[2][1] - rot[1][2]) / S
        qx = 0.25 * S
        qy = (rot[0][1] + rot[1][0]) / S
        qz = (rot[0][2] + rot[2][0]) / S
    elif (rot[1][1] > rot[2][2]):
        S = np.sqrt( 1.0 + rot[1][1] - rot[0][0] - rot[2][2]) * 2
        qw = (rot[0][2] - rot[2][0]) / S
        qx = (rot[0][1] + rot[1][0]) / S
        qy = 0.25 * S
        qz = (rot[1][2] + rot[2][1]) / S
    else:
        S = np.sqrt( 1.0 + rot[2][2] - rot[1][1] - rot[0][0]) * 2
        qw = (rot[1][0] - rot[0][1]) / S
        qx = (rot[0][2] + rot[2][0]) / S
        qy = (rot[1][2] + rot[2][1]) / S
        qz = 0.25 * S
        
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

def EulerXZY(xyz, d=1):
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
    return np.matmul(xrot,np.matmul(zrot,yrot))

def EulerYXZ(xyz, d=1):
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
    return np.matmul(yrot,np.matmul(xrot,zrot))

# Left-Handed Euler ZXY
def EulerLZXY(xyz, d=1):
    if d==0:
        x = -xyz[0]
        y = -xyz[1]
        z = -xyz[2]
    else:
        x = -xyz[0]*np.pi/180.
        y = -xyz[1]*np.pi/180.
        z = -xyz[2]*np.pi/180.
    xrot = np.array([[1.,0.,0.],[0.,np.cos(x),-np.sin(x)],[0.,np.sin(x),np.cos(x)]])
    yrot = np.array([[np.cos(y),0.,np.sin(y)],[0.,1.,0.],[-np.sin(y),0.,np.cos(y)]])
    zrot = np.array([[np.cos(z),-np.sin(z),0.],[np.sin(z),np.cos(z),0.],[0.,0.,1.]])
    return np.matmul(zrot,np.matmul(xrot,yrot))

def quat2rot( quat ):
    rot = np.array([[ quat[3]*quat[3] + quat[0]*quat[0] - quat[1]*quat[1] - quat[2]*quat[2], 2*(quat[0]*quat[1]-quat[2]*quat[3]), 2*(quat[3]*quat[1]+quat[0]*quat[2])],
                    [ 2*(quat[0]*quat[1] + quat[3]*quat[2]), quat[3]*quat[3] - quat[0]*quat[0] + quat[1]*quat[1] - quat[2]*quat[2], 2*(quat[1]*quat[2] - quat[3]*quat[0])],
                    [ 2*(quat[0]*quat[2] - quat[3]*quat[1]), 2*(quat[3]*quat[0] + quat[1]*quat[2]), quat[3]*quat[3] - quat[0]*quat[0] - quat[1]*quat[1] + quat[2]*quat[2] ]])
    return rot

def genRotMat( rm ):
    rot = np.array([[rm[0], rm[1], rm[2]],[rm[3], rm[4], rm[5]],[rm[6], rm[7], rm[8]]])
    return rot

def fromRotMat( rot ):
    rm = np.array([rot[0,0], rot[0,1], rot[0,2], rot[1,0], rot[1,1], rot[1,2], rot[2,0], rot[2,1], rot[2,2]])
    return rm

def rot2XZY( rot, d=1 ):
    # Adapted from matlab code
    x = np.arctan2( rot[2][1], rot[1][1] )
    y = np.arctan2( rot[0][2], rot[0][0] )
    z = np.arctan2( -rot[0][1], np.sqrt( rot[0][0] * rot[0][0] + rot[0][2] * rot[0][2] ) )
    
    if d == 0:
        return np.array([x, y, z])
    else:
        return np.array([x,y,z]) * 180. / np.pi

def rot2ZXY( rot, d=1 ):
    x = np.arctan2( rot[2][1], np.sqrt( rot[2][0] * rot[2][0] + rot[2][2] * rot[2][2] ) )
    y = np.arctan2( -rot[2][0], rot[2][2] )
    z = np.arctan2( -rot[0][1], rot[1][1] )
    
    if d == 0:
        return np.array([x, y, z])
    else:
        return np.array([x,y,z]) * 180. / np.pi
    return

def rot2YXZ( rot, d=1 ):
    x = np.arctan2( -rot[1][2], np.sqrt( rot[0][2] * rot[0][2] + rot[2][2] * rot[2][2] ) )
    y = np.arctan2( rot[0][2], rot[2][2] )
    z = np.arctan2( rot[1][0], rot[1][1] )
    
    if d == 0:
        return np.array([x, y, z])
    else:
        return np.array([x,y,z]) * 180. / np.pi
    return

def rot2LZXY( rot, d=1 ):
    return -rot2ZXY(rot, d)


###
# Test functionality
###
def main():
    xyz = np.random.rand(3) * 2*np.pi - np.pi
    rot = EulerXZY(xyz,0)
    print(rot)
    
    xyz_out = rot2XZY(rot, 0)
    print(xyz)
    print( xyz_out )
    
    rot = EulerZXY(xyz,0)
    print(rot)
    xyz_out = rot2ZXY(rot,0)
    print(xyz)
    print(xyz_out)
    
    rot = EulerYXZ(xyz,0)
    print(rot)
    xyz_out = rot2YXZ(rot,0)
    print(xyz)
    print(xyz_out)
    
    rot = EulerLZXY(xyz,0)
    print(rot)
    xyz_out = rot2LZXY(rot,0)
    print(xyz)
    print(xyz_out)
    
    print(rot)
    qq = rot2quat(rot)
    print(qq)
    rot_out = quat2rot(qq)
    print(rot_out)

if __name__ == "__main__":
    main()