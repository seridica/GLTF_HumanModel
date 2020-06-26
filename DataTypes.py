# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:24:49 2019

@author: Calvin
"""

def getDataTypes():
    data_types = {
            'rotmat': {'description': '3x3 rotation matrix',
                       'nvars': 9},
            'quat': {'description': 'quaternion',
                     'nvars': 4},
            'ZXY': {'description': 'ZXY euler angles',
                       'nvars': 3},
            'XZY': {'description': 'XZY euler angles',
                       'nvars': 3},
            'LZXY': {'description': 'Left handed euler angles',
                     'nvars': 3},
            'angaxis': {'description': 'Angle axis',
                       'nvars': 4},
            'pos': {'description': 'Translation',
                    'nvars': 3}
            }
    return data_types