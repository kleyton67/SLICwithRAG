/*Author: Rodrigo Gonçalves de Branco - 2016

gSLICrInterface - SWIG gSLICr (Ren et al. 2015) Python-to-C++ Interface

Project files reference: 
gSLICR - https://github.com/carlren/gSLICr
SWIG Numpy.i - http://docs.scipy.org/doc/numpy/reference/swig.interface-file.html
NumPy ndarray ⇋ OpenCV Mat conversion - https://github.com/yati-sagade/opencv-ndarray-conversion

Implementation file
*/

#ifndef _gSLICrINTERFACE_H
#define _gSLICrINTERFACE_H

#include "conversion.h"

#include "gSLICr_Lib/gSLICr.h"

//Public interface

bool checkCuda();

void setSegMaskArray(long int * _segmask, int n);

void setImage(PyObject*);

void process(int);

#endif
