/*Author: Rodrigo Gonçalves de Branco - 2016

gSLICrInterface - SWIG gSLICr (Ren et al. 2015) Python-to-C++ Interface

Project files reference: 
gSLICR - https://github.com/carlren/gSLICr
SWIG Numpy.i - http://docs.scipy.org/doc/numpy/reference/swig.interface-file.html
NumPy ndarray ⇋ OpenCV Mat conversion - https://github.com/yati-sagade/opencv-ndarray-conversion

SWIG file
*/

%module gSLICrInterface
%{
	#include "gSLICrInterface.h"
%}

%include "numpy.i"

%init %{
	import_array();
%}

//Public function - Set Python-inplace-integer-label-matrix
%apply (long int * INPLACE_ARRAY1, int DIM1) {(long int* _segmask, int n)}

%include "gSLICrInterface.h"





