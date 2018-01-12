/*Author: Rodrigo Gonçalves de Branco - 2016

gSLICrInterface - SWIG gSLICr (Ren et al. 2015) Python-to-C++ Interface

Project files reference: 
gSLICR - https://github.com/carlren/gSLICr
SWIG Numpy.i - http://docs.scipy.org/doc/numpy/reference/swig.interface-file.html
NumPy ndarray ⇋ OpenCV Mat conversion - https://github.com/yati-sagade/opencv-ndarray-conversion

Implementation file
*/

#include "gSLICrInterface.h"
#include "gSLICr_Lib/gSLICr.h"

typedef  ORUtils::Image<int> IntImage;
using namespace cv;

//Global objects/pointers
gSLICr::objects::settings my_settings;
gSLICr::engines::core_engine * gSLICr_engine = NULL;
gSLICr::UChar4Image * in_img = NULL;
long int* segmask = NULL;

//Private auxiliary function - Load a cv::Mat image to gSLICr format
void load_image(const Mat& inimg, gSLICr::UChar4Image* outimg)
{
	gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < outimg->noDims.y;y++)
		for (int x = 0; x < outimg->noDims.x; x++)
		{
			int idx = x + y * outimg->noDims.x;
			outimg_ptr[idx].b = inimg.at<Vec3b>(y, x)[0];
			outimg_ptr[idx].g = inimg.at<Vec3b>(y, x)[1];
			outimg_ptr[idx].r = inimg.at<Vec3b>(y, x)[2];
		}
}

//Private auxiliary function - Load Python-inplace-integer-label-matrix
void loadSegMask(const IntImage * inimg) {

	const int* data_ptr = inimg->GetData(MEMORYDEVICE_CPU);

	for (int i = 0; i < my_settings.img_size.x * my_settings.img_size.y; ++i) {
		segmask[i] = (long int)data_ptr[i];
	}
}

//Public function - Set Python-inplace-integer-label-matrix
void setSegMaskArray(long int * _segmask, int n) {
	segmask = _segmask;
}

//Public function - Set Python-inplace-integer-label-matrix
void setImage(PyObject* o) {
	NDArrayConverter nda;

	Mat m = nda.toMat(o);

	my_settings.img_size.x = m.cols;
	my_settings.img_size.y = m.rows;

	if(in_img)
		delete in_img;

	in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);

	load_image(m, in_img);

	//aditional parameters for gSLICr
	my_settings.coh_weight = 0.6f;
	my_settings.no_iters = 5;
	my_settings.color_space = gSLICr::RGB; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
	my_settings.seg_method = gSLICr::GIVEN_NUM; // or gSLICr::GIVEN_NUM for given number
	my_settings.do_enforce_connectivity = true; // wheter or not run the enforce connectivity step
}


void process(int n_seg) {

	//Original Python skiimage.slic call: slic(img_as_float(image), n_segments=p_segmentos, sigma=p_sigma, compactness=p_compactness)
	//How to use/transform sigma/compactness parameters?

	my_settings.no_segs = n_seg;

	if(gSLICr_engine)
		delete gSLICr_engine;

	gSLICr_engine = new gSLICr::engines::core_engine(my_settings);

	gSLICr_engine->Process_Frame(in_img);

	loadSegMask(gSLICr_engine->Get_Seg_Res());
}

bool checkCuda() {
    int devCount;
    return cudaGetDeviceCount(&devCount) == 0;
}
