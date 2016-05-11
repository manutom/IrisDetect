#ifndef DETECTFACE_H
#define DETECTFACE_H

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"


#include "settings.h"

using namespace cv;

/**
 * Author : Manu Tom
 *
 * This file contains the definition of the Iris Detector. The entire pipeline works
 * as follows:
 * Input:  Image/Video preferably containing a face
 *
 * Pipeline:
 *         1) Detect the face
 *         2) Detect the left and right eyes from the face
 *         3) Detect Iris of both the eyes
 *
 * Output: Detected Iris plotted as circles
 */



class IrisDetection {

    public:

        /**
        * face detection using haar cascades
        */
        bool detectFace(cv::Mat & eachFrame);

        /**
        * eye pair detection using haar cascades
        */
        bool detectEyePair(cv::Mat & faceROI, cv::Mat & eyePairROI);

        /**
        * eye detection using haar cascades
        */
        void detectEye(cv::Mat & faceROI);

        /**
        * iris detection using hough transform and/or contours
        */
        bool detectIris(cv::Mat & src, int flagLR);

    };


#endif // DETECTFACE_H



