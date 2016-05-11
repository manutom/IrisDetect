#include "irisDetect.h"
#include <iostream>
#include "opencv2/opencv.hpp"


//HAAR CASCADE
cv::String face_cascade_name = "../Haar/haarcascade_frontalface_alt.xml";
cv::String eyePair_cascade_big_name = "../Haar/haarcascade_mcs_eyepair_big.xml";
cv::String eyePair_cascade_small_name = "../Haar/haarcascade_mcs_eyepair_small.xml";
cv::String eye_cascade_name = "../Haar/haarcascade_eye.xml";
cv::String leftEye_cascade_name = "../Haar/haarcascade_mcs_lefteye.xml";
cv::String rightEye_cascade_name = "../Haar/haarcascade_mcs_righteye.xml";


/****************************************************
 *                  FACE DETECTION
 ***************************************************/
bool IrisDetection::detectFace(cv::Mat & inpFrame)
{
    bool faceFlag = false;

    //Pre processing 1: Reduce the noise to avoid false detection
    float sigma_x = 0.005*inpFrame.cols;
    float sigma_y = 0.005*inpFrame.rows;
    cv::Mat frame_blur;
    GaussianBlur(inpFrame, frame_blur, cv::Size(11, 11), sigma_x, sigma_y);

    //Pre processing 2: Histogram Equalization
    std::vector<cv::Mat> channels;
    cv::split(frame_blur, channels);
    cv::Mat frame_blur_gray_equalized;
    equalizeHist( channels[0], frame_blur_gray_equalized );

    //HAAR CASCADE Face Detection
    cv::CascadeClassifier face_cascade;
    if(!face_cascade.load(face_cascade_name))
    {
        std::cout<<"Error loading face cascade xml file"<<std::endl;
        return false;
    }

    std::vector<cv::Rect> faces;

#if VIDEO_MODE_ON // liberal criteria for high recall
    face_cascade.detectMultiScale( frame_blur_gray_equalized, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT,
                                   cv::Size(frame_blur_gray_equalized.cols/10, frame_blur_gray_equalized.cols/10) );
                                    // false positives at this level must be minimal
#else // strict criteria to avoid false detection
    face_cascade.detectMultiScale( frame_blur_gray_equalized, faces, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT,
                                   cv::Size(frame_blur_gray_equalized.cols/10, frame_blur_gray_equalized.cols/10) );
                                    // false positives at this level must be minimal
#endif

#if 0
    // Draw bounding box around detected face
    for(size_t i = 0; i < faces.size(); i++)
    {
        rectangle(inpFrame, faces[i],1);
    }
#endif

    cv::Mat faceROI;

    if(faces.size() > 0)// Face Detected
    {
        faceROI = inpFrame(faces[0]); // Localise
        faceFlag = true;
    }
    else // Face Not Detected
    {
        inpFrame.copyTo(faceROI);// No localisation, still proceed with the input image
    }

    //Next step, try to detect eye pair

    bool pairFlag = false;
    cv:Mat eyePairROI;

#if VIDEO_MODE_ON
    pairFlag = detectEyePair(faceROI, eyePairROI);
#endif

    if(!pairFlag)
    {
#if DEBUG_MODE_ON
        std::cout<<"Eye pair detection failed"<<std::endl;
#endif
        detectEye(faceROI);// if no eye pair detected, then detect eyes in the whole face
    }

#if DEBUG_MODE_ON
    cv::imshow("Face", faceROI);
    //cv::waitKey(0);
#endif


#if 1
    if(faces.size() <= 0)//TO DO: Reconfirm this logic
    {
        faceROI.copyTo(inpFrame);
    }
#endif

#if VIDEO_MODE_ON
    cv::imshow("Iris Detection (blue = left eye, red = righteye)", inpFrame);
#endif

    return faceFlag;

}


/****************************************************
 *                  EYE PAIR DETECTION
 ***************************************************/
bool IrisDetection::detectEyePair(cv::Mat & faceROI, cv::Mat & eyePairROI)
{
    bool pairFlag = false;

    //Pre processing 1: Smoothing to suppress noise
    float sigma_x = 0.005*faceROI.cols;
    float sigma_y = 0.005*faceROI.rows;
    cv::Mat faceROI_blur;
    GaussianBlur(faceROI, faceROI_blur, cv::Size(7, 7), sigma_x, sigma_y);

    //Pre processing 2: Histogram Equalization
    std::vector<cv::Mat> channels;
    cv::split(faceROI_blur, channels);
    cv::Mat faceROI_blur_gray_equalize;
    equalizeHist( channels[0], faceROI_blur_gray_equalize );

    /************
     * EYE PAIR
    *************/
    // Detect eye pair
    std::vector<cv::Rect> eyePair_big;
    cv::CascadeClassifier eyePair_cascade_big;
    if(!eyePair_cascade_big.load(eyePair_cascade_big_name))
    {
        std::cout<<" Error loading Eye Pair (big) cascade xml file"<<std::endl;
        return false;
    }
    eyePair_cascade_big.detectMultiScale(faceROI_blur_gray_equalize, eyePair_big, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT,
                                         cv::Size(faceROI.cols/10, faceROI.rows/10) );

    if(eyePair_big.size() > 0)
    {
        pairFlag = true;
        eyePairROI = faceROI(eyePair_big[0]);

#if DEBUG_MODE_ON
        //std::cout<<"Big Eye Pair detected"<<std::endl;
        cv::imshow("EyePair", eyePairROI);
#endif

    }
    else
    {
        std::vector<cv::Rect> eyePair_small;
        cv::CascadeClassifier eyePair_cascade_small;
        if(!eyePair_cascade_small.load(eyePair_cascade_small_name))
        {
            std::cout<<" Error loading Eye Pair small cascade xml file"<<std::endl;
            return false;
        }
        eyePair_cascade_small.detectMultiScale(faceROI_blur_gray_equalize, eyePair_small, 1.1, 1,0|CV_HAAR_SCALE_IMAGE|
                                               CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(faceROI.cols/15, faceROI.rows/15) );
        if(eyePair_small.size() > 0)
        {
            pairFlag = true;
            eyePairROI = faceROI(eyePair_small[0]);

#if DEBUG_MODE_ON
            std::cout<<"Small eye pair detected. Failed to detect Big Eye Pair"<<std::endl;
            cv::imshow("EyePair", eyePairROI);
#endif
        }
        else
        {
            faceROI.copyTo(eyePairROI);
        }
    }

    if(pairFlag)
    {
        cv::Rect leftEyeRect;
        leftEyeRect.x = 0;
        leftEyeRect.y = 0;
        leftEyeRect.width = eyePairROI.cols/2;
        leftEyeRect.height = eyePairROI.rows;

        cv::Mat leftEyeROI = eyePairROI(leftEyeRect);
        int radiusL = leftEyeRect.width/8;
        int leftEyeCenterX = leftEyeRect.x + 0.4*leftEyeRect.width;
        int leftEyeCenterY = leftEyeRect.y + leftEyeRect.height/2;

        bool leftEyeIrisFlag = detectIris(leftEyeROI, 1);
        if(!leftEyeIrisFlag)
        {
            cv::circle(eyePairROI, cv::Point(leftEyeCenterX, leftEyeCenterY), radiusL, CV_RGB(0,0,255), 5);//Blue
        }
        cv::Rect rightEyeRect;
        rightEyeRect.x = eyePairROI.cols/2;
        rightEyeRect.y = 0;
        rightEyeRect.width = eyePairROI.cols/2;
        rightEyeRect.height = eyePairROI.rows;

        cv::Mat rightEyeROI = eyePairROI(rightEyeRect);
        int radiusR = rightEyeRect.width/8;
        int rightEyeCenterX = rightEyeRect.x + 0.6*rightEyeRect.width;
        int rightEyeCenterY = rightEyeRect.y + rightEyeRect.height/2;
        bool rightEyeIrisFlag = detectIris(rightEyeROI, 0);
        if(!rightEyeIrisFlag)
        {
            cv::circle(eyePairROI, cv::Point(rightEyeCenterX, rightEyeCenterY), radiusR, CV_RGB(255,0,0), 5);//RED
        }
#if DEBUG_MODE_ON
        cv::imshow("LeftEye", leftEyeROI);
        cv::imshow("RightEye", rightEyeROI);
#endif


    }
    return pairFlag;

}


/****************************************************
 *                  EYE DETECTION
 ***************************************************/
void IrisDetection::detectEye(cv::Mat & faceROI)
{
    //Pre processing 1: Smoothing to suppress noise
    float sigma_x = 0.005*faceROI.cols;
    float sigma_y = 0.005*faceROI.rows;
    cv::Mat faceROI_blur;
    GaussianBlur(faceROI, faceROI_blur, cv::Size(7, 7), sigma_x, sigma_y);

    //Pre processing 2: Histogram Equalization
    std::vector<cv::Mat> channels;
    cv::split(faceROI_blur, channels);
    cv::Mat faceROI_blur_gray_equalised;
    equalizeHist( channels[0], faceROI_blur_gray_equalised );

    /************
     * LEFT EYE
    *************/
    // Detect left eye
    std::vector<cv::Rect> leftEye;
    cv::CascadeClassifier leftEye_cascade;
    if(!leftEye_cascade.load(leftEye_cascade_name))
    {
        std::cout<<" Error loading left Eye cascade xml file"<<std::endl;
        return;
    }
    leftEye_cascade.detectMultiScale(faceROI_blur_gray_equalised, leftEye, 1.1, 1,
                                 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT,
                                     cv::Size(faceROI_blur_gray_equalised.cols/10, faceROI_blur_gray_equalised.rows/10) );

    float leftEyeCenterX, leftEyeCenterY;

    cv::Mat _faceROI;
    faceROI.copyTo(_faceROI);

    if(leftEye.size() > 0)
    {
        cv::Mat leftEyeROI = faceROI(leftEye[0]);

#if DEBUG_MODE_ON
        cv::imshow("LeftEye", leftEyeROI);
#endif

        bool statusL = false;
        statusL = detectIris(leftEyeROI, 1);
        if(!statusL)
        {
            int radiusL = leftEye[0].width/8;
            leftEyeCenterX = leftEye[0].x + leftEye[0].width/2;
            leftEyeCenterY = leftEye[0].y + leftEye[0].height/2;
            cv::circle(faceROI, cv::Point(leftEyeCenterX, leftEyeCenterY), radiusL, CV_RGB(0,0,255), 5);//Blue
        }
    }


    /************
     *RIGHT EYE
    *************/
    // Detect right eye
    std::vector<cv::Rect> rightEye;
    cv::CascadeClassifier rightEye_cascade;
    if(!rightEye_cascade.load(rightEye_cascade_name))
    {
        std::cout<<" Error loading right Eye cascade xml file"<<std::endl;
        return;
    }
    rightEye_cascade.detectMultiScale(faceROI_blur_gray_equalised, rightEye, 1.1, 1,
                                  0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT,
                                      cv::Size(faceROI_blur_gray_equalised.cols/15, faceROI_blur_gray_equalised.rows/15) );

    float rightEyeCenterX, rightEyeCenterY;

    if(rightEye.size() > 0)
    {
        cv::Mat rightEyeROI = _faceROI(rightEye[0]);

#if DEBUG_MODE_ON
        cv::imshow("RightEye", rightEyeROI);
#endif
        bool statusR =  false;
        statusR = detectIris(rightEyeROI, 0);
        if(!statusR)
        {
            int radiusR = rightEye[0].width/8;
            rightEyeCenterX = rightEye[0].x + rightEye[0].width/2;
            rightEyeCenterY = rightEye[0].y + rightEye[0].height/2;
            if(leftEye.size() > 0) // if both eyes individually detected
            {
                leftEyeCenterX = leftEye[0].x + leftEye[0].width/2;
                leftEyeCenterY = leftEye[0].y + leftEye[0].height/2;
                float leftRightEyeDist = std::sqrt((rightEyeCenterX - leftEyeCenterX)*(rightEyeCenterX - leftEyeCenterX)
                                                   + (rightEyeCenterY - leftEyeCenterY)*(rightEyeCenterY - leftEyeCenterY));
#if DEBUG_MODE_ON
                std::cout<<"Eye Base = "<<leftRightEyeDist<<std::endl;
                std::cout<<"Both eyes detected"<<std::endl;
#endif
                if(leftRightEyeDist > rightEye[0].width)
                {
                    cv::circle(faceROI, cv::Point(rightEyeCenterX, rightEyeCenterY), radiusR, CV_RGB(255,0,0), 5);//Red
                }
            }

        }
    }

    /****************************************
     *ALL EYES (only if right or left fails)
    ****************************************/
#if 1
    if(rightEye.size() == 0 || leftEye.size() == 0)// if left and right eyes individually not detected
    {
        // Detect all eyes
        std::vector<cv::Rect> eyes;
        cv::CascadeClassifier eye_cascade;
        if(!eye_cascade.load(eye_cascade_name))
        {
            std::cout<<" Error loading Eye cascade xml file"<<std::endl;
            return;
        }
        eye_cascade.detectMultiScale(faceROI_blur_gray_equalised, eyes, 1.1, 2,
                             0|CV_HAAR_SCALE_IMAGE,
                                     cv::Size(faceROI_blur_gray_equalised.cols/15, faceROI_blur_gray_equalised.rows/15) );
    #if 0
        for(size_t i = 0; i < eyes.size(); i++)
        {
            rectangle(faceROI, eyes[i], 1);
        }
    #endif


        if(eyes.size() > 1)
        {
            int radiusEye = eyes[1].width/10;
            //cv::circle(faceROI, cv::Point(eyes[1].x + eyes[1].width/2, eyes[1].y + eyes[1].height/2),
                    //radiusEye, CV_RGB(0,255,0), 3);//Green
        }
        else if(eyes.size() > 0)
        {
            int radiusEye = eyes[0].width/10;
            //cv::circle(faceROI, cv::Point(eyes[0].x + eyes[0].width/2, eyes[0].y + eyes[0].height/2),
                //radiusEye, CV_RGB(0,255,0), 3);//Green
        }
    }
#endif

}


/****************************************************
 *                  IRIS DETECTION
 ***************************************************/
bool IrisDetection::detectIris(cv::Mat & inpEye, int flagLR)
{


#if BLOB_DETECT

    // To Single channel Gray image
    cv::Mat inpEye_gray;
    cv::cvtColor(~inpEye, inpEye_gray, CV_BGR2GRAY);
    std::vector<cv::Mat> channels;
    cv::split(inpEye_gray, channels);

    // Histogram Equalization
    cv::Mat inpEye_gray_equalized;
    equalizeHist( channels[0], inpEye_gray_equalized );

    // Thresholding for binary mask creation
    cv::Mat inpEye_mask;
    cv::threshold(inpEye_gray_equalized, inpEye_mask, 220, 255, cv::THRESH_BINARY);

    // Remove isolated mask pixels
    medianBlur( inpEye_mask, inpEye_mask, 7 );

    // Apply the eye extreme ROI mask
    cv::Mat im;
    inpEye_gray.copyTo(im, inpEye_mask);
    im =~im;

    // Setup SimpleBlobDetector parameters.
    SimpleBlobDetector::Params params;

    // Filter by Area.
    params.filterByArea = true;
    params.minArea = 500;

    // Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 255;

    // Filter by Circularity
    params.filterByCircularity = true;
    params.minCircularity = 0.25;

    // Filter by Convexity
    params.filterByConvexity = true;
    params.minConvexity = 0.03;

    // Filter by Inertia
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;

    // Storage for blobs
    std::vector<KeyPoint> keypoints;

    // Set up detector with params
    SimpleBlobDetector detector(params);

    // Detect blobs
    detector.detect( im, keypoints);

    float maxSize = 0.0f;
    int maxIndex;
    if(keypoints.size()>0)
    {
        for(int i = 0; i<keypoints.size(); i++)
        {
            maxSize = std::max(keypoints[i].size,maxSize);
            maxIndex = i;
        }
        int radius = 3*keypoints[maxIndex].size/4;
        float area = 3.14159*radius*radius;
        if(area > (inpEye.rows*inpEye.cols/25))
        {
            if(flagLR)// Left eye
            {
                cv::circle(inpEye, cv::Point(keypoints[maxIndex].pt.x, keypoints[maxIndex].pt.y), radius ,
                   CV_RGB(0,0,255), 5);//Blue
            }
            else
            {
                cv::circle(inpEye, cv::Point(keypoints[maxIndex].pt.x, keypoints[maxIndex].pt.y), radius ,
                   CV_RGB(255,0,0), 5);//Red
            }
    #if DEBUG_MODE_ON
            std::cout<<"KeyPoint Detected"<<std::endl;
    #endif
            return true;
        }
    }

#endif

    return false;
}




