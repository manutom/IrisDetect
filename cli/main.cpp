#include <iostream>
#include <stdio.h>

#include "irisDetect.h"

static void help()
{
    std::cout
        << "------------------------------------------------------------------------------" << std::endl
        << "Set the correct options in settings.h and recompile"                            << std::endl
        << "------------------------------------------------------------------------------" << std::endl
        << " Usage for images: set VIDEO_MODE_ON 0 in settings.h and recompile"             << std::endl
        << " command line:"                                                                 << std::endl
        << "./bin/cli ../dataset/images/image_name.extension"                               << std::endl
        << "------------------------------------------------------------------------------" << std::endl
        << " Usage for videos: set VIDEO_MODE_ON 1 in settings.h and recompile"             << std::endl
        << " command line:"                                                                 << std::endl
        << "./bin/cli ../dataset/videos/video_name.extension"                               << std::endl
        << "------------------------------------------------------------------------------" << std::endl
        << std::endl;
}

int main(int argc, const char** argv)
{
    help();
    cv::Mat frame;
    IrisDetection objFD;

#if DEBUG_MODE_ON
    cv::namedWindow("Face",CV_WINDOW_NORMAL);
    cv::moveWindow("Face", 10, 50);
    cv::namedWindow("LeftEye",CV_WINDOW_NORMAL);
    cv::moveWindow("LeftEye", 10, 400);
    cv::namedWindow("RightEye",CV_WINDOW_NORMAL);
    cv::moveWindow("RightEye", 950, 400);
    #if VIDEO_MODE_ON
        cv::namedWindow("EyePair",CV_WINDOW_NORMAL);
        cv::moveWindow("EyePair", 500, 400);
    #endif
#endif

#if VIDEO_MODE_ON

    #if DEBUG_MODE_ON && !WEBCAM_ON
        int faceFailcount = 0;
    #endif

    cv::namedWindow("Iris Detection (blue = left eye, red = righteye)",CV_WINDOW_NORMAL);
    cv::moveWindow("Iris Detection (blue = left eye, red = righteye)", 500, 50);


    #if WEBCAM_ON
        CvCapture* capture = cvCaptureFromCAM(-1);
        if(capture)
        {
            while(true)
            {
                frame = cvQueryFrame(capture);
    #else
        std::string filename = argv[1];

        cv::VideoCapture capture(filename);

        //Size vidSize = Size(922, 602);
        //int codec = CV_FOURCC('M', 'J', 'P', 'G');
        //VideoWriter writer("video_.avi", codec, 15, vidSize, true);

        if(!capture.isOpened())
            throw "Error reading video steam";
        else
        {

            while(true)
            {
                capture >> frame;
                //cv::resize(frame, frame, cv::Size(920, 600));

    #endif

                if(frame.empty())
                    break;

                cv::flip(frame, frame, 1);

                if(!frame.empty())
                {
                    bool faceFlag = objFD.detectFace(frame);
                    //std::cout<<frame.cols<<" x "<<frame.rows<<std::endl;
                    //writer.write(frame);

    #if DEBUG_MODE_ON
                    if(!faceFlag)
                    {
                        std::cout<<"Face detection failed"<<std::endl;
        #if !WEBCAM_ON
                        faceFailcount++;
        #endif
                    }
    #endif

                }
                else
                {
                    printf(" Frame capture error for face detection");
                    break;
                }

                cv::waitKey(10);
            }
        }

    #if DEBUG_MODE_ON && !WEBCAM_ON
        std::cout<<"Face Detection failed in : "<<faceFailcount<<" frames"<<std::endl;
    #endif


#else
        std::string filename = argv[1];
        frame = cv::imread(filename);
        bool faceFlag = objFD.detectFace(frame);
        cv::imwrite("last_output.png",frame);
        cv::namedWindow("Iris Detection (blue = left eye, red = righteye)",CV_WINDOW_NORMAL);
        cv::moveWindow("Iris Detection (blue = left eye, red = righteye)", 500, 50);
        cv::imshow("Iris Detection (blue = left eye, red = righteye)", frame);
        cv::waitKey(0);

#endif

    return 0;
}

