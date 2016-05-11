#ifndef SETTINGS_H
#define SETTINGS_H

// VIDEO_MODE_ON = 1 Loads videos as input (tested working for video formats: .mp4 .mov .avi)
// VIDEO_MODE_ON = 0 Loads images as input (tested working for image formats: .raw .png .jpg)
#define VIDEO_MODE_ON 0

#define WEBCAM_ON 0 // ON(1), OFF(0) : ACTIVE ONLY WHEN VIDEO_MODE_ON 1

//Options for Iris detection
#define BLOB_DETECT 1

//Debug mode
#define DEBUG_MODE_ON 0

#endif // SETTINGS_H
