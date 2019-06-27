# Opencv-People-counter
This project is basically designed to find no. of people leaving and boarding the bus in order to avoid fraud in ticket collection by bus conductor

Required Python3 Libraries

    NumPy
    OpenCV (v3.4+)
    dlib
    imutils
    
we feed our program with a video and SSD deep-learning based algorithm checks for people present in the given frame.

we then track these persons for a specific no. of frames as this process is comutaionally less expensive compared to SSD algorithm which is done by centroid Tracker class.

Run tracker.py
