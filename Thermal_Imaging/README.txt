Author: John Tran
Version: 25 March 2017
Requirements: OpenCV with Python 2.7 bindings

This program is a work in progress demonstrating
the ability to use FLIR thermal imaging as a method of user
interaction. Residual heat left on surfaces can be used to
analyze strokes and extrapolate meaning. Similar to a touch
screen's hand swiping or stylus; thermal signatures can
be a unique way of interacting with applications.

Current project progress is having the ability to perform
fingertip recognition and point tracking. Bayesian classification
allowed the isolation of heat signature to reduce the search
space. 

The next milestone is to extract metrics which can be done
by storing the points and using hough transformation to
analyze their trajectory and indexed position. This will allow 
measuring of the strokes' length and relative angle.