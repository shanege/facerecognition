# facerecognition
Facial Recognition using Principle Components Analysis with comparisons using Nearest Center (Euclidean distance) classifier, k-Nearest Neighbours classifier and Support Vector Machine classifier.
Works on normal, masked, and tilted faces.

# Usage 
Place the appropriate images in the appropriate folders as follows:
train - training faces
test - testing faces
masked - faces with face masks on
tilted - tilted faces

# Expected Results
Results may vary depending on quality of training images but expect approximate accuracies of:

Normal test case:
NCC - 85%
kNN - 75%
SVM - 85%

Tilted test case:
NCC - 78%
kNN - 56%
SVM - 78%

Masked test case:
NCC - 73%
kNN - 55%
SVM - 45%
