import torch
from torch.utils import data as Data
import torch.nn as nn
import cv2

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 5,
                stride = 1,
                padding =  2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x
    
def detect_num(ImageData):
    cnn = torch.load('numbers.pkl')
    TestOutput, _ = cnn(ImageData)
    predict = torch.max(TestOutput, dim=1)[1].data.numpy()
    print(predict)
    

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            continue

        # crop
        cv2.rectangle(img, (200,200), (400,400), (255,0,0), 3)
        cv2.putText(img, 'Detect Bounding', (200, 190), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 2)
        cv2.imshow('img', img)

        img = img[200:400, 200:400]
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, image = cv2.threshold(imgGray, 100, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow('BinaryImg', image)
        image = cv2.resize(image, (28, 28))
        
        testimg = torch.tensor(image)
        testimg = torch.unsqueeze(testimg, dim=0)
        testimg = torch.unsqueeze(testimg, dim=0)
        testimg = testimg.to(torch.float32)
        detect_num(testimg)

        if cv2.waitKey(3) == ord('q'):
            break 
    cv2.destroyAllWindows()    
    cap.release()
if __name__ == '__main__':
    main()