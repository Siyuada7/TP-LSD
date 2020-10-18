from .utils import *
import numpy as np

def cal_norm_vector(use_bbox, center, z=24):
    s = np.array([(use_bbox[0]-center[0]) / z, (use_bbox[1]-center[1]) / z, 1])
    e = np.array([(use_bbox[2]-center[0]) / z, (use_bbox[3]-center[1]) / z, 1])
    norm_vector = []
    tmp = np.cross(s, e)
    norm_vector.append(tmp / np.linalg.norm(tmp))
    return np.array(norm_vector)

class BoundingBox:
    def __init__(self,
                 imageName,
                 classId,
                 x1,
                 y1,
                 x2,
                 y2,
                 typeCoordinates=CoordinatesType.Absolute,
                 imgSize=None, # ->128
                 bbType=BBType.GroundTruth,
                 classConfidence=None,
                 format='GT'):

        self._imageName = imageName
        self._typeCoordinates = typeCoordinates
        if typeCoordinates == CoordinatesType.Relative and imgSize is None:
            raise IOError(
                'Parameter \'imgSize\' is required. It is necessary to inform the image size.')
        if bbType == BBType.Detected and classConfidence is None:
            raise IOError(
                'For bbType=\'Detection\', it is necessary to inform the classConfidence value.')

        self._classConfidence = classConfidence
        self._bbType = bbType
        self._classId = classId
        self._format = format

        self._width_img = imgSize[0]
        self._height_img = imgSize[1]
        self.use_res = 128
        if format == 'GT':
            self.x1 = x1 / self._width_img * self.use_res
            self.x2 = x2 / self._width_img * self.use_res
            self.y1 = y1 / self._height_img * self.use_res
            self.y2 = y2 / self._height_img * self.use_res
        else:
            raise IOError('Please Type Format')

    def getAbsoluteBoundingBox_GT(self):
        ret = {}
        ret['pos'] = [self.x1, self.y1, self.x2, self.y2]
        ret['center'] = [(self.x1 + self.x2)/2, (self.y1 + self.y2) /2]
        ret['norm'] = cal_norm_vector(ret['pos'], ret['center'], z=24)
        return ret

    def getAbsoluteBoundingBox(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def getImageName(self):
        return self._imageName

    def getConfidence(self):
        return self._classConfidence

    def getFormat(self):
        return self._format

    def getClassId(self):
        return self._classId

    def getImageSize(self):
        return (self._width_img, self._height_img)

    def getCoordinatesType(self):
        return self._typeCoordinates

    def getBBType(self):
        return self._bbType

    @staticmethod
    def compare(det1, det2):
        det1BB = det1.getAbsoluteBoundingBox()
        det1ImgSize = det1.getImageSize()
        det2BB = det2.getAbsoluteBoundingBox()
        det2ImgSize = det2.getImageSize()

        if det1.getClassId() == det2.getClassId() and \
           det1.classConfidence == det2.classConfidenc() and \
           det1BB[0] == det2BB[0] and \
           det1BB[1] == det2BB[1] and \
           det1BB[2] == det2BB[2] and \
           det1BB[3] == det2BB[3] and \
           det1ImgSize[0] == det1ImgSize[0] and \
           det2ImgSize[1] == det2ImgSize[1]:
            return True
        return False

    @staticmethod
    def clone(boundingBox):
        absBB = boundingBox.getAbsoluteBoundingBox(format=BBFormat.XYWH)
        # return (self._x,self._y,self._x2,self._y2)
        newBoundingBox = BoundingBox(
            boundingBox.getImageName(),
            boundingBox.getClassId(),
            absBB[0],
            absBB[1],
            absBB[2],
            absBB[3],
            typeCoordinates=boundingBox.getCoordinatesType(),
            imgSize=boundingBox.getImageSize(),
            bbType=boundingBox.getBBType(),
            classConfidence=boundingBox.getConfidence(),
            format=boundingBox._format)
        return newBoundingBox
