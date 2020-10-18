/**
* This file is part of LLD-SLAM.
*
* Copyright (C) 2018 Alexander Vakhitov <alexander.vakhitov at gmail dot com> (Skoltech)
* For more information see <https://github.com/alexandervakhitov/lld-slam>
*
* lld-slam is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* lld-slam is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with LLD-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORB_SLAM2_STOREDLINEEXTRACTOR_H
#define ORB_SLAM2_STOREDLINEEXTRACTOR_H


#include "LineExtractor.h"

class StoredLineExtractor : public LineExtractor
{
public:
    StoredLineExtractor(const std::string& strDetectionsStorage, bool isLeft, bool isTest = false);

    void SetFrameId(int id);
    void ExtractDetections(const cv::Mat& frame, std::vector<KeyLine>* extrLines) override;

    void ReadTXTDescriptors(std::vector<KeyLine> &extrLines, const std::string& strDescriptorsStorage, cv::Mat *lineDescs);

protected:
    int posInBatch;
    int batchId;
    std::string strDetectionsStorage;
    bool isLeft;
    int frameId;
    cv::Mat linesMat;
    bool isTest;

    void LoadDetectionsTrainDataset(int frameId);
    void LoadDetectionsTestDataset(int frameId);
};


#endif //ORB_SLAM2_STOREDLINEEXTRACTOR_H
