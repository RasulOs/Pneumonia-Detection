// DataLoader: A tool to load images from the pneumonia dataset
// Copyright (C) 2023 saccharineboi
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include "stb_image.h"

#include <vector>
#include <filesystem>
#include <algorithm>
#include <utility>
#include <random>

#include <cstdint>
#include <cstdio>
#include <cassert>

namespace Tools
{
    ////////////////////////////////////////
    enum class DatumType
    {
        TrainNormal,
        TrainBacteria,
        TrainVirus,

        TestNormal,
        TestBacteria,
        TestVirus
    };

    ////////////////////////////////////////
    struct DataLoader
    {
    private:
        std::vector<float*> mTrainNormalImages;
        std::vector<float*> mTrainBacteriaImages;
        std::vector<float*> mTrainVirusImages;

        std::vector<float*> mTestNormalImages;
        std::vector<float*> mTestBacteriaImages;
        std::vector<float*> mTestVirusImages;

        std::string mDirPath;

        int mWidth, mHeight;
        std::size_t mTotalBytesLoaded;

        float* GetXImage(std::size_t i, const std::vector<float*>& imageBuffer) const
        {
            return i < imageBuffer.size() ? imageBuffer[i] : nullptr;
        }

        void AddToBuffer(const std::string& entryPathStr,
                         float* data,
                         std::vector<float*>& normalImageBuffer,
                         std::vector<float*>& bacteriaImageBuffer,
                         std::vector<float*>& virusImageBuffer);

    public:
        explicit DataLoader(const std::string& dirPath);

        ~DataLoader();

        DataLoader(const DataLoader&) = delete;
        DataLoader& operator=(const DataLoader&) = delete;

        DataLoader(DataLoader&&) = default;
        DataLoader& operator=(DataLoader&&) = default;

        std::string GetDirPath() const { return mDirPath; }
        std::size_t GetTotalBytesLoaded() const { return mTotalBytesLoaded; }

        std::size_t GetImageCount(DatumType type) const
        {
            switch (type)
            {
                case DatumType::TrainNormal:
                    return mTrainNormalImages.size();
                case DatumType::TrainBacteria:
                    return mTrainBacteriaImages.size();
                case DatumType::TrainVirus:
                    return mTrainVirusImages.size();
                case DatumType::TestNormal:
                    return mTestNormalImages.size();
                case DatumType::TestBacteria:
                    return mTestBacteriaImages.size();
                case DatumType::TestVirus:
                    return mTestVirusImages.size();
            }
        }

        float* GetImage(DatumType type, std::size_t i) const
        {
            switch (type)
            {
                case DatumType::TrainNormal:
                    return GetXImage(i, mTrainNormalImages);
                case DatumType::TrainBacteria:
                    return GetXImage(i, mTrainBacteriaImages);
                case DatumType::TrainVirus:
                    return GetXImage(i, mTrainVirusImages);
                case DatumType::TestNormal:
                    return GetXImage(i, mTestNormalImages);
                case DatumType::TestBacteria:
                    return GetXImage(i, mTestBacteriaImages);
                case DatumType::TestVirus:
                    return GetXImage(i, mTestVirusImages);
                default:
                    std::fprintf(stderr, "ERROR: invalid DatumType\n");
                    return nullptr;
            }
        }

        std::vector<float*> GetImages(DatumType type, std::size_t start, std::size_t end) const
        {
            std::vector<float*> images;

            for (std::size_t i = start; i < end; ++i)
            {
                float* data = GetImage(type, i);
                if (data)
                {
                    images.push_back(data);
                }
            }
            return images;
        }

        std::vector<std::pair<DatumType, float*>> GetAllTrainingImagesShuffled() const
        {
            std::vector<std::pair<DatumType, float*>> data;

            for (float* imageData : mTrainNormalImages)     { data.push_back({ DatumType::TrainNormal, imageData }); }
            for (float* imageData : mTrainBacteriaImages)   { data.push_back({ DatumType::TrainBacteria, imageData }); }
            for (float* imageData : mTrainVirusImages)      { data.push_back({ DatumType::TrainVirus, imageData }); }

            std::random_device rd;
            std::mt19937 twister(rd());
            std::ranges::shuffle(data, twister);

            return data;
        }

        std::vector<std::pair<DatumType, float*>> GetAllTestingImagesShuffled() const
        {
            std::vector<std::pair<DatumType, float*>> data;

            for (float* imageData : mTestNormalImages)     { data.push_back({ DatumType::TestNormal, imageData }); }
            for (float* imageData : mTestBacteriaImages)   { data.push_back({ DatumType::TestBacteria, imageData }); }
            for (float* imageData : mTestVirusImages)      { data.push_back({ DatumType::TestVirus, imageData }); }

            std::random_device rd;
            std::mt19937 twister(rd());
            std::ranges::shuffle(data, twister);

            return data;
        }

        int GetWidth() const { return mWidth; }
        int GetHeight() const { return mHeight; }
        std::size_t GetSize() const { return static_cast<std::size_t>(mWidth * mHeight); }
        std::size_t GetCategoryCount() const { return 3; }

        std::size_t GetTrainNormalCount() const { return mTrainNormalImages.size(); }
        std::size_t GetTrainBacteriaCount() const { return mTrainBacteriaImages.size(); }
        std::size_t GetTrainVirusCount() const { return mTrainVirusImages.size(); }

        std::size_t GetTestNormalCount() const { return mTestNormalImages.size(); }
        std::size_t GetTestBacteriaCount() const { return mTestBacteriaImages.size(); }
        std::size_t GetTestVirusCount() const { return mTestVirusImages.size(); }

        std::size_t GetTrainTotalCount() const { return GetTrainNormalCount() +
                                                        GetTrainBacteriaCount() +
                                                        GetTrainVirusCount(); }

        std::size_t GetTestTotalCount() const { return GetTestNormalCount() +
                                                       GetTestBacteriaCount() +
                                                       GetTestVirusCount(); }
    };
}
