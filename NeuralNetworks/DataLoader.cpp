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

#include "DataLoader.hpp"

namespace Tools
{
    ////////////////////////////////////////
    void DataLoader::AddToBuffer(const std::string& entryPathStr,
                                 float* data,
                                 std::vector<float*>& normalImageBuffer,
                                 std::vector<float*>& bacteriaImageBuffer,
                                 std::vector<float*>& virusImageBuffer)
    {
        if (entryPathStr.find("normal") != std::string::npos)
        {
            normalImageBuffer.push_back(data);
        }
        else if (entryPathStr.find("bacteria") != std::string::npos)
        {
            bacteriaImageBuffer.push_back(data);
        }
        else if (entryPathStr.find("virus") != std::string::npos)
        {
            virusImageBuffer.push_back(data);
        }
    }

    ////////////////////////////////////////
    DataLoader::~DataLoader()
    {
        std::ranges::for_each(mTrainNormalImages, [](auto& data){ std::free(data); });
        std::ranges::for_each(mTrainBacteriaImages, [](auto& data){ std::free(data); });
        std::ranges::for_each(mTrainVirusImages, [](auto& data){ std::free(data); });

        std::ranges::for_each(mTestNormalImages, [](auto& data){ std::free(data); });
        std::ranges::for_each(mTestBacteriaImages, [](auto& data){ std::free(data); });
        std::ranges::for_each(mTestVirusImages, [](auto& data){ std::free(data); });
    }

    ////////////////////////////////////////
    DataLoader::DataLoader(const std::string& dirPath)
        : mDirPath{dirPath}, mWidth{}, mHeight{}, mTotalBytesLoaded{}
    {
        const std::filesystem::path dataPath{ dirPath };
        if (!std::filesystem::is_directory(dataPath))
        {
            std::fprintf(stderr, "ERROR: %s is not a directory\n", dirPath.c_str());
            std::exit(EXIT_FAILURE);
        }

#ifdef _DEBUG
        std::printf("DataLoader at %p has started to load data from disk...\n", reinterpret_cast<void*>(this));
#endif

        for (const auto& entry : std::filesystem::directory_iterator(dataPath))
        {
            if (std::filesystem::is_regular_file(entry) && !std::filesystem::is_empty(entry) && entry.path().extension() == ".png")
            {
                std::string entryPathStr = entry.path().string();
                int width, height, channelCount;
                stbi_uc* data = stbi_load(entryPathStr.c_str(), &width, &height, &channelCount, 0);
                if (!data)
                {
                    std::fprintf(stderr, "ERROR: failed to load %s\n", entryPathStr.c_str());
                    std::exit(EXIT_FAILURE);
                }
                if (0 == mWidth && 0 == mHeight)
                {
                    mWidth = width;
                    mHeight = height;
                }
                assert(width == height && mWidth == width && mHeight == height);

                float* floatData = static_cast<float*>(std::malloc(static_cast<std::size_t>(width * height) * sizeof(float)));
                if (!floatData)
                {
                    std::fprintf(stderr, "ERROR: not enough memory for image buffer\n");
                    std::exit(EXIT_FAILURE);
                }

                for (std::size_t i = 0; i < static_cast<std::size_t>(width * height); ++i)
                {
                    floatData[i] = static_cast<float>(data[i]) / 255.0f + 0.01f;
                    if (floatData[i] > 0.99f)
                    {
                        floatData[i] = 0.99f;
                    }
                }
                stbi_image_free(data);
                mTotalBytesLoaded += static_cast<std::size_t>(width * height) * sizeof(float);

                if (entryPathStr.find("train") != std::string::npos)
                {
                    AddToBuffer(entryPathStr, floatData, mTrainNormalImages, mTrainBacteriaImages, mTrainVirusImages);
                }
                else if (entryPathStr.find("test") != std::string::npos)
                {
                    AddToBuffer(entryPathStr, floatData, mTestNormalImages, mTestBacteriaImages, mTestVirusImages);
                }
            }
        }

#ifdef _DEBUG
        std::printf("DataLoader at %p has loaded %lu training images and %lu testing images\n", reinterpret_cast<void*>(this),
                                                                                                mTrainNormalImages.size() + mTrainBacteriaImages.size() + mTrainVirusImages.size(),
                                                                                                mTestNormalImages.size() + mTestBacteriaImages.size() + mTestVirusImages.size());
        std::printf("DataLoader at %p currently holds %lu bytes in memory\n", reinterpret_cast<void*>(this), mTotalBytesLoaded);
        std::printf("\n= = = = = = = = = = = = = = = = = = STATS = = = = = = = = = = = = = = = = = =\n");
        std::printf("[Training]\t[Normal]\t%lu\t[Bacteria]\t%lu\t[Virus]\t%lu\n", mTrainNormalImages.size(),
                                                                                                          mTrainBacteriaImages.size(),
                                                                                                          mTrainVirusImages.size());
        std::printf("[Testing]\t[Normal]\t%lu\t[Bacteria]\t%lu\t[Virus]\t%lu\n", mTestNormalImages.size(),
                                                                                                       mTestBacteriaImages.size(),
                                                                                                       mTestVirusImages.size());
        std::printf("= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n\n");
#endif
    }
}
