// Preprocessor: A tool to pre-process images in the pneumonia dataset
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

#include <vector>
#include <thread>
#include <filesystem>
#include <algorithm>

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cassert>

#include "stb_image.h"
#include "stb_image_resize.h"
#include "stb_image_write.h"

////////////////////////////////////////
#define PREPROCESSOR_VERSION_MAJOR 0
#define PREPROCESSOR_VERSION_MINOR 1

////////////////////////////////////////
struct Config
{
private:
    std::string mRootDirPath{ "." };
    bool mQuiet{};
    int mSize{ 32 };

public:
    Config(int argc, char** argv);

    std::string GetRootDirPath() const { return mRootDirPath; }
    const char* GetRootDirPathStr() const { return mRootDirPath.c_str(); }

    bool BeQuiet() const { return mQuiet; }

    int GetSize() const { return mSize; }
};

////////////////////////////////////////
static void ParseArgAssert(int argc, int i, const char* argName)
{
    if (argc - 1 == i)
    {
        std::fprintf(stderr, "ERROR: value for %s wasn't given\n", argName);
        std::exit(EXIT_FAILURE);
    }
}

////////////////////////////////////////
Config::Config(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
    {
        ////////////////////////////////////////
        /////// OPTIONS THAT TERMINATE /////////
        ////////////////////////////////////////

        if (!std::strncmp("--help", argv[i], 50))
        {
            std::printf("Usage: %s [OPTIONS...]\n\n", argv[0]);
            std::printf("Options:\n");
            std::printf("\t--help:\t\t Print this help message\n");
            std::printf("\t--license:\t Print the license\n");
            std::printf("\t--version:\t Print the version number\n");
            std::printf("\t--quiet:\t Don't write to stdout\n");
            std::printf("\t--root-dir:\t Directory containing the dataset\n");
            std::printf("\t--size:\t\t Specify a size for output images\n");

            std::exit(EXIT_SUCCESS);
        }
        else if (!std::strncmp("--license", argv[i], 50))
        {
#define LICENSE "Copyright (C) 2023 saccharineboi\n" \
                "This program is free software: you can redistribute it and/or modify\n" \
                "it under the terms of the GNU General Public License as published by\n" \
                "the Free Software Foundation, either version 3 of the License, or\n" \
                "(at your option) any later version.\n\n" \
                "This program is distributed in the hope that it will be useful,\n" \
                "but WITHOUT ANY WARRANTY; without even the implied warranty of\n" \
                "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n" \
                "GNU General Public License for more details.\n\n" \
                "You should have received a copy of the GNU General Public License\n" \
                "along with this program.  If not, see <https://www.gnu.org/licenses/>.\n"

                std::printf("%s", LICENSE);
                std::exit(EXIT_SUCCESS);
        }
        else if (!std::strncmp("--version", argv[i], 50))
        {
            std::printf("%d.%d\n", PREPROCESSOR_VERSION_MAJOR, PREPROCESSOR_VERSION_MINOR);
            std::exit(EXIT_SUCCESS);
        }

        ////////////////////////////////////////
        ///// OPTIONS THAT ENABLE/DISABLE //////
        ////////////////////////////////////////

        if (!std::strncmp("--quiet", argv[i], 50))
        {
            mQuiet = true;
        }

        ////////////////////////////////////////
        ///////// OPTIONS WITH VALUES //////////
        ////////////////////////////////////////

        if (!std::strncmp("--root-dir", argv[i], 50))
        {
            ParseArgAssert(argc, i, "--root-dir");
            mRootDirPath = argv[i + 1];
        }

        if (!std::strncmp("--size", argv[i], 50))
        {
            ParseArgAssert(argc, i, "--size");
            mSize = std::atoi(argv[i + 1]);
            if (mSize <= 0)
            {
                std::fprintf(stderr, "ERROR: %d is an invalid size\n", mSize);
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

////////////////////////////////////////
struct Processor
{
private:
    std::uint32_t mTrainNormalCount{};
    std::uint32_t mTrainVirusCount{};
    std::uint32_t mTrainBacteriaCount{};

    std::uint32_t mTestNormalCount{};
    std::uint32_t mTestVirusCount{};
    std::uint32_t mTestBacteriaCount{};

    std::string mPath;

    bool mFailed{};
    std::string mFailureMessage;

    int mOutSize;

    enum BlockType { BlockType_TrainNormal, BlockType_TrainAbnormal,
                     BlockType_TestNormal, BlockType_TestAbnormal,
                     BlockType_Count };

    std::uint8_t* mBlocks[BlockType_Count];

    void Process(const std::string&);
    void ProcessNormalPath(const std::filesystem::path&, const char*, std::uint32_t&, std::uint8_t*);
    void ProcessAbnormalPath(const std::filesystem::path&, const char*, const char*, std::uint32_t&, std::uint32_t&, std::uint8_t*);
    void ResizeAndSave(const char*, const char*, std::uint32_t, std::uint8_t*);

public:
    explicit Processor(const Config&);

    ~Processor();

    Processor(const Processor&)                 = delete;
    Processor& operator=(const Processor&)      = delete;

    Processor(Processor&&)                      = delete;
    Processor& operator=(Processor&&)           = delete;

    void Print() const;

    bool HasFailed() const                      { return mFailed; }
    std::string GetFailureMessage() const       { return mFailureMessage; }

    std::uint32_t GetTrainNormalCount() const   { return mTrainNormalCount; }
    std::uint32_t GetTrainVirusCount() const    { return mTrainVirusCount; }
    std::uint32_t GetTrainBacteriaCount() const { return mTrainBacteriaCount; }

    std::uint32_t GetTestNormalCount() const    { return mTestNormalCount; }
    std::uint32_t GetTestVirusCount() const     { return mTestVirusCount; }
    std::uint32_t GetTestBacteriaCount() const  { return mTestBacteriaCount; }

    std::uint32_t GetTotalTrainCount() const    { return mTrainNormalCount + mTrainVirusCount + mTrainBacteriaCount; }
    std::uint32_t GetTotalTestCount() const     { return mTestNormalCount + mTestVirusCount + mTestBacteriaCount; }

    std::string GetPath() const                 { return mPath; }
};

////////////////////////////////////////
Processor::~Processor()
{
    for (std::uint32_t i = 0; i < BlockType_Count; ++i)
    {
        std::free(mBlocks[i]);
    }
}

#if 0

////////////////////////////////////////
static void LoadFromBinary(const char* url, std::uint8_t* memoryBlock, std::uint32_t memoryBlockMaxSize)
{
    FILE* file = std::fopen(url, "rb");
    if (!file)
    {
        std::fprintf(stderr, "ERROR: failed to load %s\n", url);
        std::quick_exit(EXIT_FAILURE);
    }

    std::fseek(file, 0, SEEK_END);
    std::uint32_t fileSize = static_cast<std::uint32_t>(std::ftell(file));
    std::fseek(file, 0, SEEK_SET);

    if (fileSize == std::numeric_limits<std::uint32_t>::max())
    {
        std::fprintf(stderr, "ERROR: std::ftell returned an error\n");
        std::quick_exit(EXIT_FAILURE);
    }
    else if (memoryBlockMaxSize < fileSize)
    {
        std::fprintf(stderr, "ERROR: memoryBlockMaxSize (%u) is smaller than fileSize (%u) for %s\n", memoryBlockMaxSize, fileSize, url);
        std::quick_exit(EXIT_FAILURE);
    }

    std::uint32_t bytesRead = static_cast<std::uint32_t>(std::fread(memoryBlock, sizeof(std::uint8_t), fileSize, file));
    if (bytesRead != fileSize)
    {
        std::fprintf(stderr, "ERROR: Failed to read binary data from %s\n", url);
        std::quick_exit(EXIT_FAILURE);
    }

    std::fclose(file);
}

#endif

////////////////////////////////////////
void Processor::ResizeAndSave(const char* url, const char* stem, std::uint32_t count, std::uint8_t* memoryBlock)
{
    int width, height, channelCount;
    stbi_uc* pixels = stbi_load(url, &width, &height, &channelCount, 1);
    if (!pixels)
    {
        mFailed = true;
        mFailureMessage = "Failed to load " + std::string(url);
        return;
    }

    stbir_resize_uint8(pixels, width, height, 0, memoryBlock, mOutSize, mOutSize, 0, 1);
    stbi_image_free(pixels);

    std::string newFilename{ std::string(stem) + std::to_string(count) + std::string(".png") };
    stbi_write_png(newFilename.c_str(), mOutSize, mOutSize, 1, memoryBlock, 0);
}

////////////////////////////////////////
void Processor::ProcessNormalPath(const std::filesystem::path& dataPath, const char* stem, std::uint32_t& counter, std::uint8_t* memoryBlock)
{
    for (const auto& entry : std::filesystem::directory_iterator(dataPath))
    {
        if (std::filesystem::is_regular_file(entry) && !std::filesystem::is_empty(entry) && entry.path().extension() == ".jpeg")
        {
            const char* url = entry.path().c_str();
            ResizeAndSave(url, stem, counter, memoryBlock);
            ++counter;
        }
    }
}

////////////////////////////////////////
void Processor::ProcessAbnormalPath(const std::filesystem::path& dataPath, const char* bacteriaStem, const char* virusStem,
                                                                           std::uint32_t& bacteriaCounter, std::uint32_t& virusCounter,
                                                                           std::uint8_t* block)
{
    for (const auto& entry : std::filesystem::directory_iterator(dataPath))
    {
        if (std::filesystem::is_regular_file(entry) && !std::filesystem::is_empty(entry) && entry.path().extension() == ".jpeg")
        {
            const char* url = entry.path().c_str();
            if (entry.path().string().find("bacteria") != std::string::npos)
            {
                ResizeAndSave(url, bacteriaStem, bacteriaCounter, block);
                ++bacteriaCounter;
            }
            else if (entry.path().string().find("virus") != std::string::npos)
            {
                ResizeAndSave(url, virusStem, virusCounter, block);
                ++virusCounter;
            }
        }
    }
}

////////////////////////////////////////
void Processor::Process(const std::string& dirPath)
{
    const std::filesystem::path dataPath{ mPath + dirPath };
    if (!std::filesystem::is_directory(dataPath))
    {
        mFailed = true;
        mFailureMessage = dataPath.string() + " is not a directory";
        return;
    }

    bool isNormalDataset{ dirPath.find("NORMAL") != std::string::npos };
    bool isTrainingDataset{ dirPath.find("train") != std::string::npos };

    if (isNormalDataset)
    {
        if (isTrainingDataset)
        {
            ProcessNormalPath(dataPath, "normal_train_", mTrainNormalCount, mBlocks[BlockType_TrainNormal]);
        }
        else
        {
            ProcessNormalPath(dataPath, "normal_test_", mTestNormalCount, mBlocks[BlockType_TestNormal]);
        }
    }
    else
    {
        if (isTrainingDataset)
        {
            ProcessAbnormalPath(dataPath, "bacteria_train_", "virus_train_",
                                          mTrainBacteriaCount, mTrainVirusCount,
                                          mBlocks[BlockType_TrainAbnormal]);
        }
        else
        {
            ProcessAbnormalPath(dataPath, "bacteria_test_", "virus_test_",
                                          mTestBacteriaCount, mTestVirusCount,
                                          mBlocks[BlockType_TestAbnormal]);
        }
    }
}

////////////////////////////////////////
Processor::Processor(const Config& config)
    : mPath{config.GetRootDirPath()}, mOutSize{config.GetSize()}
{
    assert(mOutSize > 0);

    for (std::uint32_t i = 0; i < BlockType_Count; ++i)
    {
        mBlocks[i] = static_cast<std::uint8_t*>(std::malloc(static_cast<std::size_t>(mOutSize * mOutSize)));
        if (!mBlocks[i])
        {
            mFailureMessage = "Failed to allocate memory for blocks in the constructor";
            mFailed = true;
            return;
        }
    }

    std::vector<std::thread> threadPool;
    threadPool.push_back(std::thread(&Processor::Process, this, "/train/NORMAL"));
    threadPool.push_back(std::thread(&Processor::Process, this, "/train/PNEUMONIA"));
    threadPool.push_back(std::thread(&Processor::Process, this, "/test/NORMAL"));
    threadPool.push_back(std::thread(&Processor::Process, this, "/test/PNEUMONIA"));

    std::ranges::for_each(threadPool, [](auto& task){ task.join(); });
}

////////////////////////////////////////
int main(int argc, char** argv)
{
    Config config(argc, argv);

    if (!config.BeQuiet())
    {
        std::printf("Root Directory: %s\n", config.GetRootDirPathStr());
        std::printf("Started processing. This can take a while ...\n");
    }

    Processor processor(config);
    if (processor.HasFailed())
    {
        std::fprintf(stderr, "ERROR: preprocessor failed: %s\n", processor.GetFailureMessage().c_str());
        std::exit(EXIT_FAILURE);
    }

    if (!config.BeQuiet())
    {
        std::printf("Results:\n");

        std::printf("Total number of training images: %u\n", processor.GetTotalTrainCount());
        std::printf("Total number of test images: %u\n", processor.GetTotalTestCount());

        std::printf("Number of normal training images processed: %u\n", processor.GetTrainNormalCount());
        std::printf("Number of virus training images processed: %u\n", processor.GetTrainVirusCount());
        std::printf("Number of bacteria training images processed: %u\n", processor.GetTrainBacteriaCount());

        std::printf("Number of normal test images processed: %u\n", processor.GetTestNormalCount());
        std::printf("Number of virus test images processed: %u\n", processor.GetTestVirusCount());
        std::printf("Number of bacteria test images processed: %u\n", processor.GetTestBacteriaCount());
    }

    return 0;
}
