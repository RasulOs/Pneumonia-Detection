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

#include "stb_image.h"

#include <cstdio>
#include <cstdint>
#include <cassert>

#include <random>
#include <vector>
#include <algorithm>
#include <memory>
#include <functional>
#include <string>
#include <filesystem>

////////////////////////////////////////
struct Layer
{
private:
    std::vector<float> mNeurons;

public:
    explicit Layer(std::size_t neuronCount) : mNeurons(neuronCount) {}
    Layer(float* values, std::size_t valueCount);
    Layer(std::vector<float>&& values) : mNeurons{std::move(values)} {}

    std::size_t GetNeuronCount() const { return mNeurons.size(); }
    void Print() const { std::ranges::for_each(mNeurons, [](auto x){ std::printf("%.4f\n", x); }); }
    void ApplyActivation(std::function<float(float)> activation) { std::ranges::for_each(mNeurons, [&](auto& x){ x = activation(x); }); }

    void SetNeurons(std::vector<float>&& neurons);
    void SetNeurons(float* neurons, std::size_t size);

    std::vector<float> GetNeurons() const { return mNeurons; }

    void AddScalar(float s) { std::ranges::for_each(mNeurons, [s](auto& x){ x += s; }); }

    friend struct Weights;
};

////////////////////////////////////////
void Layer::SetNeurons(std::vector<float>&& neurons)
{
    assert(mNeurons.size() == neurons.size());
    mNeurons = std::move(neurons);
}

////////////////////////////////////////
void Layer::SetNeurons(float* neurons, std::size_t size)
{
    assert(mNeurons.size() == size);
    for (std::size_t i = 0; i < size; ++i)
    {
        mNeurons[i] = neurons[i];
    }
}

////////////////////////////////////////
Layer::Layer(float* values, std::size_t valueCount)
    : mNeurons(valueCount)
{
    assert(nullptr != values);
    for (std::size_t i = 0; i < valueCount; ++i)
    {
        mNeurons[i] = values[i];
    }
}

////////////////////////////////////////
struct Weights
{
private:
    std::size_t mRowCount, mColCount;
    std::vector<float> mWeightMatrix;

public:
    Weights(const Layer& layerA, const Layer& layerB,
            float lowest = -1.0f, float highest = 1.0f);

    Weights(std::vector<float>&& weightMatrix,
            std::size_t rowCount, std::size_t colCount)
        : mRowCount{rowCount}, mColCount{colCount},
          mWeightMatrix{std::move(weightMatrix)} {}

    void Print() const;
    Layer Dot(const Layer& layer) const;
    Weights Transpose() const;

    std::size_t GetSize() const { return mRowCount * mColCount; }
    std::size_t GetRowCount() const { return mRowCount; }
    std::size_t GetColCount() const { return mColCount; }
};

////////////////////////////////////////
using Errors = Weights;

////////////////////////////////////////
Weights::Weights(const Layer& layerA, const Layer& layerB,
                 float lowest,
                 float highest)
    : mRowCount{ layerB.GetNeuronCount() },
      mColCount{ layerA.GetNeuronCount() },
      mWeightMatrix(mRowCount * mColCount)
{
    std::random_device rd;
    std::uniform_real_distribution dist(lowest, highest);
    std::ranges::for_each(mWeightMatrix, [&](auto& x){ x = dist(rd); });
}

////////////////////////////////////////
void Weights::Print() const
{
    for (std::size_t row = 0; row < mRowCount; ++row)
    {
        for (std::size_t col = 0; col < mColCount; ++col)
        {
            std::printf("%.4f\t", mWeightMatrix[row * mColCount + col]);
        }
        std::putchar('\n');
    }
}

////////////////////////////////////////
Layer Weights::Dot(const Layer& layer) const
{
    assert(layer.GetNeuronCount() == mColCount);

    std::vector<float> newNeurons(mRowCount);
    for (std::size_t i = 0; i < mRowCount; ++i)
    {
        auto weightBegin = mWeightMatrix.begin() + static_cast<std::int32_t>(i * mColCount);
        auto weightEnd = weightBegin + static_cast<std::int32_t>(mColCount);
        newNeurons[i] = std::transform_reduce(weightBegin, weightEnd, layer.mNeurons.begin(), 0.0f);
    }
    return Layer(std::move(newNeurons));
}

////////////////////////////////////////
Weights Weights::Transpose() const
{
    std::vector<float> res(mRowCount * mColCount);

    std::size_t ind{};
    for (std::size_t col = 0; col < mColCount; ++col)
    {
        for (std::size_t row = 0; row < mRowCount; ++row)
        {
            res[ind++] = mWeightMatrix[row * mColCount + col];
        }
    }
    return Weights(std::move(res), mColCount, mRowCount);
}

////////////////////////////////////////
struct ANN
{
private:
    Layer mInputLayer;
    std::vector<Layer> mHiddenLayers;
    Layer mOutputLayer;

    std::vector<Weights> mWeights;
    std::vector<float> mBiases;

    std::size_t mParameterCount;

public:
    ANN(std::size_t inputSize, std::size_t outputSize,
        std::size_t hiddenSize, std::size_t hiddenCount,
        float biasLowest = -1.0f, float biasHighest = 1.0f);

    void SetInput(float* neurons, std::size_t neuronCount) { mInputLayer.SetNeurons(neurons, neuronCount); }
    void SetInput(std::vector<float>&& inputValues) { mInputLayer.SetNeurons(std::move(inputValues)); }
    std::vector<float> GetOutput() const { return mOutputLayer.GetNeurons(); }

    void ResetBiases(float biasesLowest, float biasesHighest);

    void ForwardPass(std::function<float(float)> activationFunction);

    std::size_t GetParameterCount() const { return mParameterCount; }
};

////////////////////////////////////////
void ANN::ResetBiases(float biasLowest, float biasHighest)
{
    std::random_device rd;
    std::uniform_real_distribution dist(biasLowest, biasHighest);
    std::ranges::for_each(mBiases, [&](auto& x){ x = dist(rd); });
}

////////////////////////////////////////
ANN::ANN(std::size_t inputSize, std::size_t outputSize,
         std::size_t hiddenSize, std::size_t hiddenCount,
         float biasLowest, float biasHighest)
    : mInputLayer(inputSize), mOutputLayer(outputSize),
      mBiases(hiddenCount + 1), mParameterCount{}
{
#ifdef _DEBUG
    std::printf("Initializing ANN at %p with %lu input neurons, %lu output neurons, %lu hidden layers each with %lu neurons, and %lu bias values\n",
                reinterpret_cast<void*>(this), inputSize, outputSize, hiddenCount, hiddenSize, mBiases.size());
#endif

    ResetBiases(biasLowest, biasHighest);

    for (std::size_t i = 0; i < hiddenCount; ++i)
    {
        mHiddenLayers.push_back(Layer(hiddenSize));
    }

    if (hiddenCount == 0)
    {
        mWeights.push_back(Weights(mInputLayer, mOutputLayer));
    }
    else
    {
        mWeights.push_back(Weights(mInputLayer, mHiddenLayers[0]));
        for (std::size_t i = 1; i < hiddenCount; ++i)
        {
            mWeights.push_back(Weights(mHiddenLayers[i - 1], mHiddenLayers[i]));
        }
        mWeights.push_back(Weights(mHiddenLayers[hiddenCount - 1], mOutputLayer));
    }

    std::ranges::for_each(mWeights, [&](const auto& weights){ mParameterCount += weights.GetSize(); });
    mParameterCount += mBiases.size();

#ifdef _DEBUG
    std::printf("ANN at %p is initialized with %lu parameters\n", reinterpret_cast<void*>(this), mParameterCount);
#endif
}

////////////////////////////////////////
void ANN::ForwardPass(std::function<float(float)> activationFunction)
{
    assert(mBiases.size() == mHiddenLayers.size() + 1);

    if (mHiddenLayers.size() == 0)
    {
        assert(mWeights.size() == 1);

        mOutputLayer = mWeights[0].Dot(mInputLayer);
        mOutputLayer.AddScalar(mBiases[0]);
        mOutputLayer.ApplyActivation(activationFunction);
    }
    else
    {
        assert(mWeights.size() == mHiddenLayers.size() + 1);

        mHiddenLayers[0] = mWeights[0].Dot(mInputLayer);
        mHiddenLayers[0].AddScalar(mBiases[0]);
        mHiddenLayers[0].ApplyActivation(activationFunction);

        for (std::size_t i = 1; i < mHiddenLayers.size(); ++i)
        {
            mHiddenLayers[i] = mWeights[i].Dot(mHiddenLayers[i - 1]);
            mHiddenLayers[i].AddScalar(mBiases[i]);
            mHiddenLayers[i].ApplyActivation(activationFunction);
        }

        mOutputLayer = mWeights[mWeights.size() - 1].Dot(mHiddenLayers[mHiddenLayers.size() - 1]);
        mOutputLayer.AddScalar(mBiases[mBiases.size() - 1]);
        mOutputLayer.ApplyActivation(activationFunction);
    }
}

////////////////////////////////////////
static float Sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

#if 0
////////////////////////////////////////
static float ReLU(float x)
{
    return std::max(x, 0.0f);
}
#endif

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
static std::vector<float> ComputeCorrectOutput(DatumType type)
{
    switch (type)
    {
        case DatumType::TrainNormal:
        case DatumType::TestNormal:
            return { 1.0f, 0.0f, 0.0f };
        case DatumType::TrainBacteria:
        case DatumType::TestBacteria:
            return { 0.0f, 1.0f, 0.0f };
        case DatumType::TrainVirus:
        case DatumType::TestVirus:
            return { 0.0f, 0.0f, 1.0f };
        default:
            std::fprintf(stderr, "ERROR: invalid DatumType\n");
            return { 0.0f, 0.0f, 0.0f };
    }
}

////////////////////////////////////////
static float ComputeDistance(const std::vector<float>& p, const std::vector<float>& q)
{
    assert(p.size() == q.size());

    float len{};
    for (std::size_t i = 0; i < p.size(); ++i)
    {
        len += std::pow(p[i] - q[i], 2.0f);
    }
    return std::sqrt(len);
}

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
        assert(i < imageBuffer.size());
        return imageBuffer[i];
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
                floatData[i] = static_cast<float>(data[i]) / 255.0f;
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

////////////////////////////////////////
static float TestANN(ANN& ann, const DataLoader& dataLoader)
{
    float error{};

    // normal
    for (std::size_t i = 0; i < dataLoader.GetTestNormalCount(); ++i)
    {
        ann.SetInput(dataLoader.GetImage(DatumType::TestNormal, i), dataLoader.GetSize());
        ann.ForwardPass(Sigmoid);

        std::vector<float> expectedOutput = ComputeCorrectOutput(DatumType::TestNormal);
        std::vector<float> actualOutput = ann.GetOutput();
        error += ComputeDistance(expectedOutput, actualOutput);
    }

    // bacteria
    for (std::size_t i = 0; i < dataLoader.GetTestBacteriaCount(); ++i)
    {
        ann.SetInput(dataLoader.GetImage(DatumType::TestBacteria, i), dataLoader.GetSize());
        ann.ForwardPass(Sigmoid);

        std::vector<float> expectedOutput = ComputeCorrectOutput(DatumType::TestBacteria);
        std::vector<float> actualOutput = ann.GetOutput();
        error += ComputeDistance(expectedOutput, actualOutput);
    }

    // virus
    for (std::size_t i = 0; i < dataLoader.GetTestVirusCount(); ++i)
    {
        ann.SetInput(dataLoader.GetImage(DatumType::TestVirus, i), dataLoader.GetSize());
        ann.ForwardPass(Sigmoid);

        std::vector<float> expectedOutput = ComputeCorrectOutput(DatumType::TestVirus);
        std::vector<float> actualOutput = ann.GetOutput();
        error += ComputeDistance(expectedOutput, actualOutput);
    }
    return error;
}

////////////////////////////////////////
int main(int argc, char** argv)
{
    DataLoader dataLoader("PneumoniaData");

    const std::size_t inputNeuronCount{ dataLoader.GetSize() };
    const std::size_t outputNeuronCount{ dataLoader.GetCategoryCount() };
    const std::size_t hiddenLayerNeuronCount{ 30 };
    const std::size_t hiddenLayerCount{ 5 };

    ANN ann(inputNeuronCount, outputNeuronCount, hiddenLayerNeuronCount, hiddenLayerCount);

    float error = TestANN(ann, dataLoader);
    std::printf("ANN Error Rate: %.4f\n", error);

    return 0;
}
