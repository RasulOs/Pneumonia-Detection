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

#include "../DataLoader/DataLoader.hpp"

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
static std::vector<float> operator*(const std::vector<float>& v1, const std::vector<float>& v2)
{
    assert(v1.size() == v2.size());
    std::vector<float> res;
    res.reserve(v1.size());
    for (std::size_t i = 0; i < v1.size(); ++i)
    {
        res.push_back(v1[i] * v2[i]);
    }
    return res;
}

////////////////////////////////////////
static std::vector<float> operator+(const std::vector<float>& v1, const std::vector<float>& v2)
{
    assert(v1.size() == v2.size());
    std::vector<float> res;
    res.reserve(v1.size());
    for (std::size_t i = 0; i < v1.size(); ++i)
    {
        res.push_back(v1[i] + v2[i]);
    }
    return res;
}

////////////////////////////////////////
static std::vector<float> operator-(const std::vector<float>& v1, const std::vector<float>& v2)
{
    assert(v1.size() == v2.size());
    std::vector<float> res;
    res.reserve(v1.size());
    for (std::size_t i = 0; i < v1.size(); ++i)
    {
        res.push_back(v1[i] - v2[i]);
    }
    return res;
}

////////////////////////////////////////
static std::vector<float> operator-(float s, const std::vector<float>& v)
{
    std::vector<float> res;
    res.reserve(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        res.push_back(s - v[i]);
    }
    return res;
}

////////////////////////////////////////
struct Layer
{
private:
    std::vector<float> mNeurons;

public:
    Layer() {}
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
    Weights(std::size_t rowCount, std::size_t colCount)
        : mRowCount{rowCount}, mColCount{colCount},
          mWeightMatrix(rowCount * colCount) {}

    Weights(const Layer& layerA, const Layer& layerB,
            float lowest = -1.0f, float highest = 1.0f);

    Weights(std::vector<float>&& weightMatrix,
            std::size_t rowCount, std::size_t colCount)
        : mRowCount{rowCount}, mColCount{colCount},
          mWeightMatrix{std::move(weightMatrix)} {}

    void Print() const;
    Layer Dot(const Layer& layer) const;
    Layer Dot(const std::vector<float>& v) const;
    Weights Dot(const Weights& weights) const;
    Weights Transpose() const;

    std::size_t GetSize() const { return mRowCount * mColCount; }
    std::size_t GetRowCount() const { return mRowCount; }
    std::size_t GetColCount() const { return mColCount; }

    std::vector<float> ExtractRow(std::size_t row) const;
    std::vector<float> ExtractCol(std::size_t col) const;

    float& operator[](std::size_t i)
    {
        assert(i < mWeightMatrix.size());
        return mWeightMatrix[i];
    }

    const float& operator[](std::size_t i) const
    {
        assert(i < mWeightMatrix.size());
        return mWeightMatrix[i];
    }

    Weights& operator+(const Weights& other)
    {
        assert(mRowCount == other.mRowCount && mColCount == other.mColCount);
        for (std::size_t i = 0; i < mWeightMatrix.size(); ++i)
        {
            mWeightMatrix[i] += other[i];
        }
        return *this;
    }

    Weights& operator=(const Weights& other)
    {
        assert(mRowCount == other.mRowCount && mColCount == other.mColCount);
        for (std::size_t i = 0; i < mWeightMatrix.size(); ++i)
        {
            mWeightMatrix[i] = other[i];
        }
        return *this;
    }

    Weights& operator+=(const Weights& other)
    {
        assert(mRowCount == other.mRowCount && mColCount == other.mColCount);
        for (std::size_t i = 0; i < mWeightMatrix.size(); ++i)
        {
            mWeightMatrix[i] += other[i];
        }
        return *this;
    }
};

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
std::vector<float> Weights::ExtractRow(std::size_t row) const
{
    assert(row < mRowCount);
    std::vector<float> res;
    res.reserve(mColCount);
    for (std::size_t i = 0; i < mColCount; ++i)
    {
        res.push_back(mWeightMatrix[row * mColCount + i]);
    }
    return res;
}

////////////////////////////////////////
std::vector<float> Weights::ExtractCol(std::size_t col) const
{
    assert(col < mColCount);
    std::vector<float> res;
    res.reserve(mRowCount);
    for (std::size_t i = 0; i < mRowCount; ++i)
    {
        res.push_back(mWeightMatrix[col + i * mColCount]);
    }
    return res;
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
Layer Weights::Dot(const std::vector<float>& v) const
{
    assert(v.size() == mColCount);

    std::vector<float> newVec(mRowCount);
    for (std::size_t i = 0; i < mRowCount; ++i)
    {
        auto weightBegin = mWeightMatrix.begin() + static_cast<std::int32_t>(i * mColCount);
        auto weightEnd = weightBegin + static_cast<std::int32_t>(mColCount);
        newVec[i] = std::transform_reduce(weightBegin, weightEnd, v.begin(), 0.0f);
    }
    return Layer(std::move(newVec));
}

////////////////////////////////////////
Weights Weights::Dot(const Weights& weights) const
{
    assert(mColCount == weights.GetRowCount());

    std::size_t newRowCount{ mRowCount };
    std::size_t newColCount{ weights.GetColCount() };

    std::vector<float> newWeights(newRowCount * newColCount);

    for (std::size_t i = 0; i < newRowCount; ++i)
    {
        for (std::size_t j = 0; j < newColCount; ++j)
        {
            std::vector<float> leftRow = ExtractRow(i);
            std::vector<float> rightCol = weights.ExtractCol(j);

            newWeights[i * newColCount + j] = std::transform_reduce(leftRow.begin(),
                                                                    leftRow.end(),
                                                                    rightCol.begin(),
                                                                    0.0f);
        }
    }
    return Weights(std::move(newWeights), newRowCount, newColCount);
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
    void BackwardPass(const std::vector<float>& expectedOutput,
                      const std::vector<float>& actualOutput,
                      float learningRate);

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

////////////////////////////////////////
static float SigmoidDerivative(float x)
{
    return Sigmoid(x) * (1.0f - Sigmoid(x));
}

////////////////////////////////////////
static std::vector<float> SigmoidDerivativeVec(const std::vector<float>& v)
{
    std::vector<float> res;
    res.reserve(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        res.push_back(SigmoidDerivative(v[i]));
    }
    return res;
}

////////////////////////////////////////
static float ReLU(float x)
{
    return std::max(x, 0.0f);
}

////////////////////////////////////////
using Errors = Layer;

////////////////////////////////////////
void ANN::BackwardPass(const std::vector<float>& expectedOutput,
                       const std::vector<float>& actualOutput,
                       float learningRate)
{
    assert(expectedOutput.size() == actualOutput.size());

    std::vector<float> outputGradient = actualOutput - expectedOutput;
    std::vector<float> outputNeurons = mOutputLayer.GetNeurons();
    std::vector<float> activationGradient = outputNeurons * (1.0f - outputNeurons);
    std::vector<float> prevNeurons = [&](std::size_t hiddenLayerCount)
    {
        if (hiddenLayerCount == 0)
        {
            return mInputLayer.GetNeurons();
        }
        else
        {
            return mHiddenLayers[mHiddenLayers.size() - 1].GetNeurons();
        }
    }(mHiddenLayers.size());

    Weights nextLayer(outputGradient * activationGradient, outputGradient.size(), 1);
    Weights prevLayer(std::move(prevNeurons), 1, prevNeurons.size());

    Weights gradient = nextLayer.Dot(prevLayer);

    Weights savedWeights = mWeights[mWeights.size() - 1];
    mWeights[mWeights.size() - 1] += gradient;

    for (std::size_t i = mWeights.size() - 1; i > 0; --i)
    {
        outputGradient = savedWeights.Transpose().Dot();
    }
}

////////////////////////////////////////
static std::vector<float> ComputeCorrectOutput(Tools::DatumType type)
{
    switch (type)
    {
        case Tools::DatumType::TrainNormal:
        case Tools::DatumType::TestNormal:
            return { 1.0f, 0.0f, 0.0f };
        case Tools::DatumType::TrainBacteria:
        case Tools::DatumType::TestBacteria:
            return { 0.0f, 1.0f, 0.0f };
        case Tools::DatumType::TrainVirus:
        case Tools::DatumType::TestVirus:
            return { 0.0f, 0.0f, 1.0f };
        default:
            std::fprintf(stderr, "ERROR: invalid DatumType\n");
            return { 0.0f, 0.0f, 0.0f };
    }
}

////////////////////////////////////////
static float TestANN(ANN& ann, const Tools::DataLoader& dataLoader, std::function<float(float)> activationFunction)
{
    float error{};

    // normal
    for (std::size_t i = 0; i < dataLoader.GetTestNormalCount(); ++i)
    {
        ann.SetInput(dataLoader.GetImage(Tools::DatumType::TestNormal, i), dataLoader.GetSize());
        ann.ForwardPass(activationFunction);

        std::vector<float> expectedOutput = ComputeCorrectOutput(Tools::DatumType::TestNormal);
        std::vector<float> actualOutput = ann.GetOutput();
        error += ComputeDistance(expectedOutput, actualOutput);
    }

    // bacteria
    for (std::size_t i = 0; i < dataLoader.GetTestBacteriaCount(); ++i)
    {
        ann.SetInput(dataLoader.GetImage(Tools::DatumType::TestBacteria, i), dataLoader.GetSize());
        ann.ForwardPass(activationFunction);

        std::vector<float> expectedOutput = ComputeCorrectOutput(Tools::DatumType::TestBacteria);
        std::vector<float> actualOutput = ann.GetOutput();
        error += ComputeDistance(expectedOutput, actualOutput);
    }

    // virus
    for (std::size_t i = 0; i < dataLoader.GetTestVirusCount(); ++i)
    {
        ann.SetInput(dataLoader.GetImage(Tools::DatumType::TestVirus, i), dataLoader.GetSize());
        ann.ForwardPass(activationFunction);

        std::vector<float> expectedOutput = ComputeCorrectOutput(Tools::DatumType::TestVirus);
        std::vector<float> actualOutput = ann.GetOutput();
        error += ComputeDistance(expectedOutput, actualOutput);
    }
    return error;
}

////////////////////////////////////////
int main(int argc, char** argv)
{
    Tools::DataLoader dataLoader("PneumoniaData");

    const std::size_t inputNeuronCount{ dataLoader.GetSize() };
    const std::size_t outputNeuronCount{ dataLoader.GetCategoryCount() };
    const std::size_t hiddenLayerNeuronCount{ 50 };
    const std::size_t hiddenLayerCount{ 3 };

    ANN ann(inputNeuronCount, outputNeuronCount, hiddenLayerNeuronCount, hiddenLayerCount);

    float error = TestANN(ann, dataLoader, Sigmoid);
    std::printf("ANN Error Rate: %.4f\n", error);

    return 0;
}
