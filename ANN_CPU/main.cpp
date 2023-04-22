#include <cstdio>
#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cassert>
#include <memory>
#include <functional>

////////////////////////////////////////
struct Layer
{
private:
    std::vector<float> mNeurons;

public:
    Layer(std::size_t neuronCount, float lowest = 0.0f, float highest = 1.0f);
    Layer(float* values, std::size_t valueCount);
    Layer(std::vector<float>&& values) : mNeurons{std::move(values)} {}

    std::size_t GetNeuronCount() const { return mNeurons.size(); }
    void Print() const { std::ranges::for_each(mNeurons, [](auto x){ std::printf("%.2f\n", x); }); }
    void ApplyActivation(std::function<float(float)> activation) { std::ranges::for_each(mNeurons, [&](auto& x){ x = activation(x); }); }

    friend struct Weights;
};

////////////////////////////////////////
Layer::Layer(std::size_t neuronCount, float lowest, float highest)
    : mNeurons(neuronCount)
{
    std::random_device rd;
    std::uniform_real_distribution dist(lowest, highest);
    std::ranges::for_each(mNeurons, [&](auto& x){ x = dist(rd); });
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
            float lowest = 0.0f, float highest = 1.0f);

    void Print() const;
    Layer Dot(const Layer& layer) const;
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
void Weights::Print() const
{
    for (std::size_t row = 0; row < mRowCount; ++row)
    {
        for (std::size_t col = 0; col < mColCount; ++col)
        {
            std::printf("%.2f\t", mWeightMatrix[row * mColCount + col]);
        }
        std::putchar('\n');
    }
}

////////////////////////////////////////
Layer Weights::Dot(const Layer& layer) const
{
    assert(layer.GetNeuronCount() == mColCount);

    std::vector<float> newNeurons(mColCount);
    for (std::size_t i = 0; i < mColCount; ++i)
    {
        auto weightBegin = mWeightMatrix.begin() + static_cast<std::int32_t>(i * mColCount);
        auto weightEnd = weightBegin + static_cast<std::int32_t>(mColCount);
        newNeurons[i] = std::transform_reduce(weightBegin, weightEnd, layer.mNeurons.begin(), 0.0f);
    }
    return Layer(std::move(newNeurons));
}

////////////////////////////////////////
static float SigmoidActivation(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

////////////////////////////////////////
int main(int argc, char** argv)
{
    Layer initialLayer(3);
    Layer outputLayer(3);

    Weights w(initialLayer, outputLayer);

    outputLayer = w.Dot(outputLayer);

    outputLayer.ApplyActivation(SigmoidActivation);

    outputLayer.Print();

    return 0;
}
