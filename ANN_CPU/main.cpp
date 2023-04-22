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
        void Print() const { std::ranges::for_each(mNeurons, [](auto x){ std::printf("%.4f\n", x); }); }
        void ApplyActivation(std::function<float(float)> activation) { std::ranges::for_each(mNeurons, [&](auto& x){ x = activation(x); }); }

        void SetNeurons(std::vector<float>&& neurons)
        {
            assert(mNeurons.size() == neurons.size());
            mNeurons = std::move(neurons);
        }

        std::vector<float> GetNeurons() const { return mNeurons; }

        void AddScalar(float s) { std::ranges::for_each(mNeurons, [s](auto& x){ x += s; }); }

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
        // assert(layer.GetNeuronCount() == mColCount);
        if (layer.GetNeuronCount() != mColCount)
        {
            std::printf("Layer Neuron Count = %lu, # columns = %lu\n", layer.GetNeuronCount(), mColCount);
        }

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
            float biasLowest = 0.0f, float biasHighest = 1.0f);

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
          mBiases(hiddenCount + 1)
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
#ifdef _DEBUG
        std::printf("ANN at %p is starting forward pass\n", reinterpret_cast<void*>(this));
#endif

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

#ifdef _DEBUG
        std::printf("ANN at %p has ended forward pass\n", reinterpret_cast<void*>(this));
#endif
    }

    ////////////////////////////////////////
    static float Sigmoid(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    ////////////////////////////////////////
    static float ReLU(float x)
    {
        return std::max(x, 0.0f);
    }

    ////////////////////////////////////////
    int main(int argc, char** argv)
    {
        constexpr std::size_t inputNeuronCount{ 16 * 16 };
        constexpr std::size_t outputNeuronCount{ 3 };
        constexpr std::size_t hiddenLayerNeuronCount{ 30 };
        constexpr std::size_t hiddenLayerCount{ 5 };

        ANN ann(inputNeuronCount, outputNeuronCount, hiddenLayerNeuronCount, hiddenLayerCount);

        ann.ForwardPass(Sigmoid);

        std::ranges::for_each(ann.GetOutput(), [](auto x){ std::printf("%.4f\t", x); });
        std::putchar('\n');

        return 0;
    }
