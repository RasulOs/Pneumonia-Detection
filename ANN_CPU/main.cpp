// ANN: An artificial neural network
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
#include <cmath>

#include <random>
#include <vector>
#include <algorithm>
#include <memory>
#include <functional>
#include <string>

namespace App
{
    ////////////////////////////////////////
    struct Matrix
    {
    private:
        std::uint32_t mRowCount;
        std::uint32_t mColCount;
        std::vector<float> mData;

    public:
        Matrix(std::uint32_t rowCount, std::uint32_t colCount);

        std::uint32_t GetRowCount() const { return mRowCount; }
        std::uint32_t GetColCount() const { return mColCount; }
        std::uint32_t GetElementCount() const { return mRowCount * mColCount; }

        float& operator[](std::uint32_t i)
        {
            assert(i < static_cast<std::uint32_t>(mData.size()));
            return mData[i];
        }

        const float& operator[](std::uint32_t i) const
        {
            assert(i < static_cast<std::uint32_t>(mData.size()));
            return mData[i];
        }

        Matrix operator+(const Matrix&) const;
        Matrix operator-(const Matrix&) const;
        Matrix operator*(const Matrix&) const;
        Matrix operator/(const Matrix&) const;

        Matrix& operator+=(const Matrix&);
        Matrix& operator-=(const Matrix&);
        Matrix& operator*=(const Matrix&);
        Matrix& operator/=(const Matrix&);

        Matrix operator+(float) const;
        Matrix operator-(float) const;
        Matrix operator*(float) const;
        Matrix operator/(float) const;

        Matrix& operator+=(float);
        Matrix& operator-=(float);
        Matrix& operator*=(float);
        Matrix& operator/=(float);

        Matrix Transpose() const;

        std::vector<float> GetRow(std::uint32_t row) const;
        std::vector<float> GetCol(std::uint32_t col) const;

        float GetElement(std::uint32_t row, std::uint32_t col) const
        {
            assert(row < mRowCount && col < mColCount);
            return mData[row * mColCount + col];
        }

        float& SetElement(std::uint32_t row, std::uint32_t col)
        {
            assert(row < mRowCount && col < mColCount);
            return mData[row * mColCount + col];
        }

        void Print() const;
    };

    ////////////////////////////////////////
    static Matrix CreateRandomMatrix(std::uint32_t rowCount, std::uint32_t colCount,
                                     float min, float max)
    {
        std::random_device rd;
        std::uniform_real_distribution dist(min, max);

        Matrix matrix(rowCount, colCount);
        for (std::uint32_t row = 0; row < rowCount; ++row)
        {
            for (std::uint32_t col = 0; col < colCount; ++col)
            {
                matrix.SetElement(row, col) = dist(rd);
            }
        }
        return matrix;
    }

    ////////////////////////////////////////
    Matrix::Matrix(std::uint32_t rowCount, std::uint32_t colCount)
        : mRowCount{rowCount}, mColCount{colCount}, mData(rowCount * colCount)
    {
        assert(mRowCount > 0 && mColCount > 0);
    }

    ////////////////////////////////////////
    Matrix& Matrix::operator+=(const Matrix& other)
    {
        *this = *this + other;
        return *this;
    }

    ////////////////////////////////////////
    Matrix& Matrix::operator-=(const Matrix& other)
    {
        *this = *this - other;
        return *this;
    }

    ////////////////////////////////////////
    Matrix& Matrix::operator*=(const Matrix& other)
    {
        *this = *this * other;
        return *this;
    }

    ////////////////////////////////////////
    Matrix& Matrix::operator/=(const Matrix& other)
    {
        *this = *this / other;
        return *this;
    }

    ////////////////////////////////////////
    Matrix& Matrix::operator+=(float s)
    {
        *this = *this + s;
        return *this;
    }

    ////////////////////////////////////////
    Matrix& Matrix::operator-=(float s)
    {
        *this = *this - s;
        return *this;
    }

    ////////////////////////////////////////
    Matrix& Matrix::operator*=(float s)
    {
        *this = *this * s;
        return *this;
    }

    ////////////////////////////////////////
    Matrix& Matrix::operator/=(float s)
    {
        *this = *this / s;
        return *this;
    }

    ////////////////////////////////////////
    Matrix Matrix::operator+(const Matrix& other) const
    {
        assert(mRowCount == other.mRowCount && mColCount == other.mColCount);

        Matrix matrix(mRowCount, mColCount);
        for (std::uint32_t row = 0; row < mRowCount; ++row)
        {
            for (std::uint32_t col = 0; col < mColCount; ++col)
            {
                matrix.SetElement(row, col) = GetElement(row, col) + other.GetElement(row, col);
            }
        }
        return matrix;
    }

    ////////////////////////////////////////
    Matrix Matrix::operator-(const Matrix& other) const
    {
        assert(mRowCount == other.mRowCount && mColCount == other.mColCount);

        Matrix matrix(mRowCount, mColCount);
        for (std::uint32_t row = 0; row < mRowCount; ++row)
        {
            for (std::uint32_t col = 0; col < mColCount; ++col)
            {
                matrix.SetElement(row, col) = GetElement(row, col) - other.GetElement(row, col);
            }
        }
        return matrix;
    }

    ////////////////////////////////////////
    Matrix Matrix::operator*(const Matrix& other) const
    {
        assert(mColCount == other.mRowCount);

        Matrix matrix(mRowCount, other.mColCount);

        std::vector<std::vector<float>> cachedRightMatrixCols;
        cachedRightMatrixCols.reserve(matrix.GetColCount());

        for (std::uint32_t row = 0; row < matrix.GetRowCount(); ++row)
        {
            for (std::uint32_t col = 0; col < matrix.GetColCount(); ++col)
            {
                auto leftMatrixBegin = mData.begin() + row * mColCount;
                auto leftMatrixEnd = leftMatrixBegin + mColCount;

                if (col >= static_cast<std::uint32_t>(cachedRightMatrixCols.size()))
                {
                    cachedRightMatrixCols.push_back(other.GetCol(col));
                }

                assert(col < static_cast<std::uint32_t>(cachedRightMatrixCols.size()));

                auto rightMatrixBegin = cachedRightMatrixCols[col].begin();

                matrix.SetElement(row, col) = std::transform_reduce(leftMatrixBegin,
                                                                    leftMatrixEnd,
                                                                    rightMatrixBegin,
                                                                    0.0f);
            }
        }
        return matrix;
    }

    ////////////////////////////////////////
    Matrix Matrix::operator/(const Matrix& other) const
    {
        assert(mRowCount == other.mRowCount && mColCount == other.mColCount);

        Matrix matrix(mRowCount, mColCount);
        for (std::uint32_t row = 0; row < mRowCount; ++row)
        {
            for (std::uint32_t col = 0; col < mColCount; ++col)
            {
                matrix.SetElement(row, col) = GetElement(row, col) / other.GetElement(row, col);
            }
        }
        return matrix;
    }

    ////////////////////////////////////////
    std::vector<float> Matrix::GetRow(std::uint32_t row) const
    {
        assert(row < mRowCount);
        std::vector<float> rowVector;
        rowVector.reserve(mColCount);
        for (std::uint32_t col = 0; col < mColCount; ++col)
        {
            rowVector.push_back(GetElement(row, col));
        }
        return rowVector;
    }

    ////////////////////////////////////////
    std::vector<float> Matrix::GetCol(std::uint32_t col) const
    {
        assert(col < mColCount);
        std::vector<float> colVector;
        colVector.reserve(mRowCount);
        for (std::uint32_t row = 0; row < mRowCount; ++row)
        {
            colVector.push_back(GetElement(row, col));
        }
        return colVector;
    }

    ////////////////////////////////////////
    Matrix Matrix::operator+(float s) const
    {
        Matrix matrix(mRowCount, mColCount);
        for (std::uint32_t row = 0; row < mRowCount; ++row)
        {
            for (std::uint32_t col = 0; col < mColCount; ++col)
            {
                matrix.SetElement(row, col) = GetElement(row, col) + s;
            }
        }
        return matrix;
    }

    ////////////////////////////////////////
    Matrix Matrix::operator-(float s) const
    {
        Matrix matrix(mRowCount, mColCount);
        for (std::uint32_t row = 0; row < mRowCount; ++row)
        {
            for (std::uint32_t col = 0; col < mColCount; ++col)
            {
                matrix.SetElement(row, col) = GetElement(row, col) - s;
            }
        }
        return matrix;
    }

    ////////////////////////////////////////
    Matrix Matrix::operator*(float s) const
    {
        Matrix matrix(mRowCount, mColCount);
        for (std::uint32_t row = 0; row < mRowCount; ++row)
        {
            for (std::uint32_t col = 0; col < mColCount; ++col)
            {
                matrix.SetElement(row, col) = GetElement(row, col) * s;
            }
        }
        return matrix;
    }

    ////////////////////////////////////////
    Matrix Matrix::operator/(float s) const
    {
        Matrix matrix(mRowCount, mColCount);
        for (std::uint32_t row = 0; row < mRowCount; ++row)
        {
            for (std::uint32_t col = 0; col < mColCount; ++col)
            {
                matrix.SetElement(row, col) = GetElement(row, col) / s;
            }
        }
        return matrix;
    }

    ////////////////////////////////////////
    Matrix Matrix::Transpose() const
    {
        Matrix matrix(mColCount, mRowCount);
        for (std::uint32_t row = 0; row < mRowCount; ++row)
        {
            for (std::uint32_t col = 0; col < mColCount; ++col)
            {
                matrix.SetElement(col, row) = GetElement(row, col);
            }
        }
        return matrix;
    }

    ////////////////////////////////////////
    void Matrix::Print() const
    {
        for (std::uint32_t row = 0; row < mRowCount; ++row)
        {
            for (std::uint32_t col = 0; col < mColCount; ++col)
            {
                std::printf("%.6f\t", GetElement(row, col));
            }
            std::putchar('\n');
        }
    }

    ////////////////////////////////////////
    static float SigmoidActivation(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    ////////////////////////////////////////
    static float SigmoidDerivative(float x)
    {
        return SigmoidActivation(x) * (1.0f - SigmoidActivation(x));
    }

    ////////////////////////////////////////
    static float ReLUActivation(float x)
    {
        return std::max(0.0f, x);
    }

    ////////////////////////////////////////
    static float ReLUDerivative(float x)
    {
        return x > 0 ? 1.0f : 0.0f;
    }

    ////////////////////////////////////////
    static Matrix ApplyActivationToMatrix(const Matrix& inputMatrix, std::function<float(float)> activationFunction)
    {
        Matrix outputMatrix(inputMatrix.GetRowCount(), inputMatrix.GetColCount());
        for (std::uint32_t row = 0; row < inputMatrix.GetRowCount(); ++row)
        {
            for (std::uint32_t col = 0; col < inputMatrix.GetColCount(); ++col)
            {
                outputMatrix.SetElement(row, col) = activationFunction(inputMatrix.GetElement(row, col));
            }
        }
        return outputMatrix;
    }

    ////////////////////////////////////////
    struct ANN
    {
    private:
        Matrix mInputLayer;
        std::vector<Matrix> mHiddenLayers;
        Matrix mOutputLayer;

        std::vector<Matrix> mWeights;
        std::vector<float> mBiases;

        void CreateHiddenLayers(std::uint32_t hiddenLayerCount, std::uint32_t hiddenLayerNeuronCount);
        void CreateRandomBiases(float min, float max);
        void CreateRandomWeights(std::uint32_t inputLayerNeuronCount,
                                 std::uint32_t outputLayerNeuronCount,
                                 std::uint32_t hiddenLayerNeuronCount,
                                 std::uint32_t hiddenLayerCount);

    public:
        ANN(std::uint32_t inputLayerNeuronCount,
            std::uint32_t outputLayerNeuronCount,
            std::uint32_t hiddenLayerNeuronCount,
            std::uint32_t hiddenLayerCount);

        std::uint32_t GetInputNeuronCount() const { return mInputLayer.GetElementCount(); }
        std::uint32_t GetHiddenNeuronCount() const { return mHiddenLayers.size() > 0 ? mHiddenLayers[0].GetElementCount() : 0; }
        std::uint32_t GetOutputNeuronCount() const { return mOutputLayer.GetElementCount(); }

        std::uint32_t GetHiddenLayerCount() const { return static_cast<std::uint32_t>(mHiddenLayers.size()); }
        std::uint32_t GetWeightCount() const { return static_cast<std::uint32_t>(mWeights.size()); }
        std::uint32_t GetBiasCount() const { return static_cast<std::uint32_t>(mBiases.size()); }

        std::uint32_t GetParameterCount() const;

        Matrix GetInputLayer() const { return mInputLayer; }
        Matrix GetOutputLayer() const { return mOutputLayer; }

        Matrix GetHiddenLayer(std::uint32_t i) const
        {
            assert(i < static_cast<std::uint32_t>(mHiddenLayers.size()));
            return mHiddenLayers[i];
        }

        Matrix GetWeightMatrix(std::uint32_t i) const
        {
            assert(i < static_cast<std::uint32_t>(mWeights.size()));
            return mWeights[i];
        }

        float GetBias(std::uint32_t i) const
        {
            assert(i < static_cast<std::uint32_t>(mBiases.size()));
            return mBiases[i];
        }

        Matrix FeedForward(const Matrix& input);
        Matrix FeedForward(const std::vector<float>& input);
        Matrix FeedForward(float* input, std::size_t size);

        Matrix BackPropagate(const Matrix& input);
    };

    ////////////////////////////////////////
    void ANN::CreateHiddenLayers(std::uint32_t hiddenLayerCount, std::uint32_t hiddenLayerNeuronCount)
    {
        assert(hiddenLayerNeuronCount > 0);

        for (std::uint32_t i = 0; i < hiddenLayerCount; ++i)
        {
            mHiddenLayers.push_back(Matrix(hiddenLayerNeuronCount, 1));
        }
    }

    ////////////////////////////////////////
    void ANN::CreateRandomBiases(float min, float max)
    {
        assert(min < max);

        std::random_device rd;
        std::uniform_real_distribution dist(min, max);

        for (std::size_t i = 0; i < mWeights.size(); ++i)
        {
            mBiases.push_back(dist(rd));
        }
    }

    ////////////////////////////////////////
    void ANN::CreateRandomWeights(std::uint32_t inputLayerNeuronCount,
                                  std::uint32_t outputLayerNeuronCount,
                                  std::uint32_t hiddenLayerNeuronCount,
                                  std::uint32_t hiddenLayerCount)
    {
        assert(inputLayerNeuronCount > 0 && outputLayerNeuronCount > 0);

        if (hiddenLayerCount == 0)
        {
            float min = -std::sqrt(2.0f / static_cast<float>(inputLayerNeuronCount));
            float max = -min;

            mWeights.push_back(CreateRandomMatrix(outputLayerNeuronCount, inputLayerNeuronCount, min, max));
        }
        else
        {
            float min = -std::sqrt(2.0f / static_cast<float>(inputLayerNeuronCount));
            float max = -min;

            mWeights.push_back(CreateRandomMatrix(hiddenLayerNeuronCount, inputLayerNeuronCount, min, max));
            for (std::uint32_t i = 1; i < hiddenLayerCount; ++i)
            {
                min = -std::sqrt(2.0f / static_cast<float>(hiddenLayerNeuronCount));
                max = -min;

                mWeights.push_back(CreateRandomMatrix(mHiddenLayers[i].GetElementCount(), mHiddenLayers[i - 1].GetElementCount(), min, max));
            }
            mWeights.push_back(CreateRandomMatrix(outputLayerNeuronCount, mHiddenLayers[hiddenLayerCount - 1].GetElementCount(), min, max));
        }
    }

    ////////////////////////////////////////
    ANN::ANN(std::uint32_t inputLayerNeuronCount,
             std::uint32_t outputLayerNeuronCount,
             std::uint32_t hiddenLayerNeuronCount,
             std::uint32_t hiddenLayerCount)
        : mInputLayer(inputLayerNeuronCount, 1),
          mOutputLayer(outputLayerNeuronCount, 1)
    {
#ifdef _DEBUG
        std::printf("Initializing ANN at %p with %u input neurons, %u output neurons, %u hidden layers and %u neurons in each hidden layer\n", reinterpret_cast<void*>(this),
                                                                                                                                               inputLayerNeuronCount,
                                                                                                                                               outputLayerNeuronCount,
                                                                                                                                               hiddenLayerCount,
                                                                                                                                               hiddenLayerNeuronCount);
#endif

        CreateHiddenLayers(hiddenLayerCount, hiddenLayerNeuronCount);
        CreateRandomWeights(inputLayerNeuronCount, outputLayerNeuronCount, hiddenLayerNeuronCount, hiddenLayerCount);
        CreateRandomBiases(0.0f, 1.0f);

#ifdef _DEBUG
        std::printf("ANN at %p is initialized\n\n", reinterpret_cast<void*>(this));
#endif
    }

    ////////////////////////////////////////
    Matrix ANN::FeedForward(const Matrix& input)
    {
        assert(input.GetRowCount() == mInputLayer.GetRowCount() &&
               input.GetColCount() == mInputLayer.GetColCount());

        mInputLayer = input;

        if (mHiddenLayers.size() == 0)
        {
            assert(mWeights.size() == 1);

            mOutputLayer = ApplyActivationToMatrix(mWeights[0] * mInputLayer, SigmoidActivation);
        }
        else
        {
            assert(mWeights.size() > 1);

            mHiddenLayers[0] = ApplyActivationToMatrix(mWeights[0] * mInputLayer, ReLUActivation);
            for (std::uint32_t i = 1; i < GetHiddenLayerCount(); ++i)
            {
                mHiddenLayers[i] = ApplyActivationToMatrix(mWeights[i] * mHiddenLayers[i - 1], ReLUActivation);
            }
            mOutputLayer = ApplyActivationToMatrix(mWeights[mWeights.size() - 1] * mHiddenLayers[mHiddenLayers.size() - 1], SigmoidActivation);
        }

        return mOutputLayer;
    }

    ////////////////////////////////////////
    Matrix ANN::FeedForward(const std::vector<float>& input)
    {
        Matrix inputMatrix(static_cast<std::uint32_t>(input.size()), 1);
        for (std::uint32_t row = 0; row < inputMatrix.GetRowCount(); ++row)
        {
            for (std::uint32_t col = 0; col < inputMatrix.GetColCount(); ++col)
            {
                inputMatrix.SetElement(row, col) = input[row];
            }
        }
        return FeedForward(inputMatrix);
    }

    ////////////////////////////////////////
    Matrix ANN::FeedForward(float* input, std::size_t size)
    {
        assert(input != nullptr && mInputLayer.GetElementCount() == static_cast<std::uint32_t>(size));

        Matrix inputMatrix(static_cast<std::uint32_t>(size), 1);
        for (std::uint32_t row = 0; row < inputMatrix.GetRowCount(); ++row)
        {
            for (std::uint32_t col = 0; col < inputMatrix.GetColCount(); ++col)
            {
                inputMatrix.SetElement(row, col) = input[row];
            }
        }
        return FeedForward(inputMatrix);
    }

    ////////////////////////////////////////
    std::uint32_t ANN::GetParameterCount() const
    {
        std::uint32_t count{};

        count += mInputLayer.GetElementCount();
        for (const Matrix& hiddenLayer : mHiddenLayers)
        {
            count += hiddenLayer.GetElementCount();
        }
        count += mOutputLayer.GetElementCount();

        for (const Matrix& weightMatrix : mWeights)
        {
            count += weightMatrix.GetElementCount();
        }

        return count;
    }

    ////////////////////////////////////////
    static Matrix ComputeCorrectOutput(Tools::DatumType type)
    {
        Matrix output(3, 1);
        switch (type)
        {
            case Tools::DatumType::TrainNormal:
            case Tools::DatumType::TestNormal:
                output.SetElement(0, 0) = 0.99f;
                output.SetElement(1, 0) = 0.01f;
                output.SetElement(2, 0) = 0.01f;
                break;
            case Tools::DatumType::TrainBacteria:
            case Tools::DatumType::TestBacteria:
                output.SetElement(0, 0) = 0.01f;
                output.SetElement(1, 0) = 0.99f;
                output.SetElement(2, 0) = 0.01f;
                break;
            case Tools::DatumType::TrainVirus:
            case Tools::DatumType::TestVirus:
                output.SetElement(0, 0) = 0.01f;
                output.SetElement(1, 0) = 0.01f;
                output.SetElement(2, 0) = 0.99f;
                break;
        }
        return output;
    }

    ////////////////////////////////////////
    static Matrix TestForSpecificCategory(ANN& ann, const Tools::DataLoader& dataLoader, Tools::DatumType category)
    {
        assert(Tools::DatumType::TestNormal == category ||
               Tools::DatumType::TestBacteria == category ||
               Tools::DatumType::TestVirus == category);

        std::size_t count = [&]() -> std::size_t
        {
            switch (category)
            {
                case Tools::DatumType::TestNormal:
                    return dataLoader.GetTestNormalCount();
                case Tools::DatumType::TestBacteria:
                    return dataLoader.GetTestBacteriaCount();
                case Tools::DatumType::TestVirus:
                    return dataLoader.GetTestVirusCount();
                case Tools::DatumType::TrainNormal:
                case Tools::DatumType::TrainBacteria:
                case Tools::DatumType::TrainVirus:
                default:
                    return 0;
            }
        }();

        Matrix error(ann.GetOutputNeuronCount(), 1);
        for (std::size_t i = 0; i < count; ++i)
        {
            Matrix actualOutput = ann.FeedForward(dataLoader.GetImage(category, i), dataLoader.GetSize());
            Matrix expectedOutput = ComputeCorrectOutput(category);

            error += expectedOutput - actualOutput;
        }
        return error;
    }

    ////////////////////////////////////////
    static Matrix TestANN(ANN& ann, const Tools::DataLoader& dataLoader)
    {
#ifdef _DEBUG
        std::printf("Starting testing ...\n");
#endif

        Matrix errorMatrix(ann.GetOutputNeuronCount(), 1);

        errorMatrix += TestForSpecificCategory(ann, dataLoader, Tools::DatumType::TestNormal);
        errorMatrix += TestForSpecificCategory(ann, dataLoader, Tools::DatumType::TestBacteria);
        errorMatrix += TestForSpecificCategory(ann, dataLoader, Tools::DatumType::TestVirus);

#ifdef _DEBUG
        std::printf("Testing has finished\n");
#endif

        return errorMatrix;
    }
}

////////////////////////////////////////
int main(int argc, char** argv)
{
    Tools::DataLoader dataLoader("PneumoniaData");

    std::uint32_t inputLayerNeuronCount{ static_cast<std::uint32_t>(dataLoader.GetSize()) };
    std::uint32_t outputLayerNeuronCount{ static_cast<std::uint32_t>(dataLoader.GetCategoryCount()) };
    std::uint32_t hiddenLayerNeuronCount{ 100 };
    std::uint32_t hiddenLayerCount{ 2 };

    App::ANN ann(inputLayerNeuronCount, outputLayerNeuronCount, hiddenLayerNeuronCount, hiddenLayerCount);

    App::Matrix errorMatrix = App::TestANN(ann, dataLoader);
    errorMatrix.Print();

    return 0;
}
