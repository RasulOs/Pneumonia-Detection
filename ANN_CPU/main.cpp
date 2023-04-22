#include <cstdio>
#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>

////////////////////////////////////////
struct Layer
{
private:
    std::vector<float> mNeurons;

public:
    Layer(std::size_t neuronCount, float lowest = 0.0f, float highest = 1.0f);

    std::size_t GetNeuronCount() const { return mNeurons.size(); }
    void Print() const { std::ranges::for_each(mNeurons, [](auto x){ std::printf("%.2f\n", x); }); }
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
int main(int argc, char** argv)
{
    Layer initialLayer(10);
    initialLayer.Print();

    return 0;
}
