#include <iostream>
#include <memory>
#include "../include/dense_layer.hpp"
#include "../include/dense_network.hpp"


int main(int argc, const char* argv[])
{
    // layer1 -> layer2
    // Dense::DenseLayer layer1(3, 2);
    // Dense::DenseLayer layer2(2, 1);
    // auto input = std::make_unique<double[]>(3);
    // input[0] = 0.5;
    // input[1] = 0.25;
    // input[2] = 0.125;

    Dense::DenseNetwork network(3, 2, 1, 3);
    auto in = std::make_unique<double[]>(2);
    auto expected = std::make_unique<double[]>(1);

    in[0] = 0;
    in[1] = 1;
    expected[0] = 1;

    std::cout << network.forwardPropError(in, expected) << std::endl;
}