#ifndef DENSE_NETWORK_HPP
#define DENSE_NETWORK_HPP

#include <memory>
#include <cassert>
#include "dense_layer.hpp"

namespace Dense
{
    class DenseNetwork
    {
        typedef double float_t;

    public:
        DenseNetwork() = default;

        explicit DenseNetwork(
            size_t layerCnt,
            size_t inputLen,
            size_t outputLen,
            size_t hiddenLen
        )   :   layerCnt(layerCnt),
                inputLen(inputLen),
                outputSize(outputLen),
                hiddenLayer(hiddenLen)
        {
            assert(layerCnt > 0);
            if (layerCnt == 1)
                outputLayer = std::make_unique<DenseLayer>(hiddenLen, outputLen);
            else
            {
                outputLayer = std::make_unique<DenseLayer>(hiddenLen, outputLen);
                auto* curLayer = &outputLayer->getPrev();
                for (size_t i = 0; i < layerCnt - 2; ++i)
                {
                    auto hidden_layer = std::make_unique<DenseLayer>(hiddenLen, hiddenLen);
                    *curLayer = std::move(hidden_layer);
                    curLayer = &(*curLayer)->getPrev();
                }

                auto firstLayer = std::make_unique<DenseLayer>(inputLen, hiddenLen);
                *curLayer = std::move(firstLayer);
            }
        }

        ~DenseNetwork() = default;

    private:
        size_t layerCnt,
            inputLen, outputSize, hiddenLayer;
        std::unique_ptr<DenseLayer> outputLayer;
        
    };
} // namespace Dense

#endif /* end of include guard :  DENSE_NETWORK_HPP */