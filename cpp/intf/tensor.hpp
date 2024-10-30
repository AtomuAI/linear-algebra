// Copyright 2024 Shane W. Mulcahy

#ifndef LINEAR_ALGEBRA_INTF_TENSOR_HPP_
#define LINEAR_ALGEBRA_INTF_TENSOR_HPP_

namespace atom {

    enum class MemoryType {
        Stack,
        Heap
    };

    enum class MemoryLocation {
        Host,
        Device
    };

    namespace la {
        template <typename T, u8 N, MemoryType M, MemoryLocation L>
        class Tensor;

        template <typename T, u8 N, MemoryType M, MemoryLocation L>
        class Tensor {
            typename std::conditional<is_bool<T>::value, std::vector<T>, std::array<T>>::type  data;
        };
    } // namespace la

} // namespace atom

#endif // LINEAR_ALGEBRA_INTF_TENSOR_HPP_